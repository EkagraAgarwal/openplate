"""Lazy, synchronous HTTP transport for CPU sample jobs (contract 10f)."""
from __future__ import annotations

import json
import base64
import binascii
import csv
import io
import math
import mimetypes
from pathlib import Path
import re
from threading import RLock
from typing import cast
import uuid
import tempfile

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response

from platesim.mcp3d.service import RunService, _safe

router = APIRouter(prefix="/api/3d")
_index_lock = RLock()
_PACKAGE = "platesim.sample3d.job"


def _bad(message, field="spec", fix="Supply a valid SampleSpec; call /validate first."):
    return HTTPException(422, {"error": str(message), "field": field, "fix": fix})


def _root(request):
    root = Path(request.app.state.data_dir) / "runs3d"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _index(request):
    path = _root(request) / "index.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _service(request):
    return RunService(_root(request))


def _validation(request, spec):
    answer = _service(request).validate_spec(spec)
    if "error" in answer:
        raise _bad(answer["error"])
    return {"ok": answer["ok"], "errors": [
        {k: e[k] for k in ("field", "message", "fix")} for e in answer["errors"]]}


def _spec(request, body):
    spec = body.get("spec")
    validation = _validation(request, spec)
    if not validation["ok"]:
        e = validation["errors"][0]
        raise _bad(e["message"], e["field"], e["fix"])
    return _service(request)._model(spec).model_dump(mode="json")


def _record(request, run_id):
    if not re.fullmatch(r"[0-9a-f-]+", run_id):
        raise HTTPException(404, "Unknown run")
    with _index_lock:
        row = _index(request).get(run_id)
    if row is None:
        raise HTTPException(404, "Unknown run")
    return row


def _status(request, run_id, row=None) -> dict:
    row = row or _record(request, run_id)
    job = request.app.state.manager.get(row["job_id"])
    if job is not None:
        public = job.to_public()
        state = {k: public[k] for k in ("status", "progress", "message", "error", "created", "finished")}
    else:
        # Active jobs are not persisted by JobManager. Never claim they are still running.
        state = dict(status="failed", progress=0., message="Interrupted run",
                     error="Server restarted before job completed; submit again.",
                     created=row["created"], finished=None)
    return cast(dict, _safe(dict(run_id=run_id, label=row["label"], spec=row["spec"], **state)))


class _CompletedRun(RunService):
    """Reuse MCP's guarded result reader without changing its ID or response contracts."""
    def _completed(self, run_id):
        return self.root


def _completed(request, run_id):
    row = _record(request, run_id)
    if _status(request, run_id, row)["status"] != "done":
        raise HTTPException(409, "Run is not finished")
    root = _root(request)
    directory = root / row["directory"]
    if directory.is_symlink() or directory.resolve().parent != root:
        raise HTTPException(404, "Invalid run directory")
    return _CompletedRun(directory)


def _read(service, name):
    return json.loads(service._file(service.root / name).read_text(encoding="utf-8"))


def _result(request, run_id, tile_id):
    service = _completed(request, run_id)
    plans = _read(service, "tiles.json")
    selected = next((p for p in plans if p["tile_id"] == tile_id), None) if tile_id else plans[0]
    if selected is None:
        raise _bad("Unknown tile_id", "tile_id", "Choose a tile from /tiles.")
    try:
        if selected == plans[0]:
            result = service._result(run_id)
        else:
            from platesim.tile3d.result import TileResult
            if not re.fullmatch(r"[A-Za-z0-9_-]+", selected["tile_id"]):
                raise ValueError("Invalid tile_id")
            path = service._file(service.root / "tiles" / selected["tile_id"])
            for child in path.rglob("*"):
                service._file(child)
            result = TileResult(path)
    except ValueError as exc:
        raise _bad(exc, "tile_id", "Choose a valid tile from /tiles.") from exc
    return result, selected


@router.get("/schema")
def schema():
    from platesim.spec3d import SampleSpec
    return {"spec_schema": SampleSpec.model_json_schema()}


@router.post("/layout/import")
def import_layout(body: dict):
    filename = body.get("filename")
    if not isinstance(filename, str) or not filename.strip():
        raise _bad("filename must be nonempty text", "filename", "Supply the original filename.")
    filename = re.sub(r"[^A-Za-z0-9_.-]", "_", filename.replace("\\", "/").split("/")[-1])
    suffix = Path(filename).suffix.lower()
    if suffix not in (".gds", ".dxf", ".png", ".tif", ".tiff", ".webp"):
        raise _bad("Unsupported layout extension", "filename", "Use GDS, DXF, PNG, TIFF or WEBP.")
    content = body.get("content_b64")
    limit = 20 * 1024 * 1024
    if not isinstance(content, str) or len(content) > 4*((limit+2)//3):
        raise _bad("Layout must be base64 and decoded size <= 20 MB", "content_b64", "Supply a smaller base64 file.")
    try:
        decoded = base64.b64decode(content, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise _bad(exc, "content_b64", "Supply valid base64 file contents.") from exc
    if len(decoded) > limit:
        raise _bad("Decoded layout exceeds 20 MB", "content_b64", "Supply a smaller file.")
    layer = body.get("layer", "etch")
    if not isinstance(layer, str) or not layer:
        raise _bad("layer must be nonempty text", "layer", "Use a named layer such as etch.")
    mapping = body.get("layer_map", {layer: 0 if suffix == ".gds" else "0"})
    if not isinstance(mapping, dict):
        raise _bad("layer_map must be an object", "layer_map", "Map named layers to source layer numbers/names.")
    try:
        from platesim.layout import import_gds, import_dxf, import_image
        from platesim.layout.to_features import fit_features
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / filename
            path.write_bytes(decoded)
            if suffix == ".gds":
                layout = import_gds(path, mapping)
                kind = "gds"
            elif suffix == ".dxf":
                layout = import_dxf(path, mapping)
                kind = "dxf"
            else:
                layout = import_image(path, body.get("pixel_um"), layer=layer)
                kind = "image"
            if not layout.features:
                raise ValueError("No polygons found; check layer_map or image intensity")
            answer = fit_features(layout, layer=layer)
            answer["source"] = dict(filename=filename, kind=kind, n_polygons=len(layout.features))
            return _safe(answer)
    except (ValueError, TypeError, OSError, ImportError, RuntimeError) as exc:
        raise _bad(exc, "layout", "Check file contents, layer_map, pixel_um and installed layout dependencies.") from exc


@router.get("/examples")
def examples():
    root = Path(__file__).resolve().parents[1] / "spec3d" / "examples"
    return {"examples": [dict(name=p.stem, title=p.stem.replace("_", " ").title(),
                              spec=json.loads(p.read_text(encoding="utf-8")))
                         for p in sorted(root.glob("*.json"))]}


@router.post("/validate")
def validate(request: Request, body: dict):
    return _validation(request, body.get("spec"))


@router.post("/plan")
def plan(request: Request, body: dict):
    validation = _validation(request, body.get("spec"))
    empty = dict(tiles=[], n_cells=0, est_time_s=0.)
    if not validation["ok"]:
        return dict(**validation, **empty)
    planned = _service(request).plan_run(body["spec"])
    if "error" in planned:
        error = dict(field=planned["field"], message=planned["error"], fix=planned["fix"])
        return dict(ok=False, errors=[error], **empty)
    keys = ("tile_id", "kind", "feature_ids", "shape", "dx_um", "n_cells", "est_time_s",
            "weight", "assumptions", "flags")
    return _safe(dict(ok=True, errors=[], tiles=[{k: t[k] for k in keys} for t in planned["tiles"]],
                      n_cells=planned["n_cells"], est_time_s=planned["est_time_s"]))


@router.post("/runs", status_code=202)
def start_run(request: Request, body: dict):
    spec = _spec(request, body)
    label = body.get("label", "")
    if not isinstance(label, str):
        raise _bad("label must be a string", "label", "Supply a text label.")
    with _index_lock:
        root = _root(request)
        directory = uuid.uuid4().hex
        out = root / directory
        out.mkdir()
        job = request.app.state.manager.submit("sample3d", _PACKAGE, {"spec": spec, "out_dir": str(out)})
        rows = _index(request)
        rows[job.id] = dict(job_id=job.id, directory=directory, label=label, spec=spec,
                            created=job.to_public()["created"])
        tmp = root / "index.json.tmp"
        tmp.write_text(json.dumps(rows, allow_nan=False), encoding="utf-8")
        tmp.replace(root / "index.json")
    return {"run_id": job.id}


@router.get("/runs")
def runs(request: Request):
    with _index_lock:
        rows = _index(request)
    keys = ("run_id", "label", "status", "progress", "message", "created", "finished")
    values = [_status(request, rid, row) for rid, row in reversed(rows.items())]
    values.sort(key=lambda r: r["created"] or "", reverse=True)
    return {"runs": [{k: r[k] for k in keys} for r in values]}


@router.get("/runs/{run_id}")
def status(request: Request, run_id: str):
    answer = _status(request, run_id)
    return {k: v for k, v in answer.items() if k not in ("created", "finished")}


@router.post("/runs/{run_id}/cancel")
def cancel(request: Request, run_id: str):
    row = _record(request, run_id)
    if request.app.state.manager.get(row["job_id"]) is not None:
        request.app.state.manager.cancel(row["job_id"])
    return status(request, run_id)


@router.get("/runs/{run_id}/manifest")
def manifest(request: Request, run_id: str):
    return _safe(_read(_completed(request, run_id), "manifest.json"))


@router.get("/runs/{run_id}/tiles")
def tiles(request: Request, run_id: str):
    from platesim.tile3d.grid import Grid3D
    service = _completed(request, run_id)
    output = []
    for tile in _read(service, "tiles.json"):
        grid = Grid3D.from_dict(_read(service, f'tiles/{tile["tile_id"]}/grid.json'))
        row = {k: tile[k] for k in ("tile_id", "kind", "feature_ids", "weight", "shape", "dx_um")}
        row.update({a + "_um": [grid.origin[i]*1e4, (grid.origin[i] + grid.shape[i]*grid.dx)*1e4]
                    for i, a in enumerate("xy")})
        output.append(row)
    return _safe({"tiles": output})


def _frame(result, frame):
    try:
        return result._index(frame)
    except (IndexError, TypeError) as exc:
        raise _bad(exc, "frame", "Choose an index from /frames (or -1 for last).") from exc


@router.get("/runs/{run_id}/height")
def height(request: Request, run_id: str, tile_id: str | None = None, frame: int = -1,
           max_n: int = Query(default=200, ge=1, le=400)):
    import numpy as np
    from platesim.analysis3d.maps import height_map
    result, tile = _result(request, run_id, tile_id)
    index = _frame(result, frame)
    mapped = height_map(result.psi(index), result.grid)
    bare = height_map(result.psi_si, result.grid)
    ix, iy = [np.linspace(0, n-1, min(max_n, n), dtype=int) for n in result.grid.shape[:2]]
    return _safe(dict(tile_id=tile["tile_id"], frame=index, t_s=result.times_s[index],
                      x_um=mapped["x_um"][ix], y_um=mapped["y_um"][iy],
                      height_um=mapped["height_um"][np.ix_(ix, iy)],
                      bare_height_um=bare["height_um"][np.ix_(ix, iy)], stats=mapped["stats"]))


@router.post("/runs/{run_id}/section")
def section(request: Request, run_id: str, body: dict):
    import numpy as np
    from platesim.analysis3d.section import cross_section
    result, _ = _result(request, run_id, body.get("tile_id"))
    frame = body.get("frame", -1)
    if not isinstance(frame, int) or isinstance(frame, bool):
        raise _bad("frame must be an integer", "frame")
    index = _frame(result, frame)
    try:
        points = np.asarray(body.get("polyline_um"), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not 2 <= len(points) <= 1024:
            raise ValueError("Supply 2..1024 [x_um, y_um] points")
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if (not np.isfinite(points).all() or not np.isfinite(lengths).all()
                or not np.isfinite(lengths.sum()) or np.any(lengths <= 0)):
            raise ValueError("Polyline vertices must be finite and segments nonzero")
        n = min(600, max(2, math.ceil(float(lengths.sum()) / (result.grid.dx * 1e4))))
        cut = cross_section(result.psi(index), result.psi_si, result.grid, points, n_samples=n)
        initial = cross_section(result.psi(0), result.psi_si, result.grid, points, n_samples=n)
    except (ValueError, TypeError) as exc:
        raise _bad(exc, "polyline_um", "Use distinct finite points inside the selected tile.") from exc
    return _safe(dict(s_um=cut["s_um"], z_um=cut["z_um"], cu_outline=cut["cu_outline"],
                      si_outline=cut["si_outline"], cu_outline_initial=initial["cu_outline"],
                      length_um=float(cut["s_um"][-1])))


@router.get("/runs/{run_id}/series")
def series(request: Request, run_id: str, tile_id: str | None = None):
    result, _ = _result(request, run_id, tile_id)
    # Unsupported diagnostics are explicit missing values, never fabricated zeros.
    values = dict(result.series)
    for key in ("V", "faraday_volume_cm3", "solid_volume_cm3", "ct_min_front", "cu_balance_rel"):
        values.setdefault(key, [None] * len(result.times_s))
    return _safe(values)


@router.get("/runs/{run_id}/frames")
def frames(request: Request, run_id: str, tile_id: str | None = None):
    result, _ = _result(request, run_id, tile_id)
    return _safe({"t_s": result.times_s})


@router.get("/runs/{run_id}/artifact")
def artifact(request: Request, run_id: str, path: str):
    service = _completed(request, run_id)
    if ".." in Path(path.replace("\\", "/")).parts or path not in {
            a["path"] for a in _read(service, "manifest.json")["artifacts"]}:
        raise HTTPException(404, "Artifact is not listed")
    try:
        file = service._file(service.root / path)
    except ValueError as exc:
        raise HTTPException(404, "Invalid artifact path") from exc
    if not file.is_file():
        raise HTTPException(404, "Artifact not found")
    media = mimetypes.guess_type(str(file))[0] or "application/octet-stream"
    return FileResponse(file, media_type=media)


@router.get("/runs/{run_id}/export")
def export_run(request: Request, run_id: str, kind: str, tile_id: str | None = None,
               frame: int = -1, polyline: str | None = None):
    result, _ = _result(request, run_id, tile_id)
    index = _frame(result, frame)
    if kind not in ("height_csv", "section_csv", "stl"):
        raise _bad("Unknown export kind", "kind", "Use height_csv, section_csv or stl.")
    filename = f"{run_id}_{kind}." + ("stl" if kind == "stl" else "csv")
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    if kind == "stl":
        from platesim.analysis3d.export import export_surface
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "surface.stl"
            export_surface(result.psi(index), result.grid, path, fmt="stl")
            return Response(path.read_bytes(), media_type="model/stl", headers=headers)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    if kind == "height_csv":
        from platesim.analysis3d.maps import height_map
        mapped = _safe(height_map(result.psi(index), result.grid))
        writer.writerow(["x_um", "y_um", "height_um"])
        for i, x in enumerate(mapped["x_um"]):
            for j, y in enumerate(mapped["y_um"]):
                writer.writerow([x, y, mapped["height_um"][i][j]])
    else:
        try:
            points = [[float(v) for v in pair.split(",")] for pair in (polyline or "").split(";")]
        except ValueError as exc:
            raise _bad(exc, "polyline", "Use x0,y0;x1,y1 inside the selected tile.") from exc
        cut = section(request, run_id, dict(polyline_um=points, tile_id=tile_id, frame=frame))
        writer.writerow(["contour", "kind", "s_um", "z_um"])
        for name, key in (("cu", "cu_outline"), ("cu_initial", "cu_outline_initial"), ("si", "si_outline")):
            for contour, points in enumerate(cut[key]):
                for s, z in points:
                    writer.writerow([contour, name, s, z])
    return Response(stream.getvalue(), media_type="text/csv", headers=headers)
