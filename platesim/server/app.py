"""FastAPI application: models registry, job queue, sweeps, shared data, WebSocket."""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import platform
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import platesim
from .jobs import (
    FINISHED_STATES,
    MAX_SWEEP_JOBS,
    JobManager,
)
from .params import validate_params
from .registry import Registry

DATA_ROOT = Path(platesim.__file__).resolve().parent / "data"

# Electron (file:// / app://) and Vite dev servers (http://localhost:*, 127.0.0.1:*).
CORS_ORIGIN_REGEX = r"^(http://(localhost|127\.0\.0\.1)(:\d+)?|file://.*|app://.*)$"

_LIT_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def resolve_data_dir(explicit: Optional[str | Path] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return (Path(base) / "platesim").resolve()
    return (Path.cwd() / "runs").resolve()


def _json_file(path: Path) -> JSONResponse:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"{path.name} is not valid JSON: {exc}") from exc
    return JSONResponse(data)


def create_app(data_dir: Optional[str | Path] = None, ready_port: Optional[int] = None,
               max_workers: Optional[int] = None) -> FastAPI:
    resolved_data_dir = resolve_data_dir(data_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        registry = Registry()
        registry.load()
        app.state.registry = registry
        manager = JobManager(resolved_data_dir, max_workers=max_workers)
        manager.start()
        app.state.manager = manager
        app.state.data_dir = resolved_data_dir
        if ready_port is not None:
            # Single machine-readable line the Electron app waits for; uvicorn has
            # bound the socket before lifespan startup runs.
            print(f"PLATESIM_SERVER_READY port={ready_port}", flush=True)
        yield
        manager.shutdown()

    app = FastAPI(title="platesim server", version=platesim.__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=CORS_ORIGIN_REGEX,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def registry_of() -> Registry:
        return app.state.registry

    def manager_of() -> JobManager:
        return app.state.manager

    def _spec_params(model_id: str) -> tuple[Any, list[dict]]:
        entry = registry_of().get(model_id)
        if entry is None:
            raise HTTPException(404, f"unknown model '{model_id}'")
        return entry, entry.spec.get("params") or []

    # ------------------------------------------------------------------ health

    @app.get("/api/health")
    def health() -> dict:
        reg = registry_of()
        return {
            "version": platesim.__version__,
            "python": platform.python_version(),
            "models_loaded": sorted(reg.models.keys()),
            "models_failed": [{"package": e["package"], "error": e["error"]} for e in reg.errors],
        }

    # ------------------------------------------------------------------ models

    @app.get("/api/models")
    def list_models() -> list[dict]:
        return registry_of().spec_list()

    @app.get("/api/models/errors")
    def model_errors() -> dict:
        return {"errors": registry_of().errors}

    @app.get("/api/models/{model_id}")
    def get_model(model_id: str) -> dict:
        entry, _ = _spec_params(model_id)
        spec = dict(entry.spec)
        spec["package"] = entry.package
        return spec

    @app.post("/api/models/{model_id}/run", status_code=202)
    def run_model(model_id: str, payload: Optional[dict] = Body(default=None)) -> dict:
        entry, spec_params = _spec_params(model_id)
        clean, errors = validate_params(spec_params, payload or {})
        if errors:
            raise HTTPException(422, {"message": "invalid parameters", "errors": errors})
        job = manager_of().submit(model_id, entry.package, clean)
        return {"job_id": job.id, "status": job.status}

    # ------------------------------------------------------------------- sweeps

    @app.post("/api/models/{model_id}/sweep")
    def start_sweep(model_id: str, body: dict) -> dict:
        entry, spec_params = _spec_params(model_id)
        base = body.get("base") or {}
        vary = body.get("vary") or {}
        if not isinstance(base, dict):
            raise HTTPException(422, {"message": "'base' must be an object", "errors": []})
        if not isinstance(vary, dict) or not (1 <= len(vary) <= 2):
            raise HTTPException(422, {
                "message": "'vary' must map 1 or 2 parameter names to non-empty value lists",
                "errors": [],
            })
        spec_by_name = {p.get("name"): p for p in spec_params}
        errors: list[str] = []
        for key, values in vary.items():
            if key not in spec_by_name:
                errors.append(f"unknown parameter '{key}' in 'vary'")
            if not isinstance(values, list) or not values:
                errors.append(f"'vary.{key}' must be a non-empty list of values")
        if errors:
            raise HTTPException(422, {"message": "invalid sweep", "errors": errors})
        combos = 1
        for values in vary.values():
            combos *= len(values)
        if combos > MAX_SWEEP_JOBS:
            raise HTTPException(422, {
                "message": f"sweep would create {combos} jobs; max is {MAX_SWEEP_JOBS}",
                "errors": [],
            })
        # Validate every combination (base + vary values) against the spec.
        keys = list(vary.keys())
        for combo in itertools.product(*(vary[k] for k in keys)):
            merged = dict(base)
            merged.update(dict(zip(keys, combo)))
            _, errs = validate_params(spec_params, merged)
            if errs:
                raise HTTPException(422, {
                    "message": f"invalid parameters for sweep value set {merged}",
                    "errors": errs,
                })
        clean_base, _ = validate_params(spec_params, base)
        clean_vary: dict[str, list] = {}
        for key, values in vary.items():
            clean_vals = []
            for v in values:
                single, errs = validate_params(spec_params, {**base, key: v})
                if errs:  # pragma: no cover - already validated above
                    raise HTTPException(422, {"message": "invalid sweep", "errors": errs})
                clean_vals.append(single[key])
            clean_vary[key] = clean_vals
        sweep_id, job_ids = manager_of().create_sweep(model_id, entry.package, clean_base, clean_vary)
        return {"sweep_id": sweep_id, "job_ids": job_ids, "n_jobs": len(job_ids)}

    @app.get("/api/sweeps")
    def list_sweeps() -> dict:
        return {"sweeps": manager_of().list_sweeps()}

    @app.get("/api/sweeps/{sweep_id}")
    def get_sweep(sweep_id: str) -> dict:
        status = manager_of().sweep_status(sweep_id)
        if status is None:
            raise HTTPException(404, f"unknown sweep '{sweep_id}'")
        return status

    # --------------------------------------------------------------------- jobs

    @app.get("/api/jobs")
    def list_jobs() -> dict:
        return {"jobs": [j.to_public() for j in manager_of().list_jobs()]}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = manager_of().get(job_id)
        if job is None:
            raise HTTPException(404, f"unknown job '{job_id}'")
        return job.to_public()

    @app.get("/api/jobs/{job_id}/result")
    def get_job_result(job_id: str) -> dict:
        job = manager_of().get(job_id)
        if job is None:
            raise HTTPException(404, f"unknown job '{job_id}'")
        return {
            "job_id": job.id,
            "model_id": job.model_id,
            "status": job.status,
            "error": job.error,
            "result": job.result,
        }

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        mgr = manager_of()
        job = mgr.get(job_id)
        if job is None:
            raise HTTPException(404, f"unknown job '{job_id}'")
        if job.status in FINISHED_STATES:
            raise HTTPException(409, f"job is already {job.status}")
        job = mgr.cancel(job_id)
        return {"job_id": job.id, "status": job.status, "message": job.message}

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> dict:
        mgr = manager_of()
        if mgr.get(job_id) is None:
            raise HTTPException(404, f"unknown job '{job_id}'")
        try:
            mgr.delete(job_id)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"deleted": job_id}

    # -------------------------------------------------------------- shared data

    @app.get("/api/data/planar_doe")
    def data_planar_doe() -> JSONResponse:
        path = DATA_ROOT / "planar_doe.json"
        if not path.is_file():
            raise HTTPException(404, "platesim/data/planar_doe.json not found")
        return _json_file(path)

    @app.get("/api/data/literature")
    def data_literature() -> dict:
        lit_dir = DATA_ROOT / "literature"
        files = []
        if lit_dir.is_dir():
            for path in sorted(lit_dir.glob("*.json")):
                entry: dict[str, Any] = {"id": path.stem, "file": path.name, "source": None}
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        entry["source"] = data.get("source")
                        entry["tag"] = data.get("tag")
                except (OSError, json.JSONDecodeError) as exc:
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                files.append(entry)
        return {"files": files}

    @app.get("/api/data/literature/{name}")
    def data_literature_item(name: str) -> JSONResponse:
        fname = name if name.endswith(".json") else f"{name}.json"
        if not _LIT_NAME_RE.match(fname):
            raise HTTPException(400, "invalid literature file name")
        path = DATA_ROOT / "literature" / fname
        if not path.is_file():
            raise HTTPException(404, f"literature file '{fname}' not found")
        return _json_file(path)

    @app.get("/api/data/sem")
    def data_sem() -> JSONResponse:
        path = DATA_ROOT / "sem_measurements.json"
        if not path.is_file():
            raise HTTPException(404, "platesim/data/sem_measurements.json not found (yet)")
        return _json_file(path)

    # ---------------------------------------------------------------- websocket

    @app.websocket("/api/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        reg = registry_of()
        mgr = manager_of()
        cursor = mgr.current_seq()
        await ws.send_json({
            "type": "hello",
            "version": platesim.__version__,
            "models_loaded": sorted(reg.models.keys()),
        })
        try:
            while True:
                events = mgr.events_after(cursor)
                if events:
                    for seq, ev in events:
                        cursor = seq
                        await ws.send_json(ev)
                else:
                    await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            pass
        except Exception:
            try:
                await ws.close()
            except Exception:
                pass

    # web mode: serve the built desktop UI (app/out/renderer) at "/" so a browser can use it
    ui_dir = Path(__file__).resolve().parents[2] / "app" / "out" / "renderer"
    if ui_dir.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    return app


# Default app instance for `uvicorn platesim.server.app:app`; the data dir is
# resolved at lifespan start, so importing this module stays side-effect free.
app = create_app()
