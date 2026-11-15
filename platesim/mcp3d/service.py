"""Transport-independent service. Solver imports occur only on demand."""
from __future__ import annotations

from functools import wraps
import json
import math
from pathlib import Path
import re
from threading import RLock, Thread


def _error(exc, field="run_id", fix="Check the run ID and wait until the run is done."):
    return {"error": str(exc), "field": field, "fix": fix}


def _safe(value):
    """Convert numpy scalars and missing crossings to strict JSON."""
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _guard(field="run_id", fix="Check the run ID and wait until the run is done."):
    def decorate(method):
        @wraps(method)
        def call(*args, **kwargs):
            try:
                return _safe(method(*args, **kwargs))
            except Exception as exc:
                return _error(exc, field, fix)
        return call
    return decorate


class RunService:
    """Manage background runs below root; injected functions use sample3d signatures.

    Active state is process-local. Completed directories survive restarts; partial
    directories after a restart are reported failed, never mistaken for running.
    """

    def __init__(self, root, runner=None, planner=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._runner, self._planner = runner, planner
        self._lock = RLock()
        self._plot_lock = RLock()
        self._states = {}

    def _path(self, run_id):
        if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{12}-[1-9][0-9]*", run_id):
            raise ValueError("Invalid run_id; expected twelve hex digits followed by -counter")
        path = self.root / run_id
        if path.resolve().parent != self.root or path.is_symlink():
            raise ValueError("Run directory must be directly below root, not a symlink")
        if not path.is_dir():
            raise ValueError("Unknown run_id")
        return path

    def _file(self, path):
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Run artifact escapes root")
        return path

    @_guard("spec", "Supply a JSON object matching SampleSpec.")
    def validate_spec(self, spec: dict) -> dict:
        """Validate a SampleSpec JSON object; return ok and actionable field errors."""
        from platesim.spec3d import validate_spec
        if not isinstance(spec, dict):
            return {"ok": False, "errors": [{"field": "spec", "value": _safe(spec),
                    "message": "Expected a JSON object", "fix": "Supply a spec object."}]}
        _, errors, _ = validate_spec(spec)
        return {"ok": not errors, "errors": [
            {"field": e["path"], "value": e["value"], "message": e["allowed"], "fix": e["fix"]}
            for e in errors]}

    def _model(self, spec):
        from platesim.spec3d import SampleSpec
        if not isinstance(spec, dict):
            raise ValueError("spec must be a JSON object")
        return SampleSpec.model_validate(spec)

    @_guard("spec", "Call validate_spec, fix its errors, or reduce tile size / increase dx_um.")
    def plan_run(self, spec: dict) -> dict:
        """Estimate tile geometry, cell count and time in seconds without running a solver."""
        model = self._model(spec)
        planner = self._planner
        if planner is None:
            from platesim.sample3d import plan_tile
            planner = plan_tile
        planned = planner(model)
        if isinstance(planned, list):
            tiles = [dict(tile) for tile in planned]
            for tile in tiles:
                if "est_seconds" in tile:
                    tile["est_time_s"] = tile.pop("est_seconds")
            return {"tiles": tiles, "n_cells": sum(t["n_cells"] for t in tiles),
                    "est_time_s": sum(t["est_time_s"] for t in tiles)}
        plan = dict(planned)
        # Section 10b uses est_seconds; agent quantities carry a suffix unit.
        if "est_seconds" in plan:
            plan["est_time_s"] = plan.pop("est_seconds")
        return plan

    @_guard("spec", "Call validate_spec and plan_run first; check sample3d is installed.")
    def run_sample(self, spec: dict) -> dict:
        """Start a background sample run; return run_id immediately. Poll get_status."""
        from platesim.spec3d import spec_hash
        model = self._model(spec)
        runner = self._runner
        if runner is None:
            from platesim.sample3d import run_sample
            runner = run_sample
        prefix = spec_hash(model)[:12]
        with self._lock:
            counter = 1
            while True:
                run_id = f"{prefix}-{counter}"
                path = self.root / run_id
                try:
                    path.mkdir()
                    break
                except FileExistsError:
                    counter += 1
            self._states[run_id] = dict(state="queued", progress=0., message="Queued", error=None)
            thread = Thread(target=self._work, args=(run_id, model, path, runner), daemon=True)
            try:
                thread.start()
            except Exception as exc:
                self._states[run_id].update(state="failed", message="Could not start worker", error=str(exc))
                raise
        return {"run_id": run_id}

    def _work(self, run_id, model, path, runner):
        def progress(fraction, message=""):
            fraction = float(fraction)
            if not math.isfinite(fraction):
                return
            with self._lock:
                state = self._states[run_id]
                state.update(progress=max(state["progress"], min(1., max(0., fraction))), message=str(message))
        with self._lock:
            self._states[run_id].update(state="running", message="Running")
        try:
            runner(model, path, progress=progress)
            from platesim.spec3d import Manifest
            Manifest.model_validate_json(self._file(path / "manifest.json").read_text(encoding="utf-8"))
            with self._lock:
                self._states[run_id].update(state="done", progress=1., message="Done")
        except Exception as exc:
            with self._lock:
                self._states[run_id].update(state="failed", message="Run failed", error=str(exc))

    @_guard()
    def get_status(self, run_id: str) -> dict:
        """Return queued/running/done/failed, dimensionless progress 0..1, message and error."""
        path = self._path(run_id)
        with self._lock:
            if run_id in self._states:
                return dict(self._states[run_id])
        manifest = self._file(path / "manifest.json")
        if manifest.is_file():
            from platesim.spec3d import Manifest
            Manifest.model_validate_json(manifest.read_text(encoding="utf-8"))
            return dict(state="done", progress=1., message="Completed run on disk", error=None)
        return dict(state="failed", progress=0., message="Interrupted run", error="No completed manifest; submit again.")

    def _completed(self, run_id):
        path = self._path(run_id)
        status = self.get_status(run_id)
        if status.get("state") != "done":
            raise ValueError(status.get("error") or "Run is not done")
        return path

    @_guard()
    def get_manifest(self, run_id: str) -> dict:
        """Read a completed run's manifest, quantities with units, checks and artifacts."""
        path = self._completed(run_id)
        return json.loads(self._file(path / "manifest.json").read_text(encoding="utf-8"))

    def _result(self, run_id):
        from platesim.tile3d.result import TileResult
        path = self._completed(run_id)
        # Result readers follow paths in this directory, so reject escaping links.
        for child in path.rglob("*"):
            self._file(child)
        if not (path / "grid.json").is_file():
            plans = json.loads(self._file(path / "tiles.json").read_text(encoding="utf-8"))
            # The first representative tile is the profile, not a trench-end
            # diagnostic. Keep old single-tile result directories readable too.
            tile_id = plans[0]["tile_id"]
            if not isinstance(tile_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", tile_id):
                raise ValueError("Invalid representative tile_id")
            path = self._file(path / "tiles" / tile_id)
        return TileResult(path)

    @_guard("polyline_um", "Use at least two distinct finite [x_um, y_um] points inside the tile.")
    def cross_section(self, run_id: str, polyline_um: list[list[float]], render: bool = True) -> dict:
        """Summarize a vertical cut along x-y points in um; optionally write a PNG (no fields returned)."""
        import numpy as np
        from platesim.analysis3d import cross_section
        result = self._result(run_id)
        points = np.asarray(polyline_um, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not 2 <= len(points) <= 1024:
            raise ValueError("polyline_um needs 2..1024 [x_um, y_um] points")
        if not np.isfinite(points).all() or np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 0):
            raise ValueError("Polyline vertices must be finite and segments nonzero")
        section = cross_section(result.psi(), result.psi_si, result.grid, points, n_samples=64)
        out = {"length_um": float(section["s_um"][-1]), "z_min_um": float(section["z_um"][0]),
               "z_max_um": float(section["z_um"][-1]), "sample_count": len(section["s_um"]),
               "cu_contour_count": len(section["cu_outline"]), "si_contour_count": len(section["si_outline"])}
        if render:
            from platesim.render3d import render_cross_section
            path = self._file(self._path(run_id) / "mcp_section.png")
            with self._plot_lock:
                render_cross_section(section, path)
            out["png_path"] = str(path)
        return out

    @_guard()
    def height_map(self, run_id: str, include_samples: bool = False) -> dict:
        """Return height stats in um and PNG; optional sampled height_um array is at most 64x64."""
        import numpy as np
        from platesim.analysis3d import height_map
        from platesim.render3d import render_height_map
        result = self._result(run_id)
        height = height_map(result.psi(), result.grid)
        path = self._file(self._path(run_id) / "mcp_height_map.png")
        with self._plot_lock:
            render_height_map(height["height_um"], height["x_um"], height["y_um"], path)
        out = {"stats": height["stats"], "png_path": str(path)}
        if include_samples:
            ix = np.linspace(0, len(height["x_um"])-1, min(64, len(height["x_um"])), dtype=int)
            iy = np.linspace(0, len(height["y_um"])-1, min(64, len(height["y_um"])), dtype=int)
            out.update(height_um=height["height_um"][np.ix_(ix, iy)],
                       x_um=height["x_um"][ix], y_um=height["y_um"][iy])
        return out

    @_guard("kind", "Choose kind surface|height_map|section and camera iso|top|front.")
    def render(self, run_id: str, kind: str = "surface", camera: str = "iso") -> dict:
        """Write a PNG: kind surface|height_map|section, camera iso|top|front (surface only)."""
        if kind not in ("surface", "height_map", "section") or camera not in ("iso", "top", "front"):
            raise ValueError("Invalid kind or camera")
        if kind == "height_map":
            response = self.height_map(run_id)
        elif kind == "section":
            result = self._result(run_id)
            x, y, _ = result.grid.centers()
            response = self.cross_section(run_id, [[float(x[0]*1e4), float(y[len(y)//2]*1e4)],
                                                  [float(x[-1]*1e4), float(y[len(y)//2]*1e4)]])
        else:
            from platesim.render3d import render_surface
            result = self._result(run_id)
            path = self._file(self._path(run_id) / f"mcp_surface_{camera}.png")
            with self._plot_lock:
                render_surface(result, out_png=path, camera=camera)
            return {"path": str(path)}
        if "error" in response:
            return response
        return {"path": response["png_path"]}

    @_guard("root", "Check run-root permissions and contents.")
    def list_runs(self) -> dict:
        """List run IDs and statuses, including completed runs from previous server sessions."""
        runs = []
        for path in sorted(self.root.iterdir()):
            if re.fullmatch(r"[0-9a-f]{12}-[1-9][0-9]*", path.name) and path.is_dir() and not path.is_symlink():
                runs.append({"run_id": path.name, **self.get_status(path.name)})
        return {"runs": runs}
