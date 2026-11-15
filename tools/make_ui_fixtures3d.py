"""Generate synthetic UI fixtures through the real HTTP API and worker pool.

Run from the repository root with PYTHONPATH=. and the spec/layout/viz dependencies.
No cached or hand-written solver responses are used.
"""
import argparse
import base64
import json
from pathlib import Path
import tempfile
import time

from fastapi.testclient import TestClient

from platesim.server.app import create_app


def generate(output):
    fixtures = {}
    with tempfile.TemporaryDirectory(prefix="platesim-ui3d-") as temporary:
        with TestClient(create_app(temporary, max_workers=1)) as client:
            def record(method, endpoint, *, key=None, **kwargs):
                response = client.request(method, "/api/3d" + endpoint, **kwargs)
                response.raise_for_status()
                value = response.json()
                fixtures[key or f"{method} /api/3d{endpoint}"] = value
                return value

            record("GET", "/schema")
            examples = record("GET", "/examples")
            spec = next(e["spec"] for e in examples["examples"] if e["name"] == "quick_trench")
            record("POST", "/validate", json={"spec": spec})
            record("POST", "/validate", key="POST /api/3d/validate errors",
                   json={"spec": dict(spec, numerics={"quality": "invalid"})})
            record("POST", "/plan", json={"spec": spec})
            rid = record("POST", "/runs", json={"spec": spec, "label": "Synthetic quick trench"})["run_id"]
            base = f"/runs/{rid}"
            deadline = time.monotonic() + 120
            while True:
                state = client.get("/api/3d" + base).json()
                if state["status"] in ("done", "failed", "cancelled"):
                    if state["status"] != "done":
                        raise RuntimeError(state)
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("quick_trench exceeded 120 s")
                time.sleep(.05)
            record("GET", "/runs")
            record("GET", base, key="GET /api/3d/runs/{id}")
            for suffix in ("manifest", "tiles", "height", "series", "frames"):
                record("GET", f"{base}/{suffix}", key=f"GET /api/3d/runs/{{id}}/{suffix}")
            h = fixtures["GET /api/3d/runs/{id}/height"]
            y = h["y_um"][len(h["y_um"])//2]
            record("POST", base + "/section", key="POST /api/3d/runs/{id}/section",
                   json={"polyline_um": [[h["x_um"][0], y], [h["x_um"][-1], y]]})
            for art in fixtures["GET /api/3d/runs/{id}/manifest"]["artifacts"]:
                if art["kind"] == "png":
                    response = client.get("/api/3d" + base + "/artifact", params={"path": art["path"]})
                    response.raise_for_status()
                    if len(response.content) < 300_000:
                        fixtures["GET /api/3d/runs/{id}/artifact"] = dict(
                            path=art["path"], media_type=response.headers["content-type"],
                            base64=base64.b64encode(response.content).decode("ascii"))
                        break
            second = client.post("/api/3d/runs", json={"spec": spec}).json()["run_id"]
            record("POST", f"/runs/{second}/cancel", key="POST /api/3d/runs/{id}/cancel")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixtures, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {output} ({len(fixtures)} responses)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("app/mock/fixtures3d.json"))
    generate(parser.parse_args().out)
