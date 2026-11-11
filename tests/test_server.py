"""End-to-end tests for platesim.server (fastapi.testclient; target < 60 s)."""
from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from platesim.server.app import create_app


def wait_job(client: TestClient, job_id: str, timeout: float = 45.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        job = r.json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not reach a terminal state within {timeout}s")


def test_health_and_models(tmp_path):
    with TestClient(create_app(data_dir=tmp_path / "psdata")) as c:
        h = c.get("/api/health")
        assert h.status_code == 200
        body = h.json()
        assert body["version"]
        assert body["python"]
        assert "demo_faraday" in body["models_loaded"]
        assert isinstance(body["models_failed"], list)

        models = c.get("/api/models").json()
        ids = [m["id"] for m in models]
        assert "demo_faraday" in ids

        spec = c.get("/api/models/demo_faraday").json()
        assert spec["id"] == "demo_faraday"
        param_names = {p["name"] for p in spec["params"]}
        assert {"j_mA_cm2", "t_min", "efficiency"} <= param_names

        errs = c.get("/api/models/errors")
        assert errs.status_code == 200
        assert "errors" in errs.json()

        assert c.get("/api/models/no_such_model").status_code == 404


def test_run_demo_job_and_result_shape(tmp_path):
    with TestClient(create_app(data_dir=tmp_path / "psdata", max_workers=2)) as c:
        r = c.post("/api/models/demo_faraday/run",
                   json={"j_mA_cm2": 25.0, "t_min": 1.0, "efficiency": 0.9})
        assert r.status_code == 202
        job_id = r.json()["job_id"]

        # Poll until finished, remembering that we saw in-flight progress.
        saw_partial_progress = False
        t0 = time.time()
        job: dict = {}
        while time.time() - t0 < 45:
            job = c.get(f"/api/jobs/{job_id}").json()
            if 0.0 < job["progress"] < 1.0:
                saw_partial_progress = True
            if job["status"] in ("done", "failed", "cancelled"):
                break
            time.sleep(0.1)

        assert job["status"] == "done", job.get("error")
        assert job["progress"] == 1.0
        assert saw_partial_progress  # the demo reports intermediate progress
        assert job["started"] and job["finished"]
        assert job["elapsed_s"] >= 1.5
        assert job["error"] is None

        res = c.get(f"/api/jobs/{job_id}/result").json()
        assert res["status"] == "done"
        out = res["result"]
        assert {"summary", "plots", "warnings", "assumptions"} <= set(out)

        # Faraday check: 25 mA/cm2 for 1 min at 90 % -> ~0.50 um
        # (1.323 um/h per mA/cm2 * 25 * 0.9 / 60 = 0.496 um)
        assert 0.45 < out["summary"]["thickness_um"] < 0.55
        assert out["summary"]["efficiency"] == 0.9

        kinds = {p["kind"] for p in out["plots"]}
        assert kinds == {"line", "heatmap", "contours"}
        line = next(p for p in out["plots"] if p["kind"] == "line")
        assert len(line["x"]) == len(line["series"][0]["y"]) == 10
        heat = next(p for p in out["plots"] if p["kind"] == "heatmap")
        assert len(heat["z"]) == len(heat["y"]) and len(heat["z"][0]) == len(heat["x"])
        contours = next(p for p in out["plots"] if p["kind"] == "contours")
        assert len(contours["frames"]) == 3
        assert all(frame["lines"] for frame in contours["frames"])

        # newest-first listing contains the job
        jobs = c.get("/api/jobs").json()["jobs"]
        assert job_id in {j["id"] for j in jobs}


def test_param_validation_errors_422(tmp_path):
    with TestClient(create_app(data_dir=tmp_path / "psdata")) as c:
        # over max
        r = c.post("/api/models/demo_faraday/run", json={"j_mA_cm2": 999.0})
        assert r.status_code == 422
        assert "above max" in json.dumps(r.json())
        # under min
        assert c.post("/api/models/demo_faraday/run", json={"efficiency": -0.5}).status_code == 422
        # wrong type
        assert c.post("/api/models/demo_faraday/run", json={"j_mA_cm2": "abc"}).status_code == 422
        # unknown param
        r = c.post("/api/models/demo_faraday/run", json={"nope": 1})
        assert r.status_code == 422
        assert "unknown parameter" in json.dumps(r.json())
        # unknown model
        assert c.post("/api/models/no_such_model/run", json={}).status_code == 404


def test_sweep_three_values(tmp_path):
    with TestClient(create_app(data_dir=tmp_path / "psdata", max_workers=4)) as c:
        r = c.post("/api/models/demo_faraday/sweep", json={
            "base": {"efficiency": 0.8, "j_mA_cm2": 15.0},
            "vary": {"t_min": [0.5, 1.0, 1.5]},
        })
        assert r.status_code == 200, r.text
        body = r.json()
        sweep_id = body["sweep_id"]
        assert body["n_jobs"] == 3
        assert len(body["job_ids"]) == 3

        t0 = time.time()
        status: dict = {}
        while time.time() - t0 < 50:
            status = c.get(f"/api/sweeps/{sweep_id}").json()
            if status["status"] in ("done", "failed", "cancelled"):
                break
            time.sleep(0.2)

        assert status["status"] == "done"
        assert status["n_jobs"] == 3
        table = status["table"]
        assert len(table) == 3
        for row in table:
            assert row["status"] == "done"
            assert row["params"]["efficiency"] == 0.8
            assert row["summary"] is not None
        tvals = sorted(row["params"]["t_min"] for row in table)
        assert tvals == [0.5, 1.0, 1.5]

        # sweep validation: >2 vary params and bad vary values are rejected
        r = c.post("/api/models/demo_faraday/sweep", json={
            "base": {}, "vary": {"t_min": [1], "j_mA_cm2": [1], "efficiency": [1]},
        })
        assert r.status_code == 422
        r = c.post("/api/models/demo_faraday/sweep", json={
            "base": {}, "vary": {"j_mA_cm2": [1e6]},
        })
        assert r.status_code == 422


def test_persistence_reload(tmp_path):
    data_dir = tmp_path / "psdata"
    with TestClient(create_app(data_dir=data_dir)) as c:
        job_id = c.post("/api/models/demo_faraday/run", json={"t_min": 0.5}).json()["job_id"]
        job = wait_job(c, job_id)
        assert job["status"] == "done"
        files = list((data_dir / "jobs").glob(f"{job_id}*.json"))
        assert len(files) == 1
        rec = json.loads(files[0].read_text(encoding="utf-8"))
        assert rec["job_id"] == job_id
        assert rec["model_id"] == "demo_faraday"
        assert rec["status"] == "done"
        assert rec["result"]["summary"]["thickness_um"] > 0

    # Fresh app on the same data dir must reload the finished job.
    with TestClient(create_app(data_dir=data_dir)) as c:
        jobs = c.get("/api/jobs").json()["jobs"]
        assert job_id in {j["id"] for j in jobs}
        job = next(j for j in jobs if j["id"] == job_id)
        assert job["status"] == "done"
        res = c.get(f"/api/jobs/{job_id}/result").json()
        assert res["status"] == "done"
        assert res["result"]["summary"]["thickness_um"] > 0


def test_cancel_and_delete_job(tmp_path):
    with TestClient(create_app(data_dir=tmp_path / "psdata", max_workers=2)) as c:
        job_id = c.post("/api/models/demo_faraday/run", json={"t_min": 5.0}).json()["job_id"]
        r = c.post(f"/api/jobs/{job_id}/cancel")
        assert r.status_code == 200
        job = wait_job(c, job_id, timeout=20)
        assert job["status"] == "cancelled"
        # cancelling again -> 409
        assert c.post(f"/api/jobs/{job_id}/cancel").status_code == 409
        # delete removes it
        assert c.delete(f"/api/jobs/{job_id}").status_code == 200
        assert c.get(f"/api/jobs/{job_id}").status_code == 404


def test_websocket_streams_job_updates(tmp_path):
    with TestClient(create_app(data_dir=tmp_path / "psdata", max_workers=2)) as c:
        with c.websocket_connect("/api/ws") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"
            assert "demo_faraday" in hello["models_loaded"]

            job_id = c.post("/api/models/demo_faraday/run", json={"t_min": 0.5}).json()["job_id"]
            seen_running = False
            seen_done = False
            t0 = time.time()
            while time.time() - t0 < 40 and not seen_done:
                ev = ws.receive_json()
                if ev.get("type") == "job_update" and ev["job"]["id"] == job_id:
                    if ev["job"]["status"] == "running":
                        seen_running = True
                    if ev["job"]["status"] == "done":
                        seen_done = True
            assert seen_running and seen_done


def test_data_endpoints(tmp_path):
    with TestClient(create_app(data_dir=tmp_path / "psdata")) as c:
        doe = c.get("/api/data/planar_doe")
        assert doe.status_code == 200
        assert "runs" in doe.json()

        lit = c.get("/api/data/literature")
        assert lit.status_code == 200
        assert isinstance(lit.json()["files"], list)

        sem = c.get("/api/data/sem")
        assert sem.status_code in (200, 404)  # file may not exist yet

        assert c.get("/api/data/literature/../secrets").status_code in (400, 404)
