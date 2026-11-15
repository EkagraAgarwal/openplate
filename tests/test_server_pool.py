"""Persistent worker-pool behaviour of platesim.server.jobs (small pools, < 60 s).

Complements test_server.py: proves that consecutive jobs reuse one long-lived
worker process (caches survive), that a cancelled job costs a worker which is
replaced at constant pool size, and that workers pin BLAS to one thread.
"""
from __future__ import annotations

import time

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
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def wait_until(predicate, timeout: float = 20.0, interval: float = 0.1) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if predicate():
            return True
        time.sleep(interval)
    return False


def live_workers(manager) -> list:
    with manager._lock:
        return [w for w in manager._workers if w.proc.is_alive()]


def submit(client: TestClient, t_min: float = 0.5) -> str:
    r = client.post("/api/models/demo_faraday/run", json={"t_min": t_min})
    assert r.status_code == 202, r.text
    return r.json()["job_id"]


def test_two_jobs_run_in_the_same_worker(tmp_path):
    app = create_app(data_dir=tmp_path / "psdata", max_workers=1)
    with TestClient(app) as client:
        jobs = []
        for _ in range(2):
            job = wait_job(client, submit(client))
            assert job["status"] == "done", job.get("error")
            jobs.append(job)

        first, second = jobs
        assert first["worker_pid"] is not None
        assert first["worker_pid"] == second["worker_pid"], (
            "two consecutive jobs must reuse one persistent worker process")

        # The child-reported meta agrees with the parent view of the job.
        out = client.get(f"/api/jobs/{second['id']}/result").json()["result"]
        assert out["meta"]["worker_pid"] == second["worker_pid"]


def test_worker_pins_blas_thread_env(tmp_path):
    app = create_app(data_dir=tmp_path / "psdata", max_workers=1)
    with TestClient(app) as client:
        job = wait_job(client, submit(client))
        assert job["status"] == "done", job.get("error")
        meta = client.get(f"/api/jobs/{job['id']}/result").json()["result"]["meta"]
        assert meta["omp_num_threads"] == "1"
        assert meta["openblas_num_threads"] == "1"
        assert meta["mkl_num_threads"] == "1"


def test_cancel_running_job_replaces_worker(tmp_path):
    app = create_app(data_dir=tmp_path / "psdata", max_workers=2)
    with TestClient(app) as client:
        manager = app.state.manager
        assert wait_until(lambda: len(live_workers(manager)) == 2)

        # Occupy both workers with long demo jobs (~2 s each).
        ids = [submit(client, t_min=5.0), submit(client, t_min=5.0)]
        assert wait_until(lambda: all(
            client.get(f"/api/jobs/{jid}").json()["status"] == "running" for jid in ids
        ))
        victim = ids[0]
        victim_pid = client.get(f"/api/jobs/{victim}").json()["worker_pid"]
        assert victim_pid is not None

        assert client.post(f"/api/jobs/{victim}/cancel").status_code == 200
        job = wait_job(client, victim, timeout=20)
        assert job["status"] == "cancelled"

        # The pool stays at max_workers and a fresh worker replaced the killed one.
        assert wait_until(lambda: len(live_workers(manager)) == 2, timeout=20)
        pids = {w.proc.pid for w in live_workers(manager)}
        assert victim_pid not in pids

        # A following job still completes on the rebuilt pool.
        follow = wait_job(client, submit(client))
        assert follow["status"] == "done", follow.get("error")
