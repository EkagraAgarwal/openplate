"""Job queue: runs model plug-ins in a pool of persistent spawned workers.

- max_workers = max(1, cpu_count - 2) long-lived child processes
  (multiprocessing spawn context). Each worker loops: take a job from its
  task queue, run it, report the result, wait for the next job. Import cost
  and the in-process module caches therefore survive between jobs.
- Workers pin OMP/OPENBLAS/MKL to one thread before numpy is imported.
- Progress flows child -> parent via a Manager().dict(); results and tracebacks
  flow back via a Manager().Queue().
- Finished jobs persist to <data_dir>/jobs/<job_id>.json and are reloaded on start.
- Cancellation is best effort: queued jobs are dropped; running jobs terminate
  the worker that executes them, and the pool spawns a replacement worker.
- In-process pub/sub event log feeds the /api/ws WebSocket.
"""
from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import os
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from queue import Empty

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
FINISHED_STATES = {DONE, FAILED, CANCELLED}

MAX_SWEEP_JOBS = 100
_BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _parse_iso(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _job_entry(job_id: str, model_package: str, params: dict,
               progress_store: Any, result_queue: Any) -> None:
    """Run one job inside a worker process. Module-level so spawn can pickle it."""
    try:
        import importlib

        mod = importlib.import_module(model_package)
        run = getattr(mod, "run", None)
        if not callable(run):
            raise AttributeError(f"{model_package} has no callable run()")

        def progress(frac: float, msg: str = "") -> None:
            try:
                progress_store[job_id] = [min(max(float(frac), 0.0), 1.0), str(msg)]
            except Exception:
                pass  # progress is best effort; never kill the run over it

        result = run(dict(params), progress)
        if isinstance(result, dict):
            meta = result.get("meta")
            if not isinstance(meta, dict):
                meta = {}
                result["meta"] = meta
            # Worker provenance, reported from the child so callers can verify
            # which persistent worker ran the job and how BLAS was pinned.
            meta["worker_pid"] = os.getpid()
            meta["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS")
            meta["openblas_num_threads"] = os.environ.get("OPENBLAS_NUM_THREADS")
            meta["mkl_num_threads"] = os.environ.get("MKL_NUM_THREADS")
        result_queue.put(("done", job_id, result))
    except Exception:
        try:
            result_queue.put(("failed", job_id, traceback.format_exc()))
        except Exception:
            pass


def _worker_entry(task_queue: Any, result_queue: Any, progress_store: Any) -> None:
    """Persistent worker loop: run jobs one at a time until a None sentinel."""
    # Must happen before any heavy import (numpy/scipy/gmsh) in this process.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    while True:
        try:
            item = task_queue.get()
        except (EOFError, OSError):
            break  # parent went away
        if item is None:
            break  # clean shutdown / recycling
        try:
            job_id, model_package, params = item
        except (TypeError, ValueError):
            continue
        _job_entry(job_id, model_package, params, progress_store, result_queue)


@dataclass
class _Worker:
    """Parent-side handle for one persistent worker process."""
    slot: int
    proc: Any
    queue: Any
    job_id: Optional[str] = None
    jobs_done: int = 0
    retiring: bool = False


@dataclass
class Job:
    id: str
    model_id: str
    model_package: str
    params: dict
    status: str = QUEUED
    progress: float = 0.0
    message: str = ""
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    error: Optional[str] = None
    result: Optional[dict] = None
    cancel_requested: bool = False
    worker_pid: Optional[int] = None  # pid of the persistent worker that ran it
    proc: Any = None  # worker process handle; not persisted

    def elapsed_s(self) -> Optional[float]:
        if self.started is None:
            return None
        end = self.finished if self.finished is not None else time.time()
        return round(end - self.started, 3)

    def to_public(self) -> dict:
        return {
            "id": self.id,
            "model_id": self.model_id,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "params": self.params,
            "created": _iso(self.created),
            "started": _iso(self.started),
            "finished": _iso(self.finished),
            "elapsed_s": self.elapsed_s(),
            "error": self.error,
            "worker_pid": self.worker_pid,
        }


class JobManager:
    def __init__(self, data_dir: Path | str, max_workers: Optional[int] = None,
                 recycle_after: Optional[int] = 200) -> None:
        self.data_dir = Path(data_dir)
        self.jobs_dir = self.data_dir / "jobs"
        cpu = os.cpu_count() or 2
        self.max_workers = max_workers if max_workers and max_workers > 0 else max(1, cpu - 2)
        # Replace a worker after this many jobs to bound slow leaks; None/<=0 disables.
        self.recycle_after = recycle_after if recycle_after and recycle_after > 0 else None
        self.jobs: dict[str, Job] = {}
        self.sweeps: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._pending: deque[str] = deque()
        self._workers: list[_Worker] = []
        self._next_slot = 0
        self._events: deque[tuple[int, dict]] = deque(maxlen=1000)
        self._event_seq = 0
        self._stop = threading.Event()
        self._ctx = mp.get_context("spawn")
        self._manager: Any = None
        self._result_queue: Any = None
        self._progress_store: Any = None
        self._scheduler: Optional[threading.Thread] = None
        self._monitor: Optional[threading.Thread] = None

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.load_history()
        self._manager = self._ctx.Manager()
        self._result_queue = self._manager.Queue()
        self._progress_store = self._manager.dict()
        self._stop.clear()
        with self._lock:
            for _ in range(self.max_workers):
                self._workers.append(self._spawn_worker())
        self._scheduler = threading.Thread(target=self._schedule_loop, name="platesim-sched", daemon=True)
        self._monitor = threading.Thread(target=self._monitor_loop, name="platesim-monitor", daemon=True)
        self._scheduler.start()
        self._monitor.start()

    def _spawn_worker(self) -> _Worker:
        """Start one persistent worker. Caller must hold self._lock."""
        queue = self._ctx.Queue()
        slot = self._next_slot
        self._next_slot += 1
        proc = self._ctx.Process(
            target=_worker_entry,
            args=(queue, self._result_queue, self._progress_store),
            name=f"platesim-worker-{slot}",
            daemon=True,
        )
        # The spawned child inherits the parent's environment at start(). Pin BLAS to one thread
        # there (the child may import numpy while unpickling, before _worker_entry runs) so
        # max_workers processes do not oversubscribe the CPU, then restore the parent's own values.
        saved = {k: os.environ.get(k) for k in _BLAS_ENV}
        os.environ.update({k: "1" for k in _BLAS_ENV})
        try:
            proc.start()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return _Worker(slot=slot, proc=proc, queue=queue)

    def shutdown(self) -> None:
        self._stop.set()
        # Stop the loops first so nothing is dispatched or respawned meanwhile.
        for t in (self._scheduler, self._monitor):
            if t is not None:
                t.join(timeout=5)
        with self._lock:
            workers = list(self._workers)
            self._workers.clear()
        for w in workers:
            try:
                w.queue.put_nowait(None)  # sentinel: exit after the current job
            except Exception:
                pass
        # One shared deadline: idle workers exit in parallel, so this is the
        # wall-clock budget for the whole pool, not per worker. Workers still
        # importing at shutdown are terminated below.
        deadline = time.time() + 1.0
        for w in workers:
            try:
                w.proc.join(timeout=max(0.0, deadline - time.time()))
            except Exception:
                pass
        for w in workers:
            try:
                if w.proc.is_alive():
                    w.proc.terminate()
                    w.proc.join(timeout=1)
            except Exception:
                pass
            try:
                w.queue.close()
            except Exception:
                pass
        if self._manager is not None:
            try:
                self._manager.shutdown()
            except Exception:
                pass
            self._manager = None

    # ------------------------------------------------------------------ submit

    def submit(self, model_id: str, package: str, params: dict) -> Job:
        job = Job(id=uuid.uuid4().hex, model_id=model_id, model_package=package, params=dict(params))
        with self._lock:
            self.jobs[job.id] = job
            self._pending.append(job.id)
        self._emit(job)
        return job

    def create_sweep(self, model_id: str, package: str, base: dict,
                     vary: dict[str, list]) -> tuple[str, list[str]]:
        """Create one child job per vary combination. Returns (sweep_id, job_ids)."""
        sweep_id = uuid.uuid4().hex[:12]
        keys = list(vary.keys())
        job_ids: list[str] = []
        for combo in itertools.product(*(vary[k] for k in keys)):
            params = dict(base)
            params.update(dict(zip(keys, combo)))
            job = self.submit(model_id, package, params)
            job_ids.append(job.id)
        with self._lock:
            self.sweeps[sweep_id] = {
                "id": sweep_id,
                "model_id": model_id,
                "base": dict(base),
                "vary": {k: list(v) for k, v in vary.items()},
                "job_ids": job_ids,
                "created": time.time(),
            }
        return sweep_id, job_ids

    def sweep_status(self, sweep_id: str) -> Optional[dict]:
        with self._lock:
            meta = self.sweeps.get(sweep_id)
            if meta is None:
                return None
            rows = []
            statuses = []
            for jid in meta["job_ids"]:
                job = self.jobs.get(jid)
                if job is None:
                    continue
                statuses.append(job.status)
                summary = (job.result or {}).get("summary") if job.status == DONE else None
                rows.append({
                    "job_id": job.id,
                    "params": job.params,
                    "status": job.status,
                    "progress": round(job.progress, 4),
                    "elapsed_s": job.elapsed_s(),
                    "summary": summary,
                    "error": job.error,
                })
        if any(s in (QUEUED, RUNNING) for s in statuses):
            status = RUNNING if any(s == RUNNING for s in statuses) else QUEUED
        elif any(s == FAILED for s in statuses):
            status = FAILED
        elif any(s == CANCELLED for s in statuses):
            status = CANCELLED
        else:
            status = DONE
        return {
            "sweep_id": meta["id"],
            "model_id": meta["model_id"],
            "base": meta["base"],
            "vary": meta["vary"],
            "status": status,
            "n_jobs": len(meta["job_ids"]),
            "table": rows,
            "created": _iso(meta["created"]),
        }

    def list_sweeps(self) -> list[dict]:
        with self._lock:
            ids = list(self.sweeps.keys())
        out = []
        for sid in ids:
            st = self.sweep_status(sid)
            if st is not None:
                out.append({k: st[k] for k in ("sweep_id", "model_id", "status", "n_jobs", "created")})
        out.sort(key=lambda s: s.get("created") or "", reverse=True)
        return out

    # ------------------------------------------------------------- cancel/delete

    def cancel(self, job_id: str) -> Job:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status in FINISHED_STATES:
                return job
            if job.status == QUEUED:
                job.status = CANCELLED
                job.cancel_requested = True
                job.finished = time.time()
                job.message = "cancelled before start"
                proc = None
            else:  # running -> terminate best effort
                job.cancel_requested = True
                job.message = "cancelling..."
                proc = job.proc
        if job.status == CANCELLED:
            self._save(job)
            self._emit(job)
            return job
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
            def _kill_later() -> None:
                time.sleep(2.0)
                try:
                    # Only force-kill while this job is still unsettled: the
                    # worker may already have moved on to the next job.
                    if job.status not in FINISHED_STATES and proc.is_alive():
                        proc.kill()
                except Exception:
                    pass
            threading.Thread(target=_kill_later, daemon=True).start()
        self._emit(job)
        return job

    def delete(self, job_id: str) -> None:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status == RUNNING:
                raise RuntimeError("job is running; cancel it first")
            if job.status == QUEUED:
                job.status = CANCELLED
                job.finished = time.time()
            self.jobs.pop(job_id, None)
        path = self.jobs_dir / f"{job_id}.json"
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    # ------------------------------------------------------------ worker loops

    def _idle_worker(self) -> Optional[_Worker]:
        """An alive worker with no job assigned. Caller must hold self._lock."""
        for worker in self._workers:
            if worker.job_id is None and not worker.retiring and worker.proc.is_alive():
                return worker
        return None

    def _schedule_loop(self) -> None:
        while not self._stop.wait(0.02):
            batch: list[tuple[_Worker, str, str, dict]] = []
            with self._lock:
                while self._pending:
                    worker = self._idle_worker()
                    if worker is None:
                        break  # pool busy; retry on the next tick
                    job_id = self._pending.popleft()
                    job = self.jobs.get(job_id)
                    if job is None or job.status != QUEUED:
                        continue
                    worker.job_id = job_id  # reserve the worker
                    job.proc = worker.proc
                    job.worker_pid = worker.proc.pid
                    job.status = RUNNING
                    if job.started is None:
                        job.started = time.time()
                    job.message = "running"
                    batch.append((worker, job_id, job.model_package, dict(job.params)))
            for worker, job_id, package, params in batch:
                try:
                    worker.queue.put((job_id, package, params))
                except Exception as exc:
                    with self._lock:
                        worker.job_id = None
                    self._fail_job(
                        job_id,
                        f"failed to dispatch job to worker: {exc!r}\n{traceback.format_exc()}",
                    )
                    continue
                with self._lock:
                    job = self.jobs.get(job_id)
                if job is not None:
                    self._emit(job)

    def _monitor_loop(self) -> None:
        while not self._stop.wait(0.05):
            # 1. results from workers (the worker stays alive for the next job)
            while True:
                try:
                    kind, job_id, payload = self._result_queue.get_nowait()
                except Empty:
                    break
                except Exception:
                    break
                self._handle_result(kind, job_id, payload)
            # 2. progress updates from running jobs
            with self._lock:
                running = [j for j in self.jobs.values() if j.status == RUNNING]
            for job in running:
                try:
                    val = self._progress_store.get(job.id)
                except Exception:
                    val = None
                if val:
                    frac, msg = float(val[0]), str(val[1])
                    if frac != job.progress or msg != job.message:
                        job.progress = frac
                        job.message = msg
                        self._emit(job)
            # 3. workers that exited without a queued result (crash / terminate /
            #    recycle): settle their in-flight job and keep the pool full.
            self._reap_workers()

    def _reap_workers(self) -> None:
        with self._lock:
            dead = [w for w in self._workers if not w.proc.is_alive()]
        for worker in dead:
            with self._lock:
                if worker not in self._workers:
                    continue
                self._workers.remove(worker)
                job = self.jobs.get(worker.job_id) if worker.job_id else None
                worker.job_id = None
            try:
                worker.proc.join(timeout=1)
            except Exception:
                pass
            if job is not None and job.status not in FINISHED_STATES:
                if job.cancel_requested:
                    job.status = CANCELLED
                    job.message = "cancelled"
                    job.error = None
                else:
                    job.status = FAILED
                    job.error = (job.error or
                                 f"worker process exited with code {worker.proc.exitcode} "
                                 "before reporting a result")
                job.finished = time.time()
                self._cleanup_progress(job.id)
                self._save(job)
                self._emit(job)
            try:
                worker.queue.close()
            except Exception:
                pass
            with self._lock:
                if (not self._stop.is_set()
                        and len(self._workers) < self.max_workers):
                    try:
                        self._workers.append(self._spawn_worker())
                    except Exception:
                        pass  # retried on the next monitor tick

    def _retire_worker(self, worker: _Worker) -> None:
        """Send a sentinel so the worker exits after its current job."""
        with self._lock:
            if worker.retiring:
                return
            worker.retiring = True
        try:
            worker.queue.put_nowait(None)
        except Exception:
            pass

    def _fail_job(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self.jobs.get(job_id)
        if job is None or job.status in FINISHED_STATES:
            return
        job.status = FAILED
        job.error = error
        job.message = "failed"
        job.finished = time.time()
        self._cleanup_progress(job_id)
        self._save(job)
        self._emit(job)

    def _handle_result(self, kind: str, job_id: str, payload: Any) -> None:
        with self._lock:
            job = self.jobs.get(job_id)
            worker = next((w for w in self._workers if w.job_id == job_id), None)
            if worker is not None:
                worker.job_id = None  # worker is free for the next job
                worker.jobs_done += 1
        if job is not None and job.status not in FINISHED_STATES:
            job.finished = time.time()
            final = DONE if kind == "done" else FAILED
            if final == DONE:
                job.result = payload if isinstance(payload, dict) else {"value": payload}
                job.progress = 1.0
                job.message = "done"
            else:
                job.error = str(payload)
                job.message = "failed"
            self._cleanup_progress(job_id)
            # Persist under the final status before publishing it, so a client
            # that observes the terminal state also finds the history file.
            self._save(job, status=final)
            job.status = final
            self._emit(job)
        if (worker is not None and self.recycle_after is not None
                and worker.jobs_done >= self.recycle_after):
            self._retire_worker(worker)

    # ------------------------------------------------------------- persistence

    def _save(self, job: Job, status: Optional[str] = None) -> None:
        status = status or job.status
        if status not in FINISHED_STATES:
            return
        rec = {
            "job_id": job.id,
            "model_id": job.model_id,
            "model_package": job.model_package,
            "params": job.params,
            "status": status,
            "progress": job.progress,
            "message": job.message,
            "created": _iso(job.created),
            "started": _iso(job.started),
            "finished": _iso(job.finished),
            "elapsed_s": job.elapsed_s(),
            "error": job.error,
            "worker_pid": job.worker_pid,
            "result": job.result,
        }
        try:
            self.jobs_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.jobs_dir / f".{job.id}.json.tmp"
            tmp.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
            tmp.replace(self.jobs_dir / f"{job.id}.json")
        except OSError:
            pass  # persistence is best effort; the in-memory job stays authoritative

    def load_history(self) -> int:
        """Reload finished jobs from <data_dir>/jobs/*.json. Returns count loaded."""
        if not self.jobs_dir.is_dir():
            return 0
        loaded = 0
        for path in sorted(self.jobs_dir.glob("*.json")):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            job_id = rec.get("job_id") or path.stem
            status = rec.get("status", FAILED)
            if status not in FINISHED_STATES:
                status = FAILED  # server died mid-run
            job = Job(
                id=job_id,
                model_id=rec.get("model_id", ""),
                model_package=rec.get("model_package", ""),
                params=rec.get("params") or {},
                status=status,
                progress=float(rec.get("progress", 1.0 if status == DONE else 0.0)),
                message=rec.get("message", ""),
                error=rec.get("error"),
                result=rec.get("result"),
                worker_pid=rec.get("worker_pid"),
            )
            job.created = _parse_iso(rec.get("created")) or path.stat().st_mtime
            job.started = _parse_iso(rec.get("started"))
            job.finished = _parse_iso(rec.get("finished")) or job.created
            if status == FAILED and not job.error:
                job.error = "job was interrupted (server restarted while it was running)"
            self.jobs[job.id] = job
            loaded += 1
        return loaded

    # ------------------------------------------------------------------- events

    def _emit(self, job: Job) -> None:
        with self._lock:
            self._event_seq += 1
            seq = self._event_seq
            self._events.append((seq, {"type": "job_update", "seq": seq, "job": job.to_public()}))

    def current_seq(self) -> int:
        with self._lock:
            return self._event_seq

    def events_after(self, seq: int) -> list[tuple[int, dict]]:
        with self._lock:
            return [(s, ev) for s, ev in self._events if s > seq]

    # ------------------------------------------------------------------ queries

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self.jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        with self._lock:
            jobs = list(self.jobs.values())
        jobs.sort(key=lambda j: j.created, reverse=True)  # newest first
        return jobs

    def _cleanup_progress(self, job_id: str) -> None:
        if self._progress_store is None:
            return
        try:
            self._progress_store.pop(job_id, None)
        except Exception:
            pass
