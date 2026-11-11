"""Job queue: runs model plug-ins in spawned child processes.

- One child process per job (multiprocessing spawn context), capped at
  max_workers = max(1, cpu_count - 2); long simulations never block the API.
- Progress flows child -> parent via a Manager().dict(); results and tracebacks
  flow back via a Manager().Queue().
- Finished jobs persist to <data_dir>/jobs/<job_id>.json and are reloaded on start.
- Cancellation is best effort: queued jobs are dropped, running jobs are
  terminated (TerminateProcess on Windows).
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
    """Child-process entry point. Must stay module-level so spawn can pickle it."""
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
        result_queue.put(("done", job_id, result))
    except Exception:
        try:
            result_queue.put(("failed", job_id, traceback.format_exc()))
        except Exception:
            pass


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
    proc: Any = None  # not persisted

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
        }


class JobManager:
    def __init__(self, data_dir: Path | str, max_workers: Optional[int] = None) -> None:
        self.data_dir = Path(data_dir)
        self.jobs_dir = self.data_dir / "jobs"
        cpu = os.cpu_count() or 2
        self.max_workers = max_workers if max_workers and max_workers > 0 else max(1, cpu - 2)
        self.jobs: dict[str, Job] = {}
        self.sweeps: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._pending: deque[str] = deque()
        self._procs: dict[str, Any] = {}
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
        self._scheduler = threading.Thread(target=self._schedule_loop, name="platesim-sched", daemon=True)
        self._monitor = threading.Thread(target=self._monitor_loop, name="platesim-monitor", daemon=True)
        self._scheduler.start()
        self._monitor.start()

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            procs = [j.proc for j in self.jobs.values()
                     if j.proc is not None and getattr(j.proc, "is_alive", lambda: False)()]
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.join(timeout=2)
            except Exception:
                pass
        for t in (self._scheduler, self._monitor):
            if t is not None:
                t.join(timeout=5)
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
                    if proc.is_alive():
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

    def _schedule_loop(self) -> None:
        while not self._stop.wait(0.02):
            batch: list[str] = []
            with self._lock:
                while self._pending and len(self._procs) < self.max_workers:
                    job_id = self._pending.popleft()
                    job = self.jobs.get(job_id)
                    if job is None or job.status != QUEUED:
                        continue
                    self._procs[job_id] = None  # reserve the slot
                    batch.append(job_id)
            for job_id in batch:
                with self._lock:
                    job = self.jobs.get(job_id)
                    if job is None or job.status != QUEUED or self._stop.is_set():
                        self._procs.pop(job_id, None)
                        continue
                    package, params = job.model_package, dict(job.params)
                try:
                    proc = self._ctx.Process(
                        target=_job_entry,
                        args=(job_id, package, params, self._progress_store, self._result_queue),
                    )
                    proc.start()
                except Exception as exc:
                    with self._lock:
                        self._procs.pop(job_id, None)
                    job.status = FAILED
                    job.error = f"failed to start worker process: {exc!r}\n{traceback.format_exc()}"
                    job.finished = time.time()
                    self._save(job)
                    self._emit(job)
                    continue
                with self._lock:
                    job.proc = proc
                    self._procs[job_id] = proc
                    job.status = RUNNING
                    if job.started is None:
                        job.started = time.time()
                    job.message = "running"
                self._emit(job)

    def _monitor_loop(self) -> None:
        while not self._stop.wait(0.05):
            # 1. results from finished children
            while True:
                try:
                    kind, job_id, payload = self._result_queue.get_nowait()
                except Empty:
                    break
                except Exception:
                    break
                self._handle_result(kind, job_id, payload)
            # 2. progress updates from running children
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
            # 3. processes that exited without a queued result (crash / terminate)
            with self._lock:
                dead = [(jid, p) for jid, p in self._procs.items() if p is not None and not p.is_alive()]
            for jid, p in dead:
                with self._lock:
                    self._procs.pop(jid, None)
                    job = self.jobs.get(jid)
                if job is None or job.status in FINISHED_STATES:
                    continue
                try:
                    p.join(timeout=1)
                except Exception:
                    pass
                if job.cancel_requested:
                    job.status = CANCELLED
                    job.message = "cancelled"
                    job.error = None
                else:
                    job.status = FAILED
                    job.error = (job.error or
                                 f"worker process exited with code {p.exitcode} before reporting a result")
                job.finished = time.time()
                self._cleanup_progress(job.id)
                self._save(job)
                self._emit(job)

    def _handle_result(self, kind: str, job_id: str, payload: Any) -> None:
        with self._lock:
            job = self.jobs.get(job_id)
        if job is None or job.status in FINISHED_STATES:
            return
        job.finished = time.time()
        if kind == "done":
            job.result = payload if isinstance(payload, dict) else {"value": payload}
            job.status = DONE
            job.progress = 1.0
            job.message = "done"
        else:
            job.status = FAILED
            job.error = str(payload)
            job.message = "failed"
        self._cleanup_progress(job_id)
        self._save(job)
        self._emit(job)

    # ------------------------------------------------------------- persistence

    def _save(self, job: Job) -> None:
        if job.status not in FINISHED_STATES:
            return
        rec = {
            "job_id": job.id,
            "model_id": job.model_id,
            "model_package": job.model_package,
            "params": job.params,
            "status": job.status,
            "progress": job.progress,
            "message": job.message,
            "created": _iso(job.created),
            "started": _iso(job.started),
            "finished": _iso(job.finished),
            "elapsed_s": job.elapsed_s(),
            "error": job.error,
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
