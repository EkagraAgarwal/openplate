"""CLI entry point: python -m platesim.server --port 8765 --host 127.0.0.1"""
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Optional


def _ensure_uvicorn() -> None:
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        print("uvicorn not found; installing via pip...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "uvicorn"])
        import uvicorn  # noqa: F401


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="platesim.server",
        description="platesim FastAPI server: plug-in simulation models over a job queue",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="bind port (default 8765)")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="job persistence directory (default: %%APPDATA%%/platesim on Windows, ./runs elsewhere)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="max parallel simulation processes (default: max(1, cpu_count-2))",
    )
    args = parser.parse_args(argv)

    _ensure_uvicorn()
    import uvicorn

    from .app import create_app

    app = create_app(data_dir=args.data_dir, ready_port=args.port, max_workers=args.max_workers)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
