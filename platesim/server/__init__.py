"""platesim.server: FastAPI backend with a process-isolated job queue for plug-in models.

Usage:
    python -m platesim.server --port 8765 --host 127.0.0.1 [--data-dir DIR]

The server discovers model packages listed in CONTRACTS.md (plus platesim.demo),
imports each lazily inside try/except, and runs every simulation in a spawned
child process so long jobs never block the API event loop.
"""
from .app import create_app, resolve_data_dir

__all__ = ["create_app", "resolve_data_dir"]
