"""Lazy model-package registry.

Imports every registered package (CONTRACTS.md list + platesim.demo) inside
try/except so a broken or not-yet-written module never kills the server;
failures are exposed via GET /api/models/errors.
"""
from __future__ import annotations

import copy
import importlib
import traceback
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

# Packages listed in CONTRACTS.md, plus the demo plug-in.
REGISTERED_PACKAGES: tuple[str, ...] = (
    "platesim.cell",
    "platesim.flow",
    "platesim.efficiency",
    "platesim.feature",
    "platesim.imaging",
    "platesim.coupon",
    "platesim.atomistic",
    "platesim.demo",
)


@dataclass
class ModelEntry:
    spec: dict
    run: Callable[..., dict]
    package: str


class Registry:
    """Holds successfully loaded models and per-package import failures."""

    def __init__(self) -> None:
        self.models: dict[str, ModelEntry] = {}
        self.errors: list[dict] = []

    def load(self, packages: Optional[Iterable[str]] = None) -> None:
        for pkg_name in list(packages if packages is not None else REGISTERED_PACKAGES):
            try:
                mod = importlib.import_module(pkg_name)
                spec = getattr(mod, "MODEL_SPEC", None)
                run = getattr(mod, "run", None)
                if not isinstance(spec, dict):
                    raise AttributeError(f"{pkg_name} has no MODEL_SPEC dict")
                if not callable(run):
                    raise AttributeError(f"{pkg_name} has no callable run(params, progress)")
                model_id = spec.get("id")
                if not isinstance(model_id, str) or not model_id:
                    raise ValueError(f"{pkg_name}.MODEL_SPEC is missing a non-empty 'id'")
                if model_id in self.models:
                    self.errors.append({
                        "package": pkg_name,
                        "error": f"duplicate model id '{model_id}' (already provided by "
                                 f"{self.models[model_id].package}); keeping the first one",
                        "traceback": None,
                    })
                    continue
                self.models[model_id] = ModelEntry(
                    spec=copy.deepcopy(spec), run=run, package=pkg_name
                )
            except Exception as exc:  # noqa: BLE001 - a broken plug-in must never kill the server
                self.errors.append({
                    "package": pkg_name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })

    def get(self, model_id: str) -> Optional[ModelEntry]:
        return self.models.get(model_id)

    def spec_list(self) -> list[dict]:
        """MODEL_SPEC dicts (with an added 'package' field), sorted by id."""
        out = []
        for mid in sorted(self.models):
            entry = self.models[mid]
            spec = copy.deepcopy(entry.spec)
            spec["package"] = entry.package
            out.append(spec)
        return out
