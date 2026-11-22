"""User-facing validation diagnostics and reproducible canonical hashes."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import SampleSpec


def _safe(value, depth=0):
    if depth >= 20:
        return "<nested input omitted>"
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v, depth + 1) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _error(path, value, allowed, fix):
    return {"path": path, "value": _safe(value), "allowed": allowed, "fix": fix}


def validate_spec(data: dict | str | Path) -> tuple[SampleSpec | None, list[dict], list[dict]]:
    """Accept a mapping, JSON text, or JSON filename; bad input never escapes as an exception."""
    try:
        if isinstance(data, Path):
            data = json.loads(data.read_text(encoding="utf-8-sig"))
        elif isinstance(data, str):
            text = data.lstrip()
            if text.startswith(("{", "[")):
                data = json.loads(text)
            else:
                data = json.loads(Path(data).read_text(encoding="utf-8-sig"))
        spec = SampleSpec.model_validate(data)
    except ValidationError as exc:
        errors = []
        for entry in exc.errors(include_url=False):
            loc = [str(p) for p in entry["loc"]
                   if not str(p).startswith("literal[") and str(p) != "constrained-int"]
            allowed = entry["msg"].removeprefix("Value error, ")
            if entry["type"] == "extra_forbidden":
                fix = "Remove this unknown field; use only fields declared in the schema."
            elif entry["type"] == "missing":
                fix = "Provide this required field using the JSON Schema."
            else:
                fix = f"Set {'.'.join(loc) or 'input'} so that: {allowed}."
            errors.append(_error(".".join(loc), entry.get("input"), allowed, fix))
        return None, errors, []
    except (OSError, ValueError, TypeError, RecursionError, OverflowError) as exc:
        return None, [_error("", data, "a readable JSON object matching SampleSpec",
                             f"Supply a valid JSON object or readable JSON file: {exc}")], []
    warnings = []
    for index, step in enumerate(spec.recipe.steps):
        if step.j_net_mA_cm2 > 60:
            warnings.append(_error(f"recipe.steps.{index}.j_net_mA_cm2", step.j_net_mA_cm2,
                                   "recommended net current <= 60 mA/cm2",
                                   "Reduce net current or independently verify high-current applicability."))
    return spec, [], warnings


def _normalise(value):
    if isinstance(value, dict):
        return {key: _normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if isinstance(value, float):
        # Exact integral floats (including negative zero) have one canonical spelling.
        return int(value) if value.is_integer() else value
    return value


def spec_hash(spec: SampleSpec | dict) -> str:
    """SHA-256 over validated, default-filled JSON; reject invalid specs."""
    model = spec if isinstance(spec, SampleSpec) else SampleSpec.model_validate(spec)
    data: Any = _normalise(model.model_dump(mode="json"))
    # A spec without additives keeps its exact pre-10g hash, whether the input
    # omitted the key or supplied it explicitly as null.
    if data.get("additives") is None:
        data.pop("additives", None)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
