"""Validate/coerce run parameters against a MODEL_SPEC param list.

Types: float, int, bool, str, enum. Enforces min/max and enum choices,
fills defaults for omitted params, rejects unknown params.
Returns (clean_params, errors); errors is empty on success.
"""
from __future__ import annotations

from typing import Any

_BOOL_TRUE = {"true", "yes", "1", "on"}
_BOOL_FALSE = {"false", "no", "0", "off"}


def _coerce(p: dict, raw: Any) -> Any:
    ptype = p.get("type", "float")
    name = p.get("name", "?")

    if ptype == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            if raw in (0, 1):
                return bool(raw)
            raise ValueError(f"'{name}': expected bool, got {raw!r}")
        if isinstance(raw, str):
            low = raw.strip().lower()
            if low in _BOOL_TRUE:
                return True
            if low in _BOOL_FALSE:
                return False
        raise ValueError(f"'{name}': expected bool, got {raw!r}")

    if ptype in ("float", "int"):
        if isinstance(raw, bool):
            raise ValueError(f"'{name}': expected {ptype}, got bool {raw!r}")
        if isinstance(raw, (int, float)):
            val: Any = float(raw)
        elif isinstance(raw, str):
            try:
                val = float(raw.strip())
            except ValueError:
                raise ValueError(f"'{name}': expected {ptype}, got {raw!r}") from None
        else:
            raise ValueError(f"'{name}': expected {ptype}, got {type(raw).__name__}")
        if ptype == "int":
            if not float(val).is_integer():
                raise ValueError(f"'{name}': expected int, got {raw!r}")
            val = int(val)
        lo, hi = p.get("min"), p.get("max")
        if lo is not None and val < lo:
            raise ValueError(f"'{name}'={val} is below min {lo}")
        if hi is not None and val > hi:
            raise ValueError(f"'{name}'={val} is above max {hi}")
        return val

    if ptype == "enum":
        choices = p.get("choices") or []
        if raw in choices:
            return raw
        raise ValueError(f"'{name}': {raw!r} is not one of {choices}")

    if ptype == "str":
        if isinstance(raw, str):
            return raw
        return str(raw)

    return raw  # unknown/extension types pass through untouched


def validate_params(spec_params: list[dict], payload: Any) -> tuple[dict, list[str]]:
    """Validate/coerce a payload against spec_params. Returns (clean, errors)."""
    errors: list[str] = []
    clean: dict[str, Any] = {}

    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return {}, ["params body must be a JSON object"]

    known = {p.get("name"): p for p in spec_params if isinstance(p, dict) and p.get("name")}
    for key in payload:
        if key not in known:
            errors.append(f"unknown parameter '{key}'")

    for name, p in known.items():
        if name in payload and payload[name] is not None:
            raw = payload[name]
        elif "default" in p:
            raw = p["default"]
        else:
            errors.append(f"missing required parameter '{name}'")
            continue
        try:
            clean[name] = _coerce(p, raw)
        except ValueError as exc:
            errors.append(str(exc))

    return clean, errors
