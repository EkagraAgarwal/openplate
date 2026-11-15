"""Portable Draft 2020-12 schema bundle for inputs and manifests."""
import json
from pathlib import Path

from pydantic.json_schema import models_json_schema

from .models import Manifest, SampleSpec


def export_schema(path: str | Path) -> None:
    _, schema = models_json_schema([(SampleSpec, "validation"), (Manifest, "validation")])
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "platesim 3-D spec and manifest"
    schema["oneOf"] = [{"$ref": "#/$defs/SampleSpec"}, {"$ref": "#/$defs/Manifest"}]
    Path(path).write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
