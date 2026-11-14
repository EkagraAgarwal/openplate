"""Run with python -m platesim.spec3d."""
import json
from pathlib import Path

import typer

from .schema import export_schema
from .validate import spec_hash, validate_spec

app = typer.Typer(help="Validate and hash 3-D sample specifications.")


@app.command("validate")
def validate(file: Path):
    _, errors, warnings = validate_spec(file)
    typer.echo(json.dumps({"errors": errors, "warnings": warnings}, indent=2))
    if errors:
        raise typer.Exit(1)


@app.command("schema")
def schema(out: Path):
    export_schema(out)


@app.command("hash")
def hash_command(file: Path):
    spec, errors, _ = validate_spec(file)
    if errors:
        typer.echo(json.dumps({"errors": errors}, indent=2))
        raise typer.Exit(1)
    typer.echo(spec_hash(spec))


if __name__ == "__main__":
    app()
