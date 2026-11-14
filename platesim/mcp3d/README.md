# 3-D plating MCP server

Install the official `mcp` Python SDK and the normal sample3d / analysis3d /
render3d dependencies in the Python environment used by the client. Real geometry
requires `shapely` (not included in the task's isolated test dependency list).
SDK v1 exposes FastMCP; SDK v2 renames it MCPServer. The wiring supports both
official APIs (use `mcp<2` if you specifically require the FastMCP class name).
Start with:

```sh
python -m platesim.mcp3d --root /absolute/path/to/runs
```

The transport is stdio; stdout belongs exclusively to MCP. For Claude Code put
this in the project `.mcp.json` (the `mcpServers` entry also works with other
stdio MCP clients, including Claude Desktop). Use absolute paths, adjust Python
to the environment with the dependencies installed, and restart the client:

```json
{
  "mcpServers": {
    "platesim-3d": {
      "command": "C:/path/to/python.exe",
      "args": ["-m", "platesim.mcp3d", "--root", "D:/platesim-runs"],
      "env": {"PYTHONPATH": "D:/Workspace/Semiconductors/openplate_3d"}
    }
  }
}
```

## Tools

Calls below are JSON tool arguments; replace `SPEC` with a SampleSpec object
(see `platesim/spec3d/examples/single_trench.json`) and the example run ID with
the ID returned by `run_sample`.

| Tool | Example arguments |
| --- | --- |
| `validate_spec(spec: dict)` | `{"spec": SPEC}` — returns `ok` and field/value/message/fix errors. |
| `plan_run(spec: dict)` | `{"spec": SPEC}` — no solver; estimates use `est_time_s`. |
| `run_sample(spec: dict)` | `{"spec": SPEC}` — immediately returns `run_id`. |
| `get_status(run_id: str)` | `{"run_id":"0123456789ab-1"}` — state, dimensionless progress, message, error. |
| `get_manifest(run_id: str)` | `{"run_id":"0123456789ab-1"}` — completed manifest. |
| `cross_section(run_id: str, polyline_um: list[list[float]], render: bool = true)` | `{"run_id":"0123456789ab-1","polyline_um":[[0,0],[32,0]],"render":true}` — length / z range in um, contour counts and optional `png_path`. |
| `height_map(run_id: str, include_samples: bool = false)` | `{"run_id":"0123456789ab-1","include_samples":true}` — stats and `png_path`; opt-in sampling is at most 64×64, with `height_um`, `x_um`, `y_um`. |
| `render(run_id: str, kind: str = "surface", camera: str = "iso")` | `{"run_id":"0123456789ab-1","kind":"surface","camera":"iso"}` — artifact `path`. |
| `list_runs()` | `{}` — `runs` containing IDs and statuses. |

`kind` is `surface`, `height_map`, or `section`; `camera` is `iso`, `top`, or
`front` (only surface uses a 3-D camera). Section rendering without a polyline
uses a centre-y cut along x; call `cross_section` for a custom cut. No full field
arrays are returned. Missing numerical crossings become JSON null.

Runs live at `root/<first 12 hex of canonical spec_hash>-<counter>`, with counters
starting at one and allocated by atomic directory creation. Use a dedicated root
outside the source tree. RunService imports sample3d lazily, so validation,
inspection and testing work while that package is still being developed.
Errors return `error`, `field`, `fix`; validation instead returns the requested
`ok` / `errors` structure. Manifests retain section 7's name/value/unit schema.

Background threads are process-local and daemonized: keep the server alive until
runs finish. After restart, valid completed manifests are rediscovered; partial
runs are marked failed and must be resubmitted. There is no cancellation tool.
Rendering is serialized within a service instance. PNG names are deterministic
and overwritten on subsequent calls, so copy an artifact if it must be retained.
Do not expose this server to untrusted clients: spec layout file paths retain the
local filesystem access of the sample pipeline; the run root is not a sandbox.
