# platesim desktop app

Electron + Vite + React + TypeScript + Plotly UI for the `platesim` Cu electroplating
simulator. Talks to the FastAPI backend in `platesim/server/` over HTTP + WebSocket.

## Scripts

| script | what it does |
| --- | --- |
| `npm install` | install deps (pinned exact versions) |
| `npm run dev` | Vite dev server + Electron (hot reload renderer) |
| `npm run dev:mock` | same, but `PLATESIM_MOCK=1` — no Python needed |
| `npm run build` | typecheck (`tsc --noEmit` × 2) + `electron-vite build` → `out/` |
| `npm run typecheck` | typecheck only (renderer `tsconfig.json` + main/preload `tsconfig.node.json`) |
| `npm start` | run Electron on the built files in `out/` (run `npm run build` first) |
| `npm test` | vitest unit tests (param coercion, plot-kind mapping) |
| `npm run package` | electron-builder Windows portable + NSIS installers → `dist/` (optional) |

## Backend lifecycle (Electron main process)

On start the main process (`src/main/backend.ts`):

1. Resolves the platesim project root: `PLATESIM_ROOT` env → settings file
   (`<userData>/platesim-settings.json`) → default = **parent folder of `app/`**.
2. Picks a free TCP port and spawns
   `python -m platesim.server --port <port> --host 127.0.0.1` with `cwd` = project root.
3. Waits for the stdout line `PLATESIM_SERVER_READY port=<port>` (60 s timeout), then
   polls `GET /api/health` until OK. Every stdout/stderr line is captured into a ring
   buffer shown in the collapsible **Logs** panel at the bottom of the window (also
   mirrored to the terminal).
4. Starts a local **gateway** proxy on another free port. The renderer only ever talks
   to the gateway; the gateway forwards HTTP and WebSocket upgrades to the real backend
   and adds permissive CORS (so the Vite dev origin works).
5. Kills the python child (and closes the gateway) on quit.

**Connect-to-existing-server mode:** Settings → Backend → “Connect to an already-running
server” + URL → Restart backend. The gateway then proxies to that URL instead of spawning.

**Mock mode:** `PLATESIM_MOCK=1` (e.g. `npm run dev:mock`) starts an in-process Node
mock server (`src/main/mockServer.ts`) serving the `demo_faraday` and
`demo_feature_fill` fixtures from `mock/fixtures.json` — models, jobs (fake progress),
results with line/heatmap/bar/contours plots, sweeps, and the three `/api/data/*`
tables — so the full UI can be used without Python. The mock does not implement
`/api/ws`, so the jobs panel automatically falls back to 1 s polling.

## Window / security

- `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true`.
- Preload (`src/preload/index.ts`) exposes only `window.platesim`: settings, backend
  state/logs/restart, LLM chat streaming, and save-dialog helpers.
- Splash overlay is shown by the renderer while the backend is starting.
- Backend logs live in a collapsible bottom panel.

## Renderer features

- **Left sidebar:** model list grouped Coupon/Cell · Flow · Efficiency · Feature fill ·
  Imaging · Other (grouping inferred from `MODEL_SPEC.package`/id).
- **Model page:** auto-generated parameter form from `MODEL_SPEC.params`
  (float/int with min/max/unit/help tooltip, bool toggle, enum select, string input),
  **Run**, presets in localStorage, **Sweep builder** (1–2 numeric params, explicit
  value lists → `POST /api/models/{id}/sweep {base, vary}`, polled via
  `GET /api/sweeps/{id}` with per-point result links).
- **Jobs panel:** live list over `WebSocket /api/ws` with 1 s polling fallback,
  progress bars, cancel, open result, compare selection (max 4).
- **Result view:** summary table, warnings/assumptions callouts, plot renderer per kind
  — `line` (multi-series, per-series `mode: markers`), `heatmap` (x/y/z + colorbar
  zlabel), `bar`, `contours` (time slider + play button, equal aspect, polylines;
  feature-fill models mirror `r → -r` to draw the full via cross-section), `image`
  treated as a gray heatmap. Export per-plot **PNG** and whole-result **JSON** via
  native save dialogs.
- **Compare:** side-by-side summary table + overlaid line plots for up to 4 results.
- **Data page:** planar DOE table, expandable literature datasets, SEM measurements
  (`/api/data/planar_doe|literature|sem`, payload-shape tolerant; missing endpoints
  degrade to an empty-state note).
- **Assistant page:** OpenAI-compatible `/chat/completions` client with SSE streaming
  (proxied through the main process — the API key is stored with Electron
  `safeStorage` and never enters the renderer). Shows “not configured” until base URL +
  model + key are set in Settings. The system prompt lists the live `/api/models`
  catalogue; the LLM may propose a run as a fenced ```` ```json ```` block
  `{"model_id", "params"}` rendered as a **Load into form** button — parameters are
  handed to the model page but never auto-run.
- Light/dark theme toggle (persisted), responsive down to 1100 px.

## Tests

`npm test` — vitest, 32 unit tests:

- `tests/coercion.test.ts` — `coerceParam` / `coerceAll` / `defaultsFromSpec` /
  `parseValueList`: numeric parsing, int rounding, clamping to min/max, bool/enum
  coercion with fallbacks, per-field error collection.
- `tests/plots.test.ts` — `normalizePlotKind` (all contract kinds + unknown),
  `lineTraces` (series mode), `heatmapTraces` (colorbar zlabel, image → gray),
  `barTraces`, `contourFrameTraces` (frame selection, clamping, r→-r mirror),
  `buildLayout` (equal aspect for contours, axis titles).

## Known limitations

- Grouping (Coupon/Cell etc.) is inferred from package/id names since `MODEL_SPEC`
  has no explicit `group` field; add `"group"` to a spec later to override is not yet
  supported — extend `groupFor()` in `src/renderer/src/types.ts` if needed.
- Sweep response shape is assumed to be `{sweep_id}` + `GET /api/sweeps/{id}` →
  `{status, results:[{params, result}]}`; the parser tolerates `{id}` instead of
  `{sweep_id}` but the backend must provide one of them.
- WS job messages are shape-tolerant (`{job}`, `{id,status}`, `{job_id,status}`,
  arrays, `{events:[]}`) but if the backend uses a wholly different envelope the UI
  still works — it falls back to refreshing the job list on every message.
- `npm run package` bundles `out/` only; if you package the app, keep Python +
  platesim installed and point Settings → project root at the checkout (mock mode
  works without Python).
- The LLM endpoint must accept browser-style CORS? No — chat is proxied through the
  main process, so any OpenAI-compatible HTTP endpoint works.
