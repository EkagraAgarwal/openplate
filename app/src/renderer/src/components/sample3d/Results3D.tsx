import { useEffect, useState } from 'react'
import {
  Sample3DApi,
  type Height3D,
  type Manifest3D,
  type Point3D,
  type Run3D,
  type Section3D,
  type Series3D,
  type Tile3D
} from '../../api/client3d'
import { decimateHeight, outlineTraces, polylineFromClicks } from '../../lib/sample3d'
import { Plot3D } from './Plot3D'
import { ManifestTables } from './ManifestTables'
import { CompareRuns } from './CompareRuns'
import { Downloads } from './Downloads'

export function Results3D({
  api,
  id,
  onError,
  runs
}: {
  api: Sample3DApi
  id: string
  onError: (e: unknown) => void
  runs: Run3D[]
}) {
  const [tiles, setTiles] = useState<Tile3D[]>([])
  const [tile, setTile] = useState('')
  const [manifest, setManifest] = useState<Manifest3D | null>(null)
  const [height, setHeight] = useState<Height3D | null>(null)
  const [series, setSeries] = useState<Series3D | null>(null)
  const [times, setTimes] = useState<number[]>([])
  const [frame, setFrame] = useState(-1)
  const [points, setPoints] = useState<Point3D[]>([])
  const [section, setSection] = useState<Section3D | null>(null)
  const [initial, setInitial] = useState<Section3D | null>(null)
  const [exaggeration, setExaggeration] = useState(1)
  const [busy, setBusy] = useState(true)
  const [cutBusy, setCutBusy] = useState(false)
  useEffect(() => {
    let alive = true
    void Promise.all([api.tiles(id), api.manifest(id)])
      .then(([t, m]) => {
        if (alive) {
          setTiles(t.tiles)
          setTile(t.tiles[0]?.tile_id ?? '')
          setManifest(m)
          if (!t.tiles.length) setBusy(false)
        }
      })
      .catch((e) => {
        if (alive) {
          onError(e)
          setBusy(false)
        }
      })
    return () => {
      alive = false
    }
  }, [api, id, onError])
  useEffect(() => {
    if (!tile) return
    let alive = true
    setPoints([])
    setSection(null)
    setInitial(null)
    setFrame(-1)
    setSeries(null)
    setTimes([])
    void Promise.all([api.frames(id, tile), api.series(id, tile)])
      .then(([f, s]) => {
        if (alive) {
          setTimes(f.t_s)
          setSeries(s)
        }
      })
      .catch((e) => {
        if (alive) onError(e)
      })
    return () => {
      alive = false
    }
  }, [api, id, tile, onError])
  useEffect(() => {
    if (!tile) return
    let alive = true
    setBusy(true)
    setHeight(null)
    void api
      .height(id, tile, frame)
      .then((h) => {
        if (alive) setHeight(h)
      })
      .catch((e) => {
        if (alive) onError(e)
      })
      .finally(() => {
        if (alive) setBusy(false)
      })
    return () => {
      alive = false
    }
  }, [api, id, tile, frame, onError])
  useEffect(() => {
    if (points.length !== 2 || !tile) {
      setSection(null)
      setInitial(null)
      return
    }
    let alive = true
    setCutBusy(true)
    setSection(null)
    setInitial(null)
    void Promise.all([api.section(id, points, tile, frame), api.section(id, points, tile, 0)])
      .then(([s, first]) => {
        if (alive) {
          setSection(s)
          setInitial(first)
        }
      })
      .catch((e) => {
        if (alive) onError(e)
      })
      .finally(() => {
        if (alive) setCutBusy(false)
      })
    return () => {
      alive = false
    }
  }, [api, id, tile, frame, points, onError])
  const grid = height ? decimateHeight(height.x_um, height.y_um, height.height_um) : null
  const finiteHeights =
    grid?.z.flat().filter((v): v is number => v !== null && Number.isFinite(v)) ?? []
  const spanX = grid ? Math.max(1e-9, Math.abs(grid.x.at(-1)! - grid.x[0])) : 1
  const spanY = grid ? Math.max(1e-9, Math.abs(grid.y.at(-1)! - grid.y[0])) : 1
  const spanZ = finiteHeights.length
    ? Math.max(1e-9, Math.max(...finiteHeights) - Math.min(...finiteHeights))
    : 1
  return (
    <section>
      <h3>Results · {id}</h3>
      <div className="row gap sample3d-controls">
        <label>
          Tile{' '}
          <select value={tile} onChange={(e) => setTile(e.target.value)}>
            {tiles.map((t) => (
              <option key={t.tile_id} value={t.tile_id}>
                {t.tile_id} · {t.kind}
              </option>
            ))}
          </select>
        </label>
        <label>
          Frame{' '}
          <input
            aria-label="Result frame"
            type="range"
            min={0}
            max={Math.max(0, times.length - 1)}
            disabled={!times.length}
            value={frame < 0 ? Math.max(0, times.length - 1) : frame}
            onChange={(e) => setFrame(Number(e.target.value))}
          />{' '}
          {times[frame < 0 ? times.length - 1 : frame] ?? '—'} s
        </label>
      </div>
      {busy && (
        <div className="sample3d-loading" role="status">
          Loading height map…
        </div>
      )}
      <p className="muted">
        Click two distinct points on the height map to define a section. {points.length}/2 points
        selected.
      </p>
      <div className="sample3d-plots">
        {grid && (
          <Plot3D
            title="Height (µm)"
            data={[
              { type: 'heatmap', ...grid, colorbar: { title: 'Height (µm)' } },
              {
                type: 'scatter',
                mode: 'lines+markers',
                x: points.map((p) => p[0]),
                y: points.map((p) => p[1]),
                line: { color: '#cf222e' },
                name: 'Section cut'
              }
            ]}
            onClick={(x, y) => setPoints((p) => polylineFromClicks(p, [x, y]))}
          />
        )}
        {section && (
          <Plot3D
            title="Cu / Si section"
            data={[
              ...outlineTraces(section.cu_outline, 'Cu selected frame', '#b87333', false, true),
              ...outlineTraces(initial?.cu_outline ?? [], 'Cu initial', '#b87333', true),
              ...outlineTraces(section.si_outline, 'Si', '#888888')
            ]}
            layout={{
              xaxis: { title: 's (µm)' },
              yaxis: { title: 'z (µm)', scaleanchor: 'x', scaleratio: 1 }
            }}
          />
        )}
      </div>
      {cutBusy && (
        <div className="sample3d-loading" role="status">
          Loading section…
        </div>
      )}
      <label>
        Z exaggeration {exaggeration}×{' '}
        <input
          aria-label="Z exaggeration"
          type="range"
          min={1}
          max={20}
          value={exaggeration}
          onChange={(e) => setExaggeration(Number(e.target.value))}
        />
      </label>
      <div className="sample3d-plots">
        {grid && (
          <Plot3D
            title="Height surface"
            data={[{ type: 'surface', ...grid, colorbar: { title: 'Height (µm)' } }]}
            layout={{
              scene: {
                xaxis: { title: 'x (µm)' },
                yaxis: { title: 'y (µm)' },
                zaxis: { title: 'Height (µm)' },
                aspectmode: 'manual',
                aspectratio: { x: 1, y: spanY / spanX, z: (exaggeration * spanZ) / spanX }
              }
            }}
          />
        )}
        {series && (
          <Plot3D
            title="Time series"
            data={[
              { type: 'scatter', mode: 'lines', name: 'V', x: series.t_s, y: series.V },
              {
                type: 'scatter',
                mode: 'lines',
                name: 'ct_min_front',
                x: series.t_s,
                y: series.ct_min_front,
                yaxis: 'y2'
              }
            ]}
            layout={{
              xaxis: { title: 'Time (s)' },
              yaxis: { title: 'V (V)' },
              yaxis2: { title: 'ct_min_front', overlaying: 'y', side: 'right' }
            }}
          />
        )}
      </div>
      {manifest && (
        <ManifestTables manifest={manifest} artifactUrl={(path) => api.artifact(id, path)} />
      )}
      <CompareRuns api={api} id={id} runs={runs} points={points} section={section} frame={frame} />
      <Downloads api={api} id={id} tile={tile} frame={frame} section={section} />
    </section>
  )
}
