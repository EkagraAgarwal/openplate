/**
 * Plot-kind mapping: normalize a backend plot object into a discriminated union
 * the renderer can switch on, and build Plotly traces/layouts for each kind.
 * Pure (no plotly import) so tests can exercise it in node. Unit-tested in
 * tests/plots.test.ts.
 */
import type {
  BarPlot,
  ContoursPlot,
  HeatmapPlot,
  ImagePlot,
  LinePlot,
  Plot,
  PlotSeries
} from '../types'

export type NormalizedKind = 'line' | 'heatmap' | 'bar' | 'contours' | 'image' | 'unknown'

export interface NormalizedPlot {
  kind: NormalizedKind
  plot: Plot | null
  title: string
}

/** Map a raw plot object's `kind` to a normalized kind. Unknown -> 'unknown'. */
export function normalizePlotKind(plot: Plot): NormalizedPlot {
  const kind = String((plot as { kind?: unknown }).kind ?? '').toLowerCase()
  const title = String((plot as { title?: unknown }).title ?? '')
  switch (kind) {
    case 'line':
      return { kind: 'line', plot, title }
    case 'heatmap':
      return { kind: 'heatmap', plot, title }
    case 'bar':
      return { kind: 'bar', plot, title }
    case 'contours':
    case 'contour':
      return { kind: 'contours', plot, title }
    case 'image':
      return { kind: 'image', plot, title }
    default:
      return { kind: 'unknown', plot: null, title }
  }
}

export interface PlotlyTrace {
  type: string
  [k: string]: unknown
}

export interface PlotlyLayout {
  title?: { text: string }
  xaxis?: Record<string, unknown>
  yaxis?: Record<string, unknown>
  [k: string]: unknown
}

const seriesMode = (s: PlotSeries): string => {
  const m = (s.mode ?? '').toLowerCase()
  if (m === 'markers' || m === 'markers+lines' || m === 'lines+markers') return m
  return 'lines'
}

export function lineTraces(plot: LinePlot): PlotlyTrace[] {
  const x = plot.x ?? []
  return (plot.series ?? []).map((s) => ({
    type: 'scatter',
    mode: seriesMode(s),
    name: s.name ?? 'series',
    x,
    y: s.y
  }))
}

export function heatmapTraces(plot: HeatmapPlot | ImagePlot, grayscale = false): PlotlyTrace[] {
  return [
    {
      type: 'heatmap',
      x: plot.x ?? [],
      y: plot.y ?? [],
      z: plot.z ?? [],
      ...(grayscale ? { colorscale: 'gray' } : {}),
      colorbar: plot.zlabel ? { title: { text: plot.zlabel } } : undefined,
      hovertemplate: '%{x:.4g}, %{y:.4g}: %{z:.4g}<extra></extra>'
    }
  ]
}

export function barTraces(plot: BarPlot): PlotlyTrace[] {
  const x = plot.x ?? []
  if (plot.series?.length) {
    return plot.series.map((s, i) => ({
      type: 'bar',
      name: s.name ?? (plot.series!.length > 1 ? `series ${i + 1}` : 'value'),
      x,
      y: s.y
    }))
  }
  return [{ type: 'bar', x, y: plot.y ?? [], name: plot.ylabel ?? 'value' }]
}

/**
 * Contours: polyline frames. When `mirror` is true (feature-fill via
 * cross-sections where only r >= 0 is simulated), each polyline is drawn twice:
 * once as-is and once with r -> -r (reversed), forming the full symmetric
 * cross-section.
 */
export function contourFrameTraces(plot: ContoursPlot, frameIndex: number, mirror = false): PlotlyTrace[] {
  const frames = plot.frames ?? []
  const frame = frames[Math.max(0, Math.min(frameIndex, frames.length - 1))]
  if (!frame) return []
  const traces: PlotlyTrace[] = []
  frame.lines.forEach((line, i) => {
    const xs = line.map((pt) => pt[0])
    const ys = line.map((pt) => pt[1])
    traces.push({ type: 'scatter', mode: 'lines', x: xs, y: ys, name: `line ${i + 1}`, hoverinfo: 'skip' })
    if (mirror && xs.some((v) => v !== 0)) {
      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: [...xs].reverse().map((v) => (v === 0 ? 0 : -v)),
        y: [...ys].reverse(),
        name: `line ${i + 1} (mirror)`,
        hoverinfo: 'skip',
        line: { dash: 'dot' }
      })
    }
  })
  return traces
}

export function buildLayout(plot: Plot, mirror = false): PlotlyLayout {
  const n = normalizePlotKind(plot)
  const base: PlotlyLayout = {
    ...(n.title ? { title: { text: n.title } } : {}),
    margin: { t: n.title ? 46 : 24, r: 24, b: 48, l: 64 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { size: 12 },
    legend: { orientation: 'h', y: -0.25 }
  }
  const p = plot as { xlabel?: string; ylabel?: string }
  if (p.xlabel) base.xaxis = { ...(base.xaxis ?? {}), title: { text: p.xlabel } }
  if (p.ylabel) base.yaxis = { ...(base.yaxis ?? {}), title: { text: p.ylabel } }

  if (n.kind === 'contours') {
    // equal aspect
    base.yaxis = { ...(base.yaxis ?? {}), scaleanchor: 'x', scaleratio: 1 }
    if (mirror) {
      base.xaxis = { ...(base.xaxis ?? {}), constrain: 'domain' }
    }
  }
  return base
}

export function config(): Record<string, unknown> {
  return {
    displaylogo: false,
    responsive: true,
    toImageButtonOptions: { format: 'png', scale: 2 }
  }
}
