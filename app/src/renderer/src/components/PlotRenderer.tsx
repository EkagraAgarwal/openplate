import { useEffect, useRef, useState } from 'react'
import type { Plot } from '../types'
import {
  barTraces,
  buildLayout,
  config,
  contourFrameTraces,
  heatmapTraces,
  lineTraces,
  normalizePlotKind
} from '../lib/plots'
import Plotly from 'plotly.js-dist-min'

export interface PlotRendererProps {
  plot: Plot
  /** mirror r -> -r (feature-fill via cross-section) */
  mirror?: boolean
  height?: number
}

function useDivPlot(data: unknown[], layout: Record<string, unknown>, deps: unknown[]) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    void Plotly.react(el, data, layout, config()).catch(() => {
      /* first render */
    })
    return () => {
      try {
        Plotly.purge(el)
      } catch {
        /* noop */
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return ref
}

/** Interactive contours: time slider + play button. */
function ContourPlot({
  plot,
  mirror = false,
  height = 340
}: PlotRendererProps & { plot: Extract<Plot, { kind: 'contours' }> }) {
  const frames = plot.frames ?? []
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const maxFrame = Math.max(0, frames.length - 1)

  useEffect(() => {
    if (!playing || frames.length < 2) return
    const id = setInterval(() => setFrame((f) => (f + 1) % frames.length), 600)
    return () => clearInterval(id)
  }, [playing, frames.length])

  const data = contourFrameTraces(plot, Math.min(frame, maxFrame), mirror)
  const layout: Record<string, unknown> = { ...buildLayout(plot, mirror), height }
  const t = frames[Math.min(frame, maxFrame)]?.t
  if (frames.length > 1) {
    layout.annotations = [
      {
        text: t !== undefined ? `t = ${t}` : '',
        xref: 'paper',
        yref: 'paper',
        x: 1,
        y: 1.1,
        showarrow: false,
        font: { size: 11 }
      }
    ]
  }
  const ref = useDivPlot(data, layout, [data, layout.height, mirror])

  if (frames.length === 0) return <div className="plot-empty">no contour frames</div>

  return (
    <div className="plot-block">
      <div className="plot-scrubber">
        <button
          type="button"
          className="btn btn-small"
          onClick={() => setPlaying((p) => !p)}
          title={playing ? 'Pause' : 'Play'}
        >
          {playing ? '❚❚' : '▶'}
        </button>
        <input
          type="range"
          min={0}
          max={maxFrame}
          value={Math.min(frame, maxFrame)}
          onChange={(e) => {
            setPlaying(false)
            setFrame(Number(e.target.value))
          }}
        />
        <span className="mono plot-frame-label">
          frame {Math.min(frame, maxFrame) + 1}/{frames.length}
          {t !== undefined ? ` · t=${t}` : ''}
        </span>
      </div>
      <div className="plot-div" ref={ref} style={{ height }} />
    </div>
  )
}

export function PlotRenderer(props: PlotRendererProps) {
  const { plot, mirror = false, height = 340 } = props
  const norm = normalizePlotKind(plot)

  if (norm.kind === 'unknown') {
    return <div className="plot-empty">Unsupported plot kind: {String((plot as { kind?: string }).kind)}</div>
  }
  if (norm.kind === 'contours' && plot.kind === 'contours') {
    return <ContourPlot {...props} plot={plot} />
  }

  const data = tracesFor(plot, mirror)
  const layout = { ...buildLayout(plot, mirror), height }
  const ref = useDivPlot(data, layout, [plot, mirror, height])

  return <div className="plot-div" ref={ref} style={{ height }} />
}

export function tracesFor(plot: Plot, mirror: boolean): unknown[] {
  const norm = normalizePlotKind(plot)
  switch (norm.kind) {
    case 'line':
      return lineTraces(plot as Extract<Plot, { kind: 'line' }>)
    case 'heatmap':
      return heatmapTraces(plot as Extract<Plot, { kind: 'heatmap' }>)
    case 'image':
      return heatmapTraces(plot as Extract<Plot, { kind: 'image' }>, true)
    case 'bar':
      return barTraces(plot as Extract<Plot, { kind: 'bar' }>)
    case 'contours':
      return contourFrameTraces(plot as Extract<Plot, { kind: 'contours' }>, 0, mirror)
    default:
      return []
  }
}
