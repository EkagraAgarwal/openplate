import { useEffect, useRef, useState } from 'react'
import Plotly from 'plotly.js-dist-min'
import { useApp } from '../../lib/context'

interface PlotElement extends HTMLDivElement {
  on: (event: string, handler: (event: { points: { x: number; y: number }[] }) => void) => void
  removeAllListeners: (event: string) => void
}
export function Plot3D({
  data,
  title,
  layout = {},
  onClick
}: {
  data: unknown[]
  title: string
  layout?: Record<string, unknown>
  onClick?: (x: number, y: number) => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { theme } = useApp()
  const [failed, setFailed] = useState(false)
  useEffect(() => {
    const el = ref.current as PlotElement | null
    if (!el) return
    let alive = true
    const css = getComputedStyle(document.documentElement)
    void Plotly.react(
      el,
      data,
      {
        title: { text: title },
        autosize: true,
        height: 360,
        paper_bgcolor: css.getPropertyValue('--bg-elev').trim(),
        plot_bgcolor: css.getPropertyValue('--bg-elev').trim(),
        font: { color: css.getPropertyValue('--fg').trim() },
        margin: { l: 60, r: 40, t: 55, b: 55 },
        xaxis: { title: 'x (µm)' },
        yaxis: { title: 'y (µm)', scaleanchor: 'x', scaleratio: 1 },
        ...layout
      },
      { responsive: true, displaylogo: false }
    )
      .then(() => {
        if (!alive) return
        setFailed(false)
        el.removeAllListeners('plotly_click')
        if (onClick)
          el.on('plotly_click', (e) => {
            const p = e.points[0]
            if (p && typeof p.x === 'number' && typeof p.y === 'number') onClick(p.x, p.y)
          })
      })
      .catch(() => {
        if (alive) setFailed(true)
      })
    // The project's minimal Plotly declaration omits the public resize API.
    const resize = (Plotly as unknown as { Plots: { resize: (el: HTMLElement) => Promise<void> } })
      .Plots.resize
    const observer = new ResizeObserver(() => {
      void resize(el).catch(() => {})
    })
    observer.observe(el)
    return () => {
      alive = false
      observer.disconnect()
      el.removeAllListeners?.('plotly_click')
      Plotly.purge(el)
    }
  }, [data, title, layout, onClick, theme])
  return (
    <figure className="sample3d-plot card">
      <div ref={ref} aria-label={title} style={{ height: 360 }} />
      {failed && (
        <figcaption className="field-error">
          Plot unavailable. Try another tile or reload the result.
        </figcaption>
      )}
    </figure>
  )
}
