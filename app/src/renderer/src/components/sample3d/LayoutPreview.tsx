import { useEffect, useRef, useState } from 'react'
import type { Plan3D, SampleSpec3D } from '../../api/client3d'
import { expandFeatures, featurePath, tileFootprints } from '../../lib/sample3d'

export function LayoutPreview({ layout, tiles = [] }: {
  layout: SampleSpec3D['layout']; tiles?: Plan3D['tiles']
}) {
  const [view, setView] = useState({ zoom: 1, x: 0, y: 0 })
  const svg = useRef<SVGSVGElement>(null)
  const drag = useRef<{ x: number; y: number; vx: number; vy: number } | null>(null)
  useEffect(() => {
    const el = svg.current
    if (!el) return
    const wheel = (e: WheelEvent) => {
      e.preventDefault()
      setView((v) => ({ ...v, zoom: Math.max(.2, Math.min(30, v.zoom * Math.exp(-e.deltaY * .001))) }))
    }
    el.addEventListener('wheel', wheel, { passive: false })
    return () => el.removeEventListener('wheel', wheel)
  }, [])
  const width = Math.max(1, Number(layout.die_um[0]) || 1)
  const height = Math.max(1, Number(layout.die_um[1]) || 1)
  const features = layout.features ?? []
  const { instances, omitted } = expandFeatures(features)
  const footprints = tileFootprints(tiles, features)
  const pad = Math.max(width, height) * .08
  const vw = (width + 2 * pad) / view.zoom, vh = (height + 2 * pad) / view.zoom
  const scale = 10 ** Math.floor(Math.log10(vw / 4))
  const font = Math.max(vw, vh) / 55
  return <section className="sample3d-layout">
    <div className="row gap"><h4>Layout preview</h4><button type="button" className="btn btn-small"
      onClick={() => setView({ zoom: 1, x: 0, y: 0 })}>Reset view</button></div>
    <svg ref={svg} aria-label="Layout preview in micrometres" role="img" className="sample3d-layout-svg"
      viewBox={`${-pad + view.x} ${-pad + view.y} ${vw} ${vh}`}
      onPointerDown={(e) => { if (e.button !== 0) return; e.currentTarget.setPointerCapture(e.pointerId); drag.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y } }}
      onPointerMove={(e) => { const d = drag.current; if (!d) return
        const box = e.currentTarget.getBoundingClientRect()
        const units = Math.max(vw / box.width, vh / box.height)
        setView((v) => ({ ...v, x: d.vx - (e.clientX - d.x) * units, y: d.vy - (e.clientY - d.y) * units })) }}
      onPointerUp={() => { drag.current = null }} onPointerCancel={() => { drag.current = null }}>
      <g transform={`translate(0 ${height}) scale(1 -1)`}>
        <rect x={0} y={0} width={width} height={height} className="sample3d-die" />
        {instances.map(({ feature: f, index, ix, iy }) => <path key={`${index}-${ix}-${iy}`}
          d={featurePath(f)} className="sample3d-feature">
          <title>{`Feature ${index} [${ix}, ${iy}] · ${f.kind === 'circle' ? `radius ${f.r_um}` : `${f.w_um} × ${f.h_um}`} µm`}</title>
        </path>)}
        {footprints.map((t) => <rect key={t.tile_id} x={t.x} y={t.y} width={t.width} height={t.height} className="sample3d-tile" />)}
      </g>
      {footprints.map((t) => <text key={t.tile_id} x={t.x} y={height - t.y - t.height} fontSize={font} className="sample3d-svg-label">{t.tile_id} (approx.)</text>)}
      <g transform={`translate(${-pad + view.x + vw * .05} ${-pad + view.y + vh * .92})`}>
        <path d={`M 0 ${-font / 2} V 0 H ${scale} V ${-font / 2}`} className="sample3d-scale" />
        <text x={0} y={font * 1.2} fontSize={font} className="sample3d-svg-label">{scale} µm</text>
      </g>
    </svg>
    <p className="muted">Wheel to zoom; drag to pan. Feature indices are zero-based. {omitted > 0 && <strong>+{omitted.toLocaleString()} more (2000 instance preview cap). </strong>}
      {footprints.length > 0 && 'Dashed tile footprints are approximate: plan origins are unavailable; centred on covered feature bounds with shape × dx margins.'}</p>
  </section>
}
