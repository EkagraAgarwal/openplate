import { useState } from 'react'
import Plotly from 'plotly.js-dist-min'
import type { Sample3DApi, Section3D } from '../../api/client3d'
import { sectionCsv } from '../../lib/sample3d'

/** Reused beside each Plotly plot, targeting its own div, never a global selector. */
export function PlotPngDownload({ element, title, disabled }: { element: () => HTMLElement | null; title: string; disabled?: boolean }) {
  const [error, setError] = useState('')
  const download = async () => {
    const el = element()
    if (!el) return
    try {
      setError('')
      await (Plotly as unknown as { downloadImage: (el: HTMLElement, opts: Record<string, unknown>) => Promise<unknown> }).downloadImage(el, {
        format: 'png', filename: title.replace(/[^\w.-]+/g, '_'), width: el.clientWidth || 800, height: el.clientHeight || 360, scale: 2
      })
    } catch { setError('PNG download failed. Reload the plot and retry.') }
  }
  return <figcaption><button type="button" className="btn btn-small" disabled={disabled} onClick={() => void download()}>Download {title} PNG</button>{error && <span role="alert" className="field-error">{error}</span>}</figcaption>
}

export function Downloads({ api, id, tile, frame, section }: {
  api: Sample3DApi; id: string; tile: string; frame: number; section: Section3D | null
}) {
  const downloadSection = () => {
    if (!section) return
    const url = URL.createObjectURL(new Blob([sectionCsv(section)], { type: 'text/csv;charset=utf-8' }))
    const a = document.createElement('a'); a.href = url; a.download = `${id}-section.csv`
    document.body.append(a); a.click(); a.remove()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  return <section className="card"><h4>Downloads</h4><div className="row gap sample3d-controls">
    <button type="button" className="btn" disabled={!section} onClick={downloadSection}>Section CSV</button>
    {tile && <>
      <a className="btn" download={`${id}-height.csv`} href={api.exportUrl(id, 'height_csv', { tile_id: tile, frame })}>Height CSV</a>
      <a className="btn" download={`${id}.stl`} href={api.exportUrl(id, 'stl', { tile_id: tile, frame })}>STL</a>
    </>}
  </div></section>
}
