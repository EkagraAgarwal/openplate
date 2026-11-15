import { useRef, useState } from 'react'
import { Sample3DApi, Sample3DError, type FieldIssue, type LayoutFit3D, type SampleSpec3D } from '../../api/client3d'
import { LayoutPreview } from './LayoutPreview'

export function LayoutImport({ api, onUse }: { api: Sample3DApi; onUse: (layout: Pick<SampleSpec3D['layout'], 'die_um' | 'features'>) => void }) {
  const [file, setFile] = useState<File | null>(null)
  const [pixel, setPixel] = useState('')
  const [map, setMap] = useState('')
  const [fit, setFit] = useState<LayoutFit3D | null>(null)
  const [error, setError] = useState<FieldIssue | null>(null)
  const [busy, setBusy] = useState(false)
  const revision = useRef(0)
  const invalidate = () => { revision.current++; setFit(null); setError(null) }
  const image = /\.(png|tiff?|webp)$/i.test(file?.name ?? '')
  const submit = async () => {
    if (!file) return
    const version = revision.current
    setBusy(true); setError(null); setFit(null)
    try {
      if (file.size > 20 * 1024 * 1024) throw new Sample3DError({ field: 'content_b64', message: 'File exceeds 20 MB.', fix: 'Choose a smaller layout file.' })
      if (image && !(Number(pixel) > 0)) throw new Sample3DError({ field: 'pixel_um', message: 'Image scale must be positive.', fix: 'Enter micrometres per pixel.' })
      const layer_map: Record<string, [number, number]> = {}
      for (const line of map.split(/\r?\n/).filter((l) => l.trim())) {
        const match = line.match(/^\s*([^=]+?)\s*=\s*(\d+)\s*\/\s*(\d+)\s*$/)
        if (!match) throw new Sample3DError({ field: 'layer_map', message: `Invalid mapping: ${line}`, fix: 'Use one name=layer/datatype per line, e.g. etch=1/0.' })
        layer_map[match[1]] = [Number(match[2]), Number(match[3])]
      }
      const result = await api.importLayout(file.name, await file.arrayBuffer(), {
        ...(image ? { pixel_um: Number(pixel) } : {}),
        ...(map.trim() ? { layer_map } : {})
      })
      if (revision.current === version) setFit(result)
    } catch (e) {
      if (revision.current === version) setError(e instanceof Sample3DError ? e.issue : { field: 'file', message: 'Could not read or import this file.', fix: 'Choose a supported file and retry.' })
    } finally { setBusy(false) }
  }
  return <details className="card sample3d-import"><summary>Import layout file</summary>
    <div className="sample3d-fields">
      <label>Layout file<input type="file" accept=".gds,.dxf,.png,.tif,.tiff,.webp" disabled={busy}
        onChange={(e) => { invalidate(); setFile(e.target.files?.[0] ?? null) }} /></label>
      <label>pixel_um (images)<input type="number" min="0" step="any" disabled={busy} value={pixel} onChange={(e) => { invalidate(); setPixel(e.target.value) }} /></label>
      <label>Layer map (optional)<textarea placeholder="etch=1/0" disabled={busy} value={map} onChange={(e) => { invalidate(); setMap(e.target.value) }} /></label>
    </div>
    <button type="button" className="btn" disabled={!file || busy} onClick={() => void submit()}>{busy ? 'Importing…' : 'Fit layout'}</button>
    {error && <p className="field-error" role="alert"><strong>{error.field}</strong>: {error.message} {error.fix}</p>}
    {fit && <>
      <p>{fit.features.length} features fitted · {fit.source.n_polygons} source polygons · {fit.source.filename}</p>
      <h4>Unfit ({fit.unfit.length})</h4><ul>{fit.unfit.map((u, i) => <li key={i}>Polygon {u.index}: area {u.area_um2} µm², relative error {u.error_rel}; bounds {u.bounds_um.join(', ')}</li>)}</ul>
      <h4>Skipped ({fit.skipped.length})</h4><ul>{fit.skipped.map((s, i) => <li key={i}>{typeof s === 'string' ? s : JSON.stringify(s)}</li>)}</ul>
      <LayoutPreview layout={{ die_um: fit.die_um, features: fit.features }} />
      <button type="button" className="btn btn-primary" onClick={() => {
        if (window.confirm('Replace layout.features and die_um with the fitted layout? Unfit and skipped polygons will not be included.')) onUse({ die_um: fit.die_um, features: fit.features })
      }}>Use these features</button>
    </>}
  </details>
}
