import { useEffect, useState } from 'react'
import { Sample3DApi, Sample3DError, type Point3D, type Run3D, type Section3D } from '../../api/client3d'
import { outlineTraces } from '../../lib/sample3d'
import { Plot3D } from './Plot3D'

export function CompareRuns({ api, id, runs, points, section, frame }: {
  api: Sample3DApi; id: string; runs: Run3D[]; points: Point3D[]; section: Section3D | null; frame: number
}) {
  const [other, setOther] = useState('')
  const [result, setResult] = useState<Section3D | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    setError(''); setBusy(false)
    if (!other || !section || points.length < 2) { setResult(null); return }
    let alive = true
    setBusy(true)
    // Do not send the current tile ID: the other run has its own tile namespace.
    void api.section(other, points, undefined, frame).then((s) => { if (alive) setResult(s) })
      .catch((e) => { if (alive) setError(e instanceof Sample3DError ? `${e.issue.field}: ${e.issue.message} ${e.issue.fix}` : 'Comparison unavailable. Check the other run and retry.') })
      .finally(() => { if (alive) setBusy(false) })
    return () => { alive = false }
  }, [api, other, points, section, frame])
  const label = (runId: string) => runs.find((r) => r.run_id === runId)?.label || runId
  return <section className="card">
    <label>Compare with another done run <select value={other} onChange={(e) => setOther(e.target.value)}>
      <option value="">No comparison</option>
      {runs.filter((r) => r.run_id !== id && ['done', 'succeeded', 'completed'].includes(r.status)).map((r) => <option key={r.run_id} value={r.run_id}>{r.label || r.run_id}</option>)}
    </select></label>
    {other && !section && <p className="muted">Choose a section cut on the height map first.</p>}
    {busy && <p role="status">Loading comparison…</p>}
    {error && <p role="alert" className="field-error">{error}</p>}
    {section && result && <Plot3D revision={`${id}:${other}`} title="Section comparison" data={[
      ...outlineTraces(section.cu_outline, `${label(id)} · Cu`, '#b87333'),
      ...outlineTraces(result.cu_outline, `${label(other)} · Cu (comparison)`, '#3276b1', true)
    ]} layout={{ xaxis: { title: 's (µm)' }, yaxis: { title: 'z (µm)', scaleanchor: 'x', scaleratio: 1 } }} />}
  </section>
}
