import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Sample3DApi,
  Sample3DError,
  type FieldIssue,
  type Plan3D,
  type Run3D,
  type SampleSpec3D
} from '../api/client3d'
import { useApp } from '../lib/context'
import { defaultSpec3D, formToSpec, specToForm } from '../lib/sample3d'
import { SpecEditor } from '../components/sample3d/SpecEditor'
import { Results3D } from '../components/sample3d/Results3D'
import { LayoutImport } from '../components/sample3d/LayoutImport'
import { estimateSpec, exampleSpec, roughRuntime } from '../lib/estimates3d'

const terminal = (status: string) =>
  ['done', 'succeeded', 'completed', 'failed', 'error', 'cancelled', 'canceled'].includes(status)
const done = (status: string) => ['done', 'succeeded', 'completed'].includes(status)
export function Sample3DPage() {
  const { api: base } = useApp()
  const api = useMemo(() => (base ? new Sample3DApi(base.baseUrl) : null), [base])
  const [spec, setSpec] = useState(() => specToForm(defaultSpec3D))
  const [examples, setExamples] = useState<{ name: string; title: string; spec: SampleSpec3D }[]>(
    []
  )
  const [errors, setErrors] = useState<FieldIssue[]>([])
  const [plan, setPlan] = useState<Plan3D | null>(null)
  const [estimate, setEstimate] = useState<Plan3D | null>(null)
  const [estimateState, setEstimateState] = useState('Estimate unavailable')
  const [exampleEstimates, setExampleEstimates] = useState<Record<string, number>>({})
  const [runs, setRuns] = useState<Run3D[]>([])
  const [run, setRun] = useState<Run3D | null>(null)
  const [label, setLabel] = useState('')
  const [busy, setBusy] = useState('')
  const [validJson, setValidJson] = useState(true)
  const [validated, setValidated] = useState(false)
  const [editorKey, setEditorKey] = useState(0)
  const reportError = useCallback((e: unknown) => {
    setErrors([
      e instanceof Sample3DError
        ? e.issue
        : {
            field: '',
            message: 'The request could not be completed.',
            fix: 'Check the service connection and retry.'
          }
    ])
  }, [])
  useEffect(() => {
    if (!api) return
    let alive = true
    setBusy('Loading examples and runs')
    void Promise.all([api.examples(), api.runs()])
      .then(([e, r]) => {
        if (alive) {
          setExamples(e.examples)
          setRuns(r.runs)
        }
      })
      .catch((e) => {
        if (alive) reportError(e)
      })
      .finally(() => {
        if (alive) setBusy('')
      })
    return () => {
      alive = false
    }
  }, [api, reportError])
  useEffect(() => {
    if (!api) return
    let alive = true
    setExampleEstimates({})
    // Sequential cheap plans avoid flooding the service when the picker loads.
    void (async () => {
      for (const example of examples) {
        if (!alive) break
        try {
          const p = await estimateSpec(api, exampleSpec(example.spec))
          if (alive && p) setExampleEstimates((old) => ({ ...old, [example.name]: p.est_time_s }))
        } catch { /* The example remains usable if planning is unavailable. */ }
      }
    })()
    return () => { alive = false }
  }, [api, examples])
  useEffect(() => {
    let alive = true
    setEstimate(null)
    setEstimateState(validJson && api ? 'Estimating…' : 'Estimate unavailable')
    if (!api || !validJson) return
    const timer = setTimeout(() => {
      void estimateSpec(api, spec).then((p) => {
        if (alive) {
          setEstimate(p)
          setEstimateState(p ? roughRuntime(p.est_time_s) : 'Estimate unavailable: check specification')
        }
      }).catch(() => { if (alive) setEstimateState('Estimate unavailable') })
    }, 400)
    return () => { alive = false; clearTimeout(timer) }
  }, [api, spec, validJson])
  const physics = spec.numerics.physics ?? 'L1'
  const runId = run?.run_id
  useEffect(() => {
    if (!api || !runId) return
    let alive = true
    let timer: ReturnType<typeof setTimeout> | undefined
    const poll = async (): Promise<void> => {
      try {
        const r = await api.status(runId)
        if (!alive) return
        setRun(r)
        if (terminal(r.status)) {
          const list = await api.runs()
          if (alive) setRuns(list.runs)
        } else timer = setTimeout(() => void poll(), 1000)
      } catch (e) {
        if (alive) {
          reportError(e)
          timer = setTimeout(() => void poll(), 1000)
        }
      }
    }
    void poll()
    return () => {
      alive = false
      if (timer) clearTimeout(timer)
    }
  }, [api, runId, reportError])
  const change = (s: SampleSpec3D): void => {
    setSpec(s)
    setPlan(null)
    setErrors([])
    setValidated(false)
  }
  const action = async (
    kind: 'Validate' | 'Plan' | 'Run' | 'Cancel' | 'Refresh',
    id?: string
  ): Promise<void> => {
    if (!api) return
    setBusy(kind)
    setErrors([])
    try {
      if (kind === 'Validate') {
        const v = await api.validate(formToSpec(spec))
        setErrors(v.errors)
        setValidated(v.ok)
      }
      if (kind === 'Plan') {
        const p = await api.plan(formToSpec(spec))
        setErrors(p.errors)
        setPlan(p)
      }
      if (kind === 'Run') {
        const r = await api.run(formToSpec(spec), label.trim() || undefined)
        setRun(await api.status(r.run_id))
        setRuns((await api.runs()).runs)
      }
      if (kind === 'Cancel' && run) {
        setRun(await api.cancel(run.run_id))
        setRuns((await api.runs()).runs)
      }
      if (kind === 'Refresh') {
        setRuns((await api.runs()).runs)
        if (id) setRun(await api.status(id))
      }
    } catch (e) {
      reportError(e)
    } finally {
      setBusy('')
    }
  }
  return (
    <div className="page sample3d-page">
      <div className="sample3d-banner callout callout-warn">
        {spec.additives
          ? `Uncalibrated: L3 physics, ${spec.additives.mode} additives (uncalibrated). Compare shapes, not absolute thickness.`
          : `Uncalibrated: ${physics} physics, no additives. Compare shapes, not absolute thickness.`}
      </div>
      <h2>3-D sample</h2>
      <div className="row gap sample3d-controls">
        <label>
          Example{' '}
          <select
            aria-label="Example"
            defaultValue=""
            disabled={Boolean(busy)}
            onChange={(e) => {
              const example = examples.find((x) => x.name === e.target.value)
              if (example) {
                change(exampleSpec(example.spec))
                setValidJson(true)
                setEditorKey((k) => k + 1)
              }
            }}
          >
            <option value="">Choose a synthetic example…</option>
            {examples.map((e) => (
              <option key={e.name} value={e.name}>
                {e.title}
                {' · '}{exampleEstimates[e.name] === undefined ? 'Estimate unavailable' : roughRuntime(exampleEstimates[e.name])}
              </option>
            ))}
          </select>
        </label>
      </div>
      <SpecEditor
        key={editorKey}
        spec={spec}
        onChange={change}
        errors={errors}
        onValid={setValidJson}
        tiles={plan?.tiles}
        estimateText={estimateState + (estimate ? ` · ${estimate.n_cells.toLocaleString()} cells` : '')}
      />
      {api && <LayoutImport api={api} onUse={(layout) => {
        change({ ...spec, layout: { ...spec.layout, ...layout } })
        setValidJson(true)
        setEditorKey((k) => k + 1)
      }} />}
      {errors.length > 0 && (
        <div className="callout callout-warn" role="alert">
          <ul>
            {errors.map((e, i) => (
              <li key={i}>
                <strong>{e.field || 'Request'}</strong>: {e.message} <span>{e.fix}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {validated && <p className="sample3d-pass">Specification valid.</p>}
      <div className="row gap sample3d-controls">
        <label>
          Run label{' '}
          <input aria-label="Run label" value={label} onChange={(e) => setLabel(e.target.value)} />
        </label>
        {(['Validate', 'Plan', 'Run'] as const).map((kind) => (
          <button
            type="button"
            key={kind}
            className={`btn ${kind === 'Run' ? 'btn-primary' : ''}`}
            disabled={!api || Boolean(busy) || !validJson}
            onClick={() => void action(kind)}
          >
            {kind}
          </button>
        ))}
      </div>
      {busy && (
        <div className="sample3d-loading" role="status">
          {busy}…
        </div>
      )}
      {plan && (
        <section className="card">
          <h3>Plan</h3>
          <p>
            {plan.n_cells.toLocaleString()} cells · {roughRuntime(plan.est_time_s)}
          </p>
          {plan.est_time_s > 1800 && (
            <div className="callout callout-warn">
              Estimated time exceeds 30 min. Consider preview quality or a coarser dx.
            </div>
          )}
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  {[
                    'Tile',
                    'Kind',
                    'Features',
                    'Shape',
                    'dx (µm)',
                    'Cells',
                    'Est. seconds',
                    'Weight',
                    'Assumptions / flags'
                  ].map((s) => (
                    <th key={s}>{s}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {plan.tiles.map((t) => (
                  <tr key={t.tile_id}>
                    <td>{t.tile_id}</td>
                    <td>{t.kind}</td>
                    <td>{t.feature_ids.join(', ')}</td>
                    <td>{t.shape.join(' × ')}</td>
                    <td>{t.dx_um}</td>
                    <td>{t.n_cells}</td>
                    <td>{t.est_time_s.toFixed(1)}</td>
                    <td>{t.weight}</td>
                    <td>{[...t.assumptions, ...t.flags].join('; ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
      <section className="card">
        <div className="row gap">
          <h3>Runs</h3>
          <button
            type="button"
            className="btn btn-small"
            disabled={!api || Boolean(busy)}
            onClick={() => void action('Refresh')}
          >
            Refresh
          </button>
        </div>
        {!runs.length && <p className="muted">No 3-D runs yet.</p>}
        <ul className="job-list">
          {runs.map((r) => (
            <li key={r.run_id} className="row gap sample3d-controls">
              <span>
                {r.label || r.run_id} · {r.status}
              </span>
              <button
                type="button"
                className="btn btn-small"
                disabled={Boolean(busy)}
                onClick={() => void action('Refresh', r.run_id)}
              >
                Open
              </button>
            </li>
          ))}
        </ul>
      </section>
      {run && (
        <section className="card">
          <h3>{run.label || run.run_id}</h3>
          <p>
            {run.status} · {run.message}
          </p>
          <progress
            aria-label="Run progress"
            max={1}
            value={Math.max(0, Math.min(1, run.progress || 0))}
          />{' '}
          <span>{Math.round((run.progress || 0) * 100)}%</span>
          {!terminal(run.status) && (
            <button
              type="button"
              className="btn"
              disabled={Boolean(busy)}
              onClick={() => void action('Cancel')}
            >
              Cancel
            </button>
          )}
          {run.error && (
            <p className="field-error">
              {typeof run.error === 'object'
                ? `${run.error.field ?? 'Run'}: ${run.error.error} ${run.error.fix ?? ''}`
                : 'Run failed. Check the specification and retry.'}
            </p>
          )}
        </section>
      )}
      {api && run && done(run.status) && (
        <Results3D key={run.run_id} api={api} id={run.run_id} runs={runs} onError={reportError} />
      )}
    </div>
  )
}
