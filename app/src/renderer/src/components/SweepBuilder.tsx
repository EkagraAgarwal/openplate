import { useEffect, useRef, useState } from 'react'
import type { ModelSpec, ParamValues, RunResult, SweepPoint } from '../types'
import { parseValueList } from '../lib/coercion'
import { useApp } from '../lib/context'

export interface SweepBuilderProps {
  spec: ModelSpec
  base: ParamValues
  onOpenResult: (label: string, result: RunResult) => void
}

/** Sweep builder: vary 1-2 numeric params over an explicit list of values. */
export function SweepBuilder({ spec, base, onOpenResult }: SweepBuilderProps) {
  const { api } = useApp()
  const numeric = (spec.params ?? []).filter((p) => p.type === 'float' || p.type === 'int')
  const [param1, setParam1] = useState('')
  const [values1, setValues1] = useState('')
  const [param2, setParam2] = useState('')
  const [values2, setValues2] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [points, setPoints] = useState<SweepPoint[]>([])
  const [status, setStatus] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (numeric.length > 0 && !param1) setParam1(numeric[0].name)
  }, [numeric, param1])

  useEffect(() => () => {
    if (pollRef.current) clearInterval(pollRef.current)
  }, [])

  const start = async (): Promise<void> => {
    if (!api) return
    setError(null)
    const v1 = parseValueList(values1)
    if (v1.length < 2) {
      setError('Enter at least 2 values for the first parameter (comma or space separated).')
      return
    }
    const vary: Record<string, number[]> = { [param1]: v1 }
    if (param2) {
      const v2 = parseValueList(values2)
      if (v2.length < 1) {
        setError('Second parameter selected but no values entered.')
        return
      }
      vary[param2] = v2
    }
    setBusy(true)
    setPoints([])
    setStatus('starting')
    try {
      const res = await api.sweep(spec.id, base, vary)
      const sweepId = res?.sweep_id ?? res?.id
      if (!sweepId) {
        setError('Sweep endpoint did not return an id (endpoint missing?)')
        setBusy(false)
        return
      }
      if (pollRef.current) clearInterval(pollRef.current)
      pollRef.current = setInterval(async () => {
        try {
          const state = await api.sweepStatus(sweepId)
          if (!state) {
            setStatus('unknown')
            return
          }
          setStatus(state.status)
          setPoints(state.results ?? [])
          const done = ['done', 'finished', 'completed', 'succeeded', 'error', 'failed'].includes(
            state.status.toLowerCase()
          )
          if (done) {
            if (pollRef.current) clearInterval(pollRef.current)
            setBusy(false)
            if (state.error) setError(state.error)
          }
        } catch {
          /* transient poll failure */
        }
      }, 1000)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setBusy(false)
    }
  }

  if (numeric.length === 0) {
    return <div className="empty-note">No numeric parameters to sweep.</div>
  }

  return (
    <div className="sweep-builder">
      <h4>Sweep</h4>
      <div className="sweep-grid">
        <label className="field">
          <span>Parameter 1</span>
          <select value={param1} onChange={(e) => setParam1(e.target.value)} disabled={busy}>
            {numeric.map((p) => (
              <option key={p.name} value={p.name}>
                {p.name}
                {p.unit ? ` (${p.unit})` : ''}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Values (comma separated)</span>
          <input
            type="text"
            placeholder={numeric[0] ? `${numeric[0].default ?? 0}, …` : ''}
            value={values1}
            onChange={(e) => setValues1(e.target.value)}
            disabled={busy}
          />
        </label>
        <label className="field">
          <span>Parameter 2 (optional)</span>
          <select value={param2} onChange={(e) => setParam2(e.target.value)} disabled={busy}>
            <option value="">— none —</option>
            {numeric
              .filter((p) => p.name !== param1)
              .map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}
                </option>
              ))}
          </select>
        </label>
        <label className="field">
          <span>Values {param2 ? '(grid)' : '(disabled)'}</span>
          <input
            type="text"
            value={values2}
            onChange={(e) => setValues2(e.target.value)}
            disabled={busy || !param2}
          />
        </label>
      </div>
      <div className="row gap">
        <button type="button" className="btn btn-primary" onClick={() => void start()} disabled={busy || !api}>
          {busy ? 'Running…' : 'Run sweep'}
        </button>
        {status && (
          <span className="mono small muted">
            status: {status}
            {points.length > 0 ? ` · ${points.length} point(s)` : ''}
          </span>
        )}
      </div>
      {error && <div className="field-error">{error}</div>}
      {points.length > 0 && (
        <div className="sweep-points">
          <h5>Points</h5>
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th>
                <th>params</th>
                <th>result</th>
              </tr>
            </thead>
            <tbody>
              {points.map((pt, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td className="mono small">
                    {Object.entries(pt.params)
                      .map(([k, v]) => `${k}=${String(v)}`)
                      .join(', ')}
                  </td>
                  <td>
                    {pt.result ? (
                      <button
                        type="button"
                        className="btn btn-small"
                        onClick={() =>
                          onOpenResult(
                            `sweep ${i + 1}: ${Object.entries(pt.params)
                              .map(([k, v]) => `${k}=${String(v)}`)
                              .join(', ')}`,
                            pt.result as RunResult
                          )
                        }
                      >
                        open
                      </button>
                    ) : (
                      <span className="muted">pending</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
