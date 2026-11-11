import { useEffect, useMemo, useRef, useState } from 'react'
import { useApp } from '../lib/context'
import { coerceAll, defaultsFromSpec } from '../lib/coercion'
import { ParamForm, validateField } from '../components/ParamForm'
import { SweepBuilder } from '../components/SweepBuilder'
import { JobsPanel } from '../components/JobsPanel'
import { ResultView } from '../components/ResultView'
import { useJobs } from '../lib/jobs'
import { deletePreset, listPresets, savePreset, type Preset } from '../lib/presets'
import type { ParamValues, RunResult } from '../types'

export function ModelPage() {
  const {
    api,
    models,
    selectedModelId,
    selectModel,
    pendingParams,
    clearPendingParams,
    openResult,
    activeResult,
    closeResult,
    compare,
    toggleCompare,
    clearCompare,
    setShowCompare
  } = useApp()

  const spec = useMemo(
    () => models.find((m) => m.id === selectedModelId) ?? null,
    [models, selectedModelId]
  )

  const [values, setValues] = useState<ParamValues>({})
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [runError, setRunError] = useState<string | null>(null)
  const [runInfo, setRunInfo] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [presets, setPresets] = useState<Preset[]>([])
  const [presetName, setPresetName] = useState('')
  const [compareIds, setCompareIds] = useState<string[]>([])
  const [localResult, setLocalResult] = useState<{ label: string; result: RunResult; model: string } | null>(null)
  const initializedFor = useRef<string | null>(null)

  const { jobs, connection, refresh, setJobs } = useJobs(api, Boolean(spec))

  // (re)initialize form when the model changes
  useEffect(() => {
    if (!spec) return
    if (initializedFor.current === spec.id && !pendingParams) return
    initializedFor.current = spec.id
    const base = defaultsFromSpec(spec.params ?? [])
    if (pendingParams && pendingParams.modelId === spec.id) {
      setValues({ ...base, ...pendingParams.params })
      clearPendingParams()
    } else {
      setValues(base)
    }
    setErrors({})
    setRunError(null)
    setRunInfo(null)
    setLocalResult(null)
    setPresets(listPresets(spec.id))
    setPresetName('')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec, pendingParams])

  // react to assistant proposals targeting a different model
  useEffect(() => {
    if (!pendingParams || !spec || pendingParams.modelId === spec.id) return
    selectModel(pendingParams.modelId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingParams, spec])

  if (!spec) {
    return (
      <div className="page">
        <div className="empty-state">
          <h3>Select a model</h3>
          <p className="muted">Pick a model from the sidebar to configure and run it.</p>
        </div>
      </div>
    )
  }

  const onFieldChange = (name: string, raw: unknown): void => {
    setValues((v) => ({ ...v, [name]: raw as string | number | boolean }))
    const err = validateField(spec, name, raw)
    setErrors((e) => {
      const next = { ...e }
      if (err) next[name] = err
      else delete next[name]
      return next
    })
  }

  const run = async (): Promise<void> => {
    if (!api) return
    const { payload, errors: errs } = coerceAll(spec.params ?? [], values)
    setErrors(errs)
    if (Object.keys(errs).length > 0) {
      setRunError('Fix the highlighted parameters first.')
      return
    }
    setRunning(true)
    setRunError(null)
    setRunInfo(null)
    try {
      const res = await api.run(spec.id, payload)
      if (!res?.job_id) {
        setRunError('Run endpoint did not return a job_id.')
      } else {
        setRunInfo(`Job ${res.job_id} started.`)
        setLocalResult(null)
        await refresh()
      }
    } catch (err) {
      setRunError(err instanceof Error ? err.message : String(err))
    } finally {
      setRunning(false)
    }
  }

  const openJobResult = async (jobId: string): Promise<void> => {
    if (!api) return
    try {
      const result = await api.result(jobId)
      if (!result) {
        setRunError(`No result available for ${jobId} (yet).`)
        return
      }
      setLocalResult({ label: `Result · ${jobId.slice(0, 12)}`, result, model: spec.id })
      closeResult()
    } catch (err) {
      setRunError(err instanceof Error ? err.message : String(err))
    }
  }

  const shown = activeResult
    ? { label: activeResult.label, result: activeResult.result, model: spec.id }
    : localResult

  const toggleCompareId = (id: string): void => {
    setCompareIds((ids) => {
      if (ids.includes(id)) return ids.filter((x) => x !== id)
      if (ids.length >= 4) return ids
      return [...ids, id]
    })
  }

  return (
    <div className="page model-page">
      <div className="model-grid">
        <div className="model-config">
          <div className="card">
            <div className="model-head">
              <div>
                <h2>{spec.title}</h2>
                <div className="mono small muted">{spec.id}</div>
              </div>
            </div>
            {spec.description && <p className="model-desc">{spec.description}</p>}

            <ParamForm
              spec={spec}
              values={values}
              errors={errors}
              onChange={onFieldChange}
              disabled={running}
            />

            {runError && <div className="field-error">{runError}</div>}
            {runInfo && <div className="ok-note">{runInfo}</div>}

            <div className="row gap run-row">
              <button
                type="button"
                className="btn btn-primary btn-run"
                onClick={() => void run()}
                disabled={running || !api}
              >
                {running ? 'Starting…' : '▶ Run'}
              </button>
              <button
                type="button"
                className="btn btn-ghost"
                onClick={() => {
                  setValues(defaultsFromSpec(spec.params ?? []))
                  setErrors({})
                }}
              >
                Reset defaults
              </button>
            </div>

            <div className="presets">
              <h4>Presets</h4>
              <div className="row gap">
                <select
                  value=""
                  onChange={(e) => {
                    const p = presets.find((x) => x.name === e.target.value)
                    if (p) {
                      setValues({ ...defaultsFromSpec(spec.params ?? []), ...p.values })
                      setErrors({})
                    }
                  }}
                >
                  <option value="">load preset…</option>
                  {presets.map((p) => (
                    <option key={p.name} value={p.name}>
                      {p.name}
                    </option>
                  ))}
                </select>
                <input
                  type="text"
                  placeholder="preset name"
                  value={presetName}
                  onChange={(e) => setPresetName(e.target.value)}
                />
                <button
                  type="button"
                  className="btn btn-small"
                  disabled={!presetName.trim()}
                  onClick={() => {
                    const name = presetName.trim()
                    if (!name) return
                    setPresets(savePreset(spec.id, name, values))
                    setPresetName('')
                  }}
                >
                  save
                </button>
                {presets.length > 0 && (
                  <select
                    value=""
                    onChange={(e) => {
                      const name = e.target.value
                      if (!name) return
                      setPresets(deletePreset(spec.id, name))
                    }}
                  >
                    <option value="">delete…</option>
                    {presets.map((p) => (
                      <option key={p.name} value={p.name}>
                        {p.name}
                      </option>
                    ))}
                  </select>
                )}
              </div>
              <div className="small muted">
                presets for this model are stored in localStorage ({presets.length})
              </div>
            </div>
          </div>

          <div className="card">
            <SweepBuilder
              spec={spec}
              base={(() => {
                const { payload } = coerceAll(spec.params ?? [], values)
                return payload
              })()}
              onOpenResult={(label, result) => setLocalResult({ label, result, model: spec.id })}
            />
          </div>
        </div>

        <div className="model-side">
          <JobsPanel
            api={api}
            jobs={jobs}
            connection={connection}
            refresh={refresh}
            onOpenResult={(id) => void openJobResult(id)}
            compareIds={compareIds}
            onToggleCompare={toggleCompareId}
          />
          {compareIds.length > 0 && (
            <div className="card compare-actions">
              <div className="row gap space-between">
                <span className="small">
                  {compareIds.length}/4 selected for comparison
                </span>
                <div className="row gap">
                  <button type="button" className="btn btn-small btn-ghost" onClick={() => setCompareIds([])}>
                    clear
                  </button>
                  <button
                    type="button"
                    className="btn btn-small btn-primary"
                    disabled={compareIds.length < 2}
                    onClick={async () => {
                      if (!api) return
                      clearCompare()
                      for (const id of compareIds) {
                        try {
                          const r = await api.result(id)
                          if (r) toggleCompare({ key: id, label: id.slice(0, 12), result: r })
                        } catch {
                          /* skip */
                        }
                      }
                      setJobs((j) => j) // no-op keep
                      setShowCompare(true)
                    }}
                  >
                    Compare →
                  </button>
                </div>
              </div>
              {compare.length > 0 && <div className="small muted">loaded: {compare.length} result(s)</div>}
            </div>
          )}
        </div>
      </div>

      {shown && (
        <div className="card result-card">
          <ResultView
            label={shown.label}
            result={shown.result}
            modelName={shown.model}
            onBack={() => {
              closeResult()
              setLocalResult(null)
            }}
          />
        </div>
      )}
    </div>
  )
}
