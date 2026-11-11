import { useMemo, useState } from 'react'
import type { ModelSpec, ParamSpec, ParamValues } from '../types'
import { coerceParam } from '../lib/coercion'

export interface ParamFormProps {
  spec: ModelSpec
  values: ParamValues
  errors: Record<string, string>
  onChange: (name: string, value: unknown) => void
  disabled?: boolean
}

function HelpTip({ text }: { text?: string }) {
  if (!text) return null
  return (
    <span className="help-tip" title={text} aria-label="help">
      ?
    </span>
  )
}

function Field({
  param,
  value,
  error,
  disabled,
  onChange
}: {
  param: ParamSpec
  value: unknown
  error?: string
  disabled?: boolean
  onChange: (v: unknown) => void
}) {
  const id = `param-${param.name}`
  const unit = param.unit ? ` (${param.unit})` : ''

  if (param.type === 'bool') {
    return (
      <div className="field field-row">
        <label className="switch" htmlFor={id}>
          <input
            id={id}
            type="checkbox"
            checked={Boolean(value)}
            disabled={disabled}
            onChange={(e) => onChange(e.target.checked)}
          />
          <span className="slider" />
        </label>
        <span className="field-label" onClick={() => !disabled && onChange(!value)}>
          {param.name}
          <HelpTip text={param.help} />
        </span>
        <span className="field-unit">{String(value)}</span>
      </div>
    )
  }

  if (param.type === 'enum') {
    return (
      <div className="field">
        <label htmlFor={id}>
          {param.name}
          {unit}
          <HelpTip text={param.help} />
        </label>
        <select
          id={id}
          value={String(value ?? '')}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
        >
          {(param.choices ?? []).map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        {error && <div className="field-error">{error}</div>}
      </div>
    )
  }

  if (param.type === 'string') {
    return (
      <div className="field">
        <label htmlFor={id}>
          {param.name}
          {unit}
          <HelpTip text={param.help} />
        </label>
        <input
          id={id}
          type="text"
          value={String(value ?? '')}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
        />
        {error && <div className="field-error">{error}</div>}
      </div>
    )
  }

  // float / int
  return (
    <div className="field">
      <label htmlFor={id}>
        {param.name}
        {unit}
        <HelpTip text={param.help} />
      </label>
      <div className="number-wrap">
        <input
          id={id}
          type="number"
          inputMode={param.type === 'int' ? 'numeric' : 'decimal'}
          step={param.type === 'int' ? 1 : 'any'}
          min={param.min}
          max={param.max}
          value={value === undefined || value === null ? '' : String(value)}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
        />
        {param.unit && <span className="unit-badge">{param.unit}</span>}
      </div>
      {(typeof param.min === 'number' || typeof param.max === 'number') && (
        <div className="field-range mono">
          {param.min ?? '−∞'} … {param.max ?? '∞'}
        </div>
      )}
      {error && <div className="field-error">{error}</div>}
    </div>
  )
}

export function ParamForm({ spec, values, errors, onChange, disabled }: ParamFormProps) {
  const [showAdvanced, setShowAdvanced] = useState(true)
  const params = spec.params ?? []

  const { numeric, rest } = useMemo(() => {
    const numeric = params.filter((p) => p.type === 'float' || p.type === 'int')
    const rest = params.filter((p) => p.type !== 'float' && p.type !== 'int')
    return { numeric, rest }
  }, [params])

  if (params.length === 0) {
    return <div className="empty-note">This model declares no parameters.</div>
  }

  return (
    <div className="param-form">
      {numeric.length > 0 && (
        <section>
          <h4 className="form-section-title">Numeric</h4>
          {numeric.map((p) => (
            <Field
              key={p.name}
              param={p}
              value={values[p.name]}
              error={errors[p.name]}
              disabled={disabled}
              onChange={(v) => onChange(p.name, v)}
            />
          ))}
        </section>
      )}
      {rest.length > 0 && (
        <section>
          <button
            type="button"
            className="section-toggle"
            onClick={() => setShowAdvanced((s) => !s)}
          >
            {showAdvanced ? '▾' : '▸'} Options
          </button>
          {showAdvanced &&
            rest.map((p) => (
              <Field
                key={p.name}
                param={p}
                value={values[p.name]}
                error={errors[p.name]}
                disabled={disabled}
                onChange={(v) => onChange(p.name, v)}
              />
            ))}
        </section>
      )}
    </div>
  )
}

/** Live-validate a single field (used while typing). */
export function validateField(spec: ModelSpec, name: string, raw: unknown): string | undefined {
  const param = spec.params.find((p) => p.name === name)
  if (!param) return undefined
  const res = coerceParam(param, raw)
  if (res.value === null) return res.error ?? 'invalid'
  if (res.error && !res.error.includes('using default')) return res.error
  return undefined
}
