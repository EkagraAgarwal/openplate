/**
 * Param-form coercion: turn raw form input (string/checkbox/number) into the
 * JSON value the backend expects for a given MODEL_SPEC param. Pure functions,
 * unit-tested in tests/coercion.test.ts.
 */
import type { ParamSpec, ParamValue, ParamValues } from '../types'

export interface CoercionResult {
  value: ParamValue | null
  error?: string
}

function clamp(v: number, spec: ParamSpec): number {
  if (typeof spec.min === 'number' && v < spec.min) return spec.min
  if (typeof spec.max === 'number' && v > spec.max) return spec.max
  return v
}

/** Coerce a single raw input for a param spec. */
export function coerceParam(spec: ParamSpec, raw: unknown): CoercionResult {
  switch (spec.type) {
    case 'float': {
      if (raw === '' || raw === null || raw === undefined) {
        return { value: spec.default !== undefined ? Number(spec.default) : null, error: 'required' }
      }
      const n = typeof raw === 'number' ? raw : Number(String(raw).trim())
      if (!Number.isFinite(n)) return { value: null, error: 'not a number' }
      return { value: clamp(n, spec) }
    }
    case 'int': {
      if (raw === '' || raw === null || raw === undefined) {
        return { value: spec.default !== undefined ? Math.round(Number(spec.default)) : null, error: 'required' }
      }
      const n = typeof raw === 'number' ? raw : Number(String(raw).trim())
      if (!Number.isFinite(n)) return { value: null, error: 'not an integer' }
      const rounded = Math.round(n)
      return { value: clamp(rounded, spec) }
    }
    case 'bool': {
      if (typeof raw === 'boolean') return { value: raw }
      const s = String(raw).trim().toLowerCase()
      if (['true', '1', 'yes', 'on'].includes(s)) return { value: true }
      if (['false', '0', 'no', 'off', ''].includes(s)) return { value: false }
      return { value: null, error: 'not a boolean' }
    }
    case 'enum': {
      const s = String(raw ?? '')
      const choices = spec.choices ?? []
      if (choices.length === 0) return { value: s }
      if (choices.includes(s)) return { value: s }
      // fall back to default when present, else first choice
      if (typeof spec.default === 'string' && choices.includes(spec.default)) {
        return { value: spec.default, error: `unknown choice "${s}" (using default)` }
      }
      return { value: choices[0], error: `unknown choice "${s}"` }
    }
    case 'string':
    default: {
      if (raw === null || raw === undefined) {
        return { value: typeof spec.default === 'string' ? spec.default : '' }
      }
      return { value: String(raw) }
    }
  }
}

/** Initial form values from the spec defaults. */
export function defaultsFromSpec(params: ParamSpec[]): ParamValues {
  const out: ParamValues = {}
  for (const p of params) {
    if (p.default !== undefined) {
      out[p.name] = p.default as ParamValue
    } else if (p.type === 'bool') {
      out[p.name] = false
    } else if (p.type === 'enum' && p.choices?.length) {
      out[p.name] = p.choices[0]
    } else if (p.type === 'string') {
      out[p.name] = ''
    } else {
      out[p.name] = 0
    }
  }
  return out
}

/** Coerce a whole form state; returns payload + per-field errors. */
export function coerceAll(
  params: ParamSpec[],
  values: ParamValues
): { payload: ParamValues; errors: Record<string, string> } {
  const payload: ParamValues = {}
  const errors: Record<string, string> = {}
  for (const p of params) {
    const raw = values[p.name]
    const res = coerceParam(p, raw)
    if (res.value === null) {
      errors[p.name] = res.error ?? 'invalid'
    } else {
      payload[p.name] = res.value
      if (res.error && !res.error.includes('using default')) errors[p.name] = res.error
    }
  }
  return { payload, errors }
}

/** Parse a whitespace/comma separated list of numbers for the sweep builder. */
export function parseValueList(text: string): number[] {
  return text
    .split(/[\s,;\n]+/)
    .map((t) => t.trim())
    .filter(Boolean)
    .map((t) => Number(t))
    .filter((n) => Number.isFinite(n))
}
