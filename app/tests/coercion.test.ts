import { describe, expect, it } from 'vitest'
import { coerceAll, coerceParam, defaultsFromSpec, parseValueList } from '@/lib/coercion'
import type { ParamSpec } from '@/types'

const floatSpec: ParamSpec = {
  name: 'I_total_mA',
  type: 'float',
  default: 48,
  min: 0.1,
  max: 500,
  unit: 'mA'
}
const intSpec: ParamSpec = { name: 'n_points', type: 'int', default: 64, min: 8, max: 512 }
const boolSpec: ParamSpec = { name: 'taped', type: 'bool', default: true }
const enumSpec: ParamSpec = {
  name: 'recipe',
  type: 'enum',
  choices: ['dc', 'plan_ramp', 'pulses'],
  default: 'dc'
}
const strSpec: ParamSpec = { name: 'note', type: 'string', default: 'hello' }

describe('coerceParam', () => {
  it('parses floats from strings', () => {
    expect(coerceParam(floatSpec, '12.5')).toEqual({ value: 12.5 })
  })

  it('rejects non-numeric float input', () => {
    expect(coerceParam(floatSpec, 'abc').value).toBeNull()
    expect(coerceParam(floatSpec, 'abc').error).toBe('not a number')
  })

  it('clamps floats to [min, max]', () => {
    expect(coerceParam(floatSpec, '1000').value).toBe(500)
    expect(coerceParam(floatSpec, '0').value).toBe(0.1)
  })

  it('falls back to default for empty float input', () => {
    expect(coerceParam(floatSpec, '').value).toBe(48)
  })

  it('rounds int input', () => {
    expect(coerceParam(intSpec, '10.6').value).toBe(11)
    expect(coerceParam(intSpec, '10.4').value).toBe(10)
  })

  it('clamps ints', () => {
    expect(coerceParam(intSpec, '2').value).toBe(8)
    expect(coerceParam(intSpec, '9999').value).toBe(512)
  })

  it('coerces bools from strings and booleans', () => {
    expect(coerceParam(boolSpec, true)).toEqual({ value: true })
    expect(coerceParam(boolSpec, 'false')).toEqual({ value: false })
    expect(coerceParam(boolSpec, 'on')).toEqual({ value: true })
    expect(coerceParam(boolSpec, '0')).toEqual({ value: false })
    expect(coerceParam(boolSpec, 'maybe').value).toBeNull()
  })

  it('validates enum choices and falls back to default', () => {
    expect(coerceParam(enumSpec, 'pulses')).toEqual({ value: 'pulses' })
    const res = coerceParam(enumSpec, 'nope')
    expect(res.value).toBe('dc')
    expect(res.error).toContain('unknown choice')
  })

  it('passes strings through', () => {
    expect(coerceParam(strSpec, 'x')).toEqual({ value: 'x' })
  })
})

describe('defaultsFromSpec', () => {
  it('produces defaults for every param', () => {
    expect(defaultsFromSpec([floatSpec, intSpec, boolSpec, enumSpec, strSpec])).toEqual({
      I_total_mA: 48,
      n_points: 64,
      taped: true,
      recipe: 'dc',
      note: 'hello'
    })
  })
})

describe('coerceAll', () => {
  it('builds a clean payload when all fields are valid', () => {
    const { payload, errors } = coerceAll([floatSpec, intSpec, boolSpec], {
      I_total_mA: '25.5',
      n_points: '32',
      taped: 'false'
    })
    expect(payload).toEqual({ I_total_mA: 25.5, n_points: 32, taped: false })
    expect(errors).toEqual({})
  })

  it('collects per-field errors and omits invalid fields', () => {
    const { payload, errors } = coerceAll([floatSpec, intSpec], {
      I_total_mA: 'abc',
      n_points: '64'
    })
    expect(payload).toEqual({ n_points: 64 })
    expect(errors.I_total_mA).toBeTruthy()
  })
})

describe('parseValueList', () => {
  it('parses comma and whitespace separated numbers', () => {
    expect(parseValueList('1, 2 3\n4;5')).toEqual([1, 2, 3, 4, 5])
  })

  it('drops non-numeric tokens', () => {
    expect(parseValueList('10, abc, 20')).toEqual([10, 20])
  })

  it('returns empty for empty input', () => {
    expect(parseValueList('')).toEqual([])
  })
})
