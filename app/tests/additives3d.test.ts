import { describe, expect, it } from 'vitest'
import { defaultSpec3D } from '@/lib/sample3d'
import {
  patchAdditives,
  setAdditivesEnabled,
  setAdditivesMode,
  setAdditivesNumber,
  setPulseModel
} from '@/lib/sample3d'

describe('L3 additives spec helpers (contract 10g)', () => {
  it('toggle creates the default L3 block and restores L1 without mutating the source', () => {
    const on = setAdditivesEnabled(defaultSpec3D, true)
    expect(on.numerics.physics).toBe('L3')
    expect(on.additives).toEqual({
      mode: 'cl_ndr',
      suppressor: true,
      pulse_coverage: 'mean',
      k_strip_s: 0,
      k_strip_a: 0,
      strip_return_a: 1
    })
    expect(defaultSpec3D.additives).toBeUndefined()
    expect(defaultSpec3D.numerics.physics).toBe('L1')
    const off = setAdditivesEnabled(on, false)
    expect(off.additives).toBeUndefined()
    expect(off.numerics.physics).toBe('L1')
    expect(on.additives?.mode).toBe('cl_ndr')
    expect(JSON.stringify(off)).not.toContain('additives')
  })

  it('switching mode away from acc removes every accelerator-only key', () => {
    let spec = setAdditivesEnabled(defaultSpec3D, true)
    spec = setAdditivesMode(spec, 'acc')
    for (const key of [
      'c_acc_mol_cm3', 'Gamma_acc_mol_cm2', 'k_acc_cm3_mol_s',
      'b_acc', 'k_inc_a_1_cm', 'D_acc_cm2_s'
    ] as const) {
      spec = setAdditivesNumber(spec, key, '0.5')
    }
    spec = setAdditivesNumber(spec, 's_bulk_mol_cm3', '0.001')
    expect(Object.keys(spec.additives!)).toContain('D_acc_cm2_s')
    spec = setAdditivesMode(spec, 'cl_ndr')
    expect(spec.additives?.mode).toBe('cl_ndr')
    expect(spec.additives?.s_bulk_mol_cm3).toBe(0.001)
    for (const key of [
      'c_acc_mol_cm3', 'Gamma_acc_mol_cm2', 'k_acc_cm3_mol_s',
      'b_acc', 'k_inc_a_1_cm', 'D_acc_cm2_s'
    ] as const) {
      expect(spec.additives).not.toHaveProperty(key)
    }
    expect(JSON.stringify(spec)).not.toMatch(/NaN|""|null/)
  })

  it('empty or non-finite numeric input removes the key instead of sending "" or NaN', () => {
    let spec = setAdditivesEnabled(defaultSpec3D, true)
    spec = setAdditivesNumber(spec, 's_bulk_mol_cm3', '0.01')
    expect(spec.additives?.s_bulk_mol_cm3).toBe(0.01)
    for (const raw of ['', '   ', 'abc', NaN, null, undefined]) {
      const cleared = setAdditivesNumber(spec, 's_bulk_mol_cm3', raw)
      expect(cleared.additives).not.toHaveProperty('s_bulk_mol_cm3')
      expect(JSON.stringify(cleared)).not.toMatch(/NaN|""|null/)
    }
    expect(spec.additives?.s_bulk_mol_cm3).toBe(0.01)
    // Contract ranges: b_acc >= -1 and strip_return_a in [0, 1].
    expect(setAdditivesNumber(spec, 'b_acc', '-5').additives?.b_acc).toBe(-1)
    expect(setAdditivesNumber(spec, 'strip_return_a', '2').additives?.strip_return_a).toBe(1)
    expect(setAdditivesNumber(spec, 'k_strip_s', '0').additives?.k_strip_s).toBe(0)
  })

  it('switching pulse model back to net resets resolved-only coverage and strip rates', () => {
    let spec = setAdditivesEnabled(defaultSpec3D, true)
    spec = patchAdditives(spec, { pulse_coverage: 'resolved' })
    spec = setAdditivesNumber(spec, 'k_strip_s', '0.2')
    spec = setAdditivesNumber(spec, 'k_strip_a', '0.3')
    expect(spec.additives).toMatchObject({
      pulse_coverage: 'resolved', k_strip_s: 0.2, k_strip_a: 0.3
    })
    spec = setPulseModel(spec, 'net')
    expect(spec.numerics.pulse_model).toBe('net')
    expect(spec.additives).toMatchObject({
      pulse_coverage: 'mean', k_strip_s: 0, k_strip_a: 0
    })
    const resolved = setPulseModel(spec, 'resolved')
    expect(resolved.numerics.pulse_model).toBe('resolved')
    expect(resolved.additives?.pulse_coverage).toBe('mean')
  })
})
