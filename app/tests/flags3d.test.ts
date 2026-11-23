import { describe, expect, it, vi } from 'vitest'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import {
  defaultSpec3D,
  patchAdditives,
  setAdditivesEnabled,
  setBuoyancy,
  setCeac,
  setCeacSmooth
} from '@/lib/sample3d'
import { SpecEditor } from '@/components/sample3d/SpecEditor'
import type { SampleSpec3D } from '@/api/client3d'

const render = (spec: SampleSpec3D) => renderToStaticMarkup(
  createElement(SpecEditor, {
    spec,
    onChange: vi.fn(),
    onValid: vi.fn(),
    errors: []
  })
)

describe('solutal buoyancy flag (contract 10h)', () => {
  it('setBuoyancy toggles the key, removes it when off and never mutates the source', () => {
    const on = setBuoyancy(defaultSpec3D, true)
    expect(on.setup.buoyancy).toBe(true)
    expect(defaultSpec3D.setup).not.toHaveProperty('buoyancy')
    const off = setBuoyancy(on, false)
    expect(off.setup).not.toHaveProperty('buoyancy')
    expect(JSON.stringify(off)).not.toContain('buoyancy')
    expect(on.setup.buoyancy).toBe(true)
  })
})

describe('CEAC flags (contract 10i)', () => {
  it('setCeac toggles the key; off also removes ceac_smooth', () => {
    let spec = setCeac(setAdditivesEnabled(defaultSpec3D, true), true)
    expect(spec.additives?.ceac).toBe(true)
    spec = setCeacSmooth(spec, true)
    expect(spec.additives?.ceac_smooth).toBe(true)
    const off = setCeac(spec, false)
    expect(off.additives).not.toHaveProperty('ceac')
    expect(off.additives).not.toHaveProperty('ceac_smooth')
    expect(spec.additives?.ceac_smooth).toBe(true)
  })

  it('setCeacSmooth toggles only the smooth key while ceac is on', () => {
    let spec = setCeac(setAdditivesEnabled(defaultSpec3D, true), true)
    const on = setCeacSmooth(spec, true)
    expect(on.additives?.ceac).toBe(true)
    expect(on.additives?.ceac_smooth).toBe(true)
    const off = setCeacSmooth(on, false)
    expect(off.additives).not.toHaveProperty('ceac_smooth')
    expect(off.additives?.ceac).toBe(true)
    // ceac_smooth cannot be set without ceac.
    spec = setAdditivesEnabled(defaultSpec3D, true)
    expect(setCeacSmooth(spec, true).additives).not.toHaveProperty('ceac_smooth')
  })

  it('resolved pulse coverage clears ceac and ceac_smooth', () => {
    let spec = setCeac(setAdditivesEnabled(defaultSpec3D, true), true)
    spec = setCeacSmooth(spec, true)
    const resolved = patchAdditives(spec, { pulse_coverage: 'resolved' })
    expect(resolved.additives?.pulse_coverage).toBe('resolved')
    expect(resolved.additives).not.toHaveProperty('ceac')
    expect(resolved.additives).not.toHaveProperty('ceac_smooth')
    expect(JSON.stringify(resolved)).not.toContain('ceac')
  })

  it('disabling additives drops the whole block including the CEAC keys', () => {
    let spec = setCeac(setAdditivesEnabled(defaultSpec3D, true), true)
    spec = setCeacSmooth(spec, true)
    const off = setAdditivesEnabled(spec, false)
    expect(off.additives).toBeUndefined()
    expect(JSON.stringify(off)).not.toContain('additives')
  })
})

describe('flag switches in the spec editor', () => {
  it('renders the buoyancy and CEAC switches, with the nested smooth option', () => {
    let spec = setBuoyancy(defaultSpec3D, true)
    spec = setCeacSmooth(setCeac(setAdditivesEnabled(spec, true), true), true)
    const html = render(spec)
    expect(html).toContain('solutal buoyancy (uncalibrated, ~6x slower)')
    expect(html).toContain('CEAC carried accelerator (uncalibrated)')
    expect(html).toContain('smooth curvature')
    expect(html).toContain('radii')
  })

  it('disables CEAC with a hint when coverage is not mean', () => {
    const spec = patchAdditives(setAdditivesEnabled(defaultSpec3D, true), {
      pulse_coverage: 'resolved'
    })
    const html = render(spec)
    expect(html).toContain('requires pulse coverage')
    expect(html).not.toContain('smooth curvature')
    expect(html).toMatch(/CEAC carried accelerator \(uncalibrated\)/)
  })
})
