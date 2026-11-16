import { describe, expect, it, vi } from 'vitest'
import { Sample3DApi } from '@/api/client3d'
import { defaultSpec3D } from '@/lib/sample3d'
import { estimateSpec, exampleSpec, roughRuntime } from '@/lib/estimates3d'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { SpecEditor } from '@/components/sample3d/SpecEditor'

describe('pre-run rough estimates', () => {
  it('shows a rough estimate in the spec editor before any run exists', () => {
    const html = renderToStaticMarkup(createElement(SpecEditor, {
      spec: defaultSpec3D, onChange: vi.fn(), onValid: vi.fn(), errors: [],
      estimateText: roughRuntime(1440)
    }))
    expect(html).toContain('~24 min (rough estimate)')
    expect(html).toContain('Planning only; runtime varies by hardware and physics.')
  })
  it('formats minute estimates and never presents invalid values as zero minutes', () => {
    expect(roughRuntime(60)).toBe('~1 min (rough estimate)')
    expect(roughRuntime(1440)).toBe('~24 min (rough estimate)')
    expect(roughRuntime(0)).toBe('~1 min (rough estimate)')
    expect(roughRuntime(NaN)).toBe('Estimate unavailable')
    expect(roughRuntime(-1)).toBe('Estimate unavailable')
  })
  it('plans the same preview spec that selecting an example loads, without solving', async () => {
    const api = new Sample3DApi('')
    const plan = { ok: true, errors: [], tiles: [], n_cells: 1000, est_time_s: 60 }
    const planner = vi.spyOn(api, 'plan').mockResolvedValue(plan)
    const solver = vi.spyOn(api, 'run')
    const source = { ...defaultSpec3D, numerics: { quality: 'fine', dx_um: 2 } }
    const preview = exampleSpec(source)
    expect(await estimateSpec(api, preview)).toBe(plan)
    expect(planner).toHaveBeenCalledWith(preview)
    expect(preview.numerics).toEqual({ quality: 'preview', dx_um: 2 })
    expect(source.numerics.quality).toBe('fine')
    expect(solver).not.toHaveBeenCalled()
  })
  it('does not display invalid plans and lets callers handle unavailable service', async () => {
    const api = new Sample3DApi('')
    const planner = vi.spyOn(api, 'plan').mockResolvedValue({
      ok: false, errors: [], tiles: [], n_cells: 0, est_time_s: 0
    })
    expect(await estimateSpec(api, defaultSpec3D)).toBeNull()
    planner.mockRejectedValueOnce(new Error('offline'))
    await expect(estimateSpec(api, defaultSpec3D)).rejects.toThrow('offline')
  })
})
