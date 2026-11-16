import { afterEach, describe, expect, it, vi } from 'vitest'

// Minimal hook host: exercise the real Plotly effects without needing WebGL or a DOM.
const host = vi.hoisted(() => ({
  el: { removeAllListeners: vi.fn(), on: vi.fn() },
  effects: [] as { effect: () => void | (() => void); deps?: unknown[] }[],
  react: vi.fn((_el: unknown, _data: unknown, _layout: unknown, _config: unknown) => Promise.resolve()), purge: vi.fn()
}))
vi.mock('react', async (original) => ({
  ...await original<typeof import('react')>(),
  useRef: () => ({ current: host.el }),
  useState: (value: unknown) => [value, vi.fn()],
  useEffect: (effect: () => void | (() => void), deps?: unknown[]) => {
    host.effects.push({ effect, deps })
  }
}))
vi.mock('plotly.js-dist-min', () => ({ default: {
  react: host.react, purge: host.purge, Plots: { resize: () => Promise.resolve() }
} }))
vi.mock('@/lib/context', () => ({ useApp: () => ({ theme: 'light' }) }))
import { Plot3D } from '@/components/sample3d/Plot3D'
import { PlotRenderer } from '@/components/PlotRenderer'

afterEach(() => { vi.unstubAllGlobals(); vi.clearAllMocks(); host.effects = [] })

describe('Plotly camera and zoom lifecycle', () => {
  it('keeps a run revision on updates and only purges the 3-D plot on unmount', async () => {
    vi.stubGlobal('document', { documentElement: {} })
    vi.stubGlobal('getComputedStyle', () => ({ getPropertyValue: () => '#fff' }))
    vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} })
    const props = { title: 'Height surface', revision: 'run1:tile1', data: [{ type: 'surface' }] }
    Plot3D(props)
    const update = host.effects[0].effect()
    const unmount = host.effects[1].effect()
    expect(host.effects[1].deps).toEqual([])
    expect(host.react.mock.calls[0][2]).toMatchObject({ uirevision: 'run1:tile1' })
    update?.()
    expect(host.purge).not.toHaveBeenCalled()
    host.effects = []
    Plot3D({ ...props, data: [{ type: 'surface', z: [[2]] }] })
    host.effects[0].effect()
    expect(host.react.mock.calls[1][2]).toMatchObject({ uirevision: 'run1:tile1' })
    host.effects = []
    Plot3D({ ...props, revision: 'run2:tile1' })
    host.effects[0].effect()
    expect(host.react.mock.calls[2][2]).toMatchObject({ uirevision: 'run2:tile1' })
    unmount?.()
    expect(host.purge).toHaveBeenCalledTimes(1)
    await Promise.resolve()
  })
  it('preserves 2-D zoom across updates rather than purging every render', () => {
    PlotRenderer({ plot: { kind: 'line', title: 'History', x: [0, 1], series: [] } })
    const update = host.effects[0].effect()
    const unmount = host.effects[1].effect()
    expect(host.react.mock.calls[0][2]).toMatchObject({ uirevision: 'History' })
    update?.()
    expect(host.purge).not.toHaveBeenCalled()
    unmount?.()
    expect(host.purge).toHaveBeenCalledTimes(1)
  })
})
