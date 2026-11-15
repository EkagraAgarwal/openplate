import { afterEach, describe, expect, it, vi } from 'vitest'
import { Sample3DApi, Sample3DError, url3d } from '@/api/client3d'
import {
  decimateHeight,
  depositFill,
  expandFeatures,
  featurePath,
  tileFootprints,
  arrayBufferToBase64,
  sectionCsv,
  defaultSpec3D,
  formToSpec,
  outlineTraces,
  parseSpec,
  polylineFromClicks,
  recipeMath,
  specToForm
} from '@/lib/sample3d'
import { startMockServer } from '../src/main/mockServer'

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})
describe('sample spec helpers', () => {
  it('expands arrays without mutation and caps work at 2000 instances', () => {
    const feature = { kind: 'rect' as const, x_um: 10, y_um: 20, w_um: 4, h_um: 2,
      array: { nx: 1000000, ny: 1000000, pitch_x_um: 5, pitch_y_um: 6 } }
    const result = expandFeatures([feature])
    expect(result.instances).toHaveLength(2000)
    expect(result.omitted).toBe(1e12 - 2000)
    expect(result.instances[1].feature.x_um).toBe(15)
    expect(feature.x_um).toBe(10)
    const small = expandFeatures([{ ...feature, array: { ...feature.array, nx: 2, ny: 2 } }], 3)
    expect(small.instances[2].feature).toMatchObject({ x_um: 10, y_um: 26, array: null })
    expect(small.omitted).toBe(1)
    expect(expandFeatures([feature], 0).instances).toEqual([])
  })
  it('generates rotated rectangle paths and rounded corners', () => {
    expect(featurePath({ kind: 'rect', x_um: 10, y_um: 20, w_um: 4, h_um: 2, rotation_deg: 90 }))
      .toBe('M 11 18 L 11 22 L 9 22 L 9 18 Z')
    const rounded = featurePath({ kind: 'rounded_rect', x_um: 0, y_um: 0, w_um: 8, h_um: 4, r_um: 99, rotation_deg: 30 })
    expect(rounded.match(/A 2 2 30 0 1/g)).toHaveLength(4)
    expect(rounded).not.toContain('NaN')
    expect(featurePath({ kind: 'circle', x_um: 0, y_um: 0, r_um: 2 })).toContain('A 2 2')
    expect(featurePath(JSON.parse('{"kind":"rect","x_um":"unfinished","y_um":0,"w_um":2,"h_um":2}'))).toBe('')
  })
  it('positions approximate tile footprints around referenced features', () => {
    const t = { tile_id: 't', kind: 'profile', feature_ids: ['feature_0000'], shape: [10, 8, 4], dx_um: 2, weight: 1, n_cells: 320, est_time_s: 1, assumptions: [], flags: [] }
    expect(tileFootprints([t], [{ kind: 'rect', x_um: 30, y_um: 20, w_um: 4, h_um: 2 }]))
      .toEqual([{ tile_id: 't', x: 20, y: 12, width: 20, height: 16, approximate: true }])
    expect(tileFootprints([{ ...t, feature_ids: ['missing'] }], defaultSpec3D.layout.features!)).toEqual([])
  })
  it('exports section CSV with contract columns, contour IDs and finite values', () => {
    expect(sectionCsv({ s_um: [], z_um: [], length_um: 1, cu_outline: [[[0, 1], [2, 3]], [[4, NaN]]], si_outline: [[[0, -1]]] }))
      .toBe('contour,kind,s_um,z_um\n0,cu,0,1\n0,cu,2,3\n1,cu,4,\n0,si,0,-1\n')
  })
  it('base64 encodes across chunk boundaries without padding intermediate chunks', () => {
    for (const length of [0, 1, 32767, 32768, 32769, 200001]) {
      const bytes = Uint8Array.from({ length }, (_, i) => i % 256)
      expect(arrayBufferToBase64(bytes.buffer)).toBe(Buffer.from(bytes).toString('base64'))
    }
  })
  it('round-trips spec/form without mutating the source or dropping extra fields', () => {
    const source = {
      ...defaultSpec3D,
      layout: { ...defaultSpec3D.layout, file: null, layer_map: { etch: 1 } }
    }
    const form = specToForm(source)
    form.layout.die_um[0] = 200
    expect(source.layout.die_um[0]).toBe(128)
    expect(formToSpec(specToForm(source))).toEqual(source)
    expect(parseSpec(JSON.stringify(source))).toEqual(source)
    expect(() => parseSpec('{')).toThrow()
    expect(() => parseSpec('{}')).toThrow()
    expect(() => parseSpec(JSON.stringify({ ...source, recipe: { steps: [null] } }))).toThrow()
    expect(() =>
      parseSpec(JSON.stringify({ ...source, layout: { die_um: [], features: {} } }))
    ).toThrow()
  })
  it('computes DC and pulse reverse net current and forward duty', () => {
    expect(recipeMath(defaultSpec3D.recipe.steps[0])).toEqual({ net: 10, duty: 1 })
    const pulse = {
      t_forward_ms: 10,
      t_reverse_ms: 2,
      t_off_ms: 8,
      j_forward_mA_cm2: 20,
      j_reverse_mA_cm2: 50,
      charge_C_cm2: 1
    }
    expect(recipeMath(pulse)).toEqual({ net: 5, duty: 0.5 })
    expect(recipeMath({ ...pulse, t_forward_ms: 0, t_reverse_ms: 0, t_off_ms: 0 })).toEqual({
      net: 0,
      duty: 0
    })
  })
  it('forms a cut from two distinct finite clicks, then starts a new cut', () => {
    expect(polylineFromClicks([], [1, 2])).toEqual([[1, 2]])
    expect(polylineFromClicks([[1, 2]], [1, 2])).toEqual([[1, 2]])
    expect(polylineFromClicks([[1, 2]], [3, 4])).toEqual([
      [1, 2],
      [3, 4]
    ])
    expect(
      polylineFromClicks(
        [
          [1, 2],
          [3, 4]
        ],
        [5, 6]
      )
    ).toEqual([[5, 6]])
    expect(polylineFromClicks([], [NaN, 1])).toEqual([])
  })
  it('produces filled copper and dashed initial outline traces', () => {
    const lines: [number, number][][] = [
      [
        [0, 0],
        [1, 2],
        [2, 0]
      ]
    ]
    expect(outlineTraces(lines, 'Cu', '#b87333', false, true)[0]).toMatchObject({
      x: [0, 1, 2],
      y: [0, 2, 0],
      fill: 'toself',
      line: { color: '#b87333', dash: 'solid' }
    })
    expect(outlineTraces(lines, 'Initial', '#888', true)[0]).toMatchObject({
      line: { dash: 'dash' }
    })
  })
  it('fills the deposit between open Cu and Si section lines', () => {
    const cu: [number, number][][] = [
      [
        [2, 3],
        [0, 1]
      ]
    ]
    const si: [number, number][][] = [
      [
        [2, 0],
        [0, -1]
      ]
    ]
    expect(depositFill(cu, si)[0]).toMatchObject({
      x: [2, 0, 0, 2, 2],
      y: [3, 1, -1, 0, 3],
      fill: 'toself'
    })
    expect(depositFill([...cu, ...cu], si)).toEqual([])
  })
  it('transposes asymmetric x-major heights and decimates preserving endpoints and nulls', () => {
    expect(
      decimateHeight(
        [0, 1, 2],
        [10, 20],
        [
          [1, 2],
          [3, null],
          [5, 6]
        ]
      )
    ).toEqual({
      x: [0, 1, 2],
      y: [10, 20],
      z: [
        [1, 3, 5],
        [2, null, 6]
      ]
    })
    expect(
      decimateHeight(
        [0, 1, 2],
        [10, 20],
        [
          [1, 2],
          [3, 4],
          [5, 6]
        ],
        2
      )
    ).toEqual({
      x: [0, 2],
      y: [10, 20],
      z: [
        [1, 5],
        [2, 6]
      ]
    })
    expect(decimateHeight([], [], [])).toEqual({ x: [], y: [], z: [] })
  })
})
describe('3-D client', () => {
  it('posts imported bytes and encodes export parameters', async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response('{}'))
    vi.stubGlobal('fetch', fetcher)
    const api = new Sample3DApi('')
    await api.importLayout('a.png', new Uint8Array([0, 255]).buffer, { pixel_um: 2, layer_map: { etch: [1, 0] } })
    expect(fetcher.mock.calls[0][0]).toBe('/api/3d/layout/import')
    expect(JSON.parse(fetcher.mock.calls[0][1].body)).toEqual({ filename: 'a.png', content_b64: 'AP8=', pixel_um: 2, layer_map: { etch: [1, 0] } })
    const url = new URL(api.exportUrl('a/b', 'section_csv', { tile_id: 'tile 1', polyline: [[0, 1], [2, 3]] }), 'http://localhost')
    expect(url.pathname).toBe('/api/3d/runs/a%2Fb/export')
    expect(url.searchParams.get('polyline')).toBe('0,1;2,3')
    expect(url.searchParams.get('frame')).toBe('-1')
  })
  it('uses gateway, web origin, or same-origin root and encodes queries', () => {
    expect(url3d('http://127.0.0.1:1234///', '/runs')).toBe('http://127.0.0.1:1234/api/3d/runs')
    expect(url3d('/', '/examples')).toBe('/api/3d/examples')
    expect(url3d('', '/runs/r/artifact', { path: 'a b/x.png' })).toBe(
      '/api/3d/runs/r/artifact?path=a+b%2Fx.png'
    )
    expect(url3d('https://example.com', '/height', { frame: -1, tile_id: undefined })).toBe(
      'https://example.com/api/3d/height?frame=-1'
    )
  })
  it('preserves structured 422 field errors without raw stacks', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            detail: {
              error: 'Bad depth',
              field: 'etch.default.depth_um',
              fix: 'Use a positive depth.'
            }
          }),
          { status: 422 }
        )
      )
    )
    await expect(new Sample3DApi('').plan(defaultSpec3D)).rejects.toMatchObject({
      issue: { field: 'etch.default.depth_um', message: 'Bad depth', fix: 'Use a positive depth.' },
      status: 422
    })
  })
  it('uses typed methods and exact section payload', async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response('{}'))
    vi.stubGlobal('fetch', fetcher)
    await new Sample3DApi('http://localhost:9').section(
      'a/b',
      [
        [0, 1],
        [2, 3]
      ],
      'profile',
      0
    )
    expect(fetcher.mock.calls[0][0]).toBe('http://localhost:9/api/3d/runs/a%2Fb/section')
    expect(JSON.parse(fetcher.mock.calls[0][1].body)).toEqual({
      polyline_um: [
        [0, 1],
        [2, 3]
      ],
      tile_id: 'profile',
      frame: 0
    })
  })
})
describe('3-D mock server smoke', () => {
  it('serves fixtures, a five-second run, results, structured errors and cancellation', async () => {
    const mock = await startMockServer(() => {})
    const api = new Sample3DApi(`http://127.0.0.1:${mock.port}`)
    try {
      expect((await api.examples()).examples.length).toBeGreaterThan(0)
      expect((await api.schema()).spec_schema).toBeTruthy()
      const fitted = await api.importLayout('test.gds', new Uint8Array([1, 2, 3]).buffer)
      expect(fitted.features).toHaveLength(1)
      expect(fitted.die_um).toEqual([128, 64])
      await expect(api.importLayout('test.png', new Uint8Array([1]).buffer)).rejects.toMatchObject({ issue: { field: 'pixel_um' } })
      expect((await api.validate(defaultSpec3D)).ok).toBe(true)
      expect((await api.plan(defaultSpec3D)).tiles.length).toBeGreaterThan(0)
      const { run_id } = await api.run(defaultSpec3D, 'Test')
      expect((await api.status(run_id)).status).toBe('running')
      await expect(api.manifest(run_id)).rejects.toBeInstanceOf(Sample3DError)
      const clock = vi.spyOn(Date, 'now').mockReturnValue(Date.now() + 5100)
      expect((await api.status(run_id)).status).toBe('done')
      clock.mockRestore()
      const csv = await fetch(api.exportUrl(run_id, 'section_csv', { polyline: [[0, 0], [10, 0]] }))
      expect(csv.headers.get('content-type')).toBe('text/csv')
      expect(await csv.text()).toContain('contour,kind,s_um,z_um')
      expect(await (await fetch(api.exportUrl(run_id, 'height_csv'))).text()).toContain('x_um,y_um,height_um')
      const tile = (await api.tiles(run_id)).tiles[0].tile_id
      expect((await api.height(run_id, tile)).height_um.length).toBeGreaterThan(0)
      const initial = await api.height(run_id, tile, 0)
      expect(initial.height_um).toEqual(initial.bare_height_um)
      expect(
        (
          await api.section(
            run_id,
            [
              [0, 0],
              [10, 0]
            ],
            tile
          )
        ).cu_outline
      ).toBeTruthy()
      expect((await api.series(run_id, tile)).t_s.length).toBeGreaterThan(0)
      expect((await api.frames(run_id, tile)).t_s.length).toBeGreaterThan(0)
      const manifest = await api.manifest(run_id)
      expect(manifest.quantities).toBeTruthy()
      const png = manifest.artifacts.find((a) => a.kind === 'png')
      if (png) expect((await fetch(api.artifact(run_id, png.path))).status).toBe(200)
      expect((await fetch(api.artifact(run_id, '../secret'))).status).toBe(404)
      const cancelled = await api.run(defaultSpec3D)
      expect((await api.cancel(cancelled.run_id)).status).toBe('cancelled')
      expect((await api.runs()).runs.some((r) => r.run_id === run_id)).toBe(true)
      await expect(
        api.run({ ...defaultSpec3D, numerics: { quality: 'invalid' } })
      ).rejects.toMatchObject({ status: 422 })
    } finally {
      mock.close()
    }
  })
})
