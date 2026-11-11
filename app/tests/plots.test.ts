import { describe, expect, it } from 'vitest'
import {
  barTraces,
  buildLayout,
  contourFrameTraces,
  heatmapTraces,
  lineTraces,
  normalizePlotKind
} from '@/lib/plots'
import type { BarPlot, ContoursPlot, HeatmapPlot, ImagePlot, LinePlot, Plot } from '@/types'

describe('normalizePlotKind', () => {
  it('maps all contract kinds', () => {
    const line: LinePlot = { kind: 'line', series: [] }
    const heat: HeatmapPlot = { kind: 'heatmap', z: [] }
    const bar: BarPlot = { kind: 'bar' }
    const contours: ContoursPlot = { kind: 'contours', frames: [] }
    const image: ImagePlot = { kind: 'image' }
    expect(normalizePlotKind(line).kind).toBe('line')
    expect(normalizePlotKind(heat).kind).toBe('heatmap')
    expect(normalizePlotKind(bar).kind).toBe('bar')
    expect(normalizePlotKind(contours).kind).toBe('contours')
    expect(normalizePlotKind(image).kind).toBe('image')
  })

  it('accepts singular "contour" and lowercases', () => {
    expect(normalizePlotKind({ kind: 'Contour' } as unknown as Plot).kind).toBe('contours')
  })

  it('marks unknown kinds as unknown with null plot', () => {
    const n = normalizePlotKind({ kind: 'sankey' } as unknown as Plot)
    expect(n.kind).toBe('unknown')
    expect(n.plot).toBeNull()
  })

  it('extracts the title', () => {
    expect(normalizePlotKind({ kind: 'line', title: 'T', series: [] }).title).toBe('T')
  })
})

describe('lineTraces', () => {
  it('emits one trace per series with x from the plot', () => {
    const plot: LinePlot = {
      kind: 'line',
      x: [1, 2, 3],
      series: [
        { name: 'a', y: [4, 5, 6] },
        { name: 'b', y: [7, 8, 9], mode: 'markers' }
      ]
    }
    const traces = lineTraces(plot)
    expect(traces).toHaveLength(2)
    expect(traces[0]).toMatchObject({ type: 'scatter', mode: 'lines', name: 'a', x: [1, 2, 3], y: [4, 5, 6] })
    // series may declare "mode":"markers"
    expect(traces[1]).toMatchObject({ mode: 'markers', name: 'b' })
  })

  it('handles a missing x array', () => {
    const traces = lineTraces({ kind: 'line', series: [{ y: [1, 2] }] })
    expect(traces[0].x).toEqual([])
  })
})

describe('heatmapTraces', () => {
  it('passes x/y/z and colorbar zlabel', () => {
    const plot: HeatmapPlot = {
      kind: 'heatmap',
      x: [0, 1],
      y: [0, 1],
      z: [
        [1, 2],
        [3, 4]
      ],
      zlabel: 'j (mA/cm2)'
    }
    const [t] = heatmapTraces(plot)
    expect(t.type).toBe('heatmap')
    expect(t.z).toEqual([
      [1, 2],
      [3, 4]
    ])
    expect((t.colorbar as { title: { text: string } }).title.text).toBe('j (mA/cm2)')
    expect(t.colorscale).toBeUndefined()
  })

  it('treats "image" like a grayscale heatmap', () => {
    const [t] = heatmapTraces({ kind: 'image', z: [[1]] }, true)
    expect(t.type).toBe('heatmap')
    expect(t.colorscale).toBe('gray')
  })
})

describe('barTraces', () => {
  it('uses series when present', () => {
    const traces = barTraces({
      kind: 'bar',
      x: ['a', 'b'],
      series: [{ name: 'h', y: [1, 2] }]
    })
    expect(traces).toHaveLength(1)
    expect(traces[0]).toMatchObject({ type: 'bar', name: 'h', y: [1, 2] })
  })

  it('falls back to bare y', () => {
    const traces = barTraces({ kind: 'bar', x: ['a'], y: [3] })
    expect(traces[0].y).toEqual([3])
  })
})

describe('contourFrameTraces', () => {
  const plot: ContoursPlot = {
    kind: 'contours',
    xlabel: 'r (um)',
    ylabel: 'z (um)',
    frames: [
      { t: 0, lines: [[[0, 0], [10, 0], [10, 40]]] },
      { t: 1, lines: [[[0, 5], [10, 5], [10, 40]]] }
    ]
  }

  it('draws the requested frame', () => {
    const traces = contourFrameTraces(plot, 0)
    expect(traces).toHaveLength(1)
    expect(traces[0].x).toEqual([0, 10, 10])
    expect(traces[0].y).toEqual([0, 0, 40])
    const t1 = contourFrameTraces(plot, 1)
    expect(t1[0].y).toEqual([5, 5, 40])
  })

  it('clamps out-of-range frame indices', () => {
    expect(contourFrameTraces(plot, 99)[0].y).toEqual([5, 5, 40])
    expect(contourFrameTraces(plot, -3)[0].y).toEqual([0, 0, 40])
  })

  it('mirrors r -> -r when requested (feature fill full cross-section)', () => {
    const traces = contourFrameTraces(plot, 0, true)
    expect(traces).toHaveLength(2)
    expect(traces[1].x).toEqual([-10, -10, 0])
    expect(traces[1].y).toEqual([40, 0, 0])
  })

  it('does not mirror when mirror=false', () => {
    expect(contourFrameTraces(plot, 0, false)).toHaveLength(1)
  })

  it('returns empty for no frames', () => {
    expect(contourFrameTraces({ kind: 'contours', frames: [] }, 0)).toEqual([])
  })
})

describe('buildLayout', () => {
  it('gives contours an equal aspect ratio', () => {
    const layout = buildLayout({ kind: 'contours', frames: [] })
    const y = layout.yaxis as { scaleanchor?: string; scaleratio?: number }
    expect(y.scaleanchor).toBe('x')
    expect(y.scaleratio).toBe(1)
  })

  it('pulls axis labels from xlabel/ylabel', () => {
    const layout = buildLayout({ kind: 'line', title: 'T', series: [], xlabel: 'x', ylabel: 'y' })
    expect((layout.xaxis as { title: { text: string } }).title.text).toBe('x')
    expect((layout.yaxis as { title: { text: string } }).title.text).toBe('y')
    expect(layout.title?.text).toBe('T')
  })
})
