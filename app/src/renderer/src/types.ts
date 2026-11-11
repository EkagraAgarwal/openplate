/** Shared types mirroring CONTRACTS.md (MODEL_SPEC / run() output / API). */

export type ParamType = 'float' | 'int' | 'bool' | 'enum' | 'string'

export interface ParamSpec {
  name: string
  type: ParamType
  default?: number | string | boolean
  min?: number
  max?: number
  unit?: string
  help?: string
  choices?: string[]
}

export interface ModelSpec {
  id: string
  title: string
  description?: string
  params: ParamSpec[]
  package?: string
  group?: string
}

export type ParamValue = number | string | boolean
export type ParamValues = Record<string, ParamValue>

// ---------------------------------------------------------------- plots ----

export interface PlotSeries {
  name?: string
  y: (number | null)[]
  mode?: string
}

export interface LinePlot {
  kind: 'line'
  title?: string
  x?: (number | string)[]
  series: PlotSeries[]
  xlabel?: string
  ylabel?: string
}

export interface HeatmapPlot {
  kind: 'heatmap'
  title?: string
  x?: number[]
  y?: number[]
  z: number[][]
  zlabel?: string
  xlabel?: string
  ylabel?: string
}

export interface BarSeries {
  name?: string
  y: (number | null)[]
}

export interface BarPlot {
  kind: 'bar'
  title?: string
  x?: (number | string)[]
  series?: BarSeries[]
  y?: (number | null)[]
  xlabel?: string
  ylabel?: string
}

export interface ContourFrame {
  t?: number
  lines: number[][][]
}

export interface ContoursPlot {
  kind: 'contours'
  title?: string
  frames: ContourFrame[]
  xlabel?: string
  ylabel?: string
}

export interface ImagePlot {
  kind: 'image'
  title?: string
  x?: number[]
  y?: number[]
  z?: number[][]
  zlabel?: string
  xlabel?: string
  ylabel?: string
}

export type Plot = LinePlot | HeatmapPlot | BarPlot | ContoursPlot | ImagePlot

export interface RunResult {
  summary: Record<string, unknown>
  plots?: Plot[]
  warnings?: string[]
  assumptions?: string[]
}

export interface Job {
  id: string
  model_id?: string
  status: string
  progress?: number
  message?: string
  elapsed_s?: number
  error?: string | null
  created_at?: string
}

export interface SweepPoint {
  params: ParamValues
  result?: RunResult
}

export interface SweepState {
  id?: string
  status: string
  results?: SweepPoint[]
  error?: string | null
}

export type ModelGroup =
  | 'Coupon/Cell'
  | 'Flow'
  | 'Efficiency'
  | 'Feature fill'
  | 'Imaging'
  | 'Other'

export const GROUP_ORDER: ModelGroup[] = [
  'Coupon/Cell',
  'Flow',
  'Efficiency',
  'Feature fill',
  'Imaging',
  'Other'
]

/** Group a model by its package name (or id prefix fallback). */
export function groupFor(model: ModelSpec): ModelGroup {
  const pkg = (model.package ?? model.id).toLowerCase()
  if (pkg.includes('cell') || pkg.includes('coupon')) return 'Coupon/Cell'
  if (pkg.includes('flow')) return 'Flow'
  if (pkg.includes('efficiency')) return 'Efficiency'
  if (pkg.includes('feature')) return 'Feature fill'
  if (pkg.includes('imaging') || pkg.includes('sem')) return 'Imaging'
  return 'Other'
}
