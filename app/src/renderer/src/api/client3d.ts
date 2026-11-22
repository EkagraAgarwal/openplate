import { arrayBufferToBase64 } from '../lib/sample3d'

export type Point3D = [number, number]
export interface Feature3D {
  kind: 'rect' | 'rounded_rect' | 'circle'
  x_um: number
  y_um: number
  w_um?: number
  h_um?: number
  r_um?: number
  rotation_deg?: number
  array?: { nx: number; ny: number; pitch_x_um: number; pitch_y_um: number } | null
  [key: string]: unknown
}
export interface Step3D {
  t_forward_ms: number
  t_reverse_ms: number
  t_off_ms: number
  j_forward_mA_cm2: number
  j_reverse_mA_cm2: number
  duration_h?: number | null
  charge_C_cm2?: number | null
}
/** L3 additives (CONTRACTS_3D.md, contract 10g). Optional numeric keys mean "Params default". */
export interface Additives {
  mode: 'cl_ndr' | 'acc'
  suppressor?: boolean
  s_bulk_mol_cm3?: number | null
  c_acc_mol_cm3?: number | null
  Gamma_acc_mol_cm2?: number | null
  k_acc_cm3_mol_s?: number | null
  b_acc?: number | null
  k_inc_a_1_cm?: number | null
  D_acc_cm2_s?: number | null
  pulse_coverage?: 'mean' | 'resolved'
  k_strip_s?: number
  k_strip_a?: number
  strip_return_a?: number
}
export interface SampleSpec3D {
  layout: { die_um: number[]; features?: Feature3D[] | null; [key: string]: unknown }
  etch: {
    default: { depth_um: number; sidewall_deg?: number; [key: string]: unknown }
    [key: string]: unknown
  }
  recipe: { steps: Step3D[] }
  bath: {
    name?: string
    cu_g_L?: number
    h2so4_g_L?: number
    cl_ppm?: number
    temperature_C?: number
  }
  setup: {
    orientation?: string
    up_edge?: string
    stir_shear_1_s?: number | null
    stir_angle_deg?: number
    coupon_position_cm?: [number, number] | null
    stir_pct?: number | null
    delta_um?: number
    area_cm2?: number
    taped?: boolean
  }
  numerics: {
    quality?: string
    dx_um?: number | null
    physics?: string
    pulse_model?: 'net' | 'resolved'
    pulse_C_dl_uF_cm2?: number
    [key: string]: unknown
  }
  /** Present exactly when numerics.physics === 'L3' (contract 10g). */
  additives?: Additives | null
}
export interface FieldIssue {
  field: string
  message: string
  fix: string
}
export interface LayoutFit3D {
  die_um: number[]
  features: Feature3D[]
  unfit: { index: number; area_um2: number; error_rel: number; bounds_um: number[] }[]
  skipped: unknown[]
  preview: { layer: string; rings_um: Point3D[][] }[]
  source: { filename: string; kind: string; n_polygons: number }
}
export interface LayoutImportOptions {
  pixel_um?: number
  layer_map?: Record<string, [number, number]>
  layer?: string
}
export interface Validation3D {
  ok: boolean
  errors: FieldIssue[]
}
export interface Tile3D {
  mirror_axes?: string[]
  tile_id: string
  kind: string
  feature_ids: string[]
  weight: number
  shape: number[]
  dx_um: number
  x_um: number[]
  y_um: number[]
}
export interface Plan3D extends Validation3D {
  tiles: (Omit<Tile3D, 'x_um' | 'y_um'> & {
    n_cells: number
    est_time_s: number
    assumptions: string[]
    flags: string[]
  })[]
  n_cells: number
  est_time_s: number
}
export interface Run3D {
  run_id: string
  label: string | null
  status: string
  progress: number
  message: string
  created?: string | number | null
  finished?: string | number | null
  error?: string | { error: string; field?: string; fix?: string } | null
  spec?: SampleSpec3D
}
export interface Height3D {
  tile_id: string
  frame: number
  t_s: number
  x_um: number[]
  y_um: number[]
  height_um: (number | null)[][]
  bare_height_um: (number | null)[][]
  stats: Record<string, unknown>
}
export interface Section3D {
  s_um: number[]
  z_um: number[]
  cu_outline: Point3D[][]
  si_outline: Point3D[][]
  length_um: number
}
export interface Series3D {
  t_s: number[]
  V: (number | null)[]
  ct_min_front: (number | null)[]
  faraday_volume_cm3: (number | null)[]
  solid_volume_cm3: (number | null)[]
  cu_balance_rel: (number | null)[]
}
export interface Manifest3D {
  quantities: { name: string; value: unknown; unit: string; feature_id: string | null }[]
  checks: { name: string; value: unknown; limit: unknown; passed: boolean }[]
  artifacts: { kind: string; path: string; title: string }[]
  summary_text: string
  [key: string]: unknown
}
export class Sample3DError extends Error {
  constructor(
    public issue: FieldIssue,
    public status?: number
  ) {
    super(issue.message)
  }
}
export function url3d(
  baseUrl: string,
  path: string,
  query: Record<string, string | number | undefined> = {}
): string {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(query))
    if (value !== undefined) params.set(key, String(value))
  return `${baseUrl.replace(/\/+$/, '')}/api/3d${path}${params.size ? `?${params}` : ''}`
}
export class Sample3DApi {
  constructor(public baseUrl: string) {}
  private async json<T>(
    path: string,
    body?: unknown,
    query?: Record<string, string | number | undefined>
  ): Promise<T> {
    let res: Response
    try {
      res = await fetch(
        url3d(this.baseUrl, path, query),
        body === undefined
          ? undefined
          : {
              method: 'POST',
              headers: { 'content-type': 'application/json' },
              body: JSON.stringify(body)
            }
      )
    } catch {
      throw new Sample3DError({
        field: '',
        message: 'Cannot reach the 3-D service.',
        fix: 'Check the backend connection and retry.'
      })
    }
    if (!res.ok) {
      const data: unknown = await res.json().catch(() => null)
      const detail =
        data &&
        typeof data === 'object' &&
        'detail' in data &&
        data.detail &&
        typeof data.detail === 'object'
          ? (data.detail as Record<string, unknown>)
          : null
      throw new Sample3DError(
        {
          field: typeof detail?.field === 'string' ? detail.field : '',
          message:
            typeof detail?.message === 'string' ? detail.message : typeof detail?.error === 'string'
              ? detail.error
              : `Request unavailable (HTTP ${res.status}).`,
          fix:
            typeof detail?.fix === 'string'
              ? detail.fix
              : 'Check the run status and input, then retry.'
        },
        res.status
      )
    }
    return res.json() as Promise<T>
  }
  schema() {
    return this.json<{ spec_schema: Record<string, unknown> }>('/schema')
  }
  importLayout(filename: string, bytes: ArrayBuffer, opts: LayoutImportOptions = {}) {
    return this.json<LayoutFit3D>('/layout/import', {
      filename, content_b64: arrayBufferToBase64(bytes), ...opts
    })
  }
  exportUrl(runId: string, kind: 'height_csv' | 'section_csv' | 'stl', params: {
    tile_id?: string; frame?: number; polyline?: Point3D[]
  } = {}) {
    return url3d(this.baseUrl, `/runs/${encodeURIComponent(runId)}/export`, {
      kind, tile_id: params.tile_id, frame: params.frame ?? -1,
      polyline: params.polyline?.map((p) => p.join(',')).join(';')
    })
  }
  examples() {
    return this.json<{ examples: { name: string; title: string; spec: SampleSpec3D }[] }>(
      '/examples'
    )
  }
  validate(spec: SampleSpec3D) {
    return this.json<Validation3D>('/validate', { spec })
  }
  plan(spec: SampleSpec3D) {
    return this.json<Plan3D>('/plan', { spec })
  }
  run(spec: SampleSpec3D, label?: string) {
    return this.json<{ run_id: string }>('/runs', { spec, label })
  }
  runs() {
    return this.json<{ runs: Run3D[] }>('/runs')
  }
  status(id: string) {
    return this.json<Run3D>(`/runs/${encodeURIComponent(id)}`)
  }
  cancel(id: string) {
    return this.json<Run3D>(`/runs/${encodeURIComponent(id)}/cancel`, {})
  }
  manifest(id: string) {
    return this.json<Manifest3D>(`/runs/${encodeURIComponent(id)}/manifest`)
  }
  tiles(id: string) {
    return this.json<{ tiles: Tile3D[] }>(`/runs/${encodeURIComponent(id)}/tiles`)
  }
  height(id: string, tile_id?: string, frame = -1, max_n = 200) {
    return this.json<Height3D>(`/runs/${encodeURIComponent(id)}/height`, undefined, {
      tile_id,
      frame,
      max_n
    })
  }
  section(id: string, polyline_um: Point3D[], tile_id?: string, frame = -1) {
    return this.json<Section3D>(`/runs/${encodeURIComponent(id)}/section`, {
      polyline_um,
      tile_id,
      frame
    })
  }
  series(id: string, tile_id?: string) {
    return this.json<Series3D>(`/runs/${encodeURIComponent(id)}/series`, undefined, { tile_id })
  }
  frames(id: string, tile_id?: string) {
    return this.json<{ t_s: number[] }>(`/runs/${encodeURIComponent(id)}/frames`, undefined, {
      tile_id
    })
  }
  artifact(id: string, path: string) {
    return url3d(this.baseUrl, `/runs/${encodeURIComponent(id)}/artifact`, { path })
  }
}
