import type { Feature3D, Plan3D, Point3D, SampleSpec3D, Section3D, Step3D } from '../api/client3d'

export const defaultSpec3D: SampleSpec3D = {
  layout: {
    die_um: [128, 64],
    features: [{ kind: 'rect', x_um: 64, y_um: 32, w_um: 32, h_um: 32 }]
  },
  etch: { default: { depth_um: 32, sidewall_deg: 90 } },
  recipe: {
    steps: [
      {
        t_forward_ms: 10,
        t_reverse_ms: 0,
        t_off_ms: 0,
        j_forward_mA_cm2: 10,
        j_reverse_mA_cm2: 0,
        duration_h: 0.001
      }
    ]
  },
  bath: { name: 'EP-EC1-65L', cu_g_L: 65, h2so4_g_L: 8, cl_ppm: 50, temperature_C: 22 },
  setup: {
    orientation: 'vertical',
    up_edge: '+y',
    stir_shear_1_s: 0,
    stir_angle_deg: 0,
    delta_um: 64,
    area_cm2: 1,
    taped: false
  },
  numerics: { quality: 'preview', dx_um: 8, physics: 'L1', device: 'cpu' }
}
export function specToForm(spec: SampleSpec3D): SampleSpec3D {
  return JSON.parse(JSON.stringify(spec)) as SampleSpec3D
}
export function formToSpec(form: SampleSpec3D): SampleSpec3D {
  return specToForm(form)
}
export function parseSpec(text: string): SampleSpec3D {
  const value = JSON.parse(text) as SampleSpec3D
  const object = (v: unknown) => v !== null && typeof v === 'object' && !Array.isArray(v)
  if (
    !object(value) ||
    !object(value.layout) ||
    !Array.isArray(value.layout.die_um) ||
    !object(value.etch?.default) ||
    !Array.isArray(value.recipe?.steps) ||
    !value.recipe.steps.every(object) ||
    (value.layout.features != null &&
      (!Array.isArray(value.layout.features) ||
        !value.layout.features.every((f) => object(f) && (f.array == null || object(f.array))))) ||
    !object(value.bath) ||
    !object(value.setup) ||
    !object(value.numerics)
  )
    throw new Error('Expected layout, etch, recipe, bath, setup and numerics objects.')
  return value
}
export function recipeMath(step: Step3D): { net: number; duty: number } {
  const period = step.t_forward_ms + step.t_reverse_ms + step.t_off_ms
  return period > 0
    ? {
        net:
          (step.t_forward_ms * step.j_forward_mA_cm2 - step.t_reverse_ms * step.j_reverse_mA_cm2) /
          period,
        duty: step.t_forward_ms / period
      }
    : { net: 0, duty: 0 }
}
export function polylineFromClicks(points: Point3D[], next: Point3D): Point3D[] {
  if (!next.every(Number.isFinite)) return points
  if (points.length !== 1) return [next]
  return points[0][0] === next[0] && points[0][1] === next[1] ? points : [points[0], next]
}
export function outlineTraces(
  lines: Point3D[][],
  name: string,
  color: string,
  dashed = false,
  filled = false
): Record<string, unknown>[] {
  return lines.map((line, i) => ({
    type: 'scatter',
    mode: 'lines',
    name,
    legendgroup: name,
    showlegend: i === 0,
    x: line.map((p) => p[0]),
    y: line.map((p) => p[1]),
    line: { color, dash: dashed ? 'dash' : 'solid' },
    ...(filled ? { fill: 'toself', fillcolor: 'rgba(184,115,51,0.45)' } : {})
  }))
}
/** Bound both axes, preserve endpoints, and transpose x-major API arrays for Plotly. */
export function decimateHeight(x: number[], y: number[], values: (number | null)[][], maxN = 200) {
  const indices = (n: number) => {
    const count = Math.min(n, Math.max(2, Math.floor(maxN)))
    return Array.from({ length: count }, (_, i) =>
      count === 1 ? 0 : Math.round((i * (n - 1)) / (count - 1))
    )
  }
  const ix = indices(x.length),
    iy = indices(y.length)
  return {
    x: ix.map((i) => x[i]),
    y: iy.map((i) => y[i]),
    z: iy.map((j) => ix.map((i) => values[i]?.[j] ?? null))
  }
}

/** Bounded expansion: even a huge array never allocates more than cap instances. */
export function expandFeatures(features: Feature3D[], cap = 2000) {
  const instances: { feature: Feature3D; index: number; ix: number; iy: number }[] = []
  let total = 0
  const count = (n: number | undefined) => n === undefined ? 1 : Number.isFinite(n) ? Math.max(0, Math.floor(n)) : 0
  const limit = Number.isFinite(cap) ? Math.max(0, Math.floor(cap)) : 2000
  features.forEach((f, index) => {
    const nx = count(f.array?.nx), ny = count(f.array?.ny)
    total += nx * ny
    const take = Math.min(nx * ny, limit - instances.length)
    for (let i = 0; i < take; i++) {
      const ix = i % nx, iy = Math.floor(i / nx)
      instances.push({ index, ix, iy, feature: { ...f, array: null,
        x_um: f.x_um + ix * (f.array?.pitch_x_um ?? 0),
        y_um: f.y_um + iy * (f.array?.pitch_y_um ?? 0) } })
    }
  })
  return { instances, omitted: Math.max(0, total - instances.length) }
}

/** Absolute SVG path, in layout µm coordinates (positive y up). */
export function featurePath(f: Feature3D): string {
  // The JSON editor is intentionally editable before server validation succeeds.
  if (![f.x_um, f.y_um, f.rotation_deg ?? 0,
    ...(f.kind === 'circle' ? [f.r_um] : [f.w_um, f.h_um]),
    ...(f.kind === 'rounded_rect' ? [f.r_um] : [])].every(Number.isFinite)) return ''
  const angle = (f.rotation_deg ?? 0) * Math.PI / 180
  const point = (x: number, y: number) => [f.x_um + x * Math.cos(angle) - y * Math.sin(angle),
    f.y_um + x * Math.sin(angle) + y * Math.cos(angle)].map((v) => Number(v.toFixed(8))).join(' ')
  if (f.kind === 'circle') {
    const r = Math.max(0, f.r_um ?? 0)
    return `M ${point(-r, 0)} A ${r} ${r} 0 1 0 ${point(r, 0)} A ${r} ${r} 0 1 0 ${point(-r, 0)} Z`
  }
  const w = Math.max(0, f.w_um ?? 0) / 2, h = Math.max(0, f.h_um ?? 0) / 2
  const r = f.kind === 'rounded_rect' ? Math.max(0, Math.min(f.r_um ?? 0, w, h)) : 0
  if (!r) return `M ${point(-w, -h)} L ${point(w, -h)} L ${point(w, h)} L ${point(-w, h)} Z`
  const arc = (x: number, y: number) => `A ${r} ${r} ${f.rotation_deg ?? 0} 0 1 ${point(x, y)}`
  return `M ${point(-w + r, -h)} L ${point(w - r, -h)} ${arc(w, -h + r)} L ${point(w, h - r)} ${arc(w - r, h)} L ${point(-w + r, h)} ${arc(-w, h - r)} L ${point(-w, -h + r)} ${arc(-w + r, -h)} Z`
}

function featureBounds(f: Feature3D): [number, number, number, number] {
  const a = (f.rotation_deg ?? 0) * Math.PI / 180
  const w = f.kind === 'circle' ? f.r_um ?? 0 : (f.w_um ?? 0) / 2
  const h = f.kind === 'circle' ? w : (f.h_um ?? 0) / 2
  const ex = f.kind === 'circle' ? w : Math.abs(w * Math.cos(a)) + Math.abs(h * Math.sin(a))
  const ey = f.kind === 'circle' ? h : Math.abs(w * Math.sin(a)) + Math.abs(h * Math.cos(a))
  const ox = Math.max(0, (f.array?.nx ?? 1) - 1) * (f.array?.pitch_x_um ?? 0)
  const oy = Math.max(0, (f.array?.ny ?? 1) - 1) * (f.array?.pitch_y_um ?? 0)
  return [f.x_um - ex + Math.min(0, ox), f.y_um - ey + Math.min(0, oy),
    f.x_um + ex + Math.max(0, ox), f.y_um + ey + Math.max(0, oy)]
}

/** Plans omit origins: centre the shape footprint on the covered feature bounds. */
export function tileFootprints(tiles: Plan3D['tiles'], features: Feature3D[]) {
  return tiles.flatMap((tile) => {
    const covered = features.filter((f, i) => featurePath(f) && tile.feature_ids.some((id) =>
      id === String(f.id ?? `feature_${String(i).padStart(4, '0')}`) || id === `f${i}` || id === String(i)))
    if (!covered.length) return []
    const b = covered.map(featureBounds)
    const loX = Math.min(...b.map((v) => v[0])), loY = Math.min(...b.map((v) => v[1]))
    const hiX = Math.max(...b.map((v) => v[2])), hiY = Math.max(...b.map((v) => v[3]))
    const width = Math.max(hiX - loX, tile.shape[0] * tile.dx_um)
    const height = Math.max(hiY - loY, tile.shape[1] * tile.dx_um)
    return [{ tile_id: tile.tile_id, x: (loX + hiX - width) / 2,
      y: (loY + hiY - height) / 2, width, height, approximate: true }]
  })
}

export function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer)
  const chunks: string[] = []
  for (let i = 0; i < bytes.length; i += 0x8000)
    chunks.push(String.fromCharCode(...bytes.subarray(i, i + 0x8000)))
  return btoa(chunks.join(''))
}

export function sectionCsv(section: Section3D): string {
  const rows = ['contour,kind,s_um,z_um']
  for (const [kind, contours] of [['cu', section.cu_outline], ['si', section.si_outline]] as const)
    contours.forEach((line, i) => line.forEach(([s, z]) => rows.push(`${i},${kind},${Number.isFinite(s) ? s : ''},${Number.isFinite(z) ? z : ''}`)))
  return rows.join('\n') + '\n'
}
