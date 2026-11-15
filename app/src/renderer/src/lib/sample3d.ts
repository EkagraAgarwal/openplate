import type { Point3D, SampleSpec3D, Step3D } from '../api/client3d'

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
