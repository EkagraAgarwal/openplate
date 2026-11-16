import type { Plan3D, Sample3DApi, SampleSpec3D } from '../api/client3d'
import { formToSpec, specToForm } from './sample3d'

/** Match the preview quality selected when loading an example into the editor. */
export function exampleSpec(spec: SampleSpec3D): SampleSpec3D {
  return { ...specToForm(spec), numerics: { ...spec.numerics, quality: 'preview' } }
}

export function roughRuntime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return 'Estimate unavailable'
  return `~${Math.max(1, Math.round(seconds / 60))} min (rough estimate)`
}

/** Planning only: never starts a solver. Failed validation is not a zero-minute run. */
export async function estimateSpec(api: Sample3DApi, spec: SampleSpec3D): Promise<Plan3D | null> {
  const plan = await api.plan(formToSpec(spec))
  return plan.ok ? plan : null
}
