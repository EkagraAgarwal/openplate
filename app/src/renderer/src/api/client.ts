/**
 * Typed API client for the platesim FastAPI backend — exactly the endpoint set
 * documented in the task. Every method tolerates missing endpoints (404 or
 * unreachable -> null / empty) so the UI degrades instead of crashing.
 */
import type { Job, ModelSpec, RunResult, SweepState, ParamValues } from '../types'

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
    public readonly path?: string
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export interface PlanarDoePayload {
  columns?: string[]
  rows?: unknown[][]
  records?: Record<string, unknown>[]
  [k: string]: unknown
}

export interface LiteratureEntry {
  name?: string
  tag?: string
  source?: string
  citation?: string
  data?: unknown
}

export interface SemPayload {
  columns?: string[]
  rows?: unknown[][]
  records?: Record<string, unknown>[]
  [k: string]: unknown
}

export class PlatesimApi {
  constructor(public baseUrl: string) {}

  setBaseUrl(url: string): void {
    this.baseUrl = url.replace(/\/+$/, '')
  }

  private url(path: string): string {
    return `${this.baseUrl.replace(/\/+$/, '')}${path}`
  }

  private async json<T>(path: string, init?: RequestInit, tolerate404 = true): Promise<T | null> {
    try {
      const res = await fetch(this.url(path), init)
      if (res.status === 404 && tolerate404) return null
      if (!res.ok) {
        let detail = ''
        try {
          const body = (await res.json()) as { detail?: string }
          detail = body.detail ? `: ${body.detail}` : ''
        } catch {
          /* body not json */
        }
        throw new ApiError(`${init?.method ?? 'GET'} ${path} -> HTTP ${res.status}${detail}`, res.status, path)
      }
      if (res.status === 204) return null
      return (await res.json()) as T
    } catch (err) {
      if (err instanceof ApiError) throw err
      // network failure: tolerated for optional endpoints
      if (tolerate404) return null
      throw new ApiError(`${path}: ${err instanceof Error ? err.message : String(err)}`, undefined, path)
    }
  }

  // ------------------------------------------------------------ endpoints --

  health(): Promise<Record<string, unknown> | null> {
    return this.json('/api/health')
  }

  async models(): Promise<ModelSpec[]> {
    const data = await this.json<ModelSpec[] | { models: ModelSpec[] }>('/api/models')
    if (!data) return []
    if (Array.isArray(data)) return data
    return Array.isArray(data.models) ? data.models : []
  }

  model(id: string): Promise<ModelSpec | null> {
    return this.json(`/api/models/${encodeURIComponent(id)}`)
  }

  run(modelId: string, params: ParamValues): Promise<{ job_id: string } | null> {
    return this.json(
      `/api/models/${encodeURIComponent(modelId)}/run`,
      { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(params) },
      false
    )
  }

  async jobs(): Promise<Job[]> {
    const data = await this.json<Job[] | { jobs: Job[] }>('/api/jobs')
    if (!data) return []
    if (Array.isArray(data)) return data
    return Array.isArray(data.jobs) ? data.jobs : []
  }

  job(id: string): Promise<Job | null> {
    return this.json(`/api/jobs/${encodeURIComponent(id)}`)
  }

  async result(id: string): Promise<RunResult | null> {
    const raw = await this.json<RunResult & { result?: RunResult }>(
      `/api/jobs/${encodeURIComponent(id)}/result`
    )
    // platesim.server wraps the model output: {job_id, model_id, status, error, result}
    return raw && raw.result && !raw.summary ? raw.result : raw
  }

  cancel(id: string): Promise<unknown> {
    return this.json(
      `/api/jobs/${encodeURIComponent(id)}/cancel`,
      { method: 'POST' },
      false
    )
  }

  sweep(
    modelId: string,
    base: ParamValues,
    vary: Record<string, ParamValueLike[]>
  ): Promise<{ sweep_id?: string; id?: string } | null> {
    return this.json(
      `/api/models/${encodeURIComponent(modelId)}/sweep`,
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ base, vary })
      },
      false
    )
  }

  async sweepStatus(id: string): Promise<SweepState | null> {
    const raw = await this.json<
      SweepState & { table?: { params: ParamValues; summary?: Record<string, unknown> }[] }
    >(`/api/sweeps/${encodeURIComponent(id)}`)
    if (!raw) return null
    // platesim.server returns rows under `table` with only the run summary
    if (!raw.results && raw.table) {
      raw.results = raw.table.map((row) => ({
        params: row.params,
        result: { summary: row.summary ?? {}, plots: [] } as unknown as RunResult
      }))
    }
    return raw
  }

  planarDoe(): Promise<PlanarDoePayload | null> {
    return this.json('/api/data/planar_doe')
  }

  literature(): Promise<LiteratureEntry[] | null> {
    return this.json('/api/data/literature')
  }

  sem(): Promise<SemPayload | null> {
    return this.json('/api/data/sem')
  }
}

type ParamValueLike = number | string | boolean

/** Build ws:// URL for the job-events socket. */
export function wsUrl(baseUrl: string): string {
  const u = new URL(baseUrl.replace(/\/+$/, ''))
  u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:'
  u.pathname = '/api/ws'
  u.search = ''
  return u.toString()
}
