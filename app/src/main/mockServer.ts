/**
 * In-process mock FastAPI-compatible server driven by mock/fixtures.json.
 * Activated with PLATESIM_MOCK=1 so the UI can be developed without Python.
 * Job results scale with a couple of params so the UI visibly reacts.
 */
import http from 'node:http'
import { randomUUID } from 'node:crypto'
import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import fixtures from '../../mock/fixtures.json'
import stub3d from '../../mock/fixtures3d.stub.json'
import { parseSpec } from '../renderer/src/lib/sample3d'
import type { Run3D, SampleSpec3D } from '../renderer/src/api/client3d'

interface MockJob {
  id: string
  model_id: string
  params: Record<string, unknown>
  status: string
  progress: number
  message: string
  elapsed_s: number
  error: string | null
  result: unknown | null
  timer: NodeJS.Timeout | null
  startedAt: number
}

interface MockSweep {
  id: string
  model_id: string
  status: string
  results: unknown[]
}

type Json = Record<string, unknown>

function scaleResult(modelId: string, params: Record<string, unknown>): unknown {
  const base = (fixtures.results as Record<string, Json>)[modelId]
  if (!base) return { summary: {}, plots: [], warnings: ['no fixture'], assumptions: [] }
  const clone = JSON.parse(JSON.stringify(base)) as Json
  const summary = { ...(clone.summary as Json) }
  if (modelId === 'demo_faraday') {
    const iTotal = Number(params.I_total_mA ?? 48)
    const k = iTotal / 48
    for (const key of ['peak_current_density_mA_cm2', 'mean_current_density_mA_cm2']) {
      if (typeof summary[key] === 'number') summary[key] = (summary[key] as number) * k
    }
    if (typeof summary.cell_voltage_V === 'number') {
      summary.cell_voltage_V = 2.6 + 0.52 * k
    }
    summary.elapsed_s = 0.02
  } else if (modelId === 'demo_feature_fill') {
    const ar = Number(params.aspect_ratio ?? 4)
    if (typeof summary.fill_time_s === 'number') summary.fill_time_s = Math.round(105 * ar)
  }
  clone.summary = summary
  return clone
}

export interface MockServer {
  port: number
  close: () => void
  log: (line: string) => void
}

export function startMockServer(log: (line: string) => void): Promise<MockServer> {
  const jobs = new Map<string, MockJob>()
  const sweeps = new Map<string, MockSweep>()
  // Source and bundled main both live two levels below app/. Prefer real API captures.
  const fixturePath = resolve(__dirname, '../../mock/fixtures3d.json')
  const data3d = (existsSync(fixturePath) ? JSON.parse(readFileSync(fixturePath, 'utf8')) : stub3d) as Record<string, unknown>
  const fixture3d = (method: string, suffix: string): unknown => data3d[`${method} /api/3d${suffix}`]
  const runs3d = new Map<string, Run3D & { startedAt?: number }>()
  const captured = fixture3d('GET', '/runs') as { runs: Run3D[] }
  for (const r of captured.runs) runs3d.set(r.run_id, { ...r, spec: (fixture3d('GET', '/examples') as { examples: { spec: SampleSpec3D }[] }).examples[0]?.spec })
  const update3d = (r: Run3D & { startedAt?: number }): Run3D => {
    if (r.status === 'running' && r.startedAt) {
      r.progress = Math.min(1, (Date.now() - r.startedAt) / 5000)
      r.message = r.progress < 1 ? 'Simulating fixture run' : 'done'
      if (r.progress === 1) { r.status = 'done'; r.finished = Date.now() / 1000 }
    }
    return r
  }

  const send = (res: http.ServerResponse, code: number, body: unknown): void => {
    const payload = JSON.stringify(body)
    res.writeHead(code, { 'content-type': 'application/json' })
    res.end(payload)
  }

  const readBody = (req: http.IncomingMessage): Promise<string> =>
    new Promise((resolve, reject) => {
      let data = ''
      req.on('data', (c) => (data += c))
      req.on('end', () => resolve(data))
      req.on('error', reject)
    })

  const finishJob = (job: MockJob): void => {
    if (job.status !== 'running') return
    job.status = 'succeeded'
    job.progress = 1
    job.message = 'done'
    job.elapsed_s = (Date.now() - job.startedAt) / 1000
    job.result = scaleResult(job.model_id, job.params)
    log(`[mock] job ${job.id} succeeded`)
  }

  const startJob = (modelId: string, params: Record<string, unknown>): MockJob => {
    const id = `job-${randomUUID().slice(0, 8)}`
    const job: MockJob = {
      id,
      model_id: modelId,
      params,
      status: 'running',
      progress: 0,
      message: 'starting',
      elapsed_s: 0,
      error: null,
      result: null,
      timer: null,
      startedAt: Date.now()
    }
    jobs.set(id, job)
    log(`[mock] job ${id} started (${modelId})`)
    job.timer = setInterval(() => {
      job.elapsed_s = (Date.now() - job.startedAt) / 1000
      job.progress = Math.min(1, job.progress + 0.12)
      job.message = `simulating (${Math.round(job.progress * 100)}%)`
      if (job.progress >= 1) {
        if (job.timer) clearInterval(job.timer)
        finishJob(job)
      }
    }, 150)
    return job
  }

  const jobView = (job: MockJob): Json => ({
    id: job.id,
    model_id: job.model_id,
    status: job.status,
    progress: job.progress,
    message: job.message,
    elapsed_s: Number(job.elapsed_s.toFixed(2)),
    error: job.error
  })

  const server = http.createServer((req, res) => {
    void (async () => {
      const url = new URL(req.url ?? '/', 'http://127.0.0.1')
      const p = url.pathname
      const method = req.method ?? 'GET'
      try {
        if (p.startsWith('/api/3d/')) {
          const suffix = p.slice('/api/3d'.length)
          const body = method === 'POST' ? JSON.parse((await readBody(req)) || '{}') : {}
          const issue = (field: string, error: string, fix: string) => send(res, 422, { detail: { error, field, fix } })
          if (['/validate', '/plan', '/runs'].includes(suffix) && method === 'POST') {
            let spec: SampleSpec3D
            try { spec = parseSpec(JSON.stringify(body.spec)) } catch { return issue('spec', 'Invalid sample specification', 'Supply layout, etch, recipe, bath, setup and numerics.') }
            const errors = []
            if (!['preview', 'standard', 'fine'].includes(spec.numerics.quality ?? 'standard')) errors.push({ field: 'numerics.quality', message: 'Unknown quality', fix: 'Choose preview, standard or fine.' })
            if (!(spec.etch.default.depth_um > 0)) errors.push({ field: 'etch.default.depth_um', message: 'Depth must be positive', fix: 'Set depth_um > 0.' })
            if (spec.numerics.dx_um != null && !(spec.numerics.dx_um > 0)) errors.push({ field: 'numerics.dx_um', message: 'Spacing must be positive', fix: 'Set dx_um > 0.' })
            if (suffix === '/validate') return send(res, 200, { ok: !errors.length, errors })
            if (errors.length) { const e = errors[0]; return issue(e.field, e.message, e.fix) }
            if (suffix === '/plan') return send(res, 200, fixture3d('POST', '/plan'))
            const run_id = `sample-${randomUUID().slice(0, 8)}`
            runs3d.set(run_id, { run_id, label: body.label || null, status: 'running', progress: 0, message: 'starting', error: null, spec, created: Date.now() / 1000, finished: null, startedAt: Date.now() })
            return send(res, 202, { run_id })
          }
          if (method === 'GET' && suffix === '/runs') return send(res, 200, { runs: [...runs3d.values()].reverse().map((r) => {
            update3d(r)
            return { run_id: r.run_id, label: r.label, status: r.status, progress: r.progress, message: r.message, created: r.created, finished: r.finished }
          }) })
          const match = suffix.match(/^\/runs\/([^/]+)(?:\/(\w+))?$/)
          if (match) {
            const r = runs3d.get(decodeURIComponent(match[1]))
            if (!r) return send(res, 404, { detail: 'unknown run' })
            update3d(r)
            const view = () => ({ run_id: r.run_id, label: r.label, status: r.status, progress: r.progress, message: r.message, error: r.error, spec: r.spec })
            if (method === 'GET' && !match[2]) return send(res, 200, view())
            if (method === 'POST' && match[2] === 'cancel') {
              if (r.status === 'running') { r.status = 'cancelled'; r.message = 'cancelled'; r.finished = Date.now() / 1000 }
              return send(res, 200, view())
            }
            if (r.status !== 'done') return send(res, 409, { detail: 'run not finished' })
            if (match[2] === 'artifact' && method === 'GET') {
              const artifact = fixture3d('GET', '/runs/{id}/artifact') as { path: string; media_type: string; base64: string } | undefined
              const path = url.searchParams.get('path')
              const manifest = fixture3d('GET', '/runs/{id}/manifest') as { artifacts: { path: string }[] }
              if (!artifact || path !== artifact.path || path.includes('..') || !manifest.artifacts.some((a) => a.path === path)) return send(res, 404, { detail: 'unknown artifact' })
              res.writeHead(200, { 'content-type': artifact.media_type }); res.end(Buffer.from(artifact.base64, 'base64')); return
            }
            const result = fixture3d(method, `/runs/{id}/${match[2]}`)
            if (result) {
              const clone = JSON.parse(JSON.stringify(result))
              // The stub/capture only stores final geometry; frame 0 is the bare surface.
              if (match[2] === 'height' && Number(url.searchParams.get('frame') ?? -1) === 0) { clone.height_um = clone.bare_height_um; clone.frame = 0; clone.t_s = 0 }
              if (match[2] === 'section' && body.frame === 0) clone.cu_outline = clone.si_outline
              return send(res, 200, clone)
            }
          }
          const result = fixture3d(method, suffix)
          if (result && ['/schema', '/examples'].includes(suffix)) return send(res, 200, result)
          return send(res, 404, { detail: 'unknown 3-D endpoint' })
        }
        if (method === 'GET' && p === '/api/health') {
          return send(res, 200, { status: 'ok', mock: true, version: '0.0-mock' })
        }
        if (method === 'GET' && p === '/api/models') {
          return send(res, 200, fixtures.models)
        }
        const modelMatch = p.match(/^\/api\/models\/([^/]+)$/)
        if (method === 'GET' && modelMatch) {
          const m = fixtures.models.find((x) => x.id === modelMatch[1])
          if (!m) return send(res, 404, { detail: 'unknown model' })
          return send(res, 200, m)
        }
        const runMatch = p.match(/^\/api\/models\/([^/]+)\/run$/)
        if (method === 'POST' && runMatch) {
          const body = (await readBody(req)) || '{}'
          const params = JSON.parse(body) as Record<string, unknown>
          const job = startJob(runMatch[1], params)
          return send(res, 200, { job_id: job.id })
        }
        const sweepMatch = p.match(/^\/api\/models\/([^/]+)\/sweep$/)
        if (method === 'POST' && sweepMatch) {
          const body = (await readBody(req)) || '{}'
          const { base = {}, vary = {} } = JSON.parse(body) as {
            base: Record<string, unknown>
            vary: Record<string, unknown[]>
          }
          const id = `sweep-${randomUUID().slice(0, 8)}`
          const sweep: MockSweep = { id, model_id: sweepMatch[1], status: 'running', results: [] }
          sweeps.set(id, sweep)
          log(`[mock] sweep ${id} started`)
          setTimeout(() => {
            const varyKeys = Object.keys(vary)
            const valueLists = varyKeys.map((k) => (Array.isArray(vary[k]) ? vary[k] : [vary[k]]))
            const combos: Record<string, unknown>[] = []
            const walk = (i: number, acc: Record<string, unknown>): void => {
              if (i >= valueLists.length) {
                combos.push({ ...acc })
                return
              }
              for (const v of valueLists[i]) {
                acc[varyKeys[i]] = v
                walk(i + 1, acc)
                delete acc[varyKeys[i]]
              }
            }
            walk(0, {})
            sweep.results = combos.slice(0, 64).map((p) => ({
              params: { ...base, ...p },
              result: scaleResult(sweep.model_id, { ...base, ...p })
            }))
            sweep.status = 'done'
            log(`[mock] sweep ${id} done (${sweep.results.length} points)`)
          }, 700)
          return send(res, 200, { sweep_id: id })
        }
        const sweepGet = p.match(/^\/api\/sweeps\/([^/]+)$/)
        if (method === 'GET' && sweepGet) {
          const s = sweeps.get(sweepGet[1])
          if (!s) return send(res, 404, { detail: 'unknown sweep' })
          return send(res, 200, { id: s.id, status: s.status, results: s.results })
        }
        if (method === 'GET' && p === '/api/jobs') {
          return send(res, 200, { jobs: [...jobs.values()].map(jobView) })
        }
        const jobGet = p.match(/^\/api\/jobs\/([^/]+)$/)
        if (method === 'GET' && jobGet) {
          const j = jobs.get(jobGet[1])
          if (!j) return send(res, 404, { detail: 'unknown job' })
          return send(res, 200, jobView(j))
        }
        const resultMatch = p.match(/^\/api\/jobs\/([^/]+)\/result$/)
        if (method === 'GET' && resultMatch) {
          const j = jobs.get(resultMatch[1])
          if (!j) return send(res, 404, { detail: 'unknown job' })
          if (j.status !== 'succeeded' || !j.result) {
            return send(res, 409, { detail: `job status is ${j.status}` })
          }
          return send(res, 200, j.result)
        }
        const cancelMatch = p.match(/^\/api\/jobs\/([^/]+)\/cancel$/)
        if (method === 'POST' && cancelMatch) {
          const j = jobs.get(cancelMatch[1])
          if (!j) return send(res, 404, { detail: 'unknown job' })
          if (j.timer) clearInterval(j.timer)
          if (j.status === 'running') {
            j.status = 'cancelled'
            j.message = 'cancelled'
          }
          return send(res, 200, jobView(j))
        }
        if (method === 'GET' && p === '/api/data/planar_doe') {
          return send(res, 200, fixtures.planar_doe)
        }
        if (method === 'GET' && p === '/api/data/literature') {
          return send(res, 200, fixtures.literature)
        }
        if (method === 'GET' && p === '/api/data/sem') {
          return send(res, 200, fixtures.sem)
        }
        return send(res, 404, { detail: `no mock route for ${method} ${p}` })
      } catch (err) {
        return send(res, 500, { detail: String(err) })
      }
    })()
  })

  return new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      const port = typeof address === 'object' && address ? address.port : 0
      log(`[mock] server listening on 127.0.0.1:${port}`)
      resolve({
        port,
        close: () => {
          for (const j of jobs.values()) if (j.timer) clearInterval(j.timer)
          server.close()
        },
        log
      })
    })
  })
}
