/**
 * In-process mock FastAPI-compatible server driven by mock/fixtures.json.
 * Activated with PLATESIM_MOCK=1 so the UI can be developed without Python.
 * Job results scale with a couple of params so the UI visibly reacts.
 */
import http from 'node:http'
import { randomUUID } from 'node:crypto'
import fixtures from '../../mock/fixtures.json'

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
