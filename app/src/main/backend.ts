/**
 * BackendManager: spawns `python -m platesim.server` (or connects to an external
 * URL, or runs the in-process mock) and fronts it with a tiny local gateway that
 * adds permissive CORS so the Vite dev origin can call it. The gateway also
 * forwards WebSocket upgrades so /api/ws works end-to-end.
 */
import { spawn, type ChildProcess } from 'node:child_process'
import http from 'node:http'
import net from 'node:net'
import type { AddressInfo } from 'node:net'
import { startMockServer, type MockServer } from './mockServer'
import type { AppSettings } from './settings'

export type BackendStatus = 'idle' | 'starting' | 'ready' | 'error' | 'stopped'

export interface BackendState {
  status: BackendStatus
  /** Base URL the renderer should call (gateway). */
  url: string
  /** Upstream (python/mock) base URL, informational. */
  upstreamUrl: string | null
  mode: 'spawn' | 'url' | 'mock'
  error: string | null
  pid: number | null
}

const READY_RE = /PLATESIM_SERVER_READY port=(\d+)/
const START_TIMEOUT_MS = 60_000
const MAX_LOG_LINES = 2000

export class BackendManager {
  private child: ChildProcess | null = null
  private mock: MockServer | null = null
  private gateway: http.Server | null = null
  private gatewayPort = 0
  private upstreamPort = 0
  private logLines: string[] = []
  private settings: AppSettings | null = null

  state: BackendState = {
    status: 'idle',
    url: '',
    upstreamUrl: null,
    mode: 'spawn',
    error: null,
    pid: null
  }

  onLog: (line: string) => void = () => {}
  onState: (state: BackendState) => void = () => {}

  getLogs(): string[] {
    return [...this.logLines]
  }

  private log(line: string): void {
    this.logLines.push(line)
    if (this.logLines.length > MAX_LOG_LINES) {
      this.logLines.splice(0, this.logLines.length - MAX_LOG_LINES)
    }
    // also mirror to stdout so `npm run start` / terminal launches show progress
    console.log(line)
    this.onLog(line)
  }

  private setState(patch: Partial<BackendState>): void {
    this.state = { ...this.state, ...patch }
    this.onState(this.state)
  }

  async start(settings: AppSettings): Promise<BackendState> {
    this.settings = settings
    this.setState({ status: 'starting', error: null, mode: settings.backendMode === 'url' ? 'url' : 'spawn' })
    try {
      if (process.env.PLATESIM_MOCK === '1') {
        await this.startMock()
      } else if (settings.backendMode === 'url' && settings.externalUrl.trim()) {
        await this.connectExternal(settings.externalUrl.trim())
      } else {
        await this.spawnPython(settings)
      }
      this.setState({ status: 'ready', error: null })
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      this.log(`[backend] ERROR: ${message}`)
      this.setState({ status: 'error', error: message })
    }
    return this.state
  }

  async stop(): Promise<void> {
    if (this.child) {
      this.log(`[backend] killing python pid ${this.child.pid}`)
      this.child.kill()
      this.child = null
    }
    if (this.mock) {
      this.mock.close()
      this.mock = null
    }
    if (this.gateway) {
      await new Promise<void>((resolve) => this.gateway?.close(() => resolve()))
      this.gateway = null
    }
    this.setState({ status: 'stopped', pid: null })
  }

  async restart(): Promise<BackendState> {
    await this.stop()
    this.logLines = []
    return this.start(this.settings ?? { backendMode: 'spawn', externalUrl: '', platesimRoot: '', pythonBin: 'python', llm: { baseUrl: '', model: '' } })
  }

  // ---------------------------------------------------------------- mock ---

  private async startMock(): Promise<void> {
    this.log('[backend] PLATESIM_MOCK=1 -> starting in-process mock server')
    this.mock = await startMockServer((line) => this.log(line))
    this.upstreamPort = this.mock.port
    this.setState({ mode: 'mock', upstreamUrl: `http://127.0.0.1:${this.mock.port}` })
    await this.startGateway()
  }

  // ------------------------------------------------------------- external ---

  private async connectExternal(url: string): Promise<void> {
    const parsed = new URL(url)
    this.upstreamPort = Number(parsed.port || (parsed.protocol === 'https:' ? 443 : 80))
    this.log(`[backend] connecting to existing server at ${url}`)
    await this.waitHealthy(url, 10_000)
    this.setState({ upstreamUrl: url })
    await this.startGateway()
  }

  // ---------------------------------------------------------------- spawn ---

  private async spawnPython(settings: AppSettings): Promise<void> {
    const port = await freePort()
    const root = settings.platesimRoot
    const bin = settings.pythonBin || 'python'
    const args = ['-m', 'platesim.server', '--port', String(port), '--host', '127.0.0.1']
    this.log(`[backend] spawn: ${bin} ${args.join(' ')}  (cwd=${root})`)
    const child = spawn(bin, args, {
      cwd: root,
      env: { ...process.env },
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe']
    })
    this.child = child
    this.setState({ pid: child.pid ?? null })

    let settled = false
    const readyPromise = new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        if (!settled) {
          settled = true
          reject(new Error(`server did not print PLATESIM_SERVER_READY within ${START_TIMEOUT_MS / 1000}s`))
        }
      }, START_TIMEOUT_MS)

      const onLine = (line: string): void => {
        this.log(`[server] ${line}`)
        const m = line.match(READY_RE)
        if (m && !settled) {
          settled = true
          clearTimeout(timeout)
          this.upstreamPort = Number(m[1])
          resolve()
        }
      }
      splitLines(child.stdout, onLine)
      splitLines(child.stderr, (line) => this.log(`[server:err] ${line}`))

      child.on('error', (err) => {
        if (!settled) {
          settled = true
          clearTimeout(timeout)
          reject(new Error(`failed to start '${bin}': ${err.message}`))
        }
      })
      child.on('exit', (code, signal) => {
        this.log(`[backend] python exited (code=${code} signal=${signal})`)
        if (!settled) {
          settled = true
          clearTimeout(timeout)
          reject(new Error(`server exited before ready (code=${code})`))
        } else if (this.state.status === 'ready') {
          this.setState({ status: 'error', error: 'backend process exited' })
        }
      })
    })

    await readyPromise
    this.setState({ upstreamUrl: `http://127.0.0.1:${this.upstreamPort}` })
    await this.waitHealthy(`http://127.0.0.1:${this.upstreamPort}`, 15_000)
    await this.startGateway()
  }

  private async waitHealthy(base: string, timeoutMs: number): Promise<void> {
    const deadline = Date.now() + timeoutMs
    let lastErr = 'unreachable'
    while (Date.now() < deadline) {
      try {
        const res = await fetch(`${base}/api/health`, { signal: AbortSignal.timeout(2000) })
        if (res.ok) {
          this.log(`[backend] health OK at ${base}`)
          return
        }
        lastErr = `HTTP ${res.status}`
      } catch (err) {
        lastErr = err instanceof Error ? err.message : String(err)
      }
      await sleep(300)
    }
    this.log(`[backend] health check failed at ${base}: ${lastErr} (continuing anyway)`)
  }

  // --------------------------------------------------------------- gateway ---

  /** Local proxy on a free port; adds CORS; forwards HTTP + WS upgrades. */
  private async startGateway(): Promise<void> {
    this.gatewayPort = await freePort()
    const gateway = http.createServer((req, res) => {
      const headers: Record<string, string> = {
        'access-control-allow-origin': '*',
        'access-control-allow-methods': 'GET,POST,PUT,PATCH,DELETE,OPTIONS',
        'access-control-allow-headers': '*'
      }
      if (req.method === 'OPTIONS') {
        res.writeHead(204, headers)
        res.end()
        return
      }
      const proxyReq = http.request(
        {
          host: '127.0.0.1',
          port: this.upstreamPort,
          path: req.url,
          method: req.method,
          headers: { ...req.headers, host: `127.0.0.1:${this.upstreamPort}` }
        },
        (proxyRes) => {
          res.writeHead(proxyRes.statusCode ?? 502, { ...proxyRes.headers, ...headers })
          proxyRes.pipe(res)
        }
      )
      proxyReq.on('error', (err) => {
        this.log(`[gateway] proxy error: ${err.message}`)
        if (!res.headersSent) {
          res.writeHead(502, { ...headers, 'content-type': 'application/json' })
        }
        res.end(JSON.stringify({ detail: `gateway proxy error: ${err.message}` }))
      })
      req.pipe(proxyReq)
    })
    gateway.on('upgrade', (req, socket, head) => {
      const upstream = net.connect(this.upstreamPort, '127.0.0.1', () => {
        const rawHeaders = req.rawHeaders
        let upgradeReq = `${req.method} ${req.url} HTTP/1.1\r\n`
        for (let i = 0; i < rawHeaders.length; i += 2) {
          upgradeReq += `${rawHeaders[i]}: ${rawHeaders[i + 1]}\r\n`
        }
        upgradeReq += '\r\n'
        upstream.write(upgradeReq)
        if (head?.length) upstream.write(head)
        socket.pipe(upstream)
        upstream.pipe(socket)
      })
      upstream.on('error', () => socket.destroy())
      socket.on('error', () => upstream.destroy())
    })
    await new Promise<void>((resolve, reject) => {
      gateway.once('error', reject)
      gateway.listen(this.gatewayPort, '127.0.0.1', () => resolve())
    })
    this.gateway = gateway
    const url = `http://127.0.0.1:${this.gatewayPort}`
    this.log(`[gateway] listening on ${url} -> upstream 127.0.0.1:${this.upstreamPort}`)
    this.setState({ url })
  }
}

// ------------------------------------------------------------------ utils ---

export function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = net.createServer()
    srv.once('error', reject)
    srv.listen(0, '127.0.0.1', () => {
      const port = (srv.address() as AddressInfo).port
      srv.close(() => resolve(port))
    })
  })
}

function splitLines(stream: NodeJS.ReadableStream | null, onLine: (line: string) => void): void {
  if (!stream) return
  let buf = ''
  stream.setEncoding('utf8')
  stream.on('data', (chunk: string) => {
    buf += chunk
    let idx = buf.indexOf('\n')
    while (idx >= 0) {
      const line = buf.slice(0, idx).replace(/\r$/, '')
      buf = buf.slice(idx + 1)
      if (line.length) onLine(line)
      idx = buf.indexOf('\n')
    }
  })
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms))
}
