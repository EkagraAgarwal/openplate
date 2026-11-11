import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AppContext, type AppState, type BackendDto, type CompareEntry, type Page } from './lib/context'
import { PlatesimApi } from './api/client'
import { GROUP_ORDER, groupFor, type ModelSpec } from './types'
import { ModelPage } from './pages/ModelPage'
import { DataPage } from './pages/DataPage'
import { AssistantPage } from './pages/AssistantPage'
import { SettingsPage } from './pages/SettingsPage'
import { ResultView } from './components/ResultView'
import { PlotRenderer } from './components/PlotRenderer'
import type { ParamValues, RunResult } from './types'

const THEME_KEY = 'platesim.theme'

function Sidebar({
  models,
  selected,
  onSelect,
  onNav,
  page
}: {
  models: ModelSpec[]
  selected: string | null
  onSelect: (id: string) => void
  onNav: (p: Page) => void
  page: Page
}) {
  const groups = useMemo(() => {
    const byGroup = new Map<string, ModelSpec[]>()
    for (const m of models) {
      const g = groupFor(m)
      const list = byGroup.get(g) ?? []
      list.push(m)
      byGroup.set(g, list)
    }
    return GROUP_ORDER.filter((g) => byGroup.has(g)).map((g) => [g, byGroup.get(g) ?? []] as const)
  }, [models])

  const navItems: [Page, string][] = [
    ['models', 'Models'],
    ['data', 'Data'],
    ['assistant', 'Assistant'],
    ['settings', 'Settings']
  ]

  return (
    <aside className="sidebar">
      <nav className="side-nav">
        {navItems.map(([id, label]) => (
          <button
            key={id}
            type="button"
            className={`side-nav-item ${page === id ? 'active' : ''}`}
            onClick={() => onNav(id)}
          >
            {label}
          </button>
        ))}
      </nav>

      <div className="side-section-title">Models</div>
      <div className="model-list">
        {models.length === 0 && <div className="empty-note">No models (server not ready?).</div>}
        {groups.map(([group, list]) => (
          <div key={group} className="model-group">
            <div className="model-group-title">{group}</div>
            {list.map((m) => (
              <button
                key={m.id}
                type="button"
                className={`model-item ${selected === m.id ? 'active' : ''}`}
                onClick={() => onSelect(m.id)}
                title={m.description ?? m.title}
              >
                <span className="model-item-title">{m.title}</span>
                <span className="mono small muted">{m.id}</span>
              </button>
            ))}
          </div>
        ))}
      </div>
    </aside>
  )
}

function CompareOverlay({ entries, onClose }: { entries: CompareEntry[]; onClose: () => void }) {
  const keys = [...new Set(entries.flatMap((e) => Object.keys(e.result.summary ?? {})))]
  return (
    <div className="overlay">
      <div className="overlay-card card">
        <div className="result-header">
          <h3>Compare ({entries.length} results)</h3>
          <button type="button" className="btn" onClick={onClose}>
            ✕ Close
          </button>
        </div>

        <div className="table-scroll">
          <table className="data-table compare-table">
            <thead>
              <tr>
                <th>key</th>
                {entries.map((e) => (
                  <th key={e.key}>{e.label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {keys.map((k) => (
                <tr key={k}>
                  <th>{k}</th>
                  {entries.map((e) => {
                    const v = e.result.summary?.[k]
                    return (
                      <td key={e.key} className="mono">
                        {v === undefined
                          ? '—'
                          : typeof v === 'object'
                            ? JSON.stringify(v)
                            : String(v)}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Overlay line plots across results when titles match */}
        {entries.length > 1 && <CompareLines entries={entries} />}

        <div className="plot-grid">
          {entries.map((e) => (
            <div key={e.key} className="compare-block">
              <div className="mono small">{e.label}</div>
              <ResultView label="" result={e.result} />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function CompareLines({ entries }: { entries: CompareEntry[] }) {
  // group line plots by title; overlay series from all entries
  const byTitle = new Map<string, CompareEntry[]>()
  for (const e of entries) {
    for (const p of e.result.plots ?? []) {
      if (p.kind !== 'line') continue
      const list = byTitle.get(p.title ?? 'plot') ?? []
      if (!list.includes(e)) list.push(e)
      byTitle.set(p.title ?? 'plot', list)
    }
  }
  if (byTitle.size === 0) return null
  return (
    <div>
      <h4>Overlaid line plots</h4>
      {[...byTitle.entries()].map(([title, list]) => {
        const series = list.flatMap((e) => {
          const p = (e.result.plots ?? []).find(
            (q) => q.kind === 'line' && (q.title ?? 'plot') === title
          )
          if (!p || p.kind !== 'line') return []
          return p.series.map((s) => ({
            name: `${e.label}: ${s.name ?? 'series'}`,
            y: s.y,
            mode: s.mode
          }))
        })
        const first = (list[0].result.plots ?? []).find(
          (q) => q.kind === 'line' && (q.title ?? 'plot') === title
        )
        if (!first || first.kind !== 'line') return null
        return (
          <figure key={title} className="plot-card">
            <PlotRenderer plot={{ ...first, series }} mirror={false} height={300} />
            <figcaption className="plot-caption small muted">{title}</figcaption>
          </figure>
        )
      })}
    </div>
  )
}

function LogsPanel({
  logs,
  open,
  toggle
}: {
  logs: string[]
  open: boolean
  toggle: () => void
}) {
  const bodyRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (open && bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight
  }, [logs, open])
  return (
    <div className={`logs-panel ${open ? 'logs-open' : ''}`}>
      <button type="button" className="logs-toggle" onClick={toggle}>
        {open ? '▾' : '▴'} Backend logs ({logs.length})
      </button>
      {open && (
        <div className="logs-body mono" ref={bodyRef}>
          {logs.length === 0 ? <div className="muted">no log output yet</div> : logs.map((l, i) => (
            <div key={i}>{l}</div>
          ))}
        </div>
      )}
    </div>
  )
}

export function App() {
  const [page, setPage] = useState<Page>('models')
  const [theme, setTheme] = useState<'light' | 'dark'>(
    () => (localStorage.getItem(THEME_KEY) as 'light' | 'dark') || 'light'
  )
  const [backend, setBackend] = useState<BackendDto | null>(null)
  const [backendLogs, setBackendLogs] = useState<string[]>([])
  const [models, setModels] = useState<ModelSpec[]>([])
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null)
  const [pendingParams, setPendingParams] = useState<{ modelId: string; params: ParamValues } | null>(null)
  const [compare, setCompare] = useState<CompareEntry[]>([])
  const [showCompare, setShowCompare] = useState(false)
  const [activeResult, setActiveResult] = useState<{ label: string; result: RunResult } | null>(null)
  const [logsOpen, setLogsOpen] = useState(false)
  const [assistantConfigured, setAssistantConfigured] = useState(false)

  const api = useMemo(
    () => (backend?.url ? new PlatesimApi(backend.url) : null),
    [backend?.url]
  )

  // theme
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem(THEME_KEY, theme)
  }, [theme])

  // backend state + logs subscriptions
  useEffect(() => {
    const p = window.platesim
    if (!p) {
      // web mode (no Electron preload): the page is served by platesim.server, or ?api=<url> points at one
      const url = new URLSearchParams(window.location.search).get('api') ?? window.location.origin
      setBackend({ status: 'ready', url, upstreamUrl: url, mode: 'url', error: null, pid: null })
      return
    }
    const offLog = p.onBackendLog((line) => setBackendLogs((l) => [...l.slice(-1999), line]))
    const offState = p.onBackendState((s) => setBackend(s))
    void p.backendState().then((s) => setBackend(s))
    void p.backendLogs().then((l) => setBackendLogs(l))
    return () => {
      offLog()
      offState()
    }
  }, [])

  // load models whenever the api url changes
  useEffect(() => {
    if (!api) return
    let alive = true
    void (async () => {
      const list = await api.models()
      if (!alive) return
      setModels(list)
      setSelectedModelId((cur) => cur ?? (list[0]?.id ?? null))
    })()
    return () => {
      alive = false
    }
  }, [api])

  const refreshAssistantConfig = useCallback(() => {
    void window.platesim
      ?.getSettings()
      .then((s) =>
        setAssistantConfigured(Boolean(s?.llm.baseUrl && s?.llm.model && s?.llmKeyStored))
      )
      .catch(() => setAssistantConfigured(false))
  }, [])

  useEffect(() => {
    refreshAssistantConfig()
  }, [refreshAssistantConfig])

  // assistant proposal -> pending params (never auto-run)
  useEffect(() => {
    const handler = (e: Event): void => {
      const detail = (e as CustomEvent<{ model_id: string; params: ParamValues }>).detail
      if (detail) {
        setPendingParams({ modelId: detail.model_id, params: detail.params })
        setPage('models')
      }
    }
    window.addEventListener('platesim:load-proposal', handler)
    return () => window.removeEventListener('platesim:load-proposal', handler)
  }, [])

  const restartBackend = useCallback(() => {
    void window.platesim?.backendRestart().then((s) => setBackend(s))
  }, [])

  const state: AppState = {
    page,
    setPage,
    theme,
    toggleTheme: () => setTheme((t) => (t === 'light' ? 'dark' : 'light')),
    backend,
    backendLogs,
    api,
    restartBackend,
    models,
    selectedModelId,
    selectModel: (id) => {
      setSelectedModelId(id)
      setPage('models')
    },
    pendingParams,
    clearPendingParams: () => setPendingParams(null),
    compare,
    toggleCompare: (entry) =>
      setCompare((c) => {
        if (c.some((x) => x.key === entry.key)) return c.filter((x) => x.key !== entry.key)
        if (c.length >= 4) return c
        return [...c, entry]
      }),
    clearCompare: () => setCompare([]),
    showCompare,
    setShowCompare,
    openResult: (label, result) => setActiveResult({ label, result }),
    closeResult: () => setActiveResult(null),
    activeResult,
    assistantConfigured,
    refreshAssistantConfig
  }

  const backendBadge = backend ? (
    <span className={`badge badge-${backend.status === 'ready' ? 'ok' : backend.status === 'error' ? 'error' : 'run'}`}>
      {backend.status}
      {backend.mode === 'mock' ? ' · mock' : ''}
      {backend.pid ? ` · pid ${backend.pid}` : ''}
    </span>
  ) : null

  return (
    <AppContext.Provider value={state}>
      <div className="app">
        {(backend === null || backend.status === 'starting' || backend.status === 'idle') && (
          <div className="splash" role="status">
            <div className="splash-inner">
              <div className="splash-mark">▣</div>
              <div className="splash-title">platesim</div>
              <div className="splash-bar">
                <span />
              </div>
              <div className="splash-note mono">
                {backend?.mode === 'mock'
                  ? 'starting mock backend…'
                  : 'waiting for backend (python -m platesim.server)…'}
              </div>
            </div>
          </div>
        )}
        {backend?.status === 'error' && (
          <div className="banner-error">
            Backend failed to start: {backend.error} — adjust connection settings (Settings →
            Restart backend) or inspect the logs panel.
          </div>
        )}
        <header className="topbar">
          <div className="brand">
            <span className="brand-mark">▣</span> platesim
            <span className="muted small">Cu electroplating simulator</span>
          </div>
          <div className="topbar-right">
            {backendBadge}
            {backend?.error && <span className="err-text small">{backend.error}</span>}
            <button type="button" className="btn btn-small" onClick={restartBackend} title="Restart backend">
              ↻ backend
            </button>
            <button
              type="button"
              className="btn btn-small"
              onClick={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))}
              title="Toggle theme"
            >
              {theme === 'light' ? '☾ dark' : '☀ light'}
            </button>
          </div>
        </header>

        <div className="app-body">
          <Sidebar
            models={models}
            selected={selectedModelId}
            onSelect={(id) => {
              setSelectedModelId(id)
              setPage('models')
            }}
            onNav={setPage}
            page={page}
          />
          <main className="main">
            {page === 'models' && <ModelPage />}
            {page === 'data' && <DataPage />}
            {page === 'assistant' && <AssistantPage />}
            {page === 'settings' && <SettingsPage />}
          </main>
        </div>

        {showCompare && compare.length >= 2 && (
          <CompareOverlay entries={compare} onClose={() => setShowCompare(false)} />
        )}

        <LogsPanel logs={backendLogs} open={logsOpen} toggle={() => setLogsOpen((o) => !o)} />
      </div>
    </AppContext.Provider>
  )
}
