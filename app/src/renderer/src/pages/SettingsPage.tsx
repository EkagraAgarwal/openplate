import { useEffect, useState } from 'react'
import { useApp } from '../lib/context'

interface Settings {
  backendMode: 'spawn' | 'url'
  externalUrl: string
  platesimRoot: string
  pythonBin: string
  llm: { baseUrl: string; model: string }
  llmKeyStored?: boolean
}

export function SettingsPage() {
  const { backend, restartBackend, refreshAssistantConfig, setPage } = useApp()
  const [s, setS] = useState<Settings | null>(null)
  const [key, setKey] = useState('')
  const [saved, setSaved] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    void window.platesim?.getSettings().then((loaded) => setS(loaded as Settings))
  }, [])

  if (!s) return <div className="page"><div className="empty-note">loading settings…</div></div>

  const patch = (p: Partial<Settings>): void => setS({ ...s, ...p })

  const save = async (): Promise<void> => {
    setSaved(null)
    setErr(null)
    try {
      let next = await window.platesim?.saveSettings(s)
      if (key.trim()) {
        const res = await window.platesim?.saveLlmKey(key.trim())
        if (!res?.ok) setErr('API key could not be encrypted/stored (safeStorage unavailable).')
        next = next ? { ...next, llmKeyStored: res?.stored } : next
        setKey('')
      }
      setS(next ?? s)
      setSaved('Saved. Restart the backend to apply connection changes.')
      refreshAssistantConfig()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="page settings-page">
      <div className="page-head">
        <h2>Settings</h2>
      </div>

      <div className="card">
        <h4>Backend</h4>
        <label className="field radio">
          <input
            type="radio"
            name="backendMode"
            checked={s.backendMode === 'spawn'}
            onChange={() => patch({ backendMode: 'spawn' })}
          />
          <span>
            <strong>Spawn local server</strong>
            <div className="small muted">
              Runs <code>python -m platesim.server --port &lt;free port&gt; --host 127.0.0.1</code>{' '}
              with cwd = platesim project root.
            </div>
          </span>
        </label>
        <label className="field radio">
          <input
            type="radio"
            name="backendMode"
            checked={s.backendMode === 'url'}
            onChange={() => patch({ backendMode: 'url' })}
          />
          <span>
            <strong>Connect to an already-running server</strong>
            <div className="small muted">Base URL, e.g. http://127.0.0.1:8000</div>
          </span>
        </label>

        {s.backendMode === 'url' && (
          <label className="field">
            <span>Server URL</span>
            <input
              type="text"
              placeholder="http://127.0.0.1:8000"
              value={s.externalUrl}
              onChange={(e) => patch({ externalUrl: e.target.value })}
            />
          </label>
        )}

        <label className="field">
          <span>platesim project root (cwd for the spawned server)</span>
          <input
            type="text"
            placeholder="default: parent of app/"
            value={s.platesimRoot}
            onChange={(e) => patch({ platesimRoot: e.target.value })}
          />
        </label>
        <div className="small muted">
          Override with the <code>PLATESIM_ROOT</code> environment variable (wins over this field).
          Backend status: <span className={`conn conn-${backend?.status ?? 'idle'}`}>{backend?.status ?? 'idle'}</span>
          {backend?.mode === 'mock' && <span className="badge">mock</span>}
          {backend?.error && <span className="err-text"> {backend.error}</span>}
        </div>
        <div className="row gap">
          <button type="button" className="btn btn-primary" onClick={() => void save()}>
            Save settings
          </button>
          <button type="button" className="btn" onClick={restartBackend}>
            Restart backend
          </button>
        </div>
        {saved && <div className="ok-note">{saved}</div>}
        {err && <div className="field-error">{err}</div>}
      </div>

      <div className="card">
        <h4>Assistant (OpenAI-compatible)</h4>
        <label className="field">
          <span>Base URL</span>
          <input
            type="text"
            placeholder="https://api.opencode.ai/v1 (or any OpenAI-compatible endpoint)"
            value={s.llm.baseUrl}
            onChange={(e) => patch({ llm: { ...s.llm, baseUrl: e.target.value } })}
          />
        </label>
        <label className="field">
          <span>Model</span>
          <input
            type="text"
            placeholder="e.g. opencode/glm-4.6"
            value={s.llm.model}
            onChange={(e) => patch({ llm: { ...s.llm, model: e.target.value } })}
          />
        </label>
        <label className="field">
          <span>API key {s.llmKeyStored ? '(stored via safeStorage ✓)' : '(not stored)'}</span>
          <input
            type="password"
            placeholder={s.llmKeyStored ? '••••••••  (enter to replace)' : 'sk-…'}
            value={key}
            onChange={(e) => setKey(e.target.value)}
          />
        </label>
        <div className="row gap">
          <button type="button" className="btn btn-primary" onClick={() => void save()}>
            Save settings
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => {
              void window.platesim?.saveLlmKey('').then(() => {
                setKey('')
                void window.platesim?.getSettings().then((x) => setS(x as Settings))
                refreshAssistantConfig()
              })
            }}
          >
            Clear stored key
          </button>
          <button type="button" className="btn btn-ghost" onClick={() => setPage('assistant')}>
            Open Assistant →
          </button>
        </div>
        <p className="small muted">
          The key never enters the renderer: chat requests are proxied through the main process with
          your stored credential.
        </p>
      </div>
    </div>
  )
}
