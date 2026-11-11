import { useEffect, useRef, useState } from 'react'
import { useApp } from '../lib/context'
import type { ModelSpec, ParamValues } from '../types'

interface Msg {
  role: 'user' | 'assistant'
  content: string
}

/** Extract the last ```json ... ``` block containing model_id + params. */
export function extractProposal(text: string): { model_id: string; params: ParamValues } | null {
  const blocks = [...text.matchAll(/```(?:json)?\s*([\s\S]*?)```/g)].map((m) => m[1])
  for (let i = blocks.length - 1; i >= 0; i--) {
    try {
      const parsed = JSON.parse(blocks[i].trim()) as { model_id?: unknown; params?: unknown }
      if (typeof parsed.model_id === 'string' && parsed.params && typeof parsed.params === 'object') {
        return { model_id: parsed.model_id, params: parsed.params as ParamValues }
      }
    } catch {
      /* try next block */
    }
  }
  // also accept an un-fenced object
  const un = text.match(/\{[\s\S]*"model_id"[\s\S]*"params"[\s\S]*\}/)
  if (un) {
    try {
      const parsed = JSON.parse(un[0]) as { model_id: string; params: ParamValues }
      if (typeof parsed.model_id === 'string' && parsed.params) return parsed
    } catch {
      /* malformed */
    }
  }
  return null
}

export function buildSystemPrompt(models: ModelSpec[]): string {
  const lines = [
    'You are the platesim assistant: a scientific-computing copilot for Cu electroplating simulations.',
    'Answer concisely and scientifically. When you recommend a simulation run, output ONLY a fenced',
    '```json code block with shape {"model_id": "<id>", "params": {<param overrides>}} — the user will',
    'review and load it manually; never claim a run was executed.',
    '',
    'Available models:'
  ]
  for (const m of models) {
    const params = (m.params ?? [])
      .map((p) => `${p.name}:${p.type}${p.unit ? ` [${p.unit}]` : ''}${typeof p.default !== 'undefined' ? ` =${p.default}` : ''}`)
      .join(', ')
    lines.push(`- ${m.id} — ${m.title}${m.description ? `: ${m.description}` : ''}`)
    if (params) lines.push(`    params: ${params}`)
  }
  return lines.join('\n')
}

export function AssistantPage() {
  const { models, assistantConfigured, setPage, pendingParams, selectModel } = useApp()
  const [messages, setMessages] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [streaming, setStreaming] = useState('')
  const [error, setError] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const unsubRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [messages, streaming])

  useEffect(() => () => unsubRef.current?.(), [])

  const send = async (): Promise<void> => {
    const text = input.trim()
    if (!text || busy) return
    const history: Msg[] = [...messages, { role: 'user', content: text }]
    setMessages(history)
    setInput('')
    setBusy(true)
    setError(null)
    setStreaming('')

    // Subscribe to streamed deltas (they append to the in-flight assistant msg).
    unsubRef.current?.()
    unsubRef.current = window.platesim?.onLlmDelta((delta) => {
      setStreaming((s) => s + delta)
    }) ?? null

    try {
      const sys = { role: 'system' as const, content: buildSystemPrompt(models) }
      const apiMsgs = [
        sys,
        ...history.map((m) => ({ role: m.role, content: m.content }))
      ]
      const baseUrl = (await window.platesim?.getSettings())?.llm.baseUrl ?? ''
      const model = (await window.platesim?.getSettings())?.llm.model ?? ''
      const full = await window.platesim?.llmChat(apiMsgs, baseUrl, model)
      setMessages([...history, { role: 'assistant', content: full ?? '' }])
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      // keep partial stream visible
      if (streaming) setMessages([...history, { role: 'assistant', content: streaming }])
    } finally {
      unsubRef.current?.()
      unsubRef.current = null
      setStreaming('')
      setBusy(false)
    }
  }

  if (!assistantConfigured) {
    return (
      <div className="page">
        <div className="page-head">
          <h2>Assistant</h2>
        </div>
        <div className="card not-configured">
          <strong>not configured</strong>
          <p className="muted">
            Set the OpenAI-compatible base URL, model and API key (stored via Electron safeStorage)
            to enable the assistant. See <button type="button" className="linklike" onClick={() => setPage('settings')}>Settings</button>.
          </p>
        </div>
      </div>
    )
  }

  const lastProposal = messages.length
    ? extractProposal(messages[messages.length - 1].content)
    : null

  return (
    <div className="page assistant-page">
      <div className="page-head">
        <h2>Assistant</h2>
        <p className="muted">
          OpenAI-compatible chat (streaming). Proposals are JSON blocks you can load into a model
          form — nothing runs automatically.
        </p>
      </div>

      <div className="chat card" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="empty-note">
            Ask e.g. “Which model estimates current distribution on a coupon?” or “Propose a via fill
            sweep.” Models: {models.map((m) => m.id).join(', ') || 'none loaded'}
          </div>
        )}
        {messages.map((m, i) => {
          const proposal =
            m.role === 'assistant' && i === messages.length - 1 ? extractProposal(m.content) : null
          return (
            <div key={i} className={`msg msg-${m.role}`}>
              <div className="msg-role mono small">{m.role}</div>
              <div className="msg-body">{m.content}</div>
              {proposal && (
                <div className="proposal">
                  <span className="mono small">
                    proposed run → {proposal.model_id} ({Object.keys(proposal.params).length} params)
                  </span>
                  <button
                    type="button"
                    className="btn btn-small btn-primary"
                    onClick={() => {
                      selectModel(proposal.model_id)
                      // hand params to ModelPage via pendingParams
                      window.dispatchEvent(
                        new CustomEvent('platesim:load-proposal', { detail: proposal })
                      )
                    }}
                  >
                    Load into form
                  </button>
                  <span className="small muted">— review before running</span>
                </div>
              )}
            </div>
          )
        })}
        {streaming && (
          <div className="msg msg-assistant">
            <div className="msg-role mono small">assistant</div>
            <div className="msg-body">{streaming}</div>
          </div>
        )}
        {error && <div className="field-error">{error}</div>}
      </div>

      <div className="chat-input">
        <textarea
          rows={3}
          value={input}
          placeholder="Ask about the models, baths, or request a run proposal…"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              void send()
            }
          }}
        />
        <div className="row gap">
          <button type="button" className="btn btn-primary" onClick={() => void send()} disabled={busy}>
            {busy ? 'Streaming…' : 'Send'}
          </button>
          {busy && (
            <button
              type="button"
              className="btn"
              onClick={() => void window.platesim?.llmCancel()}
            >
              stop
            </button>
          )}
        </div>
      </div>
      {lastProposal && (
        <div className="small muted">Latest proposal can be loaded from the message above.</div>
      )}
    </div>
  )
}
