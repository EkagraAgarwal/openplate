/**
 * OpenAI-compatible /chat/completions streaming client (used by the Assistant
 * page). The API key never leaves the main process: the renderer passes
 * messages, main injects the stored key, streams deltas back over IPC.
 */
import { BrowserWindow } from 'electron'
import { loadApiKey } from './settings'

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

function joinUrl(baseUrl: string, path: string): string {
  const trimmed = baseUrl.trim().replace(/\/+$/, '')
  if (trimmed.endsWith(path)) return trimmed
  return `${trimmed}${path}`
}

export class LlmClient {
  private controller: AbortController | null = null

  cancel(): void {
    this.controller?.abort()
    this.controller = null
  }

  /**
   * Streams a chat completion. Deltas are pushed to `win` as 'llm:delta' events;
   * resolves with the full concatenated text.
   */
  async chat(win: BrowserWindow, messages: ChatMessage[], baseUrl: string, model: string): Promise<string> {
    const key = loadApiKey()
    this.controller = new AbortController()
    const res = await fetch(joinUrl(baseUrl, '/chat/completions'), {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        ...(key ? { authorization: `Bearer ${key}` } : {})
      },
      body: JSON.stringify({ model, messages, stream: true }),
      signal: this.controller.signal
    })
    if (!res.ok || !res.body) {
      const text = await res.text().catch(() => '')
      throw new Error(`LLM HTTP ${res.status}: ${text.slice(0, 300)}`)
    }
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let full = ''
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const events = buffer.split('\n')
      buffer = events.pop() ?? ''
      for (const raw of events) {
        const line = raw.trim()
        if (!line.startsWith('data:')) continue
        const payload = line.slice(5).trim()
        if (!payload || payload === '[DONE]') continue
        try {
          const json = JSON.parse(payload) as {
            choices?: { delta?: { content?: string } }[]
          }
          const delta = json.choices?.[0]?.delta?.content
          if (delta) {
            full += delta
            if (!win.isDestroyed()) win.webContents.send('llm:delta', delta)
          }
        } catch {
          // ignore partial/keepalive frames
        }
      }
    }
    return full
  }
}
