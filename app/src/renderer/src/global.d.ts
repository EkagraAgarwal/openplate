/** Ambient types for the preload bridge (window.platesim). */

export interface BackendStateDto {
  status: 'idle' | 'starting' | 'ready' | 'error' | 'stopped'
  url: string
  upstreamUrl: string | null
  mode: 'spawn' | 'url' | 'mock'
  error: string | null
  pid: number | null
}

export interface SettingsDto {
  backendMode: 'spawn' | 'url'
  externalUrl: string
  platesimRoot: string
  pythonBin: string
  llm: { baseUrl: string; model: string }
  llmKeyStored?: boolean
}

export interface PlatesimBridge {
  getSettings(): Promise<SettingsDto>
  saveSettings(patch: Partial<SettingsDto>): Promise<SettingsDto>

  backendState(): Promise<BackendStateDto>
  backendLogs(): Promise<string[]>
  backendStart(): Promise<BackendStateDto>
  backendRestart(): Promise<BackendStateDto>
  backendStop(): Promise<BackendStateDto>
  onBackendLog(cb: (line: string) => void): () => void
  onBackendState(cb: (state: BackendStateDto) => void): () => void

  saveLlmKey(key: string): Promise<{ ok: boolean; stored: boolean }>
  llmKeyStored(): Promise<boolean>
  llmChat(
    messages: { role: string; content: string }[],
    baseUrl: string,
    model: string
  ): Promise<string>
  llmCancel(): Promise<boolean>
  onLlmDelta(cb: (delta: string) => void): () => void

  saveJson(data: unknown, suggestedName: string): Promise<string | null>
  savePng(base64: string, suggestedName: string): Promise<string | null>
}

declare global {
  interface Window {
    platesim?: PlatesimBridge
  }
}

export {}
