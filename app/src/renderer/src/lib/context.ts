import { createContext, useContext } from 'react'
import type { PlatesimApi } from '../api/client'
import type { ModelSpec, ParamValues, RunResult } from '../types'

export type Page = 'models' | 'data' | 'assistant' | 'settings'

export interface BackendDto {
  status: 'idle' | 'starting' | 'ready' | 'error' | 'stopped'
  url: string
  upstreamUrl: string | null
  mode: 'spawn' | 'url' | 'mock'
  error: string | null
  pid: number | null
}

export interface CompareEntry {
  key: string
  label: string
  result: RunResult
}

export interface AppState {
  page: Page
  setPage: (p: Page) => void
  theme: 'light' | 'dark'
  toggleTheme: () => void
  backend: BackendDto | null
  backendLogs: string[]
  api: PlatesimApi | null
  restartBackend: () => void

  models: ModelSpec[]
  selectedModelId: string | null
  selectModel: (id: string | null) => void

  /** Params handed over from the Assistant proposal; consumed by ModelPage. */
  pendingParams: { modelId: string; params: ParamValues } | null
  clearPendingParams: () => void

  compare: CompareEntry[]
  toggleCompare: (entry: CompareEntry) => void
  clearCompare: () => void
  showCompare: boolean
  setShowCompare: (v: boolean) => void

  /** Open a result overlay (job result or sweep point). */
  openResult: (label: string, result: RunResult) => void
  closeResult: () => void
  activeResult: { label: string; result: RunResult } | null

  assistantConfigured: boolean
  refreshAssistantConfig: () => void
}

export const AppContext = createContext<AppState | null>(null)

export function useApp(): AppState {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('AppContext missing')
  return ctx
}
