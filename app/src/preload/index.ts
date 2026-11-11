import { contextBridge, ipcRenderer, type IpcRendererEvent } from 'electron'

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

const api = {
  getSettings: (): Promise<SettingsDto> => ipcRenderer.invoke('settings:get'),
  saveSettings: (patch: Partial<SettingsDto>): Promise<SettingsDto> =>
    ipcRenderer.invoke('settings:save', patch),

  backendState: (): Promise<BackendStateDto> => ipcRenderer.invoke('backend:state'),
  backendLogs: (): Promise<string[]> => ipcRenderer.invoke('backend:logs'),
  backendStart: (): Promise<BackendStateDto> => ipcRenderer.invoke('backend:start'),
  backendRestart: (): Promise<BackendStateDto> => ipcRenderer.invoke('backend:restart'),
  backendStop: (): Promise<BackendStateDto> => ipcRenderer.invoke('backend:stop'),
  onBackendLog: (cb: (line: string) => void): (() => void) => {
    const h = (_e: IpcRendererEvent, line: string): void => cb(line)
    ipcRenderer.on('backend:log', h)
    return () => ipcRenderer.removeListener('backend:log', h)
  },
  onBackendState: (cb: (state: BackendStateDto) => void): (() => void) => {
    const h = (_e: IpcRendererEvent, state: BackendStateDto): void => cb(state)
    ipcRenderer.on('backend:state', h)
    return () => ipcRenderer.removeListener('backend:state', h)
  },

  saveLlmKey: (key: string): Promise<{ ok: boolean; stored: boolean }> =>
    ipcRenderer.invoke('llm:save-key', key),
  llmKeyStored: (): Promise<boolean> => ipcRenderer.invoke('llm:key-stored'),
  llmChat: (
    messages: { role: string; content: string }[],
    baseUrl: string,
    model: string
  ): Promise<string> => ipcRenderer.invoke('llm:chat', messages, baseUrl, model),
  llmCancel: (): Promise<boolean> => ipcRenderer.invoke('llm:cancel'),
  onLlmDelta: (cb: (delta: string) => void): (() => void) => {
    const h = (_e: IpcRendererEvent, delta: string): void => cb(delta)
    ipcRenderer.on('llm:delta', h)
    return () => ipcRenderer.removeListener('llm:delta', h)
  },

  saveJson: (data: unknown, suggestedName: string): Promise<string | null> =>
    ipcRenderer.invoke('dialog:save-json', data, suggestedName),
  savePng: (base64: string, suggestedName: string): Promise<string | null> =>
    ipcRenderer.invoke('dialog:save-png', base64, suggestedName)
}

export type PlatesimApi = typeof api

contextBridge.exposeInMainWorld('platesim', api)
