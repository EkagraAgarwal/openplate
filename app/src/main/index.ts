import { app, BrowserWindow, dialog, ipcMain } from 'electron'
import fs from 'node:fs'
import path from 'node:path'
import { BackendManager, type BackendState } from './backend'
import { LlmClient, type ChatMessage } from './llm'
import {
  clearApiKey,
  loadApiKey,
  loadSettings,
  saveSettings,
  storeApiKey,
  type AppSettings
} from './settings'

let mainWindow: BrowserWindow | null = null
const backend = new BackendManager()
const llm = new LlmClient()
let settings: AppSettings = loadSettings()

function broadcastLog(line: string): void {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('backend:log', line)
  }
}

function broadcastState(state: BackendState): void {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('backend:state', state)
  }
}

backend.onLog = broadcastLog
backend.onState = broadcastState

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    show: false,
    title: 'platesim',
    backgroundColor: '#f5f6f8',
    webPreferences: {
      preload: path.join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  })

  mainWindow.once('ready-to-show', () => mainWindow?.show())

  if (process.env.ELECTRON_RENDERER_URL) {
    void mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL)
  } else {
    void mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'))
  }

  mainWindow.on('closed', () => {
    mainWindow = null
  })
}

// ----------------------------------------------------------------- IPC -----

ipcMain.handle('settings:get', () => {
  const s = loadSettings()
  return { ...s, llm: { ...s.llm }, llmKeyStored: Boolean(loadApiKey()) }
})

ipcMain.handle('settings:save', (_e, patch: Partial<AppSettings>) => {
  settings = saveSettings(patch)
  return { ...settings, llm: { ...settings.llm }, llmKeyStored: Boolean(loadApiKey()) }
})

ipcMain.handle('backend:state', () => backend.state)
ipcMain.handle('backend:logs', () => backend.getLogs())
ipcMain.handle('backend:start', async () => {
  settings = loadSettings()
  return backend.start(settings)
})
ipcMain.handle('backend:restart', async () => {
  settings = loadSettings()
  return backend.restart()
})
ipcMain.handle('backend:stop', async () => {
  await backend.stop()
  return backend.state
})

ipcMain.handle('llm:save-key', (_e, key: string) => {
  if (key === '') {
    clearApiKey()
    return { ok: true, stored: false }
  }
  const ok = storeApiKey(key)
  return { ok, stored: ok }
})
ipcMain.handle('llm:key-stored', () => Boolean(loadApiKey()))
ipcMain.handle('llm:chat', async (e, messages: ChatMessage[], baseUrl: string, model: string) => {
  const win = BrowserWindow.fromWebContents(e.sender)
  if (!win) throw new Error('no window')
  return llm.chat(win, messages, baseUrl, model)
})
ipcMain.handle('llm:cancel', () => {
  llm.cancel()
  return true
})

ipcMain.handle('dialog:save-json', async (_e, data: unknown, suggestedName: string) => {
  const opts = {
    defaultPath: suggestedName,
    filters: [{ name: 'JSON', extensions: ['json'] }]
  }
  const { canceled, filePath } = mainWindow
    ? await dialog.showSaveDialog(mainWindow, opts)
    : await dialog.showSaveDialog(opts)
  if (canceled || !filePath) return null
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf8')
  return filePath
})

ipcMain.handle('dialog:save-png', async (_e, base64Data: string, suggestedName: string) => {
  const opts = {
    defaultPath: suggestedName,
    filters: [{ name: 'PNG', extensions: ['png'] }]
  }
  const { canceled, filePath } = mainWindow
    ? await dialog.showSaveDialog(mainWindow, opts)
    : await dialog.showSaveDialog(opts)
  if (canceled || !filePath) return null
  const b64 = base64Data.replace(/^data:image\/png;base64,/, '')
  fs.writeFileSync(filePath, Buffer.from(b64, 'base64'))
  return filePath
})

// -------------------------------------------------------------- lifecycle --

app.whenReady().then(async () => {
  createWindow()
  // Start backend after the window exists so splash/logs are visible.
  await backend.start(settings)
})

app.on('window-all-closed', () => {
  void backend.stop().finally(() => app.quit())
})

app.on('before-quit', () => {
  void backend.stop()
})
