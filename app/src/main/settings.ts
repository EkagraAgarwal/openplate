import { app, safeStorage } from 'electron'
import fs from 'node:fs'
import path from 'node:path'

export interface LlmSettings {
  baseUrl: string
  model: string
}

export interface AppSettings {
  /** 'spawn' = launch python backend; 'url' = connect to an already-running server. */
  backendMode: 'spawn' | 'url'
  /** Used when backendMode === 'url', e.g. http://127.0.0.1:8000 */
  externalUrl: string
  /** platesim project root (cwd for the spawned server). Empty = auto (parent of app/). */
  platesimRoot: string
  /** Python executable. */
  pythonBin: string
  llm: LlmSettings
}

const DEFAULTS: AppSettings = {
  backendMode: 'spawn',
  externalUrl: '',
  platesimRoot: '',
  pythonBin: 'python',
  llm: { baseUrl: '', model: '' }
}

function settingsPath(): string {
  return path.join(app.getPath('userData'), 'platesim-settings.json')
}

function keyPath(): string {
  return path.join(app.getPath('userData'), 'platesim-llm-key.bin')
}

/** Parent folder of app/ (dev) — env PLATESIM_ROOT wins over the settings file. */
export function defaultPlatesimRoot(): string {
  return path.resolve(app.getAppPath(), '..')
}

export function loadSettings(): AppSettings {
  let stored: Partial<AppSettings> = {}
  try {
    stored = JSON.parse(fs.readFileSync(settingsPath(), 'utf8')) as Partial<AppSettings>
  } catch {
    // first run or corrupt file -> defaults
  }
  const merged: AppSettings = {
    ...DEFAULTS,
    ...stored,
    llm: { ...DEFAULTS.llm, ...(stored.llm ?? {}) }
  }
  if (process.env.PLATESIM_ROOT) merged.platesimRoot = process.env.PLATESIM_ROOT
  if (!merged.platesimRoot) merged.platesimRoot = defaultPlatesimRoot()
  if (!merged.pythonBin) merged.pythonBin = 'python'
  return merged
}

export function saveSettings(patch: Partial<AppSettings>): AppSettings {
  const current = loadSettings()
  const next: AppSettings = {
    ...current,
    ...patch,
    llm: { ...current.llm, ...(patch.llm ?? {}) }
  }
  if (!next.platesimRoot) next.platesimRoot = defaultPlatesimRoot()
  try {
    fs.writeFileSync(settingsPath(), JSON.stringify(next, null, 2), 'utf8')
  } catch (err) {
    console.error('failed to persist settings', err)
  }
  return next
}

export function storeApiKey(plain: string): boolean {
  try {
    fs.writeFileSync(keyPath(), safeStorage.encryptString(plain))
    return true
  } catch (err) {
    console.error('safeStorage encrypt failed', err)
    return false
  }
}

export function loadApiKey(): string | null {
  try {
    return safeStorage.decryptString(fs.readFileSync(keyPath()))
  } catch {
    return null
  }
}

export function clearApiKey(): void {
  try {
    fs.unlinkSync(keyPath())
  } catch {
    // absent already
  }
}
