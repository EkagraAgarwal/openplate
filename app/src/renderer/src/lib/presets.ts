/** Named parameter presets per model, persisted in localStorage. */
import type { ParamValues } from '../types'

export interface Preset {
  name: string
  values: ParamValues
  savedAt: number
}

const KEY = 'platesim.presets.v1'

function read(): Record<string, Preset[]> {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw) as Record<string, Preset[]>
    return parsed && typeof parsed === 'object' ? parsed : {}
  } catch {
    return {}
  }
}

function write(data: Record<string, Preset[]>): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(data))
  } catch {
    /* quota */
  }
}

export function listPresets(modelId: string): Preset[] {
  return read()[modelId] ?? []
}

export function savePreset(modelId: string, name: string, values: ParamValues): Preset[] {
  const data = read()
  const list = data[modelId] ?? []
  const idx = list.findIndex((p) => p.name === name)
  const preset: Preset = { name, values: { ...values }, savedAt: Date.now() }
  if (idx >= 0) list[idx] = preset
  else list.push(preset)
  data[modelId] = list
  write(data)
  return list
}

export function deletePreset(modelId: string, name: string): Preset[] {
  const data = read()
  data[modelId] = (data[modelId] ?? []).filter((p) => p.name !== name)
  write(data)
  return data[modelId]
}
