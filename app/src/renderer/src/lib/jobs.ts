/**
 * Jobs live list: WebSocket /api/ws job events with 1 s polling fallback.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { PlatesimApi, wsUrl } from '../api/client'
import type { Job } from '../types'

export type JobsConnection = 'connecting' | 'ws' | 'polling' | 'offline'

function upsert(jobs: Job[], job: Job): Job[] {
  const i = jobs.findIndex((j) => j.id === job.id)
  if (i < 0) return [job, ...jobs].slice(0, 200)
  const next = [...jobs]
  next[i] = { ...next[i], ...job }
  return next
}

/** Accept several plausible WS message shapes; null if unrecognised. */
export function parseJobEvent(raw: unknown): Job | null {
  if (!raw || typeof raw !== 'object') return null
  const obj = raw as Record<string, unknown>
  const candidate =
    (obj.job as Record<string, unknown> | undefined) ??
    (obj.data as Record<string, unknown> | undefined) ??
    obj
  if (typeof candidate.id === 'string' && typeof candidate.status === 'string') {
    return candidate as unknown as Job
  }
  if (typeof candidate.job_id === 'string' && typeof candidate.status === 'string') {
    return { ...candidate, id: candidate.job_id } as unknown as Job
  }
  return null
}

export function useJobs(api: PlatesimApi | null, enabled: boolean) {
  const [jobs, setJobs] = useState<Job[]>([])
  const [connection, setConnection] = useState<JobsConnection>('connecting')
  const wsRef = useRef<WebSocket | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const wsWantedRef = useRef(true)

  const refresh = useCallback(async () => {
    if (!api) return
    try {
      const list = await api.jobs()
      if (list.length) setJobs((prev) => {
        // merge: keep fresher server values
        const byId = new Map(prev.map((j) => [j.id, j]))
        for (const j of list) byId.set(j.id, { ...(byId.get(j.id) ?? {}), ...j })
        // newest first (Map order would put newly seen jobs at the bottom)
        return [...byId.values()].sort((a, b) =>
          String((b as { created?: string }).created ?? '').localeCompare(String((a as { created?: string }).created ?? ''))
        )
      })
    } catch {
      setConnection((c) => (c === 'ws' ? c : 'offline'))
    }
  }, [api])

  const startPolling = useCallback(() => {
    if (pollRef.current) return
    setConnection((c) => (c === 'ws' ? c : 'polling'))
    pollRef.current = setInterval(() => void refresh(), 1000)
  }, [refresh])

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  // one-shot refresh when api becomes available
  useEffect(() => {
    if (api) void refresh()
  }, [api, refresh])

  // WebSocket lifecycle
  useEffect(() => {
    if (!api || !enabled) return
    wsWantedRef.current = true
    let retry: ReturnType<typeof setTimeout> | null = null
    let closedByEffect = false

    const connect = (): void => {
      if (!wsWantedRef.current || !api.baseUrl) return
      let ws: WebSocket
      try {
        ws = new WebSocket(wsUrl(api.baseUrl))
      } catch {
        startPolling()
        return
      }
      wsRef.current = ws
      setConnection('connecting')
      const fallback = setTimeout(() => {
        if (ws.readyState !== WebSocket.OPEN) startPolling()
      }, 3000)

      ws.onopen = () => {
        clearTimeout(fallback)
        setConnection('ws')
        stopPolling()
        void refresh()
      }
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(String(ev.data)) as unknown
          // messages may be a single job, an array, or {events: [...]}
          const items = Array.isArray(msg)
            ? msg
            : Array.isArray((msg as { events?: unknown[] })?.events)
              ? ((msg as { events: unknown[] }).events)
              : [msg]
          for (const item of items) {
            const job = parseJobEvent(item)
            if (job) setJobs((prev) => upsert(prev, job))
          }
          void refresh()
        } catch {
          void refresh()
        }
      }
      ws.onclose = () => {
        clearTimeout(fallback)
        wsRef.current = null
        if (closedByEffect || !wsWantedRef.current) return
        startPolling()
        retry = setTimeout(connect, 2000)
      }
      ws.onerror = () => {
        try {
          ws.close()
        } catch {
          /* noop */
        }
      }
    }

    connect()
    startPolling() // belt & braces until ws confirms

    return () => {
      closedByEffect = true
      wsWantedRef.current = false
      if (retry) clearTimeout(retry)
      stopPolling()
      try {
        wsRef.current?.close()
      } catch {
        /* noop */
      }
      wsRef.current = null
    }
  }, [api, enabled, refresh, startPolling, stopPolling])

  const patchJob = useCallback((id: string, patch: Partial<Job>) => {
    setJobs((prev) => prev.map((j) => (j.id === id ? { ...j, ...patch } : j)))
  }, [])

  return { jobs, connection, refresh, patchJob, setJobs }
}
