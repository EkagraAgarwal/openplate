import { useCallback, useEffect, useRef, useState } from 'react'
import type { Job } from '../types'
import type { PlatesimApi } from '../api/client'
import type { JobsConnection } from '../lib/jobs'

const ACTIVE = new Set(['queued', 'pending', 'running', 'starting', 'in_progress', 'active'])

export function isActive(job: Job): boolean {
  return ACTIVE.has(String(job.status ?? '').toLowerCase())
}

export function isDone(job: Job): boolean {
  // platesim.server reports "done"; the mock server uses "succeeded"
  return ['done', 'succeeded', 'completed', 'success'].includes(String(job.status ?? '').toLowerCase())
}

export function isFailed(job: Job): boolean {
  const s = String(job.status ?? '').toLowerCase()
  return s === 'failed' || s === 'error'
}

export interface JobsPanelProps {
  api: PlatesimApi | null
  jobs: Job[]
  connection: JobsConnection
  refresh: () => void
  onOpenResult: (jobId: string) => void
  compareIds: string[]
  onToggleCompare: (id: string) => void
  maxCompare?: number
}

export function JobsPanel({
  api,
  jobs,
  connection,
  refresh,
  onOpenResult,
  compareIds,
  onToggleCompare,
  maxCompare = 4
}: JobsPanelProps) {
  const [busyId, setBusyId] = useState<string | null>(null)
  const seenRunning = useRef<Set<string>>(new Set())

  // auto-refresh when a job transitions to a terminal state
  useEffect(() => {
    for (const j of jobs) {
      if (isActive(j)) seenRunning.current.add(j.id)
      else if (seenRunning.current.has(j.id)) {
        seenRunning.current.delete(j.id)
        refresh()
      }
    }
  }, [jobs, refresh])

  const cancel = useCallback(
    async (id: string) => {
      if (!api) return
      setBusyId(id)
      try {
        await api.cancel(id)
        await refresh()
      } catch (err) {
        console.error('cancel failed', err)
      } finally {
        setBusyId(null)
      }
    },
    [api, refresh]
  )

  const connLabel: Record<JobsConnection, string> = {
    connecting: 'connecting…',
    ws: 'live (websocket)',
    polling: 'polling 1 s',
    offline: 'offline'
  }

  return (
    <div className="jobs-panel card">
      <div className="jobs-header">
        <h4>Jobs</h4>
        <span className={`conn conn-${connection} mono small`}>{connLabel[connection]}</span>
        <button type="button" className="btn btn-small btn-ghost" onClick={refresh}>
          ↻
        </button>
      </div>
      {jobs.length === 0 ? (
        <div className="empty-note">No jobs yet. Run a model to see progress here.</div>
      ) : (
        <ul className="job-list">
          {jobs.map((job) => {
            const pct = typeof job.progress === 'number' ? Math.round(job.progress * 100) : null
            const active = isActive(job)
            const done = isDone(job)
            const failed = isFailed(job)
            const canCompare = done && !failed
            const compareOn = compareIds.includes(job.id)
            return (
              <li key={job.id} className={`job job-${failed ? 'failed' : done ? 'done' : 'active'}`}>
                <div className="job-top">
                  <span className="mono job-id">{job.id.slice(0, 12)}</span>
                  <span className={`badge badge-${failed ? 'error' : done ? 'ok' : 'run'}`}>
                    {job.status}
                  </span>
                  <span className="job-model small muted">{job.model_id ?? ''}</span>
                </div>
                {active && (
                  <div className="progress">
                    <div
                      className="progress-bar"
                      style={{ width: pct !== null ? `${pct}%` : '100%' }}
                    />
                  </div>
                )}
                <div className="job-meta small muted">
                  {pct !== null && <span>{pct}% · </span>}
                  {typeof job.elapsed_s === 'number' && <span>{job.elapsed_s.toFixed(1)} s · </span>}
                  {job.message && <span>{job.message}</span>}
                  {job.error && <span className="err-text">{job.error}</span>}
                </div>
                <div className="job-actions">
                  {active && (
                    <button
                      type="button"
                      className="btn btn-small"
                      disabled={busyId === job.id}
                      onClick={() => void cancel(job.id)}
                    >
                      cancel
                    </button>
                  )}
                  {canCompare && (
                    <button
                      type="button"
                      className="btn btn-small"
                      disabled={!compareOn && compareIds.length >= maxCompare}
                      onClick={() => onToggleCompare(job.id)}
                      title={compareOn ? 'Remove from comparison' : `Compare (max ${maxCompare})`}
                    >
                      {compareOn ? '✓ compare' : 'compare'}
                    </button>
                  )}
                  <button
                    type="button"
                    className="btn btn-small btn-primary"
                    disabled={!canCompare}
                    onClick={() => onOpenResult(job.id)}
                  >
                    open result
                  </button>
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
