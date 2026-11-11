import { useEffect, useState } from 'react'
import { useApp } from '../lib/context'
import type { LiteratureEntry, PlanarDoePayload, SemPayload } from '../api/client'

/** Turn either {columns, rows} or a list of records into a uniform table. */
export function toTable(
  payload: PlanarDoePayload | SemPayload | null | undefined
): { columns: string[]; rows: unknown[][] } | null {
  if (!payload) return null
  if (Array.isArray(payload.columns) && Array.isArray(payload.rows)) {
    return { columns: payload.columns, rows: payload.rows as unknown[][] }
  }
  if (Array.isArray(payload.records)) {
    const records = payload.records as Record<string, unknown>[]
    if (records.length === 0) return { columns: [], rows: [] }
    const columns = [...new Set(records.flatMap((r) => Object.keys(r)))]
    return { columns, rows: records.map((r) => columns.map((c) => r[c])) }
  }
  // maybe the payload itself is an array-like object of records under another key
  const firstArray = Object.values(payload).find((v) => Array.isArray(v))
  if (Array.isArray(firstArray) && firstArray.length > 0 && typeof firstArray[0] === 'object' && firstArray[0] !== null) {
    const records = firstArray as Record<string, unknown>[]
    const columns = [...new Set(records.flatMap((r) => Object.keys(r)))]
    return { columns, rows: records.map((r) => columns.map((c) => r[c])) }
  }
  return null
}

function cell(v: unknown): string {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

function DataTable({ columns, rows }: { columns: string[]; rows: unknown[][] }) {
  if (columns.length === 0 || rows.length === 0) {
    return <div className="empty-note">Empty table.</div>
  }
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {r.map((v, j) => (
                <td key={j} className={typeof v === 'number' ? 'mono num' : ''}>
                  {cell(v)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

type Tab = 'doe' | 'literature' | 'sem'

export function DataPage() {
  const { api } = useApp()
  const [tab, setTab] = useState<Tab>('doe')
  const [doe, setDoe] = useState<PlanarDoePayload | null>(null)
  const [lit, setLit] = useState<LiteratureEntry[] | null>(null)
  const [sem, setSem] = useState<SemPayload | null>(null)
  const [missing, setMissing] = useState<Record<string, boolean>>({})
  const [openLit, setOpenLit] = useState<number | null>(null)

  useEffect(() => {
    if (!api) return
    let alive = true
    void (async () => {
      const [d, l, s] = await Promise.all([api.planarDoe(), api.literature(), api.sem()])
      if (!alive) return
      setDoe(d)
      setLit(l)
      setSem(s)
      setMissing({ doe: d === null, lit: l === null, sem: s === null })
    })()
    return () => {
      alive = false
    }
  }, [api])

  const tabs: [Tab, string][] = [
    ['doe', 'Planar DOE'],
    ['literature', 'Literature'],
    ['sem', 'SEM measurements']
  ]

  return (
    <div className="page">
      <div className="page-head">
        <h2>Data</h2>
        <p className="muted">Shared datasets exposed by the backend (<code>/api/data/*</code>).</p>
      </div>

      <div className="tab-bar">
        {tabs.map(([id, label]) => (
          <button
            key={id}
            type="button"
            className={`tab ${tab === id ? 'tab-active' : ''}`}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="card">
        {tab === 'doe' &&
          (missing.doe ? (
            <div className="empty-note">Planar DOE data not available on this server.</div>
          ) : (
            (() => {
              const t = toTable(doe)
              return t ? <DataTable columns={t.columns} rows={t.rows} /> : <div className="empty-note">Unrecognised payload shape.</div>
            })()
          ))}

        {tab === 'literature' &&
          (missing.lit || !lit || lit.length === 0 ? (
            <div className="empty-note">No literature datasets returned.</div>
          ) : (
            <ul className="lit-list">
              {lit.map((entry, i) => {
                const d = entry.data as { xlabel?: string; ylabel?: string; x?: unknown[]; y?: unknown[] } | undefined
                const points = d?.x?.length ?? 0
                return (
                  <li key={i} className="lit-item">
                    <button
                      type="button"
                      className="lit-head"
                      onClick={() => setOpenLit(openLit === i ? null : i)}
                    >
                      <span className="lit-name mono">{entry.name ?? `dataset ${i + 1}`}</span>
                      <span className="badge">{entry.tag ?? 'tag'}</span>
                      <span className="muted small">{points} pts</span>
                      <span className="grow" />
                      <span className="muted">{openLit === i ? '▾' : '▸'}</span>
                    </button>
                    <div className="lit-source small">{entry.source ?? entry.citation ?? ''}</div>
                    {openLit === i && d && (
                      <div className="lit-body">
                        <div className="small muted">
                          {d.xlabel ?? 'x'} vs {d.ylabel ?? 'y'}
                        </div>
                        <DataTable
                          columns={[d.xlabel ?? 'x', d.ylabel ?? 'y']}
                          rows={(d.x ?? []).map((xv, k) => [xv, (d.y ?? [])[k]])}
                        />
                      </div>
                    )}
                  </li>
                )
              })}
            </ul>
          ))}

        {tab === 'sem' &&
          (missing.sem ? (
            <div className="empty-note">SEM measurements not available on this server.</div>
          ) : (
            (() => {
              const t = toTable(sem)
              return t ? <DataTable columns={t.columns} rows={t.rows} /> : <div className="empty-note">Unrecognised payload shape.</div>
            })()
          ))}
      </div>
    </div>
  )
}
