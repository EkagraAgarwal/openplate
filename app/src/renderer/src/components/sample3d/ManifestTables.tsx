import type { Manifest3D } from '../../api/client3d'
export function ManifestTables({
  manifest,
  artifactUrl
}: {
  manifest: Manifest3D
  artifactUrl: (path: string) => string
}) {
  const value = (v: unknown) =>
    v == null
      ? '—'
      : typeof v === 'number'
        ? Number.isInteger(v)
          ? String(v)
          : Math.abs(v) >= 1e-3 && Math.abs(v) < 1e5
            ? String(Number(v.toPrecision(4)))
            : v.toExponential(3)
        : typeof v === 'object'
          ? JSON.stringify(v)
          : String(v)
  return (
    <section className="card">
      <h3>Manifest</h3>
      <p>{manifest.summary_text}</p>
      <div className="table-scroll">
        <table className="data-table">
          <caption>Quantities</caption>
          <thead>
            <tr>
              <th>Name</th>
              <th>Value</th>
              <th>Unit</th>
              <th>Feature ID</th>
            </tr>
          </thead>
          <tbody>
            {manifest.quantities.map((q, i) => (
              <tr key={i}>
                <td>{q.name}</td>
                <td>{value(q.value)}</td>
                <td>{q.unit}</td>
                <td>{q.feature_id ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="table-scroll">
        <table className="data-table">
          <caption>Checks</caption>
          <thead>
            <tr>
              <th>Name</th>
              <th>Value</th>
              <th>Limit</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody>
            {manifest.checks.map((c, i) => (
              <tr key={i}>
                <td>{c.name}</td>
                <td>{value(c.value)}</td>
                <td>{value(c.limit)}</td>
                <td className={c.passed ? 'sample3d-pass' : 'err-text'}>
                  {c.passed ? 'Pass' : 'Fail'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="sample3d-artifacts">
        {manifest.artifacts
          .filter((a) => a.kind === 'png')
          .map((a) => (
            <a key={a.path} href={artifactUrl(a.path)} target="_blank" rel="noreferrer">
              <img loading="lazy" src={artifactUrl(a.path)} alt={a.title} />
              <span>{a.title}</span>
            </a>
          ))}
      </div>
    </section>
  )
}
