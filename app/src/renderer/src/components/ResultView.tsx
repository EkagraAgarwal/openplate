import type { Plot, RunResult } from '../types'
import { PlotRenderer } from './PlotRenderer'
import { useApp } from '../lib/context'
import Plotly from 'plotly.js-dist-min'

function summaryRows(summary: RunResult['summary']): [string, string][] {
  return Object.entries(summary ?? {}).map(([k, v]) => [
    k,
    v === null || v === undefined
      ? '—'
      : typeof v === 'object'
        ? JSON.stringify(v)
        : String(v)
  ])
}

async function exportPlotPng(plot: Plot, modelName: string): Promise<void> {
  if (!window.platesim) return
  // Use the figure that contains the clicked button.
  const btn = document.activeElement as HTMLElement | null
  const figure = btn?.closest('.plot-card') ?? null
  const plotDiv = figure?.querySelector('.plot-div') as HTMLElement | null
  if (!plotDiv) return
  try {
    const dataUrl = await Plotly.toImage(plotDiv, {
      format: 'png',
      width: plotDiv.clientWidth || 800,
      height: plotDiv.clientHeight || 400,
      scale: 2
    })
    const title = (plot as { title?: string }).title ?? 'plot'
    const safe = `${modelName}-${title}`.replace(/[^\w.-]+/g, '_').slice(0, 80)
    await window.platesim.savePng(dataUrl, `${safe}.png`)
  } catch (err) {
    console.error('PNG export failed', err)
  }
}

function PlotCard({ plot, mirror, modelName }: { plot: Plot; mirror: boolean; modelName: string }) {
  return (
    <figure className="plot-card">
      <PlotRenderer plot={plot} mirror={mirror} />
      <figcaption className="plot-caption">
        <button
          type="button"
          className="btn btn-small"
          onClick={() => void exportPlotPng(plot, modelName)}
          title="Export this plot as PNG"
        >
          PNG
        </button>
      </figcaption>
    </figure>
  )
}

export interface ResultViewProps {
  label: string
  result: RunResult
  modelName?: string
  onBack?: () => void
}

export function ResultView({ label, result, modelName = 'model', onBack }: ResultViewProps) {
  const { api } = useApp()
  const plots = result.plots ?? []
  const warnings = result.warnings ?? []
  const assumptions = result.assumptions ?? []
  // Feature-fill models return r >= 0 cross-sections; mirror to draw full via.
  const mirror = /feature/i.test(modelName)

  const exportJson = async (): Promise<void> => {
    if (!window.platesim) return
    const safe = `${modelName}-result`.replace(/[^\w.-]+/g, '_').slice(0, 80)
    await window.platesim.saveJson({ label, model: modelName, ...result }, `${safe}.json`)
  }
  void api

  return (
    <div className="result-view">
      <div className="result-header">
        <div>
          <h3>{label}</h3>
          <div className="muted small">model: {modelName}</div>
        </div>
        <div className="row gap">
          <button type="button" className="btn" onClick={() => void exportJson()}>
            Export JSON
          </button>
          {onBack && (
            <button type="button" className="btn btn-ghost" onClick={onBack}>
              ← Back
            </button>
          )}
        </div>
      </div>

      <section className="card">
        <h4>Summary</h4>
        {Object.keys(result.summary ?? {}).length === 0 ? (
          <div className="empty-note">No summary returned.</div>
        ) : (
          <table className="data-table">
            <tbody>
              {summaryRows(result.summary).map(([k, v]) => (
                <tr key={k}>
                  <th>{k}</th>
                  <td className="mono">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {warnings.length > 0 && (
        <section className="callout callout-warn">
          <strong>Warnings</strong>
          <ul>
            {warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </section>
      )}

      {assumptions.length > 0 && (
        <section className="callout callout-info">
          <strong>Assumptions</strong>
          <ul>
            {assumptions.map((a, i) => (
              <li key={i}>{a}</li>
            ))}
          </ul>
        </section>
      )}

      {plots.length > 0 && (
        <section>
          <h4>Plots</h4>
          <div className="plot-grid">
            {plots.map((p, i) => (
              <PlotCard key={i} plot={p} mirror={mirror} modelName={modelName} />
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
