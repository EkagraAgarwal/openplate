import { useState } from 'react'
import type { FieldIssue, Plan3D, SampleSpec3D } from '../../api/client3d'
import { defaultSpec3D, parseSpec, recipeMath, specToForm } from '../../lib/sample3d'
import { LayoutPreview } from './LayoutPreview'

/** Spec fields the L1 sample pipeline does not read yet (kept in the spec for later physics levels). */
const L1_UNUSED = {
  bath: 'Not used by L1 yet: cl_ppm (no additive/chloride chemistry). Cu and H2SO4 set the bulk Cu2+, H+ and SO4 2- (migration); conductivity is an uncalibrated default.',
  setup:
    'Not used by L1 yet: orientation, up_edge (no buoyancy), taped, area_cm2. Optional coupon position (centred, x across, y up) selects local current, Nernst layer and efficiency from the calibrated DC coupon model; beaker stir (%) is used only by that model (blank uses its default). Without a position, delta_um sets the bulk plane. stir_shear_1_s (if set, EXPERIMENTAL) adds resolved shear flow with the lid at the bulk plane. stir_angle_deg sets the shear direction in die axes. Sheared runs use full periodic tiles and are ~20x slower (quick_trench: 24 min instead of ~1 min).'
}

const FIELD_LABELS: Record<string, string> = {
  'layout.die_um.0': 'die width x (µm)',
  'layout.die_um.1': 'die height y (µm)',
  'setup.coupon_position_cm': 'coupon position x, y (cm)',
  'setup.stir_pct': 'beaker stir (%)',
  'numerics.pulse_model': 'pulse model',
  'numerics.pulse_C_dl_uF_cm2': 'double layer (uF/cm2, resolved only)'
}

export function SpecEditor({
  spec,
  onChange,
  errors,
  onValid,
  estimateText = 'Estimate unavailable',
  tiles = []
}: {
  spec: SampleSpec3D
  onChange: (s: SampleSpec3D) => void
  errors: FieldIssue[]
  onValid: (valid: boolean) => void
  estimateText?: string
  tiles?: Plan3D['tiles']
}) {
  const [tab, setTab] = useState('form')
  const [raw, setRaw] = useState('')
  const [jsonError, setJsonError] = useState('')
  const update = (path: string, value: unknown): void => {
    const next = specToForm(spec)
    const keys = path.split('.')
    let obj = next as unknown as Record<string, unknown>
    for (const key of keys.slice(0, -1)) {
      if (obj[key] == null) obj[key] = {}
      obj = obj[key] as Record<string, unknown>
    }
    if (value === undefined) delete obj[keys[keys.length - 1]]
    else obj[keys[keys.length - 1]] = value
    onChange(next)
  }
  const field = (path: string, value: unknown, choices?: string[], text = false) => {
    const issue = errors.find((e) => {
      const name = e.field.replace(/^spec\./, '').replace(/\[(\d+)\]/g, '.$1')
      return name === path || path.startsWith(`${name}.`)
    })
    return (
      <label key={path} className={`sample3d-field ${issue ? 'sample3d-invalid' : ''}`}>
        <span>{FIELD_LABELS[path] ?? path.split('.').slice(-1)[0]}</span>
        {choices ? (
          <select
            aria-label={path}
            aria-invalid={Boolean(issue)}
            value={String(value ?? choices[0])}
            onChange={(e) => update(path, e.target.value)}
          >
            {choices.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        ) : path === 'setup.coupon_position_cm' ? (
          <input
            key={String(value)}
            aria-label={path}
            aria-invalid={Boolean(issue)}
            type="text"
            placeholder="x, y (blank disables)"
            defaultValue={value == null ? '' : String(value)}
            onBlur={(e) => update(path, e.target.value.trim() === '' ? null :
              e.target.value.split(',').map((v) => v.trim() === '' ? null : Number(v)))}
          />
        ) : (
          <input
            aria-label={path}
            aria-invalid={Boolean(issue)}
            type={text ? 'text' : 'number'}
            step="any"
            value={value == null ? '' : String(value)}
            onChange={(e) =>
              update(
                path,
                text ? e.target.value : e.target.value === '' ? null : Number(e.target.value)
              )
            }
          />
        )}
        {issue && (
          <span className="field-error">
            {issue.message} {issue.fix}
          </span>
        )}
      </label>
    )
  }
  const fields = (prefix: string, obj: Record<string, unknown>, names: string[]) =>
    names.map((key) => field(`${prefix}.${key}`, obj[key]))
  return (
    <section className="card">
      <div className="row gap">
        <h3>Sample specification</h3>
        <p className="muted small" role="status">{estimateText}. Planning only; runtime varies by hardware and physics.</p>
        {['form', 'JSON'].map((t) => (
          <button
            type="button"
            className={`btn ${tab === t ? 'btn-primary' : ''}`}
            key={t}
            onClick={() => {
              if (t === 'JSON') setRaw(JSON.stringify(spec, null, 2))
              setTab(t)
            }}
            disabled={t === 'form' && Boolean(jsonError)}
          >
            {t}
          </button>
        ))}
      </div>
      {tab === 'JSON' ? (
        <>
          <textarea
            className="sample3d-json mono"
            aria-label="Sample JSON"
            value={raw}
            onChange={(e) => {
              setRaw(e.target.value)
              try {
                onChange(parseSpec(e.target.value))
                setJsonError('')
                onValid(true)
              } catch {
                onValid(false)
                setJsonError(
                  'Invalid spec JSON. Supply layout, etch, recipe, bath, setup and numerics; fix JSON syntax before using the form.'
                )
              }
            }}
          />
          <div className="field-error" role="alert">
            {jsonError}
          </div>
        </>
      ) : (
        <>
          <fieldset>
            <legend>Layout (µm)</legend>
            <div className="sample3d-fields">
              {field('layout.die_um.0', spec.layout.die_um[0])}
              {field('layout.die_um.1', spec.layout.die_um[1])}
            </div>
            {(spec.layout.features ?? []).map((feature, i) => (
              <div className="sample3d-subcard" key={i}>
                <div className="row gap">
                  <strong>Feature {i + 1}</strong>
                  <button
                    type="button"
                    className="btn btn-small"
                    onClick={() =>
                      update(
                        'layout.features',
                        spec.layout.features?.filter((_, j) => j !== i)
                      )
                    }
                  >
                    Remove
                  </button>
                </div>
                <div className="sample3d-fields">
                  {field(`layout.features.${i}.kind`, feature.kind, [
                    'rect',
                    'rounded_rect',
                    'circle'
                  ])}
                  {fields(`layout.features.${i}`, feature, [
                    'x_um',
                    'y_um',
                    'w_um',
                    'h_um',
                    'r_um',
                    'rotation_deg'
                  ])}
                </div>
                <label>
                  <input
                    type="checkbox"
                    checked={Boolean(feature.array)}
                    onChange={(e) =>
                      update(
                        `layout.features.${i}.array`,
                        e.target.checked ? { nx: 2, ny: 1, pitch_x_um: 40, pitch_y_um: 40 } : null
                      )
                    }
                  />{' '}
                  Array
                </label>
                {feature.array && (
                  <div className="sample3d-fields">
                    {fields(`layout.features.${i}.array`, feature.array, [
                      'nx',
                      'ny',
                      'pitch_x_um',
                      'pitch_y_um'
                    ])}
                  </div>
                )}
              </div>
            ))}
            <button
              type="button"
              className="btn"
              onClick={() =>
                update('layout.features', [
                  ...(spec.layout.features ?? []),
                  { ...defaultSpec3D.layout.features![0] }
                ])
              }
            >
              Add feature
            </button>
          </fieldset>
          <fieldset>
            <legend>Etch default</legend>
            <div className="sample3d-fields">
              {fields('etch.default', spec.etch.default, ['depth_um', 'sidewall_deg'])}
            </div>
          </fieldset>
          <fieldset>
            <legend>Recipe steps</legend>
            <div className="table-scroll">
              <table className="data-table sample3d-recipe">
                <thead>
                  <tr>
                    {[
                      'Forward ms',
                      'Reverse ms',
                      'Off ms',
                      'Forward mA/cm²',
                      'Reverse mA/cm²',
                      'Stop by',
                      'Value',
                      'Net mA/cm²',
                      'Duty',
                      ''
                    ].map((s, i) => (
                      <th key={i}>{s}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {spec.recipe.steps.map((step, i) => {
                    const math = recipeMath(step),
                      charge = step.charge_C_cm2 != null
                    return (
                      <tr key={i}>
                        {(
                          [
                            't_forward_ms',
                            't_reverse_ms',
                            't_off_ms',
                            'j_forward_mA_cm2',
                            'j_reverse_mA_cm2'
                          ] as const
                        ).map((key) => (
                          <td key={key}>{field(`recipe.steps.${i}.${key}`, step[key])}</td>
                        ))}
                        <td>
                          <select
                            aria-label={`Step ${i + 1} stop by`}
                            value={charge ? 'charge' : 'duration'}
                            onChange={(e) => {
                              const steps = spec.recipe.steps.map((s, j) =>
                                j !== i
                                  ? s
                                  : e.target.value === 'charge'
                                    ? { ...s, duration_h: undefined, charge_C_cm2: 1 }
                                    : { ...s, charge_C_cm2: undefined, duration_h: 0.001 }
                              )
                              update('recipe.steps', steps)
                            }}
                          >
                            <option value="duration">Duration (h)</option>
                            <option value="charge">Charge (C/cm²)</option>
                          </select>
                        </td>
                        <td>
                          {field(
                            `recipe.steps.${i}.${charge ? 'charge_C_cm2' : 'duration_h'}`,
                            charge ? step.charge_C_cm2 : step.duration_h
                          )}
                        </td>
                        <td>{math.net.toFixed(3)}</td>
                        <td>{(100 * math.duty).toFixed(1)}%</td>
                        <td>
                          <button
                            type="button"
                            className="btn btn-small"
                            onClick={() =>
                              update(
                                'recipe.steps',
                                spec.recipe.steps.filter((_, j) => i !== j)
                              )
                            }
                          >
                            Remove
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <button
              type="button"
              className="btn"
              onClick={() =>
                update('recipe.steps', [...spec.recipe.steps, { ...defaultSpec3D.recipe.steps[0] }])
              }
            >
              Add step
            </button>
          </fieldset>
          <fieldset>
            <legend>Bath</legend>
            <div className="sample3d-fields">
              {field('bath.name', spec.bath.name, undefined, true)}
              {fields('bath', spec.bath, ['cu_g_L', 'h2so4_g_L', 'cl_ppm', 'temperature_C'])}
            </div>
            <p className="muted">{L1_UNUSED.bath}</p>
          </fieldset>
          <fieldset>
            <legend>Setup</legend>
            <div className="sample3d-fields">
              {field('setup.orientation', spec.setup.orientation, [
                'vertical',
                'face_up',
                'face_down'
              ])}
              {field('setup.up_edge', spec.setup.up_edge, ['+x', '-x', '+y', '-y'])}
              {fields('setup', spec.setup, [
                'stir_shear_1_s',
                'stir_angle_deg',
                'coupon_position_cm',
                'stir_pct',
                'delta_um',
                'area_cm2'
              ])}
            </div>
            <label>
              <input
                type="checkbox"
                checked={Boolean(spec.setup.taped)}
                onChange={(e) => update('setup.taped', e.target.checked)}
              />{' '}
              Taped
            </label>
            <p className="muted">{L1_UNUSED.setup}</p>
          </fieldset>
          <fieldset>
            <legend>Numerics</legend>
            <div className="sample3d-fields">
              {field('numerics.quality', spec.numerics.quality ?? 'preview', [
                'preview',
                'standard',
                'fine'
              ])}
              {field('numerics.dx_um', spec.numerics.dx_um)}
              {field('numerics.pulse_model', (spec.numerics.pulse_model as string | undefined) ?? 'net', [
                'net',
                'resolved'
              ])}
              {field('numerics.pulse_C_dl_uF_cm2', (spec.numerics.pulse_C_dl_uF_cm2 as number | undefined) ?? 30)}
            </div>
            <p className="muted">
              Pulse model: net uses the cycle-average current; resolved solves forward, reverse and off phases
              each cycle (about 3x slower), with double-layer charging set by the capacitance below
              (0 turns it off). Without additives the two differ by ~0.01 in floor/field at 30 uF/cm2.
            </p>
          </fieldset>
        </>
      )}
      <LayoutPreview layout={spec.layout} tiles={tiles} />
    </section>
  )
}
