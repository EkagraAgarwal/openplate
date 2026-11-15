# L1 representative-feature pipeline

`python -m platesim.sample3d plan spec.json --quality preview` prints a **list**
of tiles without running a solver. `run spec.json --out NEW_DIRECTORY` runs
every tile sequentially. `--quality` overrides the spec; `--field-margin-um`
overrides the isolated-feature field margin. The Python equivalents are
`plan_tile(spec, field_margin_um=None)` and
`run_sample(spec, out_dir, progress=None, field_margin_um=None)` (keyword options).
Existing output directories are refused.

## Tiling rules and assumptions

- A `FeatureArray`, or a complete Cartesian lattice of identical explicit
  features, uses one pitch cell. Mirror-symmetric features use a quarter cell,
  with no-flux symmetry on all four sides (centre planes and half-pitch planes).
  Other rotations use a full cell with periodic XY boundaries. Explicit lattice
  inference requires at least two coordinates on each axis; one row cannot
  establish a transverse pitch. Missing members are never filled in implicitly.
  Edge-of-array effects are **not modelled**. Weight is the number of physical
  array members, not four times that number.
- Isolated trenches with length/width > 4 use two tiles: `trench_mid`, the
   **profile** result, has half width plus field margin, eight cells in y and
  periodic y boundaries. It models a translationally invariant middle section.
  `trench_end` extends from one width inside the end to one margin beyond it,
  with a no-flux symmetry approximation at the interior cut. Its weight is 2
  (the two identical ends); the mid tile has weight 1. These are overlapping
  local diagnostics, **not a decomposition whose volumes can be summed**.
- Compact circles, squares and short rectangles use a quarter feature plus
  field margin on both positive sides. Centre planes and far field sides are
  symmetry/no-flux. Features are solved independently in an infinite field;
  die-edge and inter-feature interactions are not solved. A neighbour whose
  polygon gap is less than twice the margin triggers the explicit
  `isolated feature in an infinite field: neighbour interaction neglected` flag.
- Default margin is `max(3 * depth, 2 * delta_um, 100)` um, with the maximum
  depth in a depth-vs-width table used conservatively. Each feature uses its own
  layer's etch. Isolated domains are in feature-local coordinates, with the
  narrow axis x and the long axis y; the source centre and rotation assumption
   are recorded. Resolved shear directions are transformed from die axes into
   these local axes (including any narrow/long-axis swap).
  Imported file layouts currently require conversion to parametric features;
  they are rejected explicitly rather than silently treated as isolated shapes.

Preview/standard/fine request 8/4/2 um, overridden by `numerics.dx_um`.
Every axis has a multiple of eight cells for multigrid coarsening. Isolated XY
extents grow only on the positive field-margin side, never moving centre or
interior cut planes; z grows upward without moving the bottom. Array spacing
is adjusted down to divide both half-pitches (or full pitches) into multiples
of eight cells; pitches are never stretched.
Preview alone may coarsen a tile in 1.25x requested-spacing steps until it fits
the 200,000-cell cap. Such quantities are flagged `preview`; all preview-quality
quantities carry that flag, otherwise `uncalibrated`. Underresolved pitches or
standard/fine tiles over the cap raise an actionable error. Thus the shipped
standard-quality single-trench/coupon specs need `--quality preview` or an
explicit coarser spacing: silent standard-quality coarsening is not permitted.
`est_seconds` remains an uncalibrated CPU budgeting heuristic.

## Outputs and interpretation

The root contains one validated `spec3d.Manifest` and a `tiles.json` artifact
containing the complete plans: tile IDs, feature IDs, side boundaries, weights,
flags, assumptions, local windows and actual spacings. The strict Manifest has
no tile-metadata field, so these are also included in `summary_text`, with
`tile_weight` quantities. Its `dx_um` is the largest tile spacing; exact
per-tile values are in `tiles.json` and each grid. Feature IDs are stable input
indices (`feature_0000`); an explicit array's ID represents that feature family.
Recognised explicit lattices retain every input feature ID in their plan and
emit the representative quantities for each input ID. Do not weight those
duplicated quantities again. End diagnostic quantity names have a `trench_end.`
prefix; the unqualified quantities are the profile result.
In particular, each feature has `floor_over_field` (dimensionless floor/field
thickness ratio) for its profile tile and `trench_end.floor_over_field` for
its end tile. Other end names include `trench_end.field_um`,
`trench_end.floor_um`, `trench_end.wall_um`, `trench_end.fill_fraction` and
`trench_end.deposited_cu_volume_um3`; the same prefix applies to all end
quantities. Undefined measurements (including ratios with zero field growth)
are omitted rather than fabricated.

Each `tiles/<tile_id>/` is a readable TileResult with original bare silicon,
step-end frames, cumulative times/charge/volumes and three PNGs (`surface`,
`height_map`, `cross_section`). The writer's empty metadata `manifest.json`
exists solely for TileResult compatibility; the validated sample Manifest is
at the root. Artifact paths are relative to that root. Centre-cut symmetry is
reflected **before analysis and rendering**, producing the full feature/full
pitch view, not a misleading quarter image. The end's interior approximation
cut is not reflected. Stored simulation frames remain the reduced domain.
`deposited_cu_volume_um3` and Faraday checks use the simulated sector volume;
void volume refers to the reflected diagnostic view. Mid-trench sections show
only the thin periodic strip, not a fabricated reconstruction of the full length.
End cross-sections span the full reflected x width, half a feature width inside
the physical end cap; they do not reflect the artificial interior y cut.
Surface PNGs show only the final electrolyte interface by default; bare Si is
an optional semi-transparent overlay (`render_surface(..., show_si=True)`).
Axes and geometry are in micrometres (`um` on VTK axes to avoid missing glyphs).

Each recipe step receives the carried front and original field mask; times,
charge and volume accumulate in a separate writer. Solver calls use no run
directory. Cu balance is the maximum absolute saved balance across steps;
Faraday error reports actual drift without volume correction. Check names
are tile-qualified. Missing thickness measurements are omitted, not zero.

The etcher consumes dicts and returns cm; the pipeline preserves local origins,
re-distances the union and extends symmetry/periodic masks before etching so
width-dependent depths see the full feature. Circle radius is derived from
diameter. Rendering metadata is filtered to the strict Artifact schema.
Only CPU L1 without a surrogate is supported. Net current, temperature, bulk
Cu and fully dissociated acid migration enhancement are mapped to the engine;
pulse timing beyond its cycle average, chloride/additives, seed thinning,
orientation and up_edge (buoyancy) are not modelled. Conductivity and kinetics
remain uncalibrated engine defaults.

## Optional resolved shear

`setup.stir_shear_1_s=None` preserves the no-flow solver path (no `flow` argument).
When set, it supplies Couette shear in 1/s; `stir_angle_deg` is measured in die
axes, not tile axes. The moving lid remains at the bulk plane set by `delta_um`;
flow updates every five solver steps, with no buoyancy and the solver's default
axial-flux constraint. Plans and manifest summaries record the shear assumption.

Mirrored tiles have no-flux sides that shear cannot cross, so a sheared spec is
planned with full tiles periodic in x and y instead: `trench_mid_periodic`
(full width, periodic image of the trench every 2*(w/2 + margin)),
`compact_periodic`, and full `array_periodic` unit cells. The trench-end
diagnostic is omitted under shear. Each plan carries `rotation_from_die_deg`,
used to turn the die-axis shear direction into tile axes. Full tiles are 2-4x
larger than mirrored ones and add Stokes solves, so sheared runs are several
times slower. A shear component normal to a remaining no-flux side is still
rejected before any tile runs. `stir_shear_1_s = 0` is the same as None.

**Experimental:** the moving lid sits at the bulk plane (delta_um above the
field). In the T2.5 comparison that convention differed from the 2-D engine's
domain-top lid by +0.03 in floor/field at 50/s, while a domain-top lid matched;
a lid-height option for sample runs is not wired yet.
