# L1 sample pipeline

`python -m platesim.sample3d plan spec.json` plans without running the solver.
`python -m platesim.sample3d run spec.json --out NEW_DIRECTORY` writes a single
TileResult plus three PNGs and a validated Manifest. Existing results are never
overwritten. Artifact paths are relative to the result directory.

One original bare-silicon frame and each recipe-step terminal frame are saved;
times, charge, deposited volume and Faraday volume are cumulative. All solver
calls use `run_dir=None`, and the final front is passed unchanged to the next
call. Fields and intermediate solver frames are not retained. Cu balance is the
maximum absolute saved solver balance across all steps. Faraday error is the
absolute cumulative volume error, without volume correction. Geometric analysis
uses occupancy for fill/voids; the deposit quantity and check use the engine's
smoothed solid-volume quadrature. Missing face measurements are omitted, not zero.

Preview/standard/fine spacing is 8/4/2 um, overridden by `numerics.dx_um`.
One single-feature array with integral, sufficiently resolved pitches uses a
full-pitch periodic window centred on the first instance whose window fits in
the die. Mixed layouts, nonintegral pitches and dies smaller than a pitch use
the whole die with symmetry sides. Whole-die dimensions and vertical extents
round outward to whole cells (at least two XY cells); this may add a sub-cell
strip outside the die. Maximum etch depth includes depth-vs-width tables.
`est_seconds` is only a heuristic; sparse linear setup dominates real runtimes.

Interface workarounds/limitations:

- Layout functions consume dicts, not Pydantic models: dump models explicitly.
- spec3d circles use diameter and permit an omitted radius; layout requires a
  positive radius: derive it from diameter.
- Etcher distances already are cm (no second conversion). Its XY origin is
  zero, sides symmetry and marching nonperiodic: construct Grid3D from the plan
  and re-distance the union with the planned periodic flags.
- Per-step run_tile treats its input as bare silicon and restarts time/charge:
  use a separate writer preserving the true original silicon and cumulative
  scalar series. Fix the original field mask across all steps.
- render_height_map takes three arrays, not the contract's height dict:
  unpack `height_um`, `x_um`, `y_um`. Rendering returns optional metadata not
  accepted by Artifact: retain only kind/path/title.
- Only CPU L1, no surrogate. Unsupported physics/GPU requests are errors, not
  silent downgrades. Temperature, bulk Cu and fully dissociated acid migration
  enhancement are mapped to Params. Bath conductivity/kinetics remain engine
  placeholders; chloride/additives, seed thinning, orientation and stirring are
  not resolved by L1. All quantities are uncalibrated.

Contract ambiguities: no rule for choosing among multiple array pitches,
nonintegral pitches or die sizes, no time-estimate calibration and no prescribed
frame count. Conservative whole-die fallback, outward rounding, a heuristic
estimate and step-end frames are used above. Setup area is sample-scale metadata;
tile Faraday volume uses the actual XY tile area, not setup.area_cm2.
