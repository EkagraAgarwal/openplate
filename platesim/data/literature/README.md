# platesim/data/literature

Index for `platesim/data/literature/`. Each dataset stored under `research:literature:<key>` is a complete JSON ready to write as `platesim/data/literature/<file>`. Tag key: [FT] full text read, [AB] abstract/metadata only, [SH] secondhand, [C] our calculation. All DOIs Crossref-verified 2026-10-28. No personal data in any URL.

## Datasets

| File | Key | Description |
| --- | --- | --- |
| `braun2019_tsv_sn_dr_parameters.json` | `research:literature:braun2019_tsv_sn_dr_parameters` | NIST LOW-k S-NDR model parameter table (D, Gamma, k+-, j0, alpha, kappa 15.26 S/m) + breakdown -60 mV/dec Cl- trend; calibrates suppressor kinetics and the voltammetric breakdown threshold for the feature model. [FT] |
| `braun2020_through_hole_parameters.json` | `research:literature:braun2020_through_hole_parameters` | NIST HIGH-k (bottom-up) parameter set (k_P+ 2500, k_P- 1e7) + complete through-hole butterfly fill simulation time series (impingement/fill times at defined j and E); regression targets for platesim feature fill. [FT] |
| `menk_hollowell2019_galvanostatic_tsv.json` | `research:literature:menk_hollowell2019_galvanostatic_tsv` | Galvanostatic TSV fill outcome matrix 80-320 mA/cm2 (via-bottom basis), 400 rpm: 160 mA/cm2 void-free for 100x600 um vias in 3 h; calibrates galvanostatic recipes and current-density basis. [FT] |
| `menk2019_potentiostatic_tsv_fill_timeseries.json` | `research:literature:menk2019_potentiostatic_tsv_fill_timeseries` | Potentiostatic fill height vs time series at -0.64 V (2/4/6 h, ~100 um/h ~ 75 mA/cm2 local j) + void/void-free outcome map -0.64..-0.74 V; primary h(t) validation series for via-fill model. [FT] |
| `josell2019_mmtsv_lowhigh_cl_timeseries.json` | `research:literature:josell2019_mmtsv_lowhigh_cl_timeseries` | mm-TSV (125x625 um) low-Cl stepped-potential protocols incl. VOID-FREE 12.5 h recipe, and high-Cl h(t) series + rpm starvation limit (<200 rpm kills bottom-up); validates stepped recipes and rpm sensitivity. [FT] |
| `josell_moffat2018_suppression_breakdown_tsv.json` | `research:literature:josell_moffat2018_suppression_breakdown_tsv` | Named priority paper: fill outcome matrix (E x TET x t) for 56 um annular TSV, S-NDR window definition, fastest published fill (15 min at 25 umol/L TET), rotation-asymmetric breakdown/repassivation criterion. [FT] |
| `schmitt2020_ar10_stepped_endpoint.json` | `research:literature:schmitt2020_ar10_stepped_endpoint` | Sandia AR10 (62.5x625 um) void-free recipe: -500->-560 mV MSE 10 mV steps, 2 h/step with 5 h dwell at -540 mV (17 h total), CT-verified; current-minimum endpoint; validates KB staged-ramp shape. [FT] |
| `kim2020_pulse_dc_tsv_fill.json` | `research:literature:kim2020_pulse_dc_tsv_fill` | DC vs pulse fill-ratio outcomes for 5x60 um AR12 vias with commercial suppressor (DC best 1 mA/cm2, PC best 3 mA/cm2 avg @ 450 mA/cm2 peak 2 us); supports PR module qualitatively. [FT] |
| `mass_transfer_anchors_rde_stirring.json` | `research:literature:mass_transfer_anchors_rde_stirring` | Mass-transfer anchors: delta=25 um @100 rpm (NIST), delta~omega^-0.5, k_m(Cu) 0.11/0.21 cm/s at 100/400 rpm [C], rotating-wafer flow speeds, D_Cu=2.65e-6 cm2/s provenance; calibrates stirring delta model. [FT]/[C] |
| `bath_conductivity_anchors.json` | `research:literature:bath_conductivity_anchors` | Conductivity anchors: 0.075 S/cm for 0.88 M CuSO4/0.18 M H2SO4 @298 K (closest to team bath) and 0.1526 S/cm high-acid; prior for kappa in ohmic model; no direct 8 g/L-acid kappa(T) found - measure via P2. [FT table/SH primary] |
| `wagner_hull_cell_kim2018.json` | `research:literature:wagner_hull_cell_kim2018` | Wagner-number uniformity method pointer (rotating-cylinder Hull cell CFD) + paywalled throwing-power leads; no numeric dataset obtained (priority 5 gap). [AB] |

## Gaps

- Planar efficiency (j, eta) tables for PEG/SPS/Cl sulfate baths: NOT FOUND in accessible literature; team planar_doe.json + vendor table remain primary. No O2-reduction contribution measurement retrieved (KB 0.15-0.5 mA/cm2 stays [C]).
- Sh-Re-Sc correlation for a magnetic stir bar: NOT FOUND; use RDE-anchored delta~rpm^-0.5 + P2 limiting-current calibration.
- kappa(T) for 65 g/L Cu / 8 g/L H2SO4: NOT FOUND; nearest anchor 0.075 S/cm @298 K (0.18 M acid). Hotlos 1988 (densities/viscosities, doi 10.1016/0304-386X(88)90012-6) may contain conductivity - not retrieved.
- Wheeler 2013 JES 160:D3260 full text not retrieved (IOP captcha); DOI verified 10.1149/2.040312jes; its Table I content is effectively covered by Braun 2019 (same group, D_Cu 2.65e-6 cm2/s).
- Figure digitisation: none performed (no image analysis possible from text fetches); Fig 6 Menk 2019, Figs 2/6 Josell 2019, Fig 4 Braun 2019, Fig 4 Kim 2020 flagged as digitisation candidates with estimated errors in each dataset notes.
