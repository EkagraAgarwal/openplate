"""platesim.imaging: Cu film thickness extraction from SEM cross-section TIFFs.

Public API follows the model plug-in contract (CONTRACTS.md): MODEL_SPEC +
run(params). Measurement internals live in .metadata and .measure.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .measure import (
    load_image_cropped,
    load_manual_lines,
    measure_band_auto,
    measure_manual,
)
from .metadata import parse_sem_metadata

# Sample Results/ lives next to the platesim project folder (project root is
# .../Semiconductors/platesim, package is .../Semiconductors/platesim/platesim).
SEM_IMAGE_DIR = Path(__file__).resolve().parents[3] / "Sample Results" / "Planar 2 (SiO2-Ti-Cu)"
MANUAL_LINES_PATH = Path(__file__).resolve().parent / "manual_lines.json"

FILE_CHOICES = [
    "planar2_9mA_top.tif",
    "planar2_9mA_bottom.tif",
    "planar2_9mA_seed.tif",
    "planar2_18mA_mid.tif",
    "planar2_18mA_bottom.tif",
    "planar2_efficiency/planar2_top_left.tif",
    "planar2_efficiency/planar2_top_left_2.tif",
    "planar2_efficiency/planar2_top_right-short.tif",
    "planar2_efficiency/planar2_bottom_left.tif",
    "planar2_efficiency/planar2_bottom_right.tif",
    "planar2_efficiency/useless.tif",
]

# per-file qualitative context (from visual inspection of the previews)
_FILE_INFO = {
    "planar2_9mA_top.tif": {"view": "cross-section (rotated 90 deg)", "run_id_guess": 1},
    "planar2_9mA_bottom.tif": {"view": "cross-section (rotated 90 deg)", "run_id_guess": 1},
    "planar2_9mA_seed.tif": {"view": "cross-section, high-mag seed/stack zoom", "run_id_guess": 1},
    "planar2_18mA_mid.tif": {"view": "cross-section (rotated 90 deg)", "run_id_guess": 2},
    "planar2_18mA_bottom.tif": {"view": "cross-section (rotated 90 deg)", "run_id_guess": 2},
    "planar2_efficiency/planar2_top_left.tif": {"view": "cross-section (horizontal stack)", "run_id_guess": 1},
    "planar2_efficiency/planar2_top_left_2.tif": {"view": "cross-section (horizontal stack)", "run_id_guess": 1},
    "planar2_efficiency/planar2_top_right-short.tif": {"view": "cross-section (rotated 90 deg)", "run_id_guess": 1},
    "planar2_efficiency/planar2_bottom_left.tif": {"view": "cross-section (horizontal stack, tilted in-plane)", "run_id_guess": 1},
    "planar2_efficiency/planar2_bottom_right.tif": {"view": "cross-section (horizontal stack)", "run_id_guess": 1},
    "planar2_efficiency/useless.tif": {"view": "unusable (no complete layer stack in frame)", "run_id_guess": None},
}

MODEL_SPEC = {
    "id": "sem_thickness",
    "title": "SEM Cu film thickness",
    "description": (
        "Extract plated Cu film thickness from SEM cross-section TIFFs "
        "(Planar 2 SiO2/Ti/Cu coupons). Reads FEI/Apreo metadata for the pixel "
        "size and stage tilt, crops the raster data bar, then measures the Cu "
        "band between the film surface and the film/substrate interface via "
        "per-column/row intensity profiles (median filter + gradient pair + "
        "robust statistics). Manual line-endpoint overrides are supported."
    ),
    "params": [
        {
            "name": "file",
            "type": "enum",
            "choices": FILE_CHOICES,
            "default": "planar2_9mA_top.tif",
            "help": "SEM TIFF (read-only, under 'Sample Results/Planar 2 (SiO2-Ti-Cu)').",
        },
        {
            "name": "method",
            "type": "enum",
            "choices": ["auto", "manual"],
            "default": "auto",
            "help": "auto = profile-based band detection; manual = user-picked line endpoints (platesim/imaging/manual_lines.json).",
        },
    ],
}


def resolve_image_path(file: str) -> Path:
    p = SEM_IMAGE_DIR / file
    if not p.exists():
        p = SEM_IMAGE_DIR / Path(file).name
    if not p.exists():
        raise FileNotFoundError(f"SEM image not found: {file} (looked in {SEM_IMAGE_DIR})")
    return p


def _f(x: float) -> float | None:
    """JSON-safe float (NaN/inf -> None)."""
    return float(x) if x is not None and math.isfinite(float(x)) else None


def _heatmap_plot(img: np.ndarray, pixel_size_nm: float, title: str) -> dict:
    """Downsampled grayscale image (<=200 px wide) as a heatmap plot."""
    h, w = img.shape
    step = max(1, int(math.ceil(w / 200)))
    small = img[::step, ::step]
    ys = np.arange(small.shape[0]) * step * pixel_size_nm / 1000.0
    xs = np.arange(small.shape[1]) * step * pixel_size_nm / 1000.0
    return {
        "kind": "heatmap",
        "title": title,
        "x": [round(float(v), 4) for v in xs],
        "y": [round(float(v), 4) for v in ys],
        "z": [[float(v) for v in row] for row in np.clip(small, 0, 255)],
        "zlabel": "grayscale (0-255)",
        "xlabel": "x (um)",
        "ylabel": "y (um)",
    }


def run(params: dict, progress=lambda frac, msg="": None) -> dict:
    file = params.get("file", "planar2_9mA_top.tif")
    method = params.get("method", "auto")
    progress(0.05, f"Reading metadata for {file}")
    path = resolve_image_path(file)
    meta = parse_sem_metadata(path)
    ps = meta["pixel_size_nm"]
    tilt = float(meta.get("tilt_deg") or 0.0)
    info = _FILE_INFO.get(file, {"view": "cross-section", "run_id_guess": None})
    img = load_image_cropped(path, meta)

    warnings: list[str] = []
    assumptions: list[str] = [
        f"pixel size from TIFF metadata: {ps:.4g} nm/px [M]",
        f"stage tilt {tilt:g} deg -> 1/sin(tilt) correction factor applied "
        f"({1.0 / np.sin(np.deg2rad(tilt)) if tilt > 1e-3 else 1.0:.3g}) [C calc]",
        "Cu film = bright band between the film/air surface and the "
        "film/substrate interface (fine-grained seed band counts as film) [J]",
    ]

    if method == "manual":
        lines = load_manual_lines(MANUAL_LINES_PATH).get(Path(file).name) or load_manual_lines(MANUAL_LINES_PATH).get(file)
        if not lines:
            raise ValueError(f"No manual lines defined for {file} in {MANUAL_LINES_PATH}")
        res = measure_manual(img, ps, lines, tilt_deg=tilt)
        profile = res["thickness_um"]
        positions = list(range(len(profile)))
        confidence = "medium (manual endpoints)"
    else:
        res = measure_band_auto(img, ps, tilt_deg=tilt)
        profile = res["thickness_um"]
        positions = res["position_um"]
        confidence = "medium (auto)"

    if "useless" in file:
        warnings.append("Image shows no complete layer stack; automatic band is not physically meaningful.")
        confidence = "none"
    if "seed" in file:
        warnings.append("High-mag seed/stack zoom: full Cu film not in frame; "
                        "operator sub-measurements on this image read ~300-810 nm features.")
        confidence = "low"
    if file.endswith("top_right-short.tif"):
        warnings.append("Operator bars on this image read 1.881/1.973 um but their right ends appear to "
                        "extend into the Si cleave face; auto profile is ambiguous (cleave steps). "
                        "Best manual estimate: 1.5-2.0 um.")

    def r3(x):
        v = _f(x)
        return round(v, 3) if v is not None else None

    summary = {
        "file": file,
        "method": method,
        "pixel_size_nm": round(ps, 4),
        "tilt_deg": tilt,
        "tilt_factor": round(res["tilt_factor"], 4),
        "view": info["view"],
        "axis": res.get("axis"),
        "thickness_um_mean": r3(res["mean_um"]),
        "thickness_um_std": r3(res["std_um"]),
        "thickness_um_min": r3(res["min_um"]),
        "thickness_um_max": r3(res["max_um"]),
        "thickness_um_median": r3(res["median_um"]),
        "n_lines": int(res["n_lines"]),
        "hv_v": meta.get("hv_v"),
        "detector": meta.get("detector"),
        "date": meta.get("date"),
        "confidence": confidence,
        "run_id_guess": info["run_id_guess"],
    }

    prof_list = [_f(v) for v in np.atleast_1d(profile)]
    plots = [
        {
            "kind": "line",
            "title": f"Cu thickness profile - {Path(file).name}",
            "x": [round(float(v), 4) for v in np.atleast_1d(positions)],
            "series": [{"name": "thickness (um)", "y": prof_list}],
            "xlabel": "position along interface (um)",
            "ylabel": "thickness (um)",
        },
        _heatmap_plot(img, ps, f"SEM image (downsampled) - {Path(file).name}"),
    ]
    progress(1.0, "done")
    return {"summary": summary, "plots": plots, "warnings": warnings, "assumptions": assumptions}


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run({"file": FILE_CHOICES[0], "method": "auto"})["summary"], indent=2))
