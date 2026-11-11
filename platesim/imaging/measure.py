"""Cu film thickness measurement from SEM cross-section images.

Strategy: crop the raster data bar, pick the thickness axis (layers horizontal
-> measure along y; layers rotated 90 deg -> measure along x) by comparing the
variation of row- vs column-median profiles, locate the film band on the median
profile as the (rise, fall) gradient pair with the largest sustained contrast
(box-smoothed gradient + pre/post window contrast, longest band wins ties),
then refine both interfaces per line (gradient extrema inside a coarse window,
re-referenced by a median-filtered boundary curve to follow tilt and surface
roughness), then robust statistics (median/MAD) over lines.

Tilt: the FEI stage tilt is ~0 deg on all Planar 2 images, so the 1/sin(tilt)
vertical-correction factor is 1.0 (recorded). With a nonzero stage tilt the
vertical chord would be scaled by 1/sin(tilt). [C calc]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from skimage.filters import threshold_multiotsu

from .metadata import image_rows, parse_sem_metadata

# ROI: fraction of the perpendicular axis skipped at each edge (frame edges,
# mount glue, comb-shaped fracture artifacts at the sample edge).
ROI_MARGIN = 0.05
# MAD outlier fence (approx normal: 3.5 * 1.4826 ~ 2.4 sigma)
MAD_Z = 3.5
# boundary-curve smoothing window (perpendicular axis, px)
BOUNDARY_SMOOTH = 51


def load_image_cropped(path: str | Path, meta: dict | None = None) -> np.ndarray:
    """Read SEM TIFF as float array with the raster data-bar rows removed."""
    img = tifffile.imread(path)
    if img.ndim == 3:  # RGB export -> grayscale
        img = img[..., :3].mean(axis=2)
    if meta is None:
        try:
            meta = parse_sem_metadata(path)
        except ValueError:
            meta = {}
    rows = image_rows(meta) if meta else img.shape[0]
    return np.asarray(img[:rows], dtype=np.float64)


def detect_thickness_axis(img: np.ndarray) -> str:
    """'y' = layers horizontal (measure along rows); 'x' = layers vertical."""
    row_prof = median_filter(np.median(img, axis=1), size=9)
    col_prof = median_filter(np.median(img, axis=0), size=9)
    return "y" if float(np.std(row_prof)) >= float(np.std(col_prof)) else "x"


def _merge_peaks(peaks: list[int], values: np.ndarray, min_dist: int) -> list[int]:
    """Keep the highest-magnitude peak inside each cluster of nearby peaks."""
    out: list[int] = []
    for i in peaks:
        if out and i - out[-1] < min_dist:
            if abs(values[i]) > abs(values[out[-1]]):
                out[-1] = i
        else:
            out.append(i)
    return out


def band_from_profile(profile: np.ndarray) -> tuple[int, int]:
    """Coarse film band = (rise, fall) pair with largest sustained contrast.

    Uses a box-smoothed gradient so that short in-band grain-boundary dips do
    not dominate; among pairs within 80% of the best contrast score the longest
    band wins (avoids locking onto a bright sub-band or an internal dark blob).
    """
    sm = gaussian_filter1d(median_filter(np.asarray(profile, dtype=np.float64), size=9), sigma=5)
    n = len(sm)
    L = max(40, int(0.05 * n))
    gb = uniform_filter1d(np.gradient(sm), size=L)
    rises = _merge_peaks(list(find_peaks(gb, height=0.2 * gb.max(), distance=L)[0]), gb, L)
    falls = _merge_peaks(list(find_peaks(-gb, height=0.2 * (-gb.min()), distance=L)[0]), gb, L)
    if not rises or not falls:
        raise ValueError("No sustained brightness transitions found in profile")
    d = max(20, L // 2)
    cands: list[tuple[float, int, int]] = []
    for r in rises:
        for f in falls:
            length = f - r
            # ignore sub-bands: a film must span a reasonable fraction of the
            # field of view (internal dark stripes are much narrower)
            if length < 0.12 * n or length > 0.9 * n:
                continue
            up = sm[min(n - 1, r + d)] - sm[max(0, r - d)]
            down = sm[max(0, f - d)] - sm[min(n - 1, f + d)]
            cands.append((up + down, r, f))
    if not cands:
        raise ValueError("No valid (rise, fall) band pair found")
    best = max(score for score, _, _ in cands)
    if best < 0.05 * float(sm.max() - sm.min()):
        raise ValueError("Band contrast too weak")
    long_ok = [(r, f) for score, r, f in cands if score >= 0.8 * best]
    r, f = max(long_ok, key=lambda rf: rf[1] - rf[0])
    return int(r), int(f)


def _profile_along(img: np.ndarray, axis: str) -> np.ndarray:
    return np.median(img, axis=1) if axis == "y" else np.median(img, axis=0)


def _line(img: np.ndarray, i: int, axis: str) -> np.ndarray:
    return img[:, i] if axis == "y" else img[i, :]


def _track_edge(
    img: np.ndarray,
    axis: str,
    ref: np.ndarray,
    window: int,
    rising: bool,
) -> np.ndarray:
    """Per-line gradient extremum inside +-window around reference curve."""
    n = img.shape[0] if axis == "y" else img.shape[1]
    m = img.shape[1] if axis == "y" else img.shape[0]
    out = np.full(m, np.nan)
    for i in range(m):
        r = ref[i]
        if not np.isfinite(r):
            continue
        line = median_filter(_line(img, i, axis), size=5)
        g = np.gradient(line)
        a = max(0, int(r) - window)
        b = min(n, int(r) + window + 1)
        if b - a < 3:
            continue
        idx = a + int(np.argmax(g[a:b]) if rising else np.argmin(g[a:b]))
        out[i] = idx
    return out


def _smooth_edge(edge: np.ndarray) -> np.ndarray:
    """Median-filter an edge curve, skipping NaNs (falls back to raw window)."""
    ok = np.isfinite(edge)
    if ok.sum() < BOUNDARY_SMOOTH:
        return edge.copy()
    filled = np.interp(np.arange(len(edge)), np.flatnonzero(ok), edge[ok])
    return median_filter(filled, size=BOUNDARY_SMOOTH)


def measure_band_auto(
    img: np.ndarray,
    pixel_size_nm: float,
    axis: str | None = None,
    tilt_deg: float = 0.0,
) -> dict:
    """Automatic film-band thickness. Per-line profile + robust stats."""
    if axis is None:
        axis = detect_thickness_axis(img)
    prof = _profile_along(img, axis)
    t0, t1 = band_from_profile(prof)
    m = img.shape[1] if axis == "y" else img.shape[0]
    # coarse window: wide enough for surface roughness / in-plane tilt,
    # then tight second pass around the smoothed boundary curve
    w1 = max(40, int(0.35 * (t1 - t0)))
    tops = _track_edge(img, axis, np.full(m, t0), w1, rising=True)
    bots = _track_edge(img, axis, np.full(m, t1), w1, rising=False)
    # consistency check: if the wide per-line search lands systematically away
    # from the global profile edge (hijacked by stronger internal texture
    # gradients), re-run tightly around the profile edge instead.
    band = max(1, t1 - t0)
    wtight = max(40, min(120, int(0.15 * band)))
    for arr, ref0, rising in ((tops, t0, True), (bots, t1, False)):
        med = np.nanmedian(arr) if np.isfinite(arr).any() else np.nan
        if not np.isfinite(med) or abs(med - ref0) > 0.05 * band:
            tight = _track_edge(img, axis, np.full(m, ref0), wtight, rising=rising)
            if np.isfinite(tight).any():
                arr[:] = np.where(np.isfinite(tight), tight, arr)
    tops_ref = _smooth_edge(tops)
    bots_ref = _smooth_edge(bots)
    w2 = max(15, min(40, w1 // 3))
    tops = _track_edge(img, axis, tops_ref, w2, rising=True)
    bots = _track_edge(img, axis, bots_ref, w2, rising=False)
    thickness = bots - tops
    thickness[(bots <= tops) | (tops < 0)] = np.nan
    # ROI margins on the perpendicular axis
    i0 = int(ROI_MARGIN * m)
    i1 = int((1 - ROI_MARGIN) * m)
    mask = np.zeros(m, dtype=bool)
    mask[i0:i1] = True
    thickness = np.where(mask, thickness, np.nan)
    # robust outlier rejection
    finite = np.isfinite(thickness)
    if finite.sum() >= 5:
        med = np.nanmedian(thickness)
        mad = np.nanmedian(np.abs(thickness - med))
        fence = MAD_Z * 1.4826 * mad if mad > 0 else np.inf
        thickness = np.where(finite & (np.abs(thickness - med) <= fence), thickness, np.nan)
    # tilt correction: with stage tilt t the true vertical thickness is
    # chord/sin(t); t ~ 0 deg here -> factor 1. [C calc]
    factor = 1.0 / np.sin(np.deg2rad(tilt_deg)) if tilt_deg > 1e-3 else 1.0
    t_um = thickness * pixel_size_nm * factor / 1000.0
    good = np.isfinite(t_um)
    pos = np.arange(m) * pixel_size_nm / 1000.0
    return {
        "axis": axis,
        "band_px": [int(t0), int(t1)],
        "tilt_factor": factor,
        "thickness_um": t_um,
        "top_px": tops,
        "bot_px": bots,
        "position_um": pos,
        "mean_um": float(np.mean(t_um[good])) if good.any() else float("nan"),
        "std_um": float(np.std(t_um[good], ddof=1)) if good.sum() > 1 else float("nan"),
        "min_um": float(np.min(t_um[good])) if good.any() else float("nan"),
        "max_um": float(np.max(t_um[good])) if good.any() else float("nan"),
        "median_um": float(np.median(t_um[good])) if good.any() else float("nan"),
        "n_lines": int(good.sum()),
    }


def measure_manual(
    img: np.ndarray,
    pixel_size_nm: float,
    lines: list[dict],
    tilt_deg: float = 0.0,
) -> dict:
    """Manual override: lines = [{x1,y1,x2,y2}, ...] in full-image pixels.

    Thickness of each line = Euclidean length in px * pixel size * tilt factor.
    """
    factor = 1.0 / np.sin(np.deg2rad(tilt_deg)) if tilt_deg > 1e-3 else 1.0
    lengths = [float(np.hypot(ln["x2"] - ln["x1"], ln["y2"] - ln["y1"])) for ln in lines]
    t_um = np.asarray(lengths) * pixel_size_nm * factor / 1000.0
    return {
        "axis": "manual",
        "tilt_factor": factor,
        "thickness_um": t_um,
        "mean_um": float(np.mean(t_um)) if len(t_um) else float("nan"),
        "std_um": float(np.std(t_um, ddof=1)) if len(t_um) > 1 else 0.0,
        "min_um": float(np.min(t_um)) if len(t_um) else float("nan"),
        "max_um": float(np.max(t_um)) if len(t_um) else float("nan"),
        "median_um": float(np.median(t_um)) if len(t_um) else float("nan"),
        "n_lines": len(t_um),
    }


def load_manual_lines(path: str | Path) -> dict:
    """JSON {file_name: [{x1,y1,x2,y2}, ...]}."""
    return json.loads(Path(path).read_text())


def band_from_otsu(profile: np.ndarray) -> tuple[int, int]:
    """Fallback: longest run of the top multi-Otsu class (bright layer)."""
    p = median_filter(np.asarray(profile, dtype=np.float64), size=9)
    try:
        thr = threshold_multiotsu(p, classes=3)
        mask = p > thr[-1]
    except ValueError:
        mask = p > np.percentile(p, 75)
    best = (0, 0)
    start = None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if best[1] - best[0] < 3:
        raise ValueError("No bright band found in profile")
    return best
