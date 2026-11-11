"""Regenerate platesim/data/sem_measurements.json, metadata_summary.json and previews.

Run from the platesim project root:
    python -m platesim.imaging.export_measurements
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import FILE_CHOICES, SEM_IMAGE_DIR, _FILE_INFO, resolve_image_path
from .measure import load_image_cropped, measure_band_auto, measure_manual, load_manual_lines
from .metadata import parse_sem_metadata

PKG_DIR = Path(__file__).resolve().parent
PREVIEW_DIR = PKG_DIR / "previews"
OUT_DATA = PKG_DIR.parent / "data" / "sem_measurements.json"

# confidence / notes decided after visual verification of the overlays
_NOTES = {
    "planar2_9mA_top.tif": (
        "Auto mean within ~3% of the operator bar (~6.55-6.65 um, label partially "
        "ambiguous). Film band: Cu surface -> fine-grain band/substrate boundary; "
        "right ~8% of the frame (comb-shaped cleave artifact) excluded by ROI."
    ),
    "planar2_9mA_bottom.tif": (
        "Auto mean within ~5% of the operator bar (6.583 um). Same band convention "
        "as 9mA_top; the film/substrate interface sits at the end of the bright "
        "fine-grain band before the cleave comb."
    ),
    "planar2_9mA_seed.tif": (
        "High-magnification zoom (HFW 3.19 um) of the seed/stack region only - the "
        "full Cu film is NOT in frame. Auto band ~0.8 um is the visible bright stack "
        "portion, not a film thickness. Operator sub-measurements on this image: "
        "~810 nm, ~537 nm, ~300 nm [M]."
    ),
    "planar2_18mA_mid.tif": (
        "Auto ~4% below the operator bar (12.43 um): per-column interface detection "
        "stops at the coarse/fine-grain transition while the operator drew to the "
        "cleave-comb start. Trust auto for the profile shape; absolute value "
        "11.9-12.4 um."
    ),
    "planar2_18mA_bottom.tif": (
        "Auto ~3% below the operator bar (12.04 um); same interface ambiguity as "
        "18mA_mid. Absolute value 11.7-12.0 um."
    ),
    "planar2_top_left.tif": (
        "Cross-section of the Oct 7 efficiency coupon (top-left corner). Band: "
        "air -> rough Cu surface -> bright fine-grain band -> dark Si stripe. "
        "Surface roughness (grain pyramids) spreads the per-column values."
    ),
    "planar2_top_left_2.tif": (
        "Second field at the same top-left corner; agrees with planar2_top_left "
        "within ~0.3 um."
    ),
    "planar2_top_right-short.tif": (
        "Anomalously thin film at the top-right corner. Auto band (1.52 um) is "
        "ambiguous because the film/substrate boundary degenerates into Si cleave "
        "steps; the operator bars read 1.881/1.973 um but their right endpoints "
        "appear to reach into the Si cleave face. Best manual estimate: 1.5-2.0 um. "
        "Filename suggests this area shorted during plating."
    ),
    "planar2_bottom_left.tif": (
        "Coupon sits tilted in-plane (~9 deg from horizontal): thickness varies "
        "systematically along the interface (left thicker than right). Auto tracks "
        "the tilted interfaces; per-column std is dominated by the tilt, not noise. "
        "Operator bars on the image read ~5.1-6.2 um."
    ),
    "planar2_bottom_right.tif": (
        "Clean horizontal stack: dark vacuum -> bright fine-grain top band -> "
        "columnar Cu -> wavy film/substrate edge -> gray Si below."
    ),
    "useless.tif": (
        "No complete layer stack in frame (too high mag, substrate interface out of "
        "view); no thickness extracted."
    ),
}

_CONFIDENCE = {
    "planar2_9mA_top.tif": "high",
    "planar2_9mA_bottom.tif": "high",
    "planar2_9mA_seed.tif": "low",
    "planar2_18mA_mid.tif": "medium",
    "planar2_18mA_bottom.tif": "medium",
    "planar2_top_left.tif": "medium-high",
    "planar2_top_left_2.tif": "medium-high",
    "planar2_top_right-short.tif": "low-medium",
    "planar2_bottom_left.tif": "medium",
    "planar2_bottom_right.tif": "medium-high",
    "useless.tif": "none",
}


def _overlay_png(img: np.ndarray, res: dict, dest: Path) -> None:
    vis = np.stack([np.asarray(img)] * 3, axis=-1)
    tops, bots = res.get("top_px"), res.get("bot_px")
    if tops is None or bots is None:
        return
    if res["axis"] == "y":
        for i in range(img.shape[1]):
            if np.isfinite(tops[i]):
                t, b = int(tops[i]), int(bots[i])
                vis[max(0, t - 1): t + 1, i] = [255, 0, 0]
                vis[max(0, b - 1): b + 1, i] = [0, 255, 0]
    else:
        for i in range(img.shape[0]):
            if np.isfinite(tops[i]):
                t, b = int(tops[i]), int(bots[i])
                vis[i, max(0, t - 1): t + 1] = [255, 0, 0]
                vis[i, max(0, b - 1): b + 1] = [0, 255, 0]
    Image.fromarray(vis.astype(np.uint8)).resize(
        (768, int(img.shape[0] * 768 / img.shape[1])), Image.Resampling.LANCZOS
    ).save(dest)


def main() -> None:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    manual_lines = load_manual_lines(PKG_DIR / "manual_lines.json")
    meta_summary: dict = {}
    entries: list[dict] = []
    for rel in FILE_CHOICES:
        path = resolve_image_path(rel)
        name = path.name
        meta = parse_sem_metadata(path)
        img = load_image_cropped(path, meta)
        ps = meta["pixel_size_nm"]
        tilt = float(meta.get("tilt_deg") or 0.0)
        hfw = meta.get("hfw_m")
        wd = meta.get("wd_m")
        meta_summary[name] = {
            "pixel_size_nm": ps,
            "tilt_deg": tilt,
            "hfw_um": hfw * 1e6 if hfw else None,
            "hv_v": meta.get("hv_v"),
            "wd_mm": (wd or 0) * 1e3,
            "detector": meta.get("detector"),
            "detector_mode": meta.get("detector_mode"),
            "date": meta.get("date"),
            "time": meta.get("time"),
            "mag_x_estimate": meta.get("mag_x_estimate"),
            "resolution": [meta.get("resolution_x"), meta.get("resolution_y")],
        }
        # full + cropped previews
        Image.fromarray(np.asarray(img, dtype=np.uint8)).resize(
            (768, int(img.shape[0] * 768 / img.shape[1])), Image.Resampling.LANCZOS
        ).save(PREVIEW_DIR / (name.replace(".tif", "") + "_crop.png"))
        # auto measurement + overlay
        try:
            res = measure_band_auto(img, ps, tilt_deg=tilt)
        except ValueError as exc:
            res = {"mean_um": float("nan"), "std_um": float("nan"), "min_um": float("nan"),
                   "max_um": float("nan"), "axis": None, "n_lines": 0}
            print(f"{name}: auto failed ({exc})")
        _overlay_png(img, res, PREVIEW_DIR / (name.replace(".tif", "") + "_overlay.png"))
        if name == "useless.tif":
            mean = std = mn = mx = None
            method = "none"
        else:
            mean = res["mean_um"] if np.isfinite(res["mean_um"]) else None
            std = res["std_um"] if np.isfinite(res["std_um"]) else None
            mn = res["min_um"] if np.isfinite(res["min_um"]) else None
            mx = res["max_um"] if np.isfinite(res["max_um"]) else None
            method = "auto"
        entries.append({
            "file": name,
            "pixel_size_nm": round(ps, 4),
            "tilt_deg": tilt,
            "view": _FILE_INFO[rel]["view"],
            "thickness_um_mean": round(mean, 3) if mean is not None else None,
            "thickness_um_std": round(std, 3) if std is not None else None,
            "thickness_um_min": round(mn, 3) if mn is not None else None,
            "thickness_um_max": round(mx, 3) if mx is not None else None,
            "method": method,
            "confidence": _CONFIDENCE[name],
            "notes": _NOTES[name],
            "run_id_guess": _FILE_INFO[rel]["run_id_guess"],
        })
        # manual (operator-bar / picked-line) companion numbers
        lines = manual_lines.get(name) or manual_lines.get(rel)
        if lines:
            mres = measure_manual(img, ps, lines, tilt_deg=tilt)
            entries[-1]["manual_lines_um"] = [round(float(v), 3) for v in mres["thickness_um"]]
            entries[-1]["manual_mean_um"] = round(mres["mean_um"], 3)
    doc = {
        "source": "platesim.imaging (auto band detection + manual line overrides); "
                  "pixel sizes from FEI/Apreo tag 34682; SEM images acquired on a "
                  "Thermo Apreo (user tvanant), 5 kV, ETD/SE [M]",
        "notes": [
            "tilt_deg is the FEI StageT stage tilt (0 deg on all images -> no 1/sin(tilt) correction needed)",
            "efficiency-corner run_id_guess=1 because the three intact corners span 6.8-8.1 um, "
            "matching the planar_doe Oct 7 coupon note (6.2-9.0 um corner-to-corner) and the "
            "9 mA run level (6.6 um), not the 18 mA level (12.0 um)",
            "planar2_top_right-short.tif is anomalous (1.5-2.0 um); if it shorted during plating "
            "it should be excluded from corner statistics",
        ],
        "measurements": entries,
    }
    OUT_DATA.write_text(json.dumps(doc, indent=2))
    (PKG_DIR / "metadata_summary.json").write_text(json.dumps(meta_summary, indent=2))
    print(f"wrote {OUT_DATA}")
    print(f"wrote {PKG_DIR / 'metadata_summary.json'}")
    for e in entries:
        print(f"  {e['file']:35s} {str(e['thickness_um_mean']):>7s} um  ({e['confidence']})")


if __name__ == "__main__":
    main()
