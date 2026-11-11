"""SEM TIFF metadata parsing (FEI/Apreo ini-style tag 34682, Zeiss 34118, Hitachi txt).

All Planar 2 images are Thermo FEI Apreo SEM exports: tag 34682 ('FEI_HELIOS')
holds a nested dict with Scan.PixelWidth (m), EBeam.HV, Stage.StageT, Detectors, etc.
Pixel size is read from Scan.PixelWidth/PixelHeight (metres -> nm). [M measured from tag]
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import tifffile
from tifffile import TiffPage

# These exports are 1536x1094 but the real image is ResolutionY=1024 rows; the
# bottom 70 rows are the raster data bar (PrivateFei.DataBarSelected).
DATABAR_FALLBACK_ROWS = 70


def _parse_zeiss_34118(value: str) -> dict:
    """Zeiss CZ_SEM tag 34118: ini-style text with 'Image Pixel Size = x nm' etc."""
    out: dict = {}
    for line in value.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    meta: dict = {"vendor": "zeiss", "raw": out}
    m = re.search(r"Image Pixel Size\s*=\s*([\d.eE+-]+)\s*nm", value, re.I)
    if m:
        meta["pixel_size_nm"] = float(m.group(1))
    m = re.search(r"AP_PIXEL_SIZE\s*=\s*([\d.eE+-]+)", value, re.I)
    if m and "pixel_size_nm" not in meta:
        meta["pixel_size_nm"] = float(m.group(1))
    return meta


def _parse_fei(value: dict) -> dict:
    """FEI/Apreo tag 34682: nested dict already decoded by tifffile."""
    scan = value.get("Scan", {}) or {}
    ebeam = value.get("EBeam", {}) or {}
    stage = value.get("Stage", {}) or {}
    det = value.get("Detectors", {}) or {}
    img = value.get("Image", {}) or {}
    user = value.get("User", {}) or {}
    system = value.get("System", {}) or {}
    pw = scan.get("PixelWidth")
    ph = scan.get("PixelHeight")
    hfw = ebeam.get("HFW")
    meta: dict = {
        "vendor": "fei",
        "system_type": system.get("SystemType"),
        "date": user.get("Date"),
        "time": user.get("Time"),
        "user": user.get("User"),
        "hv_v": ebeam.get("HV", value.get("Beam", {}).get("HV")),
        "wd_m": ebeam.get("WD"),
        "hfw_m": hfw,
        "vfw_m": ebeam.get("VFW"),
        # stage tilt (deg); SpecTilt is specimen pre-tilt; TiltCorrectionAngle when on
        "tilt_deg": stage.get("StageT", 0.0) or 0.0,
        "spec_tilt_deg": stage.get("SpecTilt", 0.0) or 0.0,
        "tilt_correction_angle_deg": ebeam.get("TiltCorrectionAngle", 0.0) or 0.0,
        "tilt_correction_is_on": str(ebeam.get("TiltCorrectionIsOn", "")).lower() in ("yes", "true", "1"),
        "detector": det.get("Name"),
        "detector_mode": det.get("Mode"),
        "resolution_x": img.get("ResolutionX"),
        "resolution_y": img.get("ResolutionY"),
    }
    if pw:
        meta["pixel_size_nm"] = float(pw) * 1e9
    if ph:
        meta["pixel_height_nm"] = float(ph) * 1e9
    # approximate magnification: FEI virtual screen width / HFW [C calc]
    disp_w = system.get("DisplayWidth")
    if disp_w and hfw:
        meta["mag_x_estimate"] = round(float(disp_w) / float(hfw))
    return meta


def parse_sem_metadata(path: str | Path) -> dict:
    """Return {pixel_size_nm, tilt_deg, hv_v, detector, ...} for one SEM TIFF.

    Supports FEI tag 34682 (dict), Zeiss tag 34118 (ini text), Hitachi .txt
    companion files, and falls back to TIFF XResolution when possible.
    Raises ValueError when no pixel size can be determined.
    """
    path = Path(path)
    meta: dict = {"file": path.name}
    with tifffile.TiffFile(path) as tf:
        page = cast(TiffPage, tf.pages[0])
        meta["shape"] = list(page.shape)
        if 34682 in page.tags:
            value = page.tags[34682].value
            if isinstance(value, dict):
                meta.update(_parse_fei(value))
            elif isinstance(value, (str, bytes)):
                text = value.decode("latin-1") if isinstance(value, bytes) else value
                if "CZ_SEM" in text or "Pixel" in text:
                    meta.update(_parse_zeiss_34118(text))
        if 34118 in page.tags and "pixel_size_nm" not in meta:
            value = page.tags[34118].value
            if isinstance(value, bytes):
                value = value.decode("latin-1", errors="replace")
            if isinstance(value, str):
                meta.update(_parse_zeiss_34118(value))
        if "pixel_size_nm" not in meta:
            # Hitachi stores a sidecar .txt; also try TIFF resolution fallback
            txt = path.with_suffix(".txt")
            if txt.exists():
                text = txt.read_text(errors="replace")
                m = re.search(r"pixel size\s*[:=]\s*([\d.]+)\s*(nm|um|mm)", text, re.I)
                if m:
                    val = float(m.group(1))
                    unit = m.group(2).lower()
                    meta["pixel_size_nm"] = val * {"nm": 1, "um": 1e3, "mm": 1e6}[unit]
                    meta["vendor"] = "hitachi"
            if "pixel_size_nm" not in meta and page.tags.get("XResolution") is not None:
                try:
                    num, den = page.tags["XResolution"].value
                    unit_tag = page.tags.get("ResolutionUnit")
                    unit = str(unit_tag.value) if unit_tag is not None else ""
                    res = num / den  # pixels per unit
                    if "MM" in unit:
                        meta["pixel_size_nm"] = 1e6 / res
                    elif "INCH" in unit:
                        meta["pixel_size_nm"] = 25.4e6 / res
                    if "pixel_size_nm" in meta:
                        meta["vendor"] = meta.get("vendor", "tiff")
                except Exception:
                    pass
    if "pixel_size_nm" not in meta or not meta["pixel_size_nm"]:
        raise ValueError(f"No pixel size found in metadata of {path}")
    meta["pixel_size_nm"] = float(meta["pixel_size_nm"])
    return meta


def image_rows(meta: dict) -> int:
    """Number of real image rows (excludes the raster data bar)."""
    ry = meta.get("resolution_y")
    if ry:
        return int(ry)
    shape = meta.get("shape") or [0]
    h = int(shape[0])
    return h - DATABAR_FALLBACK_ROWS if h == 1094 else h
