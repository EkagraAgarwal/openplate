"""3-D sample specifications, without solver or geometry imports.

Examples are synthetic. Bath concentrations (40 g/L Cu, 100 g/L acid,
50 ppm chloride), 25 C temperature and unspecified corner radii are illustrative
[J], not vendor specifications or measured team data. Array pitches are along
the die axes; x_um/y_um locate the first element's centre. Circle diameter is
w_um == h_um; r_um is optional and geometry derives radius from the diameter.
"""
from .models import (
    Artifact, Bath, Check, Etch, EtchLayer, Feature, FeatureArray, Layout,
    Manifest, Numerics, Quantity, Recipe, SampleSpec, Setup, Step,
)
from .schema import export_schema
from .validate import spec_hash, validate_spec

__all__ = [
    "Artifact", "Bath", "Check", "Etch", "EtchLayer", "Feature", "FeatureArray",
    "Layout", "Manifest", "Numerics", "Quantity", "Recipe", "SampleSpec", "Setup",
    "Step", "export_schema", "spec_hash", "validate_spec",
]
