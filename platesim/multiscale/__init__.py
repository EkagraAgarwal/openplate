"""Multiscale bridge: coupon-scale local conditions -> feature-scale trench fill.

``platesim.multiscale.bridge`` samples the calibrated coupon pipeline at a coupon
position and feeds the local current density / Nernst-layer thickness / efficiency
into either the level-set feature model (``feature.calibrate.predict``) or the
quasi-steady trench flow + transport model (``trenchflow.model.predict_site``).

``platesim.multiscale.validate`` is the CLI that evaluates a 5x5 position grid of
the Trench 1 coupon against the measured wide/narrow/flat site thicknesses and
writes ``multiscale/trench1_prediction.json``.  No parameter is fitted to the
trench data.
"""
from .bridge import (
    DEFAULT_RUN, SIM_CACHE, cache_stats, clear_cache, coupon_centre_z_cm,
    face_extent_cm, predict_trench, site_conditions,
)

__all__ = ["DEFAULT_RUN", "SIM_CACHE", "cache_stats", "clear_cache",
           "coupon_centre_z_cm", "face_extent_cm", "predict_trench",
           "site_conditions"]
