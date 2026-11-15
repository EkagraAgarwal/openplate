"""Multiscale bridge: coupon-scale local conditions -> feature-scale trench fill.

``platesim.multiscale.bridge`` samples the calibrated coupon pipeline at a coupon
position and feeds the local current density / Nernst-layer thickness / efficiency
into either the level-set feature model (``feature.calibrate.predict``) or the
quasi-steady trench flow + transport model (``trenchflow.model.predict_site``).

``platesim.multiscale.validate`` is the CLI that evaluates a 5x5 position grid of
the Trench 1 coupon against the measured wide/narrow/flat site thicknesses and
writes ``multiscale/trench1_prediction.json``.  No parameter is fitted to the
trench data.

Plan F1 adds the reverse direction (``two_way=True``, default off):
:func:`effective_area_factor` measures how much extra current one pitch of a
trench array draws (:func:`coupon_area_map` iterates that back into the coupon
current map and :func:`rescale_coupon_currents` applies it galvanostatically).
"""
from .bridge import (
    DEFAULT_F_OPEN, DEFAULT_RUN, SIM_CACHE, TWO_WAY_MAX_ITER, TWO_WAY_TOL,
    cache_stats, clear_cache, coupon_area_map, coupon_centre_z_cm,
    effective_area_factor, face_extent_cm, predict_trench,
    rescale_coupon_currents, site_conditions,
)

__all__ = ["DEFAULT_F_OPEN", "DEFAULT_RUN", "SIM_CACHE", "TWO_WAY_MAX_ITER",
           "TWO_WAY_TOL", "cache_stats", "clear_cache", "coupon_area_map",
           "coupon_centre_z_cm", "effective_area_factor", "face_extent_cm",
           "predict_trench", "rescale_coupon_currents", "site_conditions"]
