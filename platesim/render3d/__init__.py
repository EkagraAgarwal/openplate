"""Headless synthetic tile visualisation with crash-isolated OpenGL."""
from .surface import render_surface
from .plots import render_height_map, render_cross_section
from .animate import render_growth_animation

__all__ = ['render_surface', 'render_height_map', 'render_cross_section',
           'render_growth_animation']
