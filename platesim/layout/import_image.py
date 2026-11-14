"""Bright pixels are openings; image row zero is the top of the die."""
import numpy as np

from .model import make_layout


def import_image(path, pixel_um, threshold=None, layer="etch"):
    from shapely.geometry import Polygon, box
    from skimage.io import imread
    from skimage.color import rgb2gray
    from skimage.filters import threshold_otsu
    from skimage.measure import find_contours
    if pixel_um is None or not np.isfinite(pixel_um) or pixel_um <= 0:
        raise ValueError(f"pixel_um={pixel_um!r}: supply a finite pixel size > 0")
    image = imread(path)
    if image.ndim == 3:
        # Preserve native intensity range so explicit thresholds also work on RGB.
        scale = np.iinfo(image.dtype).max if image.dtype.kind in "ui" else 1
        image = rgb2gray(image[..., :3])*scale
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("image: supply a finite grayscale or RGB image")
    if threshold is None:
        threshold = threshold_otsu(image) if image.min() != image.max() else image.min()/2
    binary = image > threshold
    height, width = binary.shape
    polygons = []
    for contour in find_contours(np.pad(binary.astype(float), 1), .5,
                                 fully_connected="low"):
        # Contours index pixel centres: remove padding and add the half cell.
        points = np.column_stack(((contour[:, 1]-.5)*pixel_um,
                                  (height+.5-contour[:, 0])*pixel_um))
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        polygons.append(polygon)
    # XOR nesting handles holes, islands inside holes and disconnected features.
    geometry = Polygon()
    for polygon in sorted(polygons, key=lambda p: p.area, reverse=True):
        geometry = geometry.symmetric_difference(polygon)
    return make_layout([(layer, geometry)], box(0, 0, width*pixel_um, height*pixel_um))
