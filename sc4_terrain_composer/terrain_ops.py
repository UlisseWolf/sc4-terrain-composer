# -*- coding: utf-8 -*-
"""
terrain_ops.py
================================================================================
"Pure" numeric engine (numpy only) for DEM editing: no dependency on
PyQGIS. Contains all the delicate logic (masks, edge feathering, hole
filling, moving a region, SC4 encoding) so that it can be tested in
isolation, outside QGIS, before wiring it to the interface.

The "geometric" functions (blob polygonization, freehand drawing,
click-selection) stay in the actual plugin (dock_widget.py /
map_tools.py) because they need QGIS; here there's only array math.
================================================================================
"""

import numpy as np

# dtype used for all calculations: elevations are in meters, no need for
# float64 precision (which would needlessly double memory usage —
# relevant on large working DEMs, on the order of tens of millions of
# pixels like a regional mosaic at 10 m resolution).
WORK_DTYPE = np.float32


# ------------------------------------------------------------------------
# Land/sea masks
# ------------------------------------------------------------------------

def build_land_mask(dem: np.ndarray, sea_level: float = 0.0) -> np.ndarray:
    """Returns a boolean mask: True where the DEM is dry land."""
    return dem > sea_level


# ------------------------------------------------------------------------
# Separable blur via cumulative sum (O(n), no scipy)
# ------------------------------------------------------------------------

def _box_blur_1d(arr: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """1D box blur along one axis, via cumulative sum (fast, exact)."""
    if radius <= 0:
        return arr
    arr = np.moveaxis(arr, axis, 0)
    n = arr.shape[0]
    pad = np.pad(arr, [(radius, radius)] + [(0, 0)] * (arr.ndim - 1), mode="edge")
    csum = np.cumsum(pad, axis=0)
    csum = np.concatenate([np.zeros_like(csum[:1]), csum], axis=0)
    window = 2 * radius + 1
    out = (csum[window:] - csum[:-window]) / float(window)
    return np.moveaxis(out, 0, axis)


def box_blur(arr: np.ndarray, radius: int) -> np.ndarray:
    """Separable 2D box blur (rows, then columns)."""
    out = _box_blur_1d(arr.astype(WORK_DTYPE), radius, axis=0)
    out = _box_blur_1d(out, radius, axis=1)
    return out


def gaussian_like_blur(arr: np.ndarray, radius: int, passes: int = 3) -> np.ndarray:
    """Approximates a Gaussian blur with N sequential box blurs (classic
    trick: 3 box blurs ~ 1 Gaussian blur), staying pure numpy."""
    out = arr.astype(WORK_DTYPE)
    for _ in range(passes):
        out = box_blur(out, radius)
    return out


def feather_mask(mask: np.ndarray, feather_px: int = 8) -> np.ndarray:
    """Turns a crisp boolean mask into a 0..1 alpha channel with feathered
    edges, to avoid a visible 'seam' when pasting/deleting a piece of
    terrain. WARNING: cost scales with the size of the array passed in —
    always call this on a SMALL crop around the selection, never on the
    whole canvas (see move_region)."""
    if feather_px <= 0:
        return mask.astype(WORK_DTYPE)
    alpha = gaussian_like_blur(mask.astype(WORK_DTYPE), radius=feather_px, passes=3)
    return np.clip(alpha, 0.0, 1.0)


# ------------------------------------------------------------------------
# Hole filling (inpainting) — iterative Jacobi diffusion
# ------------------------------------------------------------------------

def inpaint_fill(dem: np.ndarray, hole_mask: np.ndarray, iterations: int = 300,
                  tol: float = 1e-3) -> np.ndarray:
    """Fills the pixels in hole_mask by propagating inward the elevation
    of the valid pixels at the hole's edges (average of the 4 neighbors
    at each iteration, anchored to the non-hole pixels, which stay
    fixed). Converges to a smooth surface that blends the edges, no
    sharp 'step'.

    Note: very large holes need more iterations to reach the center (the
    diffusion is local, one iteration propagates by one pixel); for an
    interactive editor on reasonably sized selections (a hill, a tile
    fragment) this is enough and stays pure numpy.
    """
    out = dem.astype(WORK_DTYPE).copy()
    hole = hole_mask.astype(bool)
    if not hole.any():
        return out

    # initialize the hole with the average of the valid edge pixels, to
    # start already close to the solution and converge sooner
    valid = ~hole
    if valid.any():
        out[hole] = out[valid].mean()

    prev = out.copy()
    for _ in range(iterations):
        up = np.roll(out, 1, axis=0)
        down = np.roll(out, -1, axis=0)
        left = np.roll(out, 1, axis=1)
        right = np.roll(out, -1, axis=1)
        avg = (up + down + left + right) / 4.0
        out = np.where(hole, avg, out)  # non-hole pixels stay fixed
        delta = np.abs(out - prev)[hole]
        prev = out
        if delta.size == 0 or delta.max() < tol:
            break
    return out


def _bbox_of_mask(mask: np.ndarray, margin: int, shape):
    """Bounding box (with margin) of the True pixels in mask, clipped to
    the array's bounds. Returns None if the mask is empty."""
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    r0 = max(int(rows.min()) - margin, 0)
    r1 = min(int(rows.max()) + 1 + margin, shape[0])
    c0 = max(int(cols.min()) - margin, 0)
    c1 = min(int(cols.max()) + 1 + margin, shape[1])
    return r0, r1, c0, c1


def rotation_safe_bbox(mask: np.ndarray, feather_px: int, shape, rotation_degrees: float = 0.0):
    """Like selection_bbox, but if rotation_degrees is nonzero it widens
    the box just enough to contain the selection rotated by EXACTLY that
    angle (standard bounding-box-of-a-rotated-rectangle formula), not the
    absolute worst case (a 90° rotation).

    The worst-case margin scales with the selection's DIAGONAL: on a very
    elongated shape (e.g. a chain of islands) it can be huge, and it used
    to be applied even for a small rotation like 5°, where in reality a
    much smaller margin is needed — a real, concrete cause of interface
    freezes encountered in practice."""
    margin_base = max(feather_px * 3, 15)
    if abs(rotation_degrees) < 1e-6:
        return _bbox_of_mask(mask, margin_base, shape)

    tight = _bbox_of_mask(mask, 0, shape)
    if tight is None:
        return None
    r0, r1, c0, c1 = tight
    h_sel, w_sel = r1 - r0, c1 - c0
    hh, hw = h_sel / 2.0, w_sel / 2.0

    # (axis-aligned) bounding box of an hh x hw rectangle rotated by
    # rotation_degrees: standard |cos|+|sin| formula weighted on both axes
    angle_rad = np.deg2rad(rotation_degrees)
    cos_a, sin_a = abs(np.cos(angle_rad)), abs(np.sin(angle_rad))
    new_half_h = hh * cos_a + hw * sin_a
    new_half_w = hw * cos_a + hh * sin_a

    extra = max(new_half_h - hh, new_half_w - hw, 0.0)
    margin = int(np.ceil(extra)) + margin_base
    return _bbox_of_mask(mask, margin, shape)


def _bilinear_sample(arr: np.ndarray, src_y: np.ndarray, src_x: np.ndarray,
                      fill: float = 0.0) -> np.ndarray:
    """Samples 'arr' at (src_y, src_x), non-integer coordinates, with
    bilinear interpolation; returns 'fill' outside the bounds."""
    h, w = arr.shape
    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1

    valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y1, 0, h - 1)

    wx = (src_x - x0).astype(WORK_DTYPE)
    wy = (src_y - y0).astype(WORK_DTYPE)

    top = arr[y0c, x0c] * (1 - wx) + arr[y0c, x1c] * wx
    bot = arr[y1c, x0c] * (1 - wx) + arr[y1c, x1c] * wx
    out = top * (1 - wy) + bot * wy
    return np.where(valid, out, fill).astype(WORK_DTYPE)


def rotate_crop(values: np.ndarray, alpha: np.ndarray, angle_degrees: float,
                 center_yx=None):
    """Rotates 'values' and 'alpha' (same shape) around a center, by
    angle_degrees, with bilinear interpolation (pure numpy, via inverse
    mapping: for each destination pixel, compute where to sample from in
    the source).

    center_yx: rotation center in coordinates LOCAL to the crop (row,
    column). If None, uses the crop's geometric center ((h-1)/2, (w-1)/2)
    — warning: this is correct ONLY if the crop is symmetric around the
    selection. When the selection is near the canvas edge, the crop
    widened for rotation can end up clipped asymmetrically on one side:
    in that case the selection's true center must be passed explicitly
    (see move_region), otherwise the rotation happens around the wrong
    point.

    Convention: a positive angle_degrees rotates CLOCKWISE as displayed
    on screen (a raster's row axis points downward, not upward like in a
    standard Cartesian plane) — the same intuitive convention as a
    rotation angle in a graphics editor.

    Outside the original shape, alpha stays 0 (transparent), so any
    'empty' area that rotates into the crop produces no visible artifacts
    once composited: only where alpha>0 matters."""
    h, w = values.shape
    if center_yx is None:
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    else:
        cy, cx = center_yx
    angle_rad = np.deg2rad(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    yy, xx = np.mgrid[0:h, 0:w].astype(WORK_DTYPE)
    y_rel, x_rel = yy - cy, xx - cx

    # inverse rotation: for a destination pixel (yy,xx), where to sample
    # in the source to get a rotation of +angle
    src_x = cx + x_rel * cos_a + y_rel * sin_a
    src_y = cy - x_rel * sin_a + y_rel * cos_a

    values_r = _bilinear_sample(values, src_y, src_x, fill=0.0)
    alpha_r = _bilinear_sample(alpha, src_y, src_x, fill=0.0)
    alpha_r = np.clip(alpha_r, 0.0, 1.0)
    return values_r, alpha_r


def mask_center(mask: np.ndarray, shape=None):
    """Center (row, column) of the mask's TIGHT bounding box — used as a
    stable rotation center, independent of any asymmetry introduced by a
    wider crop (e.g. near the canvas edge). Returns None if the mask is
    empty."""
    if shape is None:
        shape = mask.shape
    tight = _bbox_of_mask(mask, 0, shape)
    if tight is None:
        return None
    r0, r1, c0, c1 = tight
    return (r0 + r1 - 1) / 2.0, (c0 + c1 - 1) / 2.0


def selection_bbox(mask: np.ndarray, feather_px: int = 6, margin: int = None, shape=None):
    """Public wrapper around _bbox_of_mask: also used outside this module
    (by the plugin) to know WHICH area an operation will touch before
    running it — needed for cheap in-memory undo/redo (saves only the
    affected crop, not the whole canvas)."""
    if margin is None:
        margin = max(feather_px * 3, 15)
    if shape is None:
        shape = mask.shape
    return _bbox_of_mask(mask, margin, shape)


def dst_bbox_from_src(src_bbox, dst_row_offset: int, dst_col_offset: int, shape):
    """Given a move's source bbox and the applied offset, computes the
    destination bbox, clipped to the canvas bounds. Returns None if the
    destination falls completely outside."""
    r0, r1, c0, c1 = src_bbox
    dr0, dc0 = r0 + dst_row_offset, c0 + dst_col_offset
    dr1, dc1 = dr0 + (r1 - r0), dc0 + (c1 - c0)
    dr0, dc0 = max(dr0, 0), max(dc0, 0)
    dr1, dc1 = min(dr1, shape[0]), min(dc1, shape[1])
    if dr1 <= dr0 or dc1 <= dc0:
        return None
    return dr0, dr1, dc0, dc1


def inpaint_fill_windowed(dem: np.ndarray, hole_mask: np.ndarray, iterations: int = 300,
                           tol: float = 1e-3, margin: int = None,
                           feather_px: int = 6) -> np.ndarray:
    """Like inpaint_fill, but operates ONLY on a crop around the hole
    (bounding box + margin) instead of the whole DEM.

    The iterative solver's cost grows with the size of the area it runs
    on: running it on the whole canvas when the selection is small (an
    island on a huge regional mosaic) is needlessly slow — this is
    exactly what used to cause the interface freeze. Here the
    computation is instead limited to a box sized to the selection, so it
    stays fast regardless of how large the loaded DEM is."""
    if margin is None:
        margin = max(feather_px * 3, 15)
    bbox = _bbox_of_mask(hole_mask, margin, dem.shape)
    out = dem.astype(WORK_DTYPE).copy()
    if bbox is None:
        return out
    r0, r1, c0, c1 = bbox
    filled_crop = inpaint_fill(out[r0:r1, c0:c1], hole_mask[r0:r1, c0:c1],
                                iterations=iterations, tol=tol)
    out[r0:r1, c0:c1] = filled_crop
    return out


# ------------------------------------------------------------------------
# Move a selected region elsewhere on the canvas
# ------------------------------------------------------------------------

def move_region(dem: np.ndarray, selection_mask: np.ndarray,
                 dst_row_offset: int, dst_col_offset: int,
                 feather_px: int = 6, sea_level: float = 0.0,
                 inpaint_iterations: int = 300, margin: int = None,
                 rotation_degrees: float = 0.0) -> np.ndarray:
    """Moves (and optionally rotates) the 'dem' pixels selected by
    selection_mask. With dst_row_offset=dst_col_offset=0 and
    rotation_degrees different from zero, rotates the selection IN
    PLACE without moving it — exactly what the plugin's 'Rotate
    selection' button does, reusing this same function.

    The hole left at the source is closed with inpaint_fill; the
    insertion at the destination (rotated or not) uses a feathered edge
    (feather_mask) instead of a hard paste.

    IMPORTANT: every heavy computation (blur, rotation, diffusion)
    operates ONLY on a crop around the selection (bounding box + margin,
    widened just enough not to clip the corners when rotating), never on
    the whole canvas — see rotation_safe_bbox.

    Returns a new array (the original DEM is not modified)."""
    h, w = dem.shape
    out = dem.astype(WORK_DTYPE).copy()

    if margin is not None:
        bbox = _bbox_of_mask(selection_mask, margin, dem.shape)
    else:
        bbox = rotation_safe_bbox(selection_mask, feather_px, dem.shape, rotation_degrees)
    if bbox is None:
        return out  # empty selection, nothing to move/rotate
    r0, r1, c0, c1 = bbox

    # 1) source crop's values and alpha, taken BEFORE closing the hole
    values_crop = out[r0:r1, c0:c1].copy()
    mask_crop = selection_mask[r0:r1, c0:c1]
    alpha_crop = feather_mask(mask_crop, feather_px=feather_px)

    # 2) close the hole left at the source (in-place, on the crop only),
    #    using the ORIGINAL (non-rotated) mask: the hole is where the
    #    land used to be, regardless of where/how it gets relocated
    out[r0:r1, c0:c1] = inpaint_fill(out[r0:r1, c0:c1], mask_crop,
                                      iterations=inpaint_iterations)

    # 2b) rotate the content to be relocated, if requested — around the
    #     TRUE center of the selection (not the crop's geometric center,
    #     which can be asymmetric if one side was clipped by the canvas
    #     edge)
    if abs(rotation_degrees) > 1e-6:
        center = mask_center(mask_crop)
        center_local = None if center is None else center  # already in crop-local coordinates
        values_crop, alpha_crop = rotate_crop(values_crop, alpha_crop, rotation_degrees,
                                               center_yx=center_local)

    # 3) position of the crop at the destination, with reciprocal
    #    clipping if it partly falls outside the canvas bounds
    dr0, dc0 = r0 + dst_row_offset, c0 + dst_col_offset
    dr1, dc1 = dr0 + (r1 - r0), dc0 + (c1 - c0)

    crop_r0, crop_c0 = 0, 0
    if dr0 < 0:
        crop_r0 = -dr0
        dr0 = 0
    if dc0 < 0:
        crop_c0 = -dc0
        dc0 = 0
    crop_r1 = (r1 - r0) - max(dr1 - h, 0)
    crop_c1 = (c1 - c0) - max(dc1 - w, 0)
    dr1 = min(dr1, h)
    dc1 = min(dc1, w)

    if crop_r1 <= crop_r0 or crop_c1 <= crop_c0:
        return out  # destination completely outside the canvas

    v = values_crop[crop_r0:crop_r1, crop_c0:crop_c1]
    a = alpha_crop[crop_r0:crop_r1, crop_c0:crop_c1]

    # 4) alpha-blend compositing, only on the small destination box
    region = out[dr0:dr1, dc0:dc1]
    out[dr0:dr1, dc0:dc1] = region * (1 - a) + v * a
    return out


def delete_region(dem: np.ndarray, selection_mask: np.ndarray,
                   inpaint_iterations: int = 300, feather_px: int = 6) -> np.ndarray:
    """Deletes (flattens) the selected region by filling it via diffusion
    from its edges — no visible hole, no fixed 'stamped-on' value.
    Operates on a local crop (see inpaint_fill_windowed), so it stays
    fast even on a very large working DEM."""
    return inpaint_fill_windowed(dem, selection_mask, iterations=inpaint_iterations,
                                  feather_px=feather_px)


# ------------------------------------------------------------------------
# Elevation -> SC4 grayscale encoding — VALUES AND FORMULA VERIFIED
# AGAINST SC4MAPPER'S REAL SOURCE CODE (app.py, class CreateRgnFromFile
# and method CreateRgnFromGrey):
#   - the 'scale factor' presets are these exact values (the 'scales'
#     dictionary in app.py, function GetImageFactor), not a freely
#     chosen 'meters per level' value as assumed in an earlier version
#     of this plugin;
#   - the conversion formula is LINEAR FROM ZERO, with no sea-level
#     offset subtracted beforehand: elevation_m = grayscale * scale
#     (from app.py, around line 1421 'r * (10*scale)', and from the
#     0.1m precision of the SC4M format documented in doc/readme.html:
#     the internal value is in tenths of a meter, so /10 gives meters);
#   - SC4Mapper's 'sea level' (waterLevel) is hardcoded to 250
#     internally (app.py line 2043) and is NOT editable in the
#     scale-factor selection window: it's used elsewhere (rendering /
#     flooding city edges), not inside this conversion formula.
# ------------------------------------------------------------------------

SC4MAPPER_SCALE_FACTORS = {
    "100m": 1.3725, "250m": 1.9608, "500m": 2.9412, "Default factor": 3.0,
    "1000m": 4.9020, "1500m": 6.8627, "2000m": 8.8235, "import.dat": 9.7832,
    "2500m": 10.7843, "3000m": 12.7451, "3500m": 14.7059, "4000m": 16.6667,
    "4500m": 18.6275, "5000m": 20.5882,
}


def elevation_to_sc4_gray(elev_m: np.ndarray, scale_factor: float,
                           coastal_offset_m: float = 0.0) -> np.ndarray:
    """elevation_m = grayscale * scale_factor (formula verified against
    SC4Mapper's source code) — no sea-level offset subtracted by
    default: grayscale 0 -> elevation 0 m. Negative elevations (below
    sea level) are clipped to 0, since the same formula in SC4Mapper has
    no way to represent them.

    coastal_offset_m: added to elev_m BEFORE encoding, to push low-lying
    coastal land above SC4Mapper's own fixed water-rendering threshold.
    Verified against the source code: SC4Mapper's water/land rendering
    (terrain.py, onePassColors: 'water = H < waterLevel') compares the
    RAW height buffer written into the actual game file
    (region.py, City.Save: 'self.heightMap.tobytes()', no further
    conversion) against a hardcoded waterLevel=250 (app.py, line 2043).
    Since that raw buffer equals grayscale*10*scale_factor (app.py,
    ~line 1421), the water condition reduces to
    'elevation_m < 250/10 = 25', REGARDLESS of the chosen scale_factor:
    any exported land below 25 real meters renders as submerged in
    SC4Mapper (and in the actual saved region, since the raw buffer is
    written unchanged). This is a real, code-verified SC4Mapper
    behavior, not a bug in this formula — coastal_offset_m is an
    optional way to compensate for it if you don't want that effect."""
    gray = np.round((elev_m + coastal_offset_m) / scale_factor)
    return np.clip(gray, 0, 255).astype(np.uint8)


SC4MAPPER_WATER_THRESHOLD_M = 25.0
"""Elevation (in meters, scale-independent) below which SC4Mapper always
renders exported land as underwater — see elevation_to_sc4_gray's
docstring for the exact derivation from the source code."""


def suggest_scale(elev_max_m: float, scale_factor: float,
                   coastal_offset_m: float = 0.0) -> float | None:
    """If the chosen scale factor would clip the peaks (max elevation +
    coastal_offset_m beyond 255 * scale_factor), computes the minimum
    value needed to fit them in 255 levels. Not restricted to the
    presets listed in SC4MAPPER_SCALE_FACTORS: SC4Mapper also accepts a
    hand-typed number instead of a preset (see doc/readme.html), so an
    'in-between' value is still valid — just type the same number into
    SC4Mapper's own window too."""
    effective_max = elev_max_m + coastal_offset_m
    max_representable = 255 * scale_factor
    if effective_max <= max_representable:
        return None
    needed = effective_max / 255.0
    import math
    return math.ceil(needed * 10000) / 10000.0


# ------------------------------------------------------------------------
# Elevation -> OpenTTD grayscale encoding
# ------------------------------------------------------------------------
# OpenTTD convention (different from SC4): black (0) = sea level / lowest
# point, white (255) = highest point of the map — min-max normalization
# between sea level and a maximum elevation, not fixed increments in
# meters. The image's edges must stay at sea level, because OpenTTD
# requires the map's edges to be at elevation 0.

OPENTTD_VALID_SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def elevation_to_openttd_gray(elev_m: np.ndarray, sea_level_m: float,
                               max_elev_m: float) -> np.ndarray:
    """Linear normalization: sea_level_m -> 0 (black), max_elev_m -> 255
    (white). Elevations below sea level are clipped to 0 (OpenTTD does
    not represent bathymetry in a simple heightmap)."""
    span = max(max_elev_m - sea_level_m, 1e-6)
    gray = np.round((elev_m - sea_level_m) / span * 255.0)
    return np.clip(gray, 0, 255).astype(np.uint8)


def force_sea_level_border(gray: np.ndarray, border_px: int = 2) -> np.ndarray:
    """Forces the image's border pixels to 0 (sea level): OpenTTD
    requires the map's edges to be at sea level, otherwise the import can
    behave unexpectedly along the margins."""
    out = gray.copy()
    out[:border_px, :] = 0
    out[-border_px:, :] = 0
    out[:, :border_px] = 0
    out[:, -border_px:] = 0
    return out


def nearest_openttd_size(n: int) -> int:
    """Rounds n to the nearest valid OpenTTD size (a power of 2 between
    64 and 4096)."""
    return min(OPENTTD_VALID_SIZES, key=lambda v: abs(v - n))


def find_optimal_rotation(mask: np.ndarray, angle_step: float = 1.0, max_points: int = 20000):
    """Searches for the rotation angle that minimizes the axis-aligned
    bounding box area of 'mask''s content — useful for aligning a
    diagonal coastline to an export rectangle's edges, reducing wasted
    sea at the corners without manual trial and error.

    IMPORTANT for performance: uses at most max_points points
    (subsampled at regular intervals if the number of land pixels
    exceeds it), not EVERY land pixel. The first version of this
    function worked on every land pixel's coordinates: on a large
    mosaic (all of Calabria can have hundreds of millions of land
    pixels) this meant repeating a huge computation for 180 candidate
    angles — a real, concrete cause of interface freezes encountered in
    practice. To estimate the outline's general shape, a sample of
    ~20,000 points is more than enough: the result doesn't change
    appreciably, only the speed does (from minutes to fractions of a
    second).

    Returns (angle_degrees, bbox_area_at_that_angle). Angle in the same
    clockwise convention as rotate_crop/rotate_full."""
    rows, cols = np.where(mask)
    if rows.size == 0:
        return 0.0, 0
    if rows.size > max_points:
        step = rows.size // max_points
        rows = rows[::step]
        cols = cols[::step]
    cy, cx = rows.mean(), cols.mean()
    y = (rows - cy).astype(np.float64)
    x = (cols - cx).astype(np.float64)

    best_angle, best_area = 0.0, None
    angle = 0.0
    while angle < 180.0:
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        # rotate the points by -angle: equivalent to looking at the
        # content as if it had been rotated by +angle (same convention
        # as rotate_crop)
        rx = x * cos_a + y * sin_a
        ry = -x * sin_a + y * cos_a
        area = (rx.max() - rx.min()) * (ry.max() - ry.min())
        if best_area is None or area < best_area:
            best_area, best_angle = area, angle
        angle += angle_step
    return best_angle, best_area


# ------------------------------------------------------------------------
# Pre-export processing: automatic cropping of excess sea and rotation of
# the whole content, to better align a diagonal coastline to the export
# rectangle (avoids huge empty sea/black corners, typical of a region
# elongated diagonally like Calabria).
# ------------------------------------------------------------------------

def autocrop_to_content(dem: np.ndarray, sea_level: float = 0.0, margin_px: int = 10):
    """Crops 'dem' to the bounding box of the useful content (land above
    sea_level), with a margin. Returns (cropped_array, bbox) — bbox is
    None if there's no land (all sea)."""
    mask = dem > sea_level
    bbox = _bbox_of_mask(mask, margin_px, dem.shape)
    if bbox is None:
        return dem.astype(WORK_DTYPE), None
    r0, r1, c0, c1 = bbox
    return dem[r0:r1, c0:c1].astype(WORK_DTYPE), bbox


def rotate_full(dem: np.ndarray, angle_degrees: float, center_yx=None) -> np.ndarray:
    """Rotates the ENTIRE array, by default around its own geometric
    center (same convention as rotate_crop: positive angle = clockwise
    on screen). Areas that rotate in from outside the original bounds
    are filled with 0 (sea).

    center_yx: if the content isn't centered in the array (e.g. because
    the rotation safety margin got clipped asymmetrically on one side —
    see autocrop_to_content_rotation_safe), the content's TRUE center
    must be passed explicitly, otherwise the rotation happens around the
    wrong point and cuts away real land (real bug encountered: with a
    small angle like -3.9° on content close to the loaded DEM's edge,
    the cut ran halfway across the mainland).

    WARNING: rotates WITHIN the same canvas passed in, without
    enlarging it — if the content (e.g. an elongated shape) extends
    beyond that canvas once rotated, that part is CUT AWAY. Before
    calling this on content that's already tightly cropped
    (autocrop_to_content with margin_px=0), a sufficient margin must be
    added using the same criterion as rotation_safe_bbox — see
    autocrop_to_content_rotation_safe.

    Must be called on an array that's ALREADY CROPPED (see
    autocrop_to_content), not on the whole original mosaic: same memory
    caution already seen for move_region — rotating a huge canvas
    allocates temporary arrays proportional to its size."""
    if abs(angle_degrees) < 1e-6:
        return dem.astype(WORK_DTYPE).copy()
    alpha_dummy = np.ones(dem.shape, dtype=WORK_DTYPE)
    rotated_values, _ = rotate_crop(dem.astype(WORK_DTYPE), alpha_dummy, angle_degrees,
                                     center_yx=center_yx)
    return rotated_values


def rotation_padding_needed(h: int, w: int, rotation_degrees: float) -> int:
    """Margin (in pixels, isotropic) to add to an h x w content before
    rotating it with rotate_full, so as not to cut part of it away —
    same rotated-rectangle bounding-box formula already used in
    rotation_safe_bbox, applied here to the whole content before export
    instead of to a single selection."""
    if abs(rotation_degrees) < 1e-6:
        return 0
    hh, hw = h / 2.0, w / 2.0
    rad = np.deg2rad(rotation_degrees)
    cos_a, sin_a = abs(np.cos(rad)), abs(np.sin(rad))
    new_half_h = hh * cos_a + hw * sin_a
    new_half_w = hw * cos_a + hh * sin_a
    extra = max(new_half_h - hh, new_half_w - hw, 0.0)
    return int(np.ceil(extra))


def autocrop_to_content_rotation_safe(dem: np.ndarray, sea_level: float = 0.0,
                                       rotation_degrees: float = 0.0):
    """Like autocrop_to_content, but if rotation_degrees is nonzero it
    adds the necessary margin (rotation_padding_needed) so as not to cut
    the content away when rotate_full rotates it — ALWAYS use this, not
    autocrop_to_content with margin_px=0, as the first crop before a
    rotation at export time.

    Returns (padded_array, padded_bbox, content_true_center). The true
    center must ALWAYS be passed to rotate_full(..., center_yx=...): if
    the margin ends up clipped asymmetrically on one side (because the
    content touches the original DEM's edge), the padded crop's
    geometric center no longer coincides with the content's true center
    — using the wrong one cuts away real land during rotation (real bug
    encountered and fixed)."""
    if abs(rotation_degrees) < 1e-6:
        cropped, bbox = autocrop_to_content(dem, sea_level, margin_px=0)
        if bbox is None:
            return cropped, None, None
        h, w = cropped.shape
        return cropped, bbox, ((h - 1) / 2.0, (w - 1) / 2.0)

    tight, bbox_tight = autocrop_to_content(dem, sea_level, margin_px=0)
    if bbox_tight is None:
        return tight, None, None
    h1, w1 = tight.shape
    margin = rotation_padding_needed(h1, w1, rotation_degrees)
    padded, bbox_padded = autocrop_to_content(dem, sea_level, margin_px=margin)

    # center of the TIGHT content, re-expressed in coordinates LOCAL to
    # the padded crop (may not be the latter's geometric center, if the
    # margin was clipped asymmetrically)
    tr0, tr1, tc0, tc1 = bbox_tight
    pr0, pr1, pc0, pc1 = bbox_padded
    center_y = (tr0 + tr1 - 1) / 2.0 - pr0
    center_x = (tc0 + tc1 - 1) / 2.0 - pc0
    return padded, bbox_padded, (center_y, center_x)


def apply_border_falloff(dem: np.ndarray, sea_level: float, falloff_px: int) -> np.ndarray:
    """Smoothly brings the terrain down to sea level as it approaches the
    image's 4 edges, over a falloff_px-pixel band (linear fade: 0
    exactly at the edge, 1 starting falloff_px pixels inward). Used to
    avoid the terrain ending in a hard cut against the export's edge —
    especially useful after a tight automatic crop around the content
    (autocrop_to_content), where the selected land's most extreme pixel
    can sit exactly on the image's border."""
    if falloff_px <= 0:
        return dem.astype(WORK_DTYPE).copy()
    h, w = dem.shape
    y = np.arange(h, dtype=WORK_DTYPE)
    x = np.arange(w, dtype=WORK_DTYPE)
    fade_y = np.clip(np.minimum(y, (h - 1) - y) / falloff_px, 0.0, 1.0)
    fade_x = np.clip(np.minimum(x, (w - 1) - x) / falloff_px, 0.0, 1.0)
    fade = np.minimum(fade_y[:, None], fade_x[None, :])
    out = sea_level + (dem.astype(WORK_DTYPE) - sea_level) * fade
    return out


def compute_export_bbox_corners(dem: np.ndarray, sea_level: float = 0.0,
                                 rotation_degrees: float = 0.0, max_points: int = 20000):
    """Computes the 4 corners of the final export rectangle (crop +
    optional rotation), in the PIXEL coordinate system of the ORIGINAL
    array passed in — not of the intermediate crop. Used to draw a
    preview of the result (e.g. as an outline on the QGIS canvas)
    BEFORE actually exporting, without having to generate the file.

    IMPORTANT for performance: with a nonzero rotation, this does NOT
    use rotate_full on the whole crop (which would do a pixel-by-pixel
    bilinear resampling over a potentially huge area — the same class of
    freeze already fixed for 'Find optimal angle'). For a preview, it's
    enough to know WHERE the land pixels end up after rotation, not
    their values: so only the COORDINATES of a sample of land pixels
    (up to max_points) are rotated, an operation orders of magnitude
    lighter. The real export (_prepare_export_array) still uses
    rotate_full for the actual values, where pixel-perfect accuracy
    matters; here only preview speed matters.

    If rotation_degrees is nonzero, the returned rectangle is itself
    rotated (the 4 corners no longer form an axis-aligned rectangle in
    the original coordinate system) — which is exactly what you'd
    expect to see in a preview.

    Order of the returned points: top-left, top-right, bottom-right,
    bottom-left (row, column coordinates). Returns None if there's no
    land."""
    crop1, bbox1 = autocrop_to_content(dem, sea_level, margin_px=0)
    if bbox1 is None:
        return None
    r0, r1, c0, c1 = bbox1
    h1, w1 = r1 - r0, c1 - c0

    if abs(rotation_degrees) < 1e-6:
        corners_local = [(0.0, 0.0), (0.0, float(w1)),
                          (float(h1), float(w1)), (float(h1), 0.0)]
    else:
        mask1 = crop1 > sea_level
        rows, cols = np.where(mask1)
        if rows.size == 0:
            corners_local = [(0.0, 0.0), (0.0, float(w1)),
                              (float(h1), float(w1)), (float(h1), 0.0)]
        else:
            if rows.size > max_points:
                step = rows.size // max_points
                rows = rows[::step]
                cols = cols[::step]
            cy, cx = (h1 - 1) / 2.0, (w1 - 1) / 2.0
            y0 = rows.astype(np.float64) - cy
            x0 = cols.astype(np.float64) - cx

            rad = np.deg2rad(rotation_degrees)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            # FORWARD mapping (same convention as rotate_crop/
            # rotate_full): where each land point ends up after
            # rotation, in the rotated frame relative to the same center
            x1 = x0 * cos_a - y0 * sin_a
            y1 = x0 * sin_a + y0 * cos_a
            r0b, r1b = float(y1.min()), float(y1.max())
            c0b, c1b = float(x1.min()), float(x1.max())

            corners_rot = [(r0b, c0b), (r0b, c1b), (r1b, c1b), (r1b, c0b)]
            # maps these corners (in the ROTATED frame, same origin as
            # crop1) back to crop1's ORIGINAL frame — INVERSE mapping,
            # same sampling formula as rotate_crop
            corners_local = []
            for ry, rx in corners_rot:
                src_x = rx * cos_a + ry * sin_a
                src_y = -rx * sin_a + ry * cos_a
                corners_local.append((src_y + cy, src_x + cx))

    return [(y + r0, x + c0) for y, x in corners_local]
