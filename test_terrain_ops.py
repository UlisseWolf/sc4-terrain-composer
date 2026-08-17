import numpy as np
import sys
# terrain_ops.py has no Qt/QGIS dependency and is byte-identical between
# the qgis3/ and qgis4/ builds (only dock_widget.py and
# sc4_terrain_composer.py differ, for Qt6 enum-scoping reasons — see
# qgis4/README.md) — testing against the qgis3 copy exercises the exact
# same code the qgis4 build ships too.
sys.path.insert(0, "qgis3/sc4_terrain_composer")
from terrain_ops import (
    build_land_mask, feather_mask, box_blur, gaussian_like_blur,
    inpaint_fill, move_region, delete_region,
    elevation_to_sc4_gray, suggest_scale, SC4MAPPER_SCALE_FACTORS,
    SC4MAPPER_WATER_THRESHOLD_M,
    rotate_crop, rotation_safe_bbox, selection_bbox,
    elevation_to_openttd_gray, force_sea_level_border, nearest_openttd_size,
    autocrop_to_content, rotate_full, find_optimal_rotation,
    compute_export_bbox_corners, autocrop_to_content_rotation_safe,
    rotation_padding_needed, prepare_export, transform_points_with_move,
)


def make_test_dem(h=200, w=200):
    """Synthetic DEM: flat sea at 0 with two dome-shaped 'islands'."""
    y, x = np.mgrid[0:h, 0:w]
    dem = np.zeros((h, w), dtype=np.float64)

    def dome(cx, cy, r, peak):
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return np.clip(peak * (1 - d / r), 0, None)

    dem += dome(50, 50, 35, 800)     # island A
    dem += dome(150, 140, 25, 500)   # island B
    return dem


def max_gradient(arr):
    """Max absolute difference between adjacent pixels (proxy for a 'step')."""
    gx = np.abs(np.diff(arr, axis=1))
    gy = np.abs(np.diff(arr, axis=0))
    return max(gx.max(), gy.max())


def test_build_land_mask():
    dem = make_test_dem()
    mask = build_land_mask(dem, sea_level=0.0)
    assert mask.dtype == bool
    assert mask[50, 50] == True   # center of island A: land
    assert mask[0, 0] == False    # corner: sea
    print("OK build_land_mask")


def test_box_blur_preserves_mean():
    arr = np.random.default_rng(0).normal(size=(50, 50))
    blurred = box_blur(arr, radius=5)
    # a blur must not significantly change the overall mean
    assert abs(arr.mean() - blurred.mean()) < 0.05
    assert blurred.shape == arr.shape
    print("OK box_blur (shape unchanged, mean preserved)")


def test_feather_mask_smooth_transition():
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True
    alpha = feather_mask(mask, feather_px=8)
    # at the selection's edges, alpha must be somewhere in between
    # (feathered), not a hard 0/1 jump
    edge_val = alpha[30, 50]
    assert 0.05 < edge_val < 0.95, f"edge not feathered: alpha={edge_val}"
    # at the center it must stay close to 1, far from the edge close to 0
    assert alpha[50, 50] > 0.9
    assert alpha[5, 5] < 0.05
    print(f"OK feather_mask (feathered edge, alpha at boundary={edge_val:.2f})")


def test_inpaint_fill_no_hard_step():
    dem = make_test_dem()
    hole = np.zeros_like(dem, dtype=bool)
    hole[40:60, 40:60] = True  # hole inside island A
    before_max_grad = max_gradient(dem)
    filled = inpaint_fill(dem, hole, iterations=500)
    assert not np.isnan(filled).any()
    # no pixel of the hole should remain at the initial "raw" stepped value
    grad_at_hole_border = max_gradient(filled[35:65, 35:65])
    print(f"OK inpaint_fill (max gradient in the filled area: {grad_at_hole_border:.2f}, "
          f"max original scene gradient: {before_max_grad:.2f})")
    assert grad_at_hole_border < before_max_grad * 1.5


def test_delete_region_removes_bump_smoothly():
    dem = make_test_dem()
    mask = build_land_mask(dem) & (np.arange(dem.shape[1])[None, :] < 100)  # island A only
    result = delete_region(dem, mask, inpaint_iterations=400)
    # island A must disappear (values close to 0, no longer at the peak of 800)
    assert result[50, 50] < 50, f"island not deleted: residual value {result[50,50]:.1f}"
    # island B (outside the selection) must remain untouched
    assert abs(result[140, 150] - dem[140, 150]) < 1e-6
    # no sharp step at the deletion's edges
    grad = max_gradient(result[10:90, 10:90])
    print(f"OK delete_region (island removed, max residual gradient={grad:.2f})")
    assert grad < 60  # much less than the original jump (~800 over a few pixels at a hard edge)


def test_move_region_relocates_island_without_seam():
    dem = make_test_dem()
    mask = np.zeros_like(dem, dtype=bool)
    # select only island A with a generous bbox
    mask[15:85, 15:85] = build_land_mask(dem)[15:85, 15:85]

    moved = move_region(dem, mask, dst_row_offset=0, dst_col_offset=100,
                         feather_px=6, inpaint_iterations=400)

    # 1) the island must have disappeared from its original spot (hole closed)
    assert moved[50, 50] < 50, f"hole not closed at the source: {moved[50,50]:.1f}"

    # 2) and it must appear near the new position (moved by +100 columns)
    assert moved[50, 150] > 300, f"island not found at destination: {moved[50,150]:.1f}"

    # 3) island B, untouched by the selection, must remain intact
    assert abs(moved[140, 150] - dem[140, 150]) < 5.0, "island B altered by mistake"

    # 4) no hard step at either the old or the new position
    grad_src = max_gradient(moved[10:90, 10:90])
    grad_dst = max_gradient(moved[10:90, 110:190])
    print(f"OK move_region (source hole closed, island reappeared at destination, "
          f"source gradient={grad_src:.2f}, destination gradient={grad_dst:.2f})")
    assert grad_src < 60
    assert grad_dst < 250  # the dome has its own slope, but no extra hard step


def test_sc4_encoding_and_scale_suggestion():
    # formula verified against SC4Mapper's real source code:
    # elevation_m = grayscale * scale_factor, no offset, negative
    # elevations clipped to 0 (not representable)
    scale_2000m = SC4MAPPER_SCALE_FACTORS["2000m"]  # 8.8235, the preset from the screenshot
    elev = np.array([-50, 0, 500, 1000, 2000, 2247.5])  # last = 255*8.8235, the exact ceiling
    gray = elevation_to_sc4_gray(elev, scale_2000m)
    assert gray[0] == 0  # negative elevation -> clipped to 0
    assert gray[1] == 0  # elevation 0 -> gray 0 (no offset)
    assert gray[-1] == 255  # exactly at the representable ceiling
    assert gray.min() >= 0 and gray.max() <= 255
    # approximate round-trip: gray*scale should come back close to the original elevation
    reconstructed = gray.astype(np.float64) * scale_2000m
    assert abs(reconstructed[2] - 500) < scale_2000m  # within one gray level

    # scale suggestion when the chosen preset isn't enough
    suggested = suggest_scale(2100, scale_2000m)  # 2100m > 255*8.8235=2250? no, 2100<2250: no suggestion
    none_expected = suggest_scale(2000, scale_2000m)
    real_suggestion = suggest_scale(3000, scale_2000m)  # 3000m > 2250 -> needs a wider scale
    assert none_expected is None
    assert real_suggestion is not None and real_suggestion > scale_2000m
    print(f"OK elevation_to_sc4_gray + suggest_scale (formula verified against SC4Mapper: "
          f"scale '2000m'={scale_2000m}, elevation 500m -> gray {gray[2]}, "
          f"scale suggested for 3000m with insufficient '2000m' preset: {real_suggestion})")


def test_sc4_water_threshold_and_coastal_offset():
    # Verifies the water-rendering threshold discovered in SC4Mapper's
    # source code (terrain.py's onePassColors: 'water = H < waterLevel',
    # with waterLevel=250 hardcoded in app.py and the raw height buffer
    # written unchanged to the actual game file equal to
    # grayscale*10*scale_factor). This reduces, independent of the
    # chosen scale_factor, to: any real elevation below 25m renders as
    # underwater in SC4Mapper. coastal_offset_m is meant to compensate.
    scale = SC4MAPPER_SCALE_FACTORS["2000m"]
    elev = np.array([0.0, 10.0, 24.0, 25.0, 26.0, 100.0])

    gray_no_offset = elevation_to_sc4_gray(elev, scale, coastal_offset_m=0.0)
    reconstructed_no_offset = gray_no_offset.astype(np.float64) * scale
    # without an offset, elevations well below the 25m threshold are
    # still encoded (my formula has no lower bound of its own), but
    # SC4Mapper's own fixed threshold would still show them as sea
    assert reconstructed_no_offset[0] < SC4MAPPER_WATER_THRESHOLD_M

    gray_with_offset = elevation_to_sc4_gray(elev, scale, coastal_offset_m=SC4MAPPER_WATER_THRESHOLD_M)
    reconstructed_with_offset = gray_with_offset.astype(np.float64) * scale
    # with the offset applied, EVERY elevation (including 0m, true sea
    # level) must decode to at least the water threshold, so none of it
    # renders as submerged in SC4Mapper
    assert (reconstructed_with_offset >= SC4MAPPER_WATER_THRESHOLD_M - 1e-6).all()

    print(f"OK SC4 water threshold ({SC4MAPPER_WATER_THRESHOLD_M}m, verified against source): "
          f"without offset, 0m decodes to {reconstructed_no_offset[0]:.1f}m (below threshold, "
          f"would render as sea); with coastal_offset_m={SC4MAPPER_WATER_THRESHOLD_M}, "
          f"0m decodes to {reconstructed_with_offset[0]:.1f}m (at/above threshold)")


def test_prepare_export_no_processing_leaves_points_unchanged():
    dem = np.zeros((200, 200))
    dem[50:150, 50:150] = 300
    arr, pts, info = prepare_export(dem, sea_level=0, rotation_degrees=0, do_autocrop=False,
                                     points_rc=[(100, 100), (5, 5)])
    assert pts == [(100.0, 100.0), (5.0, 5.0)]
    print("OK prepare_export with no processing leaves points unchanged")


def test_prepare_export_crop_only_shifts_and_clips_points():
    dem = np.zeros((200, 200))
    dem[50:150, 50:150] = 300
    arr, pts, info = prepare_export(dem, sea_level=0, rotation_degrees=0, do_autocrop=True,
                                     points_rc=[(100, 100), (0, 0)])
    _, bbox = autocrop_to_content(dem, sea_level=0, margin_px=0)
    r0, c0 = bbox[0], bbox[2]
    assert abs(pts[0][0] - (100 - r0)) < 1e-6 and abs(pts[0][1] - (100 - c0)) < 1e-6
    assert pts[1] is None  # (0,0) falls outside the crop
    print(f"OK prepare_export crop-only (bbox={bbox}, point shifted correctly, "
          f"out-of-bounds point became None)")


def test_prepare_export_marker_matches_real_pixel_after_rotation():
    # the strongest possible check: place a distinguishable marker in the
    # DEM at a known point, run it through prepare_export ALONGSIDE that
    # same point as points_rc, and verify the returned point coincides
    # with where the marker pixel actually ended up in the transformed
    # array — guarantees the point-tracking math stays in sync with the
    # real array transform, not just internally self-consistent
    dem = np.zeros((400, 400))
    dem[150:250, 100:350] = 300.0  # off-center, non-square shape
    marker_row, marker_col = 180, 320
    dem[marker_row, marker_col] = 900.0  # distinguishable marker

    angle = 23.0
    arr, pts, info = prepare_export(dem, sea_level=0, rotation_degrees=angle, do_autocrop=True,
                                     points_rc=[(marker_row, marker_col)])
    tracked = pts[0]
    ry, rx = np.unravel_index(np.argmax(arr), arr.shape)
    dist = np.hypot(tracked[0] - ry, tracked[1] - rx)
    print(f"OK prepare_export marker tracking after {angle}° rotation: "
          f"tracked point={tracked}, real marker pixel=({ry},{rx}), distance={dist:.2f}px")
    assert dist < 2.0


def test_rotate_crop_90_degrees():
    arr = np.zeros((21, 21), dtype=np.float64)
    arr[10, 15] = 100.0  # point to the right of the center (10,10), same row
    alpha = (arr > 0).astype(np.float64)
    rotated_vals, rotated_alpha = rotate_crop(arr, alpha, 90.0)
    # code convention: positive angle = CLOCKWISE rotation as displayed
    # on screen (row axis pointing down) -> a point to the right of the
    # center must end up BELOW the center
    ry, rx = np.unravel_index(np.argmax(rotated_vals), rotated_vals.shape)
    print(f"OK rotate_crop 90° clockwise (point moved from (10,15) to ({ry},{rx}), expected close to (15,10))")
    assert abs(ry - 15) <= 1 and abs(rx - 10) <= 1


def test_rotate_crop_360_returns_close_to_original():
    rng = np.random.default_rng(1)
    arr = rng.normal(size=(40, 40)).astype(np.float64)
    alpha = np.ones_like(arr)
    rotated_vals, _ = rotate_crop(arr, alpha, 360.0)
    diff = np.abs(rotated_vals[5:-5, 5:-5] - arr[5:-5, 5:-5]).max()
    print(f"OK rotate_crop 360° (max difference at the center: {diff:.4f}, expected close to 0)")
    assert diff < 0.5


def test_rotation_safe_bbox_bigger_than_normal_when_rotating():
    mask = np.zeros((200, 200), dtype=bool)
    mask[80:120, 60:140] = True  # non-square rectangle, 40x80
    bbox_norm = selection_bbox(mask, feather_px=6, shape=mask.shape)
    bbox_rot = rotation_safe_bbox(mask, feather_px=6, shape=mask.shape, rotation_degrees=45)
    area_norm = (bbox_norm[1]-bbox_norm[0]) * (bbox_norm[3]-bbox_norm[2])
    area_rot = (bbox_rot[1]-bbox_rot[0]) * (bbox_rot[3]-bbox_rot[2])
    print(f"OK rotation_safe_bbox (normal area={area_norm}, area with rotation margin={area_rot})")
    assert area_rot > area_norm


def test_rotation_safe_bbox_scales_with_angle_not_worst_case():
    # VERY elongated selection (like a chain of islands): the old code
    # always used the worst-case margin (90°) for ANY nonzero angle — a
    # real, concrete cause of freezes encountered with small rotations
    # (5°) on elongated selections
    mask = np.zeros((6000, 6000), dtype=bool)
    mask[2750:3250, 500:5500] = True  # 500 x 5000 px

    bbox_5 = rotation_safe_bbox(mask, feather_px=6, shape=mask.shape, rotation_degrees=5)
    bbox_90 = rotation_safe_bbox(mask, feather_px=6, shape=mask.shape, rotation_degrees=90)
    area_5 = (bbox_5[1]-bbox_5[0]) * (bbox_5[3]-bbox_5[2])
    area_90 = (bbox_90[1]-bbox_90[0]) * (bbox_90[3]-bbox_90[2])
    print(f"OK rotation_safe_bbox scales with the angle (area at 5°={area_5:,}px, "
          f"area at 90°={area_90:,}px, ratio={area_90/area_5:.1f}x)")
    assert area_5 < area_90 * 0.3  # at 5° it must be noticeably smaller than at 90°


def test_rotate_small_angle_on_elongated_selection_no_clipping():
    # GEOMETRIC verification (not "how much mass is left", which would
    # confuse a real cut with the intentional feathering attenuation at
    # the edges): the rotated selection's corners must fall INSIDE the
    # box computed by rotation_safe_bbox, for any angle.
    h_sel, w_sel = 500, 5000
    mask = np.zeros((6000, 6000), dtype=bool)
    r0, c0 = 2750, 500
    mask[r0:r0+h_sel, c0:c0+w_sel] = True

    cy, cx = r0 + h_sel/2.0, c0 + w_sel/2.0
    corners = [(-h_sel/2, -w_sel/2), (-h_sel/2, w_sel/2),
               (h_sel/2, -w_sel/2), (h_sel/2, w_sel/2)]

    for angle in [5, 15, 45, 90]:
        bbox = rotation_safe_bbox(mask, feather_px=4, shape=mask.shape, rotation_degrees=angle)
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        all_inside = True
        for dy, dx in corners:
            # same convention as rotate_crop (clockwise on screen)
            rot_y = cy + (-dx * sin_a + dy * cos_a)
            rot_x = cx + (dx * cos_a + dy * sin_a)
            if not (bbox[0] <= rot_y <= bbox[1] and bbox[2] <= rot_x <= bbox[3]):
                all_inside = False
        assert all_inside, f"angle {angle}°: a corner of the rotated shape falls outside the bbox {bbox}"
    print("OK no geometric clipping for an elongated selection at 5°/15°/45°/90° "
          "(rotated shape's corners always inside the computed box)")


def test_transform_points_with_move_pure_translation():
    mask = np.zeros((200, 200), dtype=bool)
    mask[50:100, 60:140] = True
    pts = [(70, 90), (10, 10)]  # first INSIDE the selection, second OUTSIDE
    out = transform_points_with_move(pts, mask, dst_row_offset=30, dst_col_offset=-20,
                                      rotation_degrees=0.0)
    assert out[0] == (100, 70)
    assert out[1] == (10, 10)  # untouched: not part of the moved selection
    print("OK transform_points_with_move (pure translation): point inside moved, "
          "point outside left untouched")


def test_transform_points_with_move_matches_real_rotated_pixel():
    # strongest possible check, same principle as prepare_export's marker
    # test: place a marker in the RASTER at a known point, rotate it with
    # move_region, and verify a VECTOR point at that same original
    # location ends up exactly where the real marker pixel landed
    mask = np.zeros((200, 200), dtype=bool)
    mask[50:100, 60:140] = True
    dem = np.zeros((200, 200))
    dem[50:100, 60:140] = 300.0
    dem[70, 135] = 900.0

    angle = 40.0
    rotated = move_region(dem, mask, dst_row_offset=0, dst_col_offset=0, rotation_degrees=angle,
                           feather_px=4, inpaint_iterations=300)
    ry, rx = np.unravel_index(np.argmax(rotated), rotated.shape)

    out = transform_points_with_move([(70, 135)], mask, dst_row_offset=0, dst_col_offset=0,
                                      rotation_degrees=angle)
    tracked = out[0]
    dist = np.hypot(tracked[0] - ry, tracked[1] - rx)
    print(f"OK transform_points_with_move matches the real rotated raster pixel "
          f"(tracked={tracked}, real marker=({ry},{rx}), distance={dist:.2f}px)")
    assert dist < 2.0


def test_move_region_with_rotation_in_place():
    dem = make_test_dem()
    # island A isn't symmetric relative to its own tight bbox: add a
    # small asymmetric bump so we can verify the rotation actually has
    # an effect (otherwise a rotated circle would be indistinguishable)
    dem2 = dem.copy()
    dem2[35, 70] += 400  # bump to the right of island A

    mask = np.zeros_like(dem2, dtype=bool)
    mask[10:90, 10:90] = build_land_mask(dem2)[10:90, 10:90]

    rotated_in_place = move_region(dem2, mask, dst_row_offset=0, dst_col_offset=0,
                                    feather_px=4, rotation_degrees=90, inpaint_iterations=400)

    # the bump must have moved (no longer at the same spot) and the
    # island must have stayed roughly in the same area (center unmoved,
    # since it's an in-place rotation)
    assert rotated_in_place[35, 70] < 400  # the bump is no longer there
    assert rotated_in_place[50, 50] > 300  # the island's body is still centered
    print("OK move_region with rotation_degrees=90 in place "
          "(bump moved, island body still centered)")


def test_find_optimal_rotation_recovers_known_angle():
    h, w = 1200, 1200
    dem = np.zeros((h, w), dtype=np.float64)
    dem[585:615, 350:850] = 500.0  # 30x500 axis-aligned rectangle

    true_angle = 30.0
    rotated_dem = rotate_full(dem, true_angle)
    mask = rotated_dem > 100

    found_angle, area_found = find_optimal_rotation(mask, angle_step=1.0)
    bbox0 = selection_bbox(mask, feather_px=0, margin=0, shape=mask.shape)
    area_0 = (bbox0[1] - bbox0[0]) * (bbox0[3] - bbox0[2])

    print(f"OK find_optimal_rotation: built at {true_angle}°, found {found_angle}°, "
          f"area reduction {100*(1-area_found/area_0):.0f}% compared to no rotation")
    assert abs(found_angle - true_angle) <= 1.0
    assert area_found < area_0 * 0.5


def test_find_optimal_rotation_fast_on_huge_mask():
    import time
    # Simulates a huge DEM with A LOT of land pixels (tens of millions),
    # like all of Calabria at full resolution — exactly the scenario
    # that used to freeze the interface. Using an ASYMMETRIC shape (a
    # long strip + a wider block at one end, an 'L' shape) instead of a
    # plain rectangle: a pure rectangle has TWO equally optimal
    # orientations (the true angle and true angle+90°, because swapping
    # the long/short side gives the same bounding box) — an 'L' shape
    # breaks this symmetry and has a single truly best angle, more
    # representative of a real coastline.
    h, w = 6000, 6000
    dem = np.zeros((h, w), dtype=np.float64)
    true_angle = 22.0
    dem[2900:3100, 500:5500] = 500.0   # long strip (~10 million px)
    dem[2900:3600, 5200:5500] = 500.0  # perpendicular block at one end
    rotated_dem = rotate_full(dem, true_angle)
    mask = rotated_dem > 100
    n_land = int(mask.sum())

    t0 = time.time()
    found_angle, _ = find_optimal_rotation(mask, angle_step=1.0, max_points=20000)
    dt = time.time() - t0

    print(f"OK find_optimal_rotation on {n_land:,} land pixels: {dt:.2f}s "
          f"(angle found: {found_angle}°, expected ~{true_angle}°)")
    assert dt < 3.0, f"too slow: {dt:.2f}s for {n_land:,} land pixels"
    assert abs(found_angle - true_angle) <= 2.0  # slightly wider tolerance (subsampled)


def test_save_uint8_bmp_readable_by_pillow():
    # The BMP is now written by hand in raster_io.py (no more GDAL
    # driver, suspected cause of 'This is not a valid file' in
    # SC4Mapper, which opens the file with Pillow — see app.py,
    # OnBrowseFile). Verifies with REAL Pillow that the file opens and
    # that the pixels come back exact, including edge cases (0, 255) and
    # a width that is NOT a multiple of 4 (tests the BMP format's row
    # padding).
    import sys, types, os
    if "osgeo" not in sys.modules:
        fake_osgeo = types.ModuleType("osgeo")
        fake_gdal = types.ModuleType("osgeo.gdal")
        fake_gdal.UseExceptions = lambda: None
        fake_gdal.GDT_Float32 = 6
        fake_gdal.GDT_Byte = 1
        fake_gdal.GA_ReadOnly = 0
        fake_gdal.GRIORA_Bilinear = 1
        fake_osr = types.ModuleType("osgeo.osr")
        fake_osgeo.gdal = fake_gdal
        fake_osgeo.osr = fake_osr
        sys.modules["osgeo"] = fake_osgeo
        sys.modules["osgeo.gdal"] = fake_gdal
        sys.modules["osgeo.osr"] = fake_osr

    sys.path.insert(0, "qgis3/sc4_terrain_composer")
    from raster_io import save_uint8_bmp
    from PIL import Image

    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(300, 517), dtype=np.uint8)  # width not a multiple of 4
    arr[0, 0] = 0
    arr[-1, -1] = 255
    arr[150, 250] = 128

    out_path = "_test_sc4_export.bmp"
    try:
        save_uint8_bmp(arr, out_path)
        im = Image.open(out_path)  # must open WITHOUT exceptions, just like SC4Mapper does
        arr_back = np.array(im.convert("L"))
        assert arr_back.shape == arr.shape
        assert np.array_equal(arr, arr_back)
        print(f"OK save_uint8_bmp openable with Pillow, exact pixel round-trip "
              f"(mode={im.mode}, size={im.size})")
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def test_estimate_utm_epsg_global_cases():
    import sys, types
    # raster_io.py imports osgeo.gdal, not available in this test
    # sandbox: we insert a fake stub into sys.modules so the import
    # succeeds and we test the REAL function from raster_io.py (which
    # doesn't actually call any GDAL function), not a duplicated copy of
    # the logic.
    if "osgeo" not in sys.modules:
        fake_osgeo = types.ModuleType("osgeo")
        fake_gdal = types.ModuleType("osgeo.gdal")
        fake_gdal.UseExceptions = lambda: None
        # constants used as default parameters at class/function
        # definition time in raster_io.py: any placeholder is enough,
        # they're never actually executed in this test (we don't call
        # GDAL, only estimate_utm_epsg, which is pure math)
        fake_gdal.GDT_Float32 = 6
        fake_gdal.GDT_Byte = 1
        fake_gdal.GA_ReadOnly = 0
        fake_gdal.GRIORA_Bilinear = 1
        fake_osr = types.ModuleType("osgeo.osr")
        fake_osgeo.gdal = fake_gdal
        fake_osgeo.osr = fake_osr
        sys.modules["osgeo"] = fake_osgeo
        sys.modules["osgeo.gdal"] = fake_gdal
        sys.modules["osgeo.osr"] = fake_osr

    sys.path.insert(0, "qgis3/sc4_terrain_composer")
    from raster_io import estimate_utm_epsg
    cases = [
        ((16.6, 38.1), 32633, "Calabria"),
        ((-0.1, 51.5), 32630, "London"),
        ((151.2, -33.9), 32756, "Sydney"),
        ((-74.0, 40.7), 32618, "New York"),
    ]
    for (lon, lat), expected, label in cases:
        result = estimate_utm_epsg(lon, lat)
        assert result == expected, f"{label}: expected {expected}, got {result}"
    print("OK estimate_utm_epsg on global cases (Calabria, London, Sydney, New York) "
          "[real function from raster_io.py, imported with an osgeo stub]")


def test_openttd_encoding():
    elev = np.array([-50, 0, 500, 1000, 2000], dtype=np.float64)
    gray = elevation_to_openttd_gray(elev, sea_level_m=0, max_elev_m=1000)
    # below/at sea level -> 0 (black); at the max elevation -> 255 (white)
    assert gray[0] == 0 and gray[1] == 0
    assert gray[3] == 255
    assert gray[4] == 255  # beyond the max -> clipped to 255, not truncated/overflowed
    assert gray.min() >= 0 and gray.max() <= 255
    print(f"OK elevation_to_openttd_gray: {elev.tolist()} -> {gray.tolist()}")


def test_openttd_border_forced_to_sea_level():
    gray = np.full((20, 20), 200, dtype=np.uint8)
    bordered = force_sea_level_border(gray, border_px=2)
    assert (bordered[:2, :] == 0).all()
    assert (bordered[-2:, :] == 0).all()
    assert (bordered[:, :2] == 0).all()
    assert (bordered[:, -2:] == 0).all()
    assert bordered[10, 10] == 200  # the center must not be touched
    print("OK force_sea_level_border (border at 0, center untouched)")


def test_nearest_openttd_size():
    assert nearest_openttd_size(1000) == 1024
    assert nearest_openttd_size(70) == 64
    assert nearest_openttd_size(3000) in (2048, 4096)
    print("OK nearest_openttd_size")


def test_autocrop_to_content_removes_empty_sea():
    # huge canvas (proportionally) with a small piece of land in a
    # corner, similar to a diagonal region lost in a big sea rectangle
    dem = np.zeros((500, 500), dtype=np.float64)
    dem[50:100, 60:120] = 300.0  # small island in a corner
    cropped, bbox = autocrop_to_content(dem, sea_level=0.0, margin_px=10)
    print(f"OK autocrop_to_content: canvas {dem.shape} -> crop {cropped.shape} (bbox={bbox})")
    assert cropped.shape[0] < 200 and cropped.shape[1] < 200  # much smaller than the canvas
    assert cropped.max() == 300.0  # the useful content wasn't lost


def test_autocrop_to_content_empty_returns_unchanged():
    dem = np.zeros((50, 50), dtype=np.float64)  # all sea, no land
    cropped, bbox = autocrop_to_content(dem, sea_level=0.0, margin_px=10)
    assert bbox is None
    assert cropped.shape == dem.shape
    print("OK autocrop_to_content on an all-sea area (no crop, no crash)")


def test_rotate_full_reduces_bbox_of_diagonal_strip():
    # diagonal strip, like the Calabrian coastline in a north-aligned
    # canvas: the tight bbox takes up almost the whole canvas until it's rotated
    n = 200
    dem = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = i  # diagonal i==j
        dem[max(i-3,0):i+3, max(j-3,0):j+3] = 500.0
    _, bbox_before = autocrop_to_content(dem, 0.0, margin_px=0)
    area_before = (bbox_before[1]-bbox_before[0]) * (bbox_before[3]-bbox_before[2])

    rotated = rotate_full(dem, 45.0)  # aligns the diagonal ~horizontally
    _, bbox_after = autocrop_to_content(rotated, 0.0, margin_px=0)
    area_after = (bbox_after[1]-bbox_after[0]) * (bbox_after[3]-bbox_after[2])

    print(f"OK rotate_full (useful area before={area_before}px, after 45° rotation={area_after}px, "
          f"reduction={100*(1-area_after/area_before):.0f}%)")
    assert area_after < area_before * 0.6  # substantial reduction of the waste


def test_compute_export_bbox_corners_no_rotation_matches_autocrop():
    dem = np.zeros((300, 400), dtype=np.float64)
    dem[50:200, 80:350] = 300.0  # rectangular block of land, not centered

    corners = compute_export_bbox_corners(dem, sea_level=0.0, rotation_degrees=0.0)
    cropped, bbox = autocrop_to_content(dem, sea_level=0.0, margin_px=0)
    r0, r1, c0, c1 = bbox

    expected = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
    for (y, x), (ey, ex) in zip(corners, expected):
        assert abs(y - ey) < 1e-6 and abs(x - ex) < 1e-6
    print(f"OK compute_export_bbox_corners with no rotation matches the plain crop {bbox}")


def test_compute_export_bbox_corners_with_rotation_is_tilted_quad():
    dem = np.zeros((400, 400), dtype=np.float64)
    dem[150:250, 50:350] = 300.0  # horizontal strip

    corners = compute_export_bbox_corners(dem, sea_level=0.0, rotation_degrees=20.0)
    assert corners is not None and len(corners) == 4

    # with rotation, the quadrilateral must NOT be axis-aligned: the
    # sides aren't parallel to the axes (otherwise we wouldn't really be
    # showing a rotation in the preview)
    (y0, x0), (y1, x1), (y2, x2), (y3, x3) = corners
    top_edge_not_horizontal = abs(y0 - y1) > 1.0
    print(f"OK compute_export_bbox_corners with rotation produces a tilted "
          f"quadrilateral (top-edge height difference: {abs(y0-y1):.1f}px)")
    assert top_edge_not_horizontal


def test_compute_export_bbox_corners_fast_on_huge_content():
    import time
    # same L-shape used for the find_optimal_rotation test: large enough
    # to make rotate_full (the old approach) slow
    h, w = 6000, 6000
    dem = np.zeros((h, w), dtype=np.float64)
    dem[2900:3100, 500:5500] = 500.0
    dem[2900:3600, 5200:5500] = 500.0

    t0 = time.time()
    corners = compute_export_bbox_corners(dem, sea_level=0.0, rotation_degrees=22.0)
    dt = time.time() - t0
    print(f"OK compute_export_bbox_corners on large content: {dt:.2f}s "
          f"(before, with rotate_full on the whole crop, it would have been much slower)")
    assert corners is not None
    assert dt < 3.0, f"too slow: {dt:.2f}s"


def _shoelace_area(corners):
    """Area of a polygon from its vertices (shoelace formula) — used in
    the tests to correctly compare the area of a ROTATED quadrilateral
    (whose 4 corners don't form an axis-aligned rectangle) against a
    reference area. Using (max-min) on rotated corners would instead
    compute the AXIS-ALIGNED bounding box area of those points, which is
    always larger than the tilted quadrilateral's true area — a
    conceptual mistake that made an earlier test below fail, not a bug
    in the code."""
    n = len(corners)
    s = 0.0
    for i in range(n):
        y1, x1 = corners[i]
        y2, x2 = corners[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def test_compute_export_bbox_corners_matches_exact_rotate_full():
    # direct comparison against the "exact" method (actually rotates the
    # array), which to be correct must use
    # autocrop_to_content_rotation_safe (with margin) BEFORE rotating —
    # otherwise rotate_full would cut away part of the content (real bug
    # discovered while writing this very comparison: without the margin,
    # up to 20% of the mass was lost). With the correct margin, the fast
    # method (point-based) and the exact one must come out close —
    # comparing the quadrilateral's TRUE area (shoelace), not the
    # axis-aligned bounding box of its 4 corners (which would always
    # overestimate, being tilted).
    dem = np.zeros((700, 700), dtype=np.float64)
    dem[300:400, 150:550] = 400.0
    dem[300:500, 480:550] = 400.0  # L-shape, no 90° symmetry

    angle = 17.0
    fast_corners = compute_export_bbox_corners(dem, sea_level=0.0, rotation_degrees=angle)
    fast_area = _shoelace_area(fast_corners)

    padded, bbox1, center_yx = autocrop_to_content_rotation_safe(dem, sea_level=0.0, rotation_degrees=angle)
    rotated_exact = rotate_full(padded, angle, center_yx=center_yx)
    _, bbox2_exact = autocrop_to_content(rotated_exact, sea_level=0.0, margin_px=0)
    exact_area = (bbox2_exact[1]-bbox2_exact[0]) * (bbox2_exact[3]-bbox2_exact[2])

    diff_pct = 100 * abs(fast_area - exact_area) / exact_area
    print(f"OK compute_export_bbox_corners (fast) vs rotate_full with the correct "
          f"margin (exact): exact area={exact_area:.0f}px, "
          f"fast quadrilateral's true area={fast_area:.0f}px, "
          f"difference {diff_pct:.1f}%")
    assert diff_pct < 10.0  # within 10%, more than enough for a preview


def test_rotate_full_without_padding_clips_content():
    # explicitly documents the discovered bug: rotate_full on a crop
    # WITHOUT a margin loses content. Meant to prevent someone from
    # accidentally removing the safety margin in _prepare_export_array
    # in the future.
    dem = np.zeros((500, 500), dtype=np.float64)
    dem[200:300, 50:450] = 400.0
    dem[200:400, 380:450] = 400.0

    tight, _ = autocrop_to_content(dem, sea_level=0.0, margin_px=0)
    rotated_unsafe = rotate_full(tight, 17.0)
    mass_lost_pct = 100 * (1 - rotated_unsafe.sum() / tight.sum())
    print(f"OK confirmed: rotate_full WITHOUT a margin loses {mass_lost_pct:.0f}% "
          f"of the mass (which is why _prepare_export_array always uses "
          f"autocrop_to_content_rotation_safe before rotating)")
    assert mass_lost_pct > 5  # confirms the issue is real and not negligible

    padded, _, center_yx = autocrop_to_content_rotation_safe(dem, sea_level=0.0, rotation_degrees=17.0)
    rotated_safe = rotate_full(padded, 17.0, center_yx=center_yx)
    mass_lost_safe_pct = 100 * (1 - rotated_safe.sum() / padded.sum())
    print(f"OK with the safety margin, the loss drops to {mass_lost_safe_pct:.1f}%")
    assert mass_lost_safe_pct < 1


def test_rotate_full_center_correct_when_content_near_edge():
    # reproduces EXACTLY the scenario of the bug reported in practice: a
    # small angle (here 4°, like the -3.9° encountered) on content that
    # touches the original DEM's edge, such that the rotation safety
    # margin ends up clipped ASYMMETRICALLY on one side. Verified via
    # translation invariance: the same shape, rotated by the same angle,
    # must give the same result (same mass, same dimensions) whether it
    # sits comfortably at the center of a large DEM or is pushed against
    # the edge of a smaller one — if the rotation center were wrong for
    # the 'near the edge' case, the result would differ (real land cut
    # away).
    shape_h, shape_w = 120, 300
    angle = 4.0

    def make_dem(canvas_h, canvas_w, row0, col0):
        d = np.zeros((canvas_h, canvas_w), dtype=np.float64)
        d[row0:row0+shape_h, col0:col0+shape_w] = 500.0
        d[row0:row0+shape_h, col0+shape_w-40:col0+shape_w] = 500.0  # small bump, breaks the 90° symmetry
        d[row0+shape_h:row0+shape_h+60, col0+shape_w-40:col0+shape_w] = 500.0
        return d

    # case A: content comfortably at the center of a large DEM (no margin clipping)
    dem_centered = make_dem(800, 800, 340, 250)
    padded_c, _, center_c = autocrop_to_content_rotation_safe(dem_centered, 0.0, angle)
    rotated_c = rotate_full(padded_c, angle, center_yx=center_c)
    mass_centered = rotated_c[rotated_c > 100].sum()

    # case B: the SAME shape, but pushed against the left/top edge of a
    # smaller DEM, such that the margin gets clipped asymmetrically
    dem_edge = make_dem(220, 340, 2, 2)
    padded_e, _, center_e = autocrop_to_content_rotation_safe(dem_edge, 0.0, angle)
    rotated_e = rotate_full(padded_e, angle, center_yx=center_e)
    mass_edge = rotated_e[rotated_e > 100].sum()

    diff_pct = 100 * abs(mass_centered - mass_edge) / mass_centered
    print(f"OK rotate_full with the correct center: mass at center={mass_centered:.0f}, "
          f"mass near the edge={mass_edge:.0f}, difference {diff_pct:.1f}% "
          f"(expected close to 0: same result regardless of position)")
    assert diff_pct < 2.0


if __name__ == "__main__":
    test_build_land_mask()
    test_box_blur_preserves_mean()
    test_feather_mask_smooth_transition()
    test_inpaint_fill_no_hard_step()
    test_delete_region_removes_bump_smoothly()
    test_move_region_relocates_island_without_seam()
    test_sc4_encoding_and_scale_suggestion()
    test_sc4_water_threshold_and_coastal_offset()
    test_prepare_export_no_processing_leaves_points_unchanged()
    test_prepare_export_crop_only_shifts_and_clips_points()
    test_prepare_export_marker_matches_real_pixel_after_rotation()
    test_transform_points_with_move_pure_translation()
    test_transform_points_with_move_matches_real_rotated_pixel()
    test_rotate_crop_90_degrees()
    test_rotate_crop_360_returns_close_to_original()
    test_rotation_safe_bbox_bigger_than_normal_when_rotating()
    test_rotation_safe_bbox_scales_with_angle_not_worst_case()
    test_rotate_small_angle_on_elongated_selection_no_clipping()
    test_move_region_with_rotation_in_place()
    test_openttd_encoding()
    test_openttd_border_forced_to_sea_level()
    test_nearest_openttd_size()
    test_autocrop_to_content_removes_empty_sea()
    test_autocrop_to_content_empty_returns_unchanged()
    test_rotate_full_reduces_bbox_of_diagonal_strip()
    test_find_optimal_rotation_recovers_known_angle()
    test_find_optimal_rotation_fast_on_huge_mask()
    test_compute_export_bbox_corners_no_rotation_matches_autocrop()
    test_compute_export_bbox_corners_with_rotation_is_tilted_quad()
    test_compute_export_bbox_corners_fast_on_huge_content()
    test_compute_export_bbox_corners_matches_exact_rotate_full()
    test_rotate_full_without_padding_clips_content()
    test_rotate_full_center_correct_when_content_near_edge()
    test_save_uint8_bmp_readable_by_pillow()
    test_estimate_utm_epsg_global_cases()
    print("\nALL TESTS PASSED")
