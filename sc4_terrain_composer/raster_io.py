# -*- coding: utf-8 -*-
"""
raster_io.py
================================================================================
Raster read/write via GDAL (osgeo.gdal), which is already included in
QGIS's Python installation: no extra dependency to install, unlike the
standalone script that used rasterio.
================================================================================
"""

from osgeo import gdal, osr
import numpy as np
import struct

gdal.UseExceptions()


def estimate_utm_epsg(centroid_lon: float, centroid_lat: float) -> int:
    """Computes the best-fit UTM EPSG code (metric projection) for a
    WGS84 lon/lat point — pure math, no GDAL dependency.

    Used to guarantee that the final mosaic is ALWAYS in a metric
    projection, regardless of the source DEMs' native CRS: some agencies
    (Copernicus DEM, SRTM) distribute tiles in geographic coordinates
    (degrees, EPSG:4326) instead of already projected. Working in degrees
    would make all the plugin's 'in pixel' parameters (feather, crop)
    inconsistent between the X and Y axes, because a degree of longitude
    doesn't measure the same as a degree of latitude (except at the
    equator)."""
    zone = int((centroid_lon + 180) / 6) % 60 + 1
    if centroid_lat >= 0:
        return 32600 + zone
    return 32700 + zone


class RasterHandle:
    """Small wrapper around a GeoTIFF: holds a numpy array + georeferencing,
    and knows how to rewrite itself to disk so QGIS can reload the layer."""

    def __init__(self, path: str, nodata_mode: str = "sea",
                 fill_max_search_dist: int = 100):
        """nodata_mode:
        - 'sea': no-data pixels become 0m (correct for DEMs that cover
          only dry land and use no-data for the sea, like TINITALY or
          many national DEMs).
        - 'interpolate': no-data pixels are filled with the native GDAL
          FillNodata algorithm (inverse-distance interpolation from
          nearby valid pixels, searching ONLY locally around each hole up
          to fill_max_search_dist pixels — not a whole-canvas operation,
          so the cost depends on the size of the holes, not the size of
          the mosaic). More correct for global DEMs like SRTM/ASTER,
          where no-data represents real gaps in the data even inland,
          not the sea."""
        self.path = path
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            raise IOError(f"Could not open the raster: {path}")
        band = ds.GetRasterBand(1)
        self.nodata = band.GetNoDataValue()
        self.n_nodata_filled = 0
        self.n_nodata_left_as_sea = 0

        scale = band.GetScale()
        offset = band.GetOffset()
        scale = 1.0 if not scale else scale
        offset = 0.0 if not offset else offset

        if nodata_mode == "interpolate" and self.nodata is not None:
            # FillNodata needs a writable band: work on an in-memory copy
            # (MEM driver), the original file is never touched
            mem_ds = gdal.GetDriverByName("MEM").CreateCopy("", ds)
            mem_band = mem_ds.GetRasterBand(1)
            raw_before = mem_band.ReadAsArray()
            hole_mask = (raw_before == self.nodata)
            self.n_nodata_filled = int(hole_mask.sum())
            if self.n_nodata_filled > 0:
                gdal.FillNodata(targetBand=mem_band, maskBand=None,
                                 maxSearchDist=fill_max_search_dist,
                                 smoothingIterations=0)
            raw = mem_band.ReadAsArray().astype(np.float32)
            mem_ds = None
            self.nodata_mask = None

            # safety net: a hole wider than fill_max_search_dist might
            # not have been reached by FillNodata and could remain at the
            # original no-data sentinel value (e.g. -32767) — if left as
            # is, that absurd value would corrupt any later calculation
            # (min/max, SC4/OpenTTD encoding, etc.). Fall back to 0m
            # (sea) only for these leftovers, not for the whole hole.
            still_nodata = (raw == self.nodata)
            if still_nodata.any():
                self.n_nodata_left_as_sea = int(still_nodata.sum())
                raw[still_nodata] = 0.0
        else:
            raw = band.ReadAsArray().astype(np.float32)
            # the no-data mask must be computed on the RAW values (before
            # scale/offset): that's how GDAL defines the no-data value
            self.nodata_mask = (raw == self.nodata) if self.nodata is not None else None
            if self.nodata_mask is not None and nodata_mode == "sea":
                raw[self.nodata_mask] = 0.0
                self.nodata_mask = None

        if scale != 1.0 or offset != 0.0:
            raw = raw * np.float32(scale) + np.float32(offset)

        self.array = raw
        self.geotransform = ds.GetGeoTransform()
        self.projection = ds.GetProjection()
        self.width = ds.RasterXSize
        self.height = ds.RasterYSize
        ds = None

    def pixel_to_geo(self, col: float, row: float):
        gt = self.geotransform
        x = gt[0] + col * gt[1] + row * gt[2]
        y = gt[3] + col * gt[4] + row * gt[5]
        return x, y

    def geo_to_pixel(self, x: float, y: float):
        gt = self.geotransform
        det = gt[1] * gt[5] - gt[2] * gt[4]
        col = (gt[5] * (x - gt[0]) - gt[2] * (y - gt[3])) / det
        row = (gt[1] * (y - gt[3]) - gt[4] * (x - gt[0])) / det
        return col, row

    def save_as(self, out_path: str, array: np.ndarray = None, dtype=gdal.GDT_Float32):
        arr = self.array if array is None else array
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(out_path, self.width, self.height, 1, dtype)
        out_ds.SetGeoTransform(self.geotransform)
        out_ds.SetProjection(self.projection)
        out_ds.GetRasterBand(1).WriteArray(arr)
        out_ds.FlushCache()
        out_ds = None


def mosaic_tiles_auto_utm(paths, out_path: str, resample_alg: str = "near"):
    """Mosaics a list of rasters into a single GeoTIFF.

    If the tiles are ALL ALREADY in the same projected (metric) CRS —
    the common case of a mosaic made of TINITALY-only tiles, already in
    UTM — NO reprojection is applied: the mosaic is built directly in
    that CRS. Reprojecting when it isn't needed introduces a slight
    misalignment (an automatically computed UTM zone's 'north' is never
    perfectly identical to the tiles' original zone, especially near
    zone boundaries), visible as a tilted mosaic — real bug encountered:
    before this automatic reprojection was introduced, a mosaic of
    TINITALY-only tiles came out as a perfect rectangle; afterwards, it
    was tilted.

    Reprojection (to the UTM zone best suited to the area's centroid) is
    applied ONLY when actually needed: tiles with different CRSs from
    each other, or not already projected (e.g. degrees, EPSG:4326 —
    common for Copernicus/SRTM). This is what still makes the plugin
    usable with DEMs from different sources in the same session.

    Returns the EPSG code of the CRS used for the mosaic (None if not
    determinable, with reprojection skipped)."""
    srs_list = []
    for p in paths:
        ds = gdal.Open(p, gdal.GA_ReadOnly)
        if ds is None:
            raise IOError(f"Could not open the raster: {p}")
        srs = osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjection())
        srs_list.append(srs)
        ds = None

    all_projected = all(srs.IsProjected() for srs in srs_list)
    all_same_crs = all(srs.IsSame(srs_list[0]) for srs in srs_list[1:])

    if all_projected and all_same_crs:
        # no reprojection needed: mosaic directly in the common native
        # CRS, so the result stays a clean rectangle
        gdal.Warp(out_path, list(paths), format="GTiff", resampleAlg=resample_alg)
        auth_code = srs_list[0].GetAuthorityCode(None)
        return int(auth_code) if auth_code else None

    first_ds = gdal.Open(paths[0], gdal.GA_ReadOnly)
    src_srs = srs_list[0]
    gt = first_ds.GetGeoTransform()
    w, h = first_ds.RasterXSize, first_ds.RasterYSize
    cx = gt[0] + gt[1] * w / 2 + gt[2] * h / 2
    cy = gt[3] + gt[4] * w / 2 + gt[5] * h / 2
    first_ds = None

    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    # guarantees (lon, lat) order regardless of the GDAL version (GDAL 3+
    # would otherwise use the (lat, lon) order for EPSG:4326)
    wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(src_srs, wgs84)
    lon, lat, _ = transform.TransformPoint(cx, cy)

    epsg = estimate_utm_epsg(lon, lat)
    gdal.Warp(out_path, list(paths), format="GTiff",
              dstSRS=f"EPSG:{epsg}", resampleAlg=resample_alg)
    return epsg


def array_to_mem_dataset(array: np.ndarray, geotransform, projection):
    """Creates an in-memory GDAL dataset (MEM driver) from a numpy array,
    useful for passing the land/sea mask to gdal.Polygonize without
    touching the disk."""
    driver = gdal.GetDriverByName("MEM")
    ds = driver.Create("", array.shape[1], array.shape[0], 1, gdal.GDT_Byte)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)
    ds.GetRasterBand(1).WriteArray(array.astype(np.uint8))
    return ds


def save_uint8_bmp(array: np.ndarray, out_path: str):
    """Saves a uint8 array as an 8-bit grayscale BMP, written BY HAND in
    pure Python (standard BITMAPINFOHEADER format, 256-level grayscale
    palette), WITHOUT going through GDAL's BMP driver.

    Reason: SC4Mapper opens the file with Pillow (Image.open) and shows
    the generic error 'This is not a valid file' if that open fails for
    ANY reason (verified in SC4Mapper's source code, app.py,
    OnBrowseFile) — real bug encountered: GDAL's BMP driver is fairly
    old and can write format variants that Pillow doesn't reliably read.
    Writing the file by hand in a minimal, widely standard BMP format
    removes the risk of incompatibility."""
    h, w = array.shape
    arr = np.ascontiguousarray(array, dtype=np.uint8)

    row_size = (w + 3) // 4 * 4  # every BMP row is 4-byte aligned
    pixel_data_size = row_size * h
    palette_size = 256 * 4
    header_size = 14 + 40 + palette_size
    file_size = header_size + pixel_data_size

    with open(out_path, "wb") as f:
        # File header (14 bytes)
        f.write(b"BM")
        f.write(struct.pack("<I", file_size))
        f.write(struct.pack("<HH", 0, 0))
        f.write(struct.pack("<I", header_size))

        # DIB header — BITMAPINFOHEADER (40 bytes), the most widely
        # supported variant
        f.write(struct.pack("<I", 40))
        f.write(struct.pack("<i", w))
        f.write(struct.pack("<i", h))  # positive = standard bottom-up order
        f.write(struct.pack("<H", 1))   # planes
        f.write(struct.pack("<H", 8))   # bits per pixel
        f.write(struct.pack("<I", 0))   # BI_RGB, no compression
        f.write(struct.pack("<I", pixel_data_size))
        f.write(struct.pack("<i", 2835))  # ~72 DPI, arbitrary but standard value
        f.write(struct.pack("<i", 2835))
        f.write(struct.pack("<I", 256))  # colors in the palette
        f.write(struct.pack("<I", 0))    # "important" colors: all of them

        # Grayscale palette (B, G, R, reserved)
        for i in range(256):
            f.write(struct.pack("<BBBB", i, i, i, 0))

        # Pixel data, bottom-up order, each row 4-byte aligned
        pad = b"\x00" * (row_size - w)
        for row in range(h - 1, -1, -1):
            f.write(arr[row].tobytes())
            if pad:
                f.write(pad)


def save_uint8_png(array: np.ndarray, out_path: str):
    """Saves a uint8 array as a grayscale PNG (OpenTTD heightmap format).
    GDAL's PNG driver only supports CreateCopy, not a direct Create: an
    in-memory dataset (MEM driver) is used as a go-between."""
    h, w = array.shape
    mem_ds = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    mem_ds.GetRasterBand(1).WriteArray(array)
    png_ds = gdal.GetDriverByName("PNG").CreateCopy(out_path, mem_ds)
    png_ds = None
    mem_ds = None


def sieve_small_land(dem: np.ndarray, sea_level: float, threshold_px: int,
                      connectedness: int = 8):
    """Brings connected land fragments smaller than threshold_px pixels
    back down to sea level, using the native GDAL SieveFilter algorithm
    (the same principle used to clean up small artifacts from classified
    maps). Useful at export time: scattered islets far from the main body
    would otherwise force the automatic crop to include huge empty sea
    areas just to fit them all in a single rectangle.

    Does not modify the original array — returns a filtered copy of it,
    plus the number of removed pixels (to report in the export message)."""
    if threshold_px <= 0:
        return dem, 0

    h, w = dem.shape
    land_mask = (dem > sea_level).astype(np.uint8)

    src_ds = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    src_ds.GetRasterBand(1).WriteArray(land_mask)

    out_ds = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    out_band = out_ds.GetRasterBand(1)
    gdal.SieveFilter(srcBand=src_ds.GetRasterBand(1), maskBand=None,
                      dstBand=out_band, threshold=threshold_px,
                      connectedness=connectedness)
    sieved_mask = out_band.ReadAsArray().astype(bool)
    src_ds = None
    out_ds = None

    removed = land_mask.astype(bool) & ~sieved_mask
    result = dem.copy()
    result[removed] = sea_level
    return result, int(removed.sum())


def resample_array(array: np.ndarray, target_w: int, target_h: int,
                    resample_alg=None) -> np.ndarray:
    """Resamples a 2D float32 array to the target size, via an in-memory
    GDAL dataset (no real georeferencing is needed: only the resampling
    matters, not the coordinate transformation)."""
    if resample_alg is None:
        resample_alg = gdal.GRIORA_Bilinear
    h, w = array.shape
    mem_ds = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Float32)
    mem_ds.GetRasterBand(1).WriteArray(array.astype(np.float32))
    resampled = mem_ds.GetRasterBand(1).ReadAsArray(
        buf_xsize=target_w, buf_ysize=target_h, resample_alg=resample_alg)
    mem_ds = None
    return resampled.astype(np.float32)
