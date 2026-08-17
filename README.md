# SC4 Terrain Composer

**A QGIS plugin for composing and exporting DEM heightmaps for SimCity 4
(SC4Mapper) and OpenTTD.**

`QGIS 3.34+ / 4.0+` · `Python 3` · `License: MIT`

SC4 Terrain Composer turns a DEM mosaic into a game-ready heightmap
without leaving QGIS or touching an image editor. It automatically finds
separate pieces of land in your source tiles (islands, disconnected
fragments), lets you pick which ones to keep, move, rotate or delete them
with seamless blending, and exports an 8-bit BMP or PNG that matches
SC4Mapper's and OpenTTD's exact elevation-encoding rules — rules that were
reverse-engineered directly from SC4Mapper's own source code, not guessed.

## Which folder do I need?

This repository ships two parallel builds, since QGIS 4.0 "Norrköping"
(March 2026) moved from Qt5 to Qt6 and old-style Qt enum access
(`Qt.UserRole`) breaks under it:

| Folder | For | 
|---|---|
| [`qgis3/`](qgis3/) | QGIS 3.34 up to 3.99 (Qt5) |
| [`qgis4/`](qgis4/) | QGIS 4.0 and later (Qt6) |

Each folder contains a self-contained `sc4_terrain_composer/` plugin
directory — zip *that* subfolder (with itself as the ZIP's top level) to
install via **Plugins → Manage and Install Plugins → Install from ZIP**.
`terrain_ops.py`, `raster_io.py` and `map_tools.py` are byte-identical
between the two builds; only `dock_widget.py`, `sc4_terrain_composer.py`
and `metadata.txt` differ, for the Qt6 enum-scoping reasons documented in
`qgis4/README.md`.

## Why this exists

Composing a "collage" heightmap — for example, mainland terrain plus an
offshore archipelago repositioned onto the coast — normally means
cropping, blending and rescaling rasters by hand across several tools.
This plugin keeps the whole workflow inside QGIS: load your DEM tiles,
select and rearrange the pieces you want on the map canvas, and export
directly to a format your target game will accept.

## Features

- **Multi-source DEM loading** — mosaics one or more raster layers
  already open in QGIS (or a file from disk). Reprojects to a consistent
  metric CRS automatically, but *only* when the source tiles actually
  need it, so a clean single-source mosaic never ends up needlessly
  tilted.
- **Configurable no-data handling** — treat no-data as sea level, or fill
  it via GDAL's native `FillNodata` (local inverse-distance
  interpolation, so it stays fast regardless of mosaic size).
- **Automatic land-blob detection** (via `gdal:polygonize`) — every
  separate piece of land becomes an individually selectable polygon,
  using QGIS's native selection tool.
- **Manual freehand lasso** for land pieces that are physically joined
  and need to be split by hand.
- **Move / Rotate / Delete** a selection with edge feathering and
  diffusion-based hole filling — every operation works on a local crop
  around the selection, not the whole canvas, so it stays fast on large
  regional mosaics.
- **Undo / Redo** via a lightweight in-memory patch stack.
- **Pre-export processing** — automatic cropping of excess sea, an
  export-only rotation that never touches the working DEM, an
  optimal-rotation finder, and an on-canvas preview of the export
  rectangle before you generate the file.
- **SC4 export** — 8-bit BMP snapped to the `64x+1` / `256x+1` dimensions
  SC4 requires, with the exact scale-factor presets from SC4Mapper's own
  source code and automatic scale adjustment when the terrain would
  otherwise be clipped. Includes a **coastal offset** option that
  compensates for a real, source-verified SC4Mapper behavior: it renders
  any exported land below 25 real meters of elevation as submerged,
  regardless of the chosen scale factor (see
  [Technical notes](#technical-notes)).
- **OpenTTD export** — 8-bit PNG snapped to a power-of-2 size (up to
  16384 with extended-limit clients such as JGRPP), black = sea level,
  white = the chosen maximum elevation.
- **OpenTTD town data export** — exports a point layer (e.g. loaded from
  OpenStreetMap via QuickOSM) as OpenTTD's *official* town-import JSON
  format (`Scenario Editor → Town Generation → Load from file`), with
  each town's tile position computed through the exact same crop/
  rotation pipeline as the heightmap export, so they land in the right
  spot even on a rotated or cropped map.
- **Linked vector layer** — optionally, moving or rotating a piece of
  terrain also moves/rotates any point feature (e.g. a town marker) that
  currently sits on top of it, by the identical transform, using the
  same underlying math validated for the heightmap and JSON exports.

## Installation

1. Pick the right folder for your QGIS version (see
   [Which folder do I need?](#which-folder-do-i-need) above) — `qgis3/`
   or `qgis4/`.
2. Clone this repository, or download it as a ZIP and extract it, then
   zip *that* folder's `sc4_terrain_composer/` subfolder on its own
   (with itself as the new ZIP's top level).
3. In QGIS: **Plugins → Manage and Install Plugins → Install from ZIP**,
   select the file.
4. Enable the plugin from the list if it doesn't activate automatically.
5. A toolbar icon / menu entry **"SC4 Terrain Composer"** appears and
   opens the side panel.

For development, you can instead symlink or copy the matching
`sc4_terrain_composer/` folder directly into QGIS's plugin profile
folder (on Windows, typically
`C:\Users\<you>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\`
for QGIS 3.x — the QGIS 4.x path uses a similar layout under its own
profile directory), then enable it from **Plugins → Manage and Install
Plugins → Installed**.

## Basic workflow

1. **DEM** — check one or more raster layers already loaded in QGIS (or
   load a file from disk) and mosaic them.
2. **Selection** — detect connected land blobs and click the ones you
   want, or draw a manual lasso for land pieces that are physically
   joined.
3. **Move/Delete** — move the selection to a new spot on the canvas,
   rotate it in place, or delete it; feathering and diffusion-based
   filling keep the result seamless.
4. **Pre-export** — crop excess sea, optionally rotate the export window
   only, and preview the resulting rectangle before exporting.
5. **SC4 / OpenTTD** — export in the format your target game expects.

## DEM source compatibility

Built and tested primarily against [TINITALY](http://tinitaly.pi.ingv.it/)
tiles, but designed to also work with DEMs from other agencies
(Copernicus DEM, USGS SRTM/3DEP, ASTER GDEM):

- Reprojection to a metric UTM CRS only happens when the loaded tiles
  actually need it (different CRSs from each other, or a geographic CRS
  like plain degrees) — a single-source mosaic that's already projected
  is left untouched, avoiding unnecessary distortion.
- No-data handling is configurable between "treat as sea level" (correct
  for land-only DEMs like TINITALY) and "fill by interpolation" (correct
  for global DEMs where no-data represents real data gaps, not the sea).
- Any scale/offset factor present in the source GeoTIFF's GDAL metadata
  is applied automatically on load.

## Technical notes

A few implementation details are worth documenting for anyone extending
this plugin or debugging an export:

- **SC4Mapper's elevation formula** (`elevation_m = grayscale ×
  scale_factor`) and its exact scale-factor presets were verified line
  by line against SC4Mapper's own Python source (`app.py`), not assumed
  from general SC4 community knowledge — an earlier version of this
  plugin used an incorrect formula before the source was available for
  review.
- **SC4Mapper's water-rendering threshold**: SC4Mapper always renders
  exported land below **25 real meters of elevation** as submerged,
  regardless of the chosen scale factor. This comes from a hardcoded
  `waterLevel = 250` constant compared directly against the raw height
  buffer (`grayscale × 10 × scale_factor`) that gets written unchanged
  into the actual game file (see `terrain.py`'s `onePassColors` and
  `region.py`'s `City.Save` in SC4Mapper's source). The plugin's
  "coastal offset" export option compensates for this.
- All performance-sensitive raster operations (feathering, inpainting,
  rotation, blob detection) are windowed to operate on a crop around the
  affected area rather than the whole canvas, so the plugin stays
  responsive on large regional mosaics (tested up to hundreds of
  millions of pixels).

## Testing

`terrain_ops.py` — masks, edge feathering, hole filling, move/rotate,
SC4/OpenTTD elevation encoding, and export-rectangle geometry — is pure
NumPy with no QGIS dependency, and is covered by an automated test suite
that runs standalone:

```bash
python3 test_terrain_ops.py
```

No QGIS installation is required to run these tests (36 checks, covering
masks, feathering, inpainting, move/rotate math, SC4/OpenTTD encoding,
export-geometry consistency, and point-tracking through the export
pipeline). The PyQGIS integration layer (`dock_widget.py`, `raster_io.py`,
`map_tools.py`, `sc4_terrain_composer.py`) depends on QGIS and GDAL's
Python bindings and is exercised inside QGIS itself.

## Known limitations

- No-data interpolation and reprojection depend on GDAL's Python
  bindings (`osgeo`), which ship with QGIS but aren't independently
  testable outside it in every environment.
- A single rectangular export can't avoid all wasted sea around a
  non-rectangular coastline — cropping and rotation minimize it, but a
  concave coastline (bays, inlets) will always leave some sea inside the
  rectangle.

## Contributing

Issues and pull requests are welcome. If you're changing anything in
`terrain_ops.py`, please add or update a test in `test_terrain_ops.py` —
it's fast to run and doesn't require a QGIS install.

## Acknowledgments

- [TINITALY](http://tinitaly.pi.ingv.it/) (INGV) for the DEM data this
  plugin was primarily developed and tested against.
- [SC4Mapper-2013](https://github.com/wouanagaine/SC4Mapper-2013) by
  Wouanagaine and JoeST, and its Python 3 modernization "SC4Mapper-2026"
  — the region-creation tool this plugin's SC4 export is designed to
  feed. Its source code was essential for getting the elevation encoding
  exactly right (see [Technical notes](#technical-notes) above).

## License

Released under the [MIT License](LICENSE).
