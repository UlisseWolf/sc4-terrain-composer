# SC4 Terrain Composer — QGIS 4.0 build

This is a **separate copy** of the plugin, adapted for QGIS 4.0
"Norrköping" (released March 6, 2026), which moved QGIS from Qt5 to Qt6.
It is **not** meant to replace the main plugin folder used for QGIS 3.x —
install whichever one matches your QGIS version.

## What was actually changed here

QGIS 4.0 kept most of its own API stable ("deprecated APIs retained
where possible" per the official announcement), so the vast majority of
this code — all of `terrain_ops.py` (pure numpy, no Qt/QGIS dependency
at all) and `raster_io.py` (GDAL only) — needed **zero changes**.

The concrete, documented breaking change is in **raw Qt/PyQt enum
access**: PyQt6 nests enum members under their enum class (e.g.
`Qt.ItemDataRole.UserRole`) instead of exposing them directly on `Qt`
(`Qt.UserRole`, valid under PyQt5/Qt5). This plugin already imported Qt
symbols through `qgis.PyQt` (QGIS's own compatibility shim) everywhere,
which is the correct practice and handles *which* library gets loaded —
but it does not rewrite call sites that relied on PyQt5's flatter enum
access. Those call sites needed updating by hand:

| File | Change |
|---|---|
| `dock_widget.py` | `Qt.red` → `Qt.GlobalColor.red`; `Qt.transparent` → `Qt.GlobalColor.transparent`; `Qt.UserRole` → `Qt.ItemDataRole.UserRole` (×3); `Qt.Checked`/`Qt.Unchecked` → `Qt.CheckState.Checked`/`Qt.CheckState.Unchecked` (×6); `Qt.ItemIsUserCheckable` → `Qt.ItemFlag.ItemIsUserCheckable` |
| `sc4_terrain_composer.py` | Replaced a raw magic-number dock area argument (`0x2`, with a comment claiming it meant `Qt.RightDockWidgetArea`) with the actual named, scoped constant `Qt.DockWidgetArea.RightDockWidgetArea` — found and fixed while auditing this file for Qt6 compatibility, unrelated to the Qt5→Qt6 change itself but worth cleaning up while here |
| `map_tools.py` | No changes needed (only imports `pyqtSignal`, no enum access) |

`metadata.txt` was updated to `qgisMinimumVersion=4.0` /
`qgisMaximumVersion=4.99`.

## Honest limitations

I do not have a QGIS 4.0 installation available to actually run this
plugin end to end — the same limitation that applied to the QGIS 3.x
build throughout this project's development. What I verified:

- The changes above are exactly what QGIS's own migration tooling
  (`pyqt5_to_pyqt6.py`, mentioned in QGIS's official migration guide)
  is documented to handle — this isn't a guess, it's the known,
  predictable class of change for this transition.
- Every file still compiles (`python3 -m py_compile`) after the edits.
- A cross-reference check confirms no other raw `Qt.<enum>` access
  remains anywhere in the codebase outside of what's listed above.

What I could **not** verify, for the same reason as always in this
project (no QGIS installed in this environment): whether QGIS 4.0's own
API surface (`QgsRasterLayer`, `QgsVectorLayer`, `QgsRubberBand`,
`QgsMapLayerComboBox`, `QgsCoordinateTransform`, etc.) has any breaking
changes of its own beyond the Qt6 enum scoping — QGIS's announcement
says deprecated APIs were retained, which is reassuring, but "retained"
isn't the same as "identical," and some methods could behave slightly
differently or emit deprecation warnings. The first real test of this
build should be inside an actual QGIS 4.0 installation, the same
caveat that has applied to every PyQGIS-layer change made throughout
this whole project.

## Installation

Same as the main build: zip this `sc4_terrain_composer/` folder (with
this folder itself as the ZIP's top level) and use QGIS 4.0's
**Plugins → Manage and Install Plugins → Install from ZIP**.
