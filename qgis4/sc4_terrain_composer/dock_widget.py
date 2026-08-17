# -*- coding: utf-8 -*-
"""
dock_widget.py
================================================================================
Main plugin panel. Orchestrates the whole workflow:

  1. Load DEM (GeoTIFF)                          -> RasterHandle
  2. Detect connected land blobs                  -> gdal:polygonize (processing)
     -> vector layer, selectable with the native QGIS tool
  3. Alternative: manual lasso                    -> native QGIS "Add
     Freehand Feature" on a scratch layer
  4. Apply selection -> raster mask
  5. Move (click destination) / Delete            -> terrain_ops.py
  6. Undo/redo (in-memory array stack)
  7. Export for SC4                               -> 8-bit BMP + scale suggestion

Every heavy operation (masks, feathering, inpainting, encoding) lives in
terrain_ops.py (pure numpy, tested in isolation). This file only wires
that logic to the UI and to QGIS.
================================================================================
"""

import os
import copy
import tempfile
import json
import numpy as np

from qgis.PyQt.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QDoubleSpinBox, QSpinBox, QFileDialog, QMessageBox, QGroupBox, QFormLayout,
    QListWidget, QListWidgetItem, QAbstractItemView, QScrollArea, QComboBox,
    QCheckBox, QTabWidget,
)
from qgis.core import (
    QgsProject, QgsRasterLayer, QgsVectorLayer, QgsField, QgsFeature,
    QgsGeometry, QgsWkbTypes, QgsFillSymbol, QgsCategorizedSymbolRenderer,
    QgsRendererCategory, QgsRandomColorRamp, edit, QgsCoordinateTransform,
    QgsPointXY, QgsMapLayerProxyModel,
)
from qgis.PyQt.QtCore import QVariant, Qt
from osgeo import gdal, ogr

from . import terrain_ops
from .raster_io import (
    RasterHandle, array_to_mem_dataset, save_uint8_bmp, save_uint8_png,
    resample_array, mosaic_tiles_auto_utm, sieve_small_land,
)
from .map_tools import PointPickTool
from qgis.gui import QgsRubberBand, QgsMapLayerComboBox

MAX_UNDO_STATES = 25


class Sc4TerrainComposerDock(QDockWidget):

    def __init__(self, iface, parent=None):
        super().__init__("SC4 Terrain Composer", parent)
        self.iface = iface
        self.canvas = iface.mapCanvas()

        self.raster: RasterHandle | None = None
        self.raster_layer: QgsRasterLayer | None = None
        self.working_path: str | None = None

        self.blob_layer: QgsVectorLayer | None = None
        self.manual_layer: QgsVectorLayer | None = None
        self.current_mask: np.ndarray | None = None
        self._selection_anchor_rc = None  # (row, col) selection centroid

        self._undo_stack = []
        self._redo_stack = []

        self._point_tool = PointPickTool(self.canvas)
        self._point_tool.point_picked.connect(self._on_destination_picked)
        self._pending_move = False

        # export rectangle preview outline (temporary, not a permanent
        # layer): created once, reused on every recalculation
        self._preview_rubber_band = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
        self._preview_rubber_band.setColor(Qt.GlobalColor.red)
        self._preview_rubber_band.setWidth(2)
        self._preview_rubber_band.setFillColor(Qt.GlobalColor.transparent)

        # working folder independent of where the loaded files live
        # (important now that the DEM can come from a mosaic of several
        # layers already open in QGIS, not a single file on disk)
        self._work_dir = tempfile.mkdtemp(prefix="sc4_terrain_composer_")

        self._build_ui()

        # active cleanup: any orphaned "Working DEM" left over in the
        # project from a previous plugin session (e.g. panel closed and
        # reopened) is removed right away, not just hidden from the list —
        # otherwise it stays selectable as if it were a raw tile, and if it
        # still carries the effects of an earlier mistaken operation it
        # contaminates the next mosaic from the very first load (real bug
        # encountered)
        self._remove_orphaned_working_layers()

        # the list of available DEMs stays up to date automatically
        # whenever the user loads/removes layers elsewhere in QGIS
        QgsProject.instance().layersAdded.connect(self.refresh_layer_list)
        QgsProject.instance().layersRemoved.connect(self.refresh_layer_list)
        self.refresh_layer_list()

    def _remove_orphaned_working_layers(self):
        to_remove = [
            lyr.id() for lyr in QgsProject.instance().mapLayers().values()
            if isinstance(lyr, QgsRasterLayer) and lyr.name().startswith("Working DEM")
        ]
        if to_remove:
            QgsProject.instance().removeMapLayers(to_remove)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        tabs = QTabWidget()

        tabs.addTab(self._build_tab_dem(), "1. DEM")
        tabs.addTab(self._build_tab_selection(), "2. Selection")
        tabs.addTab(self._build_tab_edit(), "3. Move/Delete")
        tabs.addTab(self._build_tab_preexport(), "4. Pre-export")
        tabs.addTab(self._build_tab_sc4(), "5. SC4")
        tabs.addTab(self._build_tab_openttd(), "6. OpenTTD")

        # Explicit stylesheet instead of inheriting QGIS's theme: some
        # dark QGIS themes have shown buttons with invisible text (text
        # color too close to the inherited background) — without a live
        # QGIS to reproduce and diagnose this with certainty, the most
        # robust fix is to not depend on the inherited theme and fix
        # readable colors/contrast explicitly for the whole panel.
        tabs.setStyleSheet("""
            QPushButton {
                color: #f0f0f0;
                background-color: #3a3a3a;
                border: 1px solid #5a5a5a;
                border-radius: 3px;
                padding: 4px 6px;
                min-height: 20px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton:pressed { background-color: #2a2a2a; }
            QLabel { color: #e0e0e0; }
            QTabWidget::pane { border: 1px solid #5a5a5a; }
            QTabBar::tab {
                color: #f0f0f0;
                background-color: #3a3a3a;
                padding: 4px 8px;
                border: 1px solid #5a5a5a;
                border-bottom: none;
            }
            QTabBar::tab:selected { background-color: #4a4a4a; }
            QComboBox, QSpinBox, QDoubleSpinBox, QListWidget {
                color: #f0f0f0;
                background-color: #2a2a2a;
                border: 1px solid #5a5a5a;
            }
        """)

        # Tabs keep the panel COMPACT (one section at a time, not all six
        # stacked): the QGIS window no longer needs to be enlarged just to
        # reach the last section. Each tab still sits inside its own
        # scroll area as a safety net, in case a tab's content exceeds
        # the available height anyway.
        self.setWidget(tabs)

    def _wrap_scroll(self, inner: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        return scroll

    # --- Tab 1: DEM --------------------------------------------------------
    def _build_tab_dem(self):
        root = QWidget()
        l = QVBoxLayout(root)

        l.addWidget(QLabel(
            "Raster already loaded in QGIS (check one or more tiles: if "
            "you check more than one they are mosaicked automatically):"
        ))
        self.list_layers = QListWidget()
        self.list_layers.setMaximumHeight(120)
        self.list_layers.setSelectionMode(QAbstractItemView.NoSelection)
        self.list_layers.setToolTip(
            "Also mosaics different sources together (e.g. TINITALY + "
            "Copernicus + SRTM), automatically reprojecting everything to "
            "a consistent metric CRS."
        )
        l.addWidget(self.list_layers)

        hbox_layers = QHBoxLayout()
        btn_refresh = QPushButton("Refresh list")
        btn_refresh.clicked.connect(self.refresh_layer_list)
        hbox_layers.addWidget(btn_refresh)
        btn_select_all = QPushButton("Select all")
        btn_select_all.clicked.connect(self.select_all_layers)
        hbox_layers.addWidget(btn_select_all)
        btn_select_none = QPushButton("Deselect all")
        btn_select_none.clicked.connect(self.deselect_all_layers)
        hbox_layers.addWidget(btn_select_none)
        l.addLayout(hbox_layers)

        row_nodata = QHBoxLayout()
        row_nodata.addWidget(QLabel("No-data pixels:"))
        self.combo_nodata_mode = QComboBox()
        self.combo_nodata_mode.addItem("Sea level (0 m) — TINITALY etc.", "sea")
        self.combo_nodata_mode.addItem("Interpolation (GDAL FillNodata) — SRTM/ASTER", "interpolate")
        self.combo_nodata_mode.setToolTip(
            "'Sea level': correct for DEMs that cover only dry land "
            "(TINITALY and similar). 'Interpolation': fills holes by "
            "searching ONLY locally (radius next to it) with the native "
            "GDAL FillNodata algorithm — stays fast no matter how large "
            "the mosaic is. Correct for global DEMs (SRTM/ASTER) where "
            "no-data is a real gap in the data, not the sea. If a hole is "
            "larger than the radius, the unreached pixels are still left "
            "at sea level."
        )
        row_nodata.addWidget(self.combo_nodata_mode, stretch=1)
        l.addLayout(row_nodata)

        row_search = QHBoxLayout()
        row_search.addWidget(QLabel("Interpolation search radius:"))
        self.spin_fill_search_dist = QSpinBox()
        self.spin_fill_search_dist.setRange(1, 5000)
        self.spin_fill_search_dist.setValue(100)
        self.spin_fill_search_dist.setSuffix(" px")
        row_search.addWidget(self.spin_fill_search_dist)
        l.addLayout(row_search)

        btn_use_layers = QPushButton("Use checked layers")
        btn_use_layers.clicked.connect(self.use_selected_layers)
        l.addWidget(btn_use_layers)

        btn_load = QPushButton("...or load DEM from file (not yet in QGIS)")
        btn_load.clicked.connect(self.load_dem)
        l.addWidget(btn_load)

        self.lbl_dem = QLabel("No DEM loaded")
        self.lbl_dem.setWordWrap(True)
        l.addWidget(self.lbl_dem)
        l.addStretch()
        return self._wrap_scroll(root)

    # --- Tab 2: Selection ----------------------------------------------------
    def _build_tab_selection(self):
        root = QWidget()
        form = QFormLayout(root)

        self.spin_sea_level = QDoubleSpinBox()
        self.spin_sea_level.setRange(-1000, 1000)
        self.spin_sea_level.setValue(0.0)
        self.spin_sea_level.setSuffix(" m")
        form.addRow("Sea level:", self.spin_sea_level)

        btn_blobs = QPushButton("Detect connected land blobs")
        btn_blobs.clicked.connect(self.detect_land_blobs)
        btn_blobs.setToolTip(
            "Select the resulting shapes with the native QGIS 'Select "
            "Features' tool (click one or more shapes)."
        )
        form.addRow(btn_blobs)

        btn_manual = QPushButton("Activate manual lasso (freehand drawing)")
        btn_manual.clicked.connect(self.start_manual_lasso)
        btn_manual.setToolTip(
            "'manual_selection' layer → 'Add Polygon Feature' → R key "
            "(Digitize Freehand) → draw while holding the mouse button "
            "down."
        )
        form.addRow(btn_manual)

        btn_apply_sel = QPushButton("Apply current selection → mask")
        btn_apply_sel.clicked.connect(self.apply_selection_to_mask)
        form.addRow(btn_apply_sel)
        self.lbl_mask = QLabel("No active selection")
        self.lbl_mask.setWordWrap(True)
        form.addRow(self.lbl_mask)

        return self._wrap_scroll(root)

    # --- Tab 3: Move/Delete -----------------------------------------------
    def _build_tab_edit(self):
        root = QWidget()
        form = QFormLayout(root)

        self.spin_feather = QSpinBox()
        self.spin_feather.setRange(0, 50)
        self.spin_feather.setValue(6)
        self.spin_feather.setSuffix(" px")
        form.addRow("Edge feathering:", self.spin_feather)

        self.spin_rotation = QDoubleSpinBox()
        self.spin_rotation.setRange(-180, 180)
        self.spin_rotation.setValue(0)
        self.spin_rotation.setSuffix(" °")
        self.spin_rotation.setToolTip(
            "Rotation angle (clockwise on screen). Applies both to "
            "'Rotate selection' and — if different from 0 — to 'Move "
            "selection', to rotate while moving."
        )
        form.addRow("Rotation:", self.spin_rotation)

        self.combo_linked_vector_layer = QgsMapLayerComboBox()
        self.combo_linked_vector_layer.setFilters(QgsMapLayerProxyModel.PointLayer)
        self.combo_linked_vector_layer.setAllowEmptyLayer(True)
        self.combo_linked_vector_layer.setToolTip(
            "Optional. If set, moving or rotating a raster selection also "
            "moves/rotates — by the EXACT same transform — any point "
            "feature from this layer that currently falls inside the "
            "selection (e.g. town markers sitting on the piece of "
            "terrain you're relocating). Features outside the selection "
            "are never touched. The layer is edited and saved directly."
        )
        form.addRow("Linked vector layer (optional):", self.combo_linked_vector_layer)

        btn_move = QPushButton("Move selection (then click the destination)")
        btn_move.clicked.connect(self.start_move)
        form.addRow(btn_move)

        btn_rotate = QPushButton("Rotate selection (in place)")
        btn_rotate.clicked.connect(self.rotate_selection)
        form.addRow(btn_rotate)

        btn_delete = QPushButton("Delete selection (fill by diffusion)")
        btn_delete.clicked.connect(self.delete_selection)
        form.addRow(btn_delete)

        hbox = QHBoxLayout()
        btn_undo = QPushButton("Undo")
        btn_undo.clicked.connect(self.undo)
        btn_redo = QPushButton("Redo")
        btn_redo.clicked.connect(self.redo)
        hbox.addWidget(btn_undo)
        hbox.addWidget(btn_redo)
        form.addRow(hbox)

        return self._wrap_scroll(root)

    # --- Tab 4: Pre-export processing ---------------------------------------
    def _build_tab_preexport(self):
        root = QWidget()
        form = QFormLayout(root)

        self.chk_autocrop = QCheckBox("Crop the excess sea (clean edge on the land)")
        self.chk_autocrop.setChecked(True)
        self.chk_autocrop.setToolTip(
            "The export rectangle ends exactly where the land ends, like "
            "a cross-section cut — no extra sea margin at the edges."
        )
        form.addRow(self.chk_autocrop)

        self.spin_export_rotation = QDoubleSpinBox()
        self.spin_export_rotation.setRange(-180, 180)
        self.spin_export_rotation.setValue(0)
        self.spin_export_rotation.setSuffix(" °")
        self.spin_export_rotation.setToolTip(
            "Rotates ONLY the export window (a disposable copy), never "
            "the actual working DEM: no risk of 'baking in' a wrong "
            "rotation into the data, unlike the old whole-canvas "
            "rotation. Useful to align a diagonal coastline to the "
            "rectangle's edges and reduce wasted sea at the corners."
        )
        form.addRow("Rotate export ONLY:", self.spin_export_rotation)

        btn_auto_angle = QPushButton("Find optimal angle")
        btn_auto_angle.clicked.connect(self.find_optimal_export_rotation)
        btn_auto_angle.setToolTip(
            "Automatically computes the angle that minimizes the "
            "rectangle needed to contain all the land present in the "
            "loaded DEM — instead of guessing the angle by trial and "
            "error."
        )
        form.addRow(btn_auto_angle)

        self.lbl_optimal_angle = QLabel("")
        self.lbl_optimal_angle.setWordWrap(True)
        form.addRow(self.lbl_optimal_angle)

        btn_preview = QPushButton("Recalculate and show export rectangle")
        btn_preview.clicked.connect(self.show_export_rectangle_preview)
        btn_preview.setToolTip(
            "Draws on the QGIS canvas the outline of the rectangle that "
            "would be generated with the current settings (crop + "
            "rotation) — without actually exporting the file. Useful to "
            "check the result before generating the final BMP/PNG."
        )
        form.addRow(btn_preview)

        btn_clear_preview = QPushButton("Hide preview")
        btn_clear_preview.clicked.connect(self.clear_export_rectangle_preview)
        form.addRow(btn_clear_preview)

        return self._wrap_scroll(root)

    def show_export_rectangle_preview(self):
        if self.raster is None:
            QMessageBox.warning(self, "Warning", "Load a DEM first.")
            return
        sea_level = self.spin_sea_level.value()
        rotation = self.spin_export_rotation.value()
        corners = terrain_ops.compute_export_bbox_corners(self.raster.array, sea_level, rotation)
        if corners is None:
            QMessageBox.warning(self, "Warning", "No land found in the loaded DEM.")
            return

        points = [QgsPointXY(*self.raster.pixel_to_geo(x, y)) for (y, x) in corners]
        geom = QgsGeometry.fromPolygonXY([points])
        self._preview_rubber_band.setToGeometry(geom, None)
        self.canvas.refresh()

    def clear_export_rectangle_preview(self):
        self._preview_rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        self.canvas.refresh()

    def find_optimal_export_rotation(self):
        if self.raster is None:
            QMessageBox.warning(self, "Warning", "Load a DEM first.")
            return
        sea_level = self.spin_sea_level.value()
        mask = terrain_ops.build_land_mask(self.raster.array, sea_level)
        angle, area = terrain_ops.find_optimal_rotation(mask, angle_step=1.0)
        bbox0 = terrain_ops.selection_bbox(mask, feather_px=0, margin=0, shape=mask.shape)
        area0 = (bbox0[1] - bbox0[0]) * (bbox0[3] - bbox0[2]) if bbox0 else 0
        self.spin_export_rotation.setValue(angle)
        reduction = 100 * (1 - area / area0) if area0 else 0
        self.lbl_optimal_angle.setText(
            f"Optimal angle: {angle:.0f}° (reduces the required rectangle "
            f"by {reduction:.0f}% compared to 0°). Value already set above."
        )

    # --- Tab 5: SC4 export ----------------------------------------------------
    def _build_tab_sc4(self):
        root = QWidget()
        form = QFormLayout(root)

        self.spin_n_citta_x = QSpinBox()
        self.spin_n_citta_x.setRange(1, 500)
        self.spin_n_citta_x.setValue(10)
        form.addRow("N. cities (width):", self.spin_n_citta_x)
        self.spin_n_citta_y = QSpinBox()
        self.spin_n_citta_y.setRange(1, 500)
        self.spin_n_citta_y.setValue(10)
        form.addRow("N. cities (height):", self.spin_n_citta_y)
        self.spin_celle_citta = QSpinBox()
        self.spin_celle_citta.setRange(1, 1000)
        self.spin_celle_citta.setValue(64)
        self.spin_celle_citta.setToolTip("64 = small city, 256 = large city")
        form.addRow("Cells per city:", self.spin_celle_citta)

        self.combo_sc4_scale = QComboBox()
        for label in terrain_ops.SC4MAPPER_SCALE_FACTORS:
            self.combo_sc4_scale.addItem(label)
        self.combo_sc4_scale.setCurrentText("Default factor")
        self.combo_sc4_scale.setEditable(True)
        self.combo_sc4_scale.setToolTip(
            "Same exact presets as the 'Scale factor' window in "
            "SC4Mapper (verified against the source code) — pick HERE "
            "the same one you'll pick in SC4Mapper at import time, "
            "without having to retype it by hand. You can also type a "
            "custom number: SC4Mapper accepts that too, just type the "
            "same value in both."
        )
        form.addRow("Scale factor (as in SC4Mapper):", self.combo_sc4_scale)

        self.spin_sc4_coastal_offset = QDoubleSpinBox()
        self.spin_sc4_coastal_offset.setRange(0, 500)
        self.spin_sc4_coastal_offset.setValue(25.0)
        self.spin_sc4_coastal_offset.setSuffix(" m")
        self.spin_sc4_coastal_offset.setToolTip(
            "Added to every elevation BEFORE encoding, to push low-lying "
            "coastal land above SC4Mapper's own fixed water-rendering "
            "threshold. Verified against the source code: SC4Mapper "
            "always shows any exported land below 25 real meters as "
            "submerged (terrain.py: 'water = height < waterLevel', with "
            "waterLevel=250 hardcoded and the raw height buffer written "
            "unchanged into the actual game file) — this happens "
            "regardless of the scale factor chosen above. Set to 0 if "
            "you want a literal 1:1 elevation match instead (accepting "
            "that anything below 25m will show as shallow water/beach "
            "in SC4Mapper)."
        )
        form.addRow("Coastal offset (avoid SC4's water threshold):", self.spin_sc4_coastal_offset)

        note_scale = QLabel(
            "Formula verified against SC4Mapper's source code: "
            "elevation_m = (grayscale × scale_factor) − coastal offset. "
            "SC4Mapper always renders exported land below 25 real "
            "meters as underwater, independent of the scale factor — "
            "not a bug in this formula, but a fixed rendering/water "
            "threshold hardcoded in SC4Mapper itself. The coastal "
            "offset above compensates for it if you don't want that "
            "effect."
        )
        note_scale.setWordWrap(True)
        form.addRow(note_scale)

        btn_export = QPushButton("Export 8-bit BMP for SC4Mapper...")
        btn_export.clicked.connect(self.export_sc4)
        form.addRow(btn_export)

        return self._wrap_scroll(root)

    # --- Tab 6: OpenTTD export -------------------------------------------------
    def _build_tab_openttd(self):
        root = QWidget()
        form = QFormLayout(root)

        self.combo_ottd_w = QComboBox()
        self.combo_ottd_h = QComboBox()
        for combo, default in ((self.combo_ottd_w, 1024), (self.combo_ottd_h, 1024)):
            for size in terrain_ops.OPENTTD_VALID_SIZES:
                combo.addItem(str(size))
            combo.setCurrentText(str(default))
        tip = ("Power of 2. 'Vanilla' OpenTTD tops out at 4096x4096 — "
               "8192 and 16384 only work with clients that have an "
               "extended limit (e.g. JGRPP).")
        self.combo_ottd_w.setToolTip(tip)
        self.combo_ottd_h.setToolTip(tip)
        form.addRow("Map width:", self.combo_ottd_w)
        form.addRow("Map height:", self.combo_ottd_h)

        self.spin_ottd_max_elev = QDoubleSpinBox()
        self.spin_ottd_max_elev.setRange(0, 9000)
        self.spin_ottd_max_elev.setValue(0)
        self.spin_ottd_max_elev.setSuffix(" m")
        self.spin_ottd_max_elev.setToolTip(
            "0 = automatic, uses the maximum of the loaded DEM. The sea "
            "level used is the one set in tab 2 (black = sea level)."
        )
        form.addRow("Elevation → white (255):", self.spin_ottd_max_elev)

        btn_export_ottd = QPushButton("Export 8-bit PNG for OpenTTD...")
        btn_export_ottd.clicked.connect(self.export_openttd)
        form.addRow(btn_export_ottd)

        note_towns = QLabel("<b>Towns (optional)</b>")
        form.addRow(note_towns)

        self.combo_towns_layer = QgsMapLayerComboBox()
        self.combo_towns_layer.setFilters(QgsMapLayerProxyModel.PointLayer)
        self.combo_towns_layer.setAllowEmptyLayer(True)
        self.combo_towns_layer.setToolTip(
            "A point vector layer with town/city locations (e.g. loaded "
            "via QuickOSM, or any 'populated places' dataset already in "
            "the project). Only the ones that fall inside the exported "
            "area are included."
        )
        self.combo_towns_layer.layerChanged.connect(self._refresh_towns_field_combo)
        form.addRow("Towns layer:", self.combo_towns_layer)

        self.combo_towns_field = QComboBox()
        self.combo_towns_field.setToolTip("Attribute field that holds each town's name.")
        form.addRow("Name field:", self.combo_towns_field)

        self.combo_towns_pop_field = QComboBox()
        self.combo_towns_pop_field.setToolTip(
            "Optional. Attribute field with population figures. If left "
            "empty (or set to '(none)'), every town uses the default "
            "population below instead."
        )
        form.addRow("Population field (optional):", self.combo_towns_pop_field)

        self.spin_towns_default_pop = QSpinBox()
        self.spin_towns_default_pop.setRange(1, 1000000)
        self.spin_towns_default_pop.setValue(1000)
        self.spin_towns_default_pop.setToolTip(
            "Used for towns with no population value (missing field, or "
            "no population field selected). OpenTTD scales this down "
            "internally, so it doesn't need to match a real-world figure "
            "exactly."
        )
        form.addRow("Default population:", self.spin_towns_default_pop)

        self.spin_towns_city_threshold = QSpinBox()
        self.spin_towns_city_threshold.setRange(0, 1000000)
        self.spin_towns_city_threshold.setValue(5000)
        self.spin_towns_city_threshold.setToolTip(
            "Towns with a population at or above this value are marked "
            "as 'city' (grows faster, larger max size) in OpenTTD's "
            "import format."
        )
        form.addRow("City population threshold:", self.spin_towns_city_threshold)

        btn_export_towns = QPushButton("Export towns JSON...")
        btn_export_towns.clicked.connect(self.export_openttd_towns_json)
        btn_export_towns.setToolTip(
            "Exports the towns from the selected layer in OpenTTD's "
            "official town-data JSON format (Scenario Editor → Town "
            "Generation → Load from file), computed through the SAME "
            "crop/rotation pipeline as the PNG above — so they line up "
            "with the heightmap even if you rotated or cropped the "
            "export."
        )
        form.addRow(btn_export_towns)

        return self._wrap_scroll(root)


    # ------------------------------------------------------------------
    # 1. Loading the DEM — from file, or from raster layers already in QGIS
    # ------------------------------------------------------------------
    def refresh_layer_list(self, *_):
        """Repopulates the list with all rasters currently in the project
        (also called automatically when the user loads/removes layers
        elsewhere in QGIS, not just from the 'Refresh list' button)."""
        # remember which ones were checked, so the selection isn't lost on every refresh
        checked_ids = {
            self.list_layers.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.list_layers.count())
            if self.list_layers.item(i).checkState() == Qt.CheckState.Checked
        }
        self.list_layers.clear()
        for lyr in QgsProject.instance().mapLayers().values():
            if not isinstance(lyr, QgsRasterLayer):
                continue
            if lyr.providerType() != "gdal":
                continue  # excludes WMS/WMTS/other providers: this needs a local file
            if lyr.name().startswith("Working DEM"):
                # excludes ANY layer generated by the plugin, not just the
                # current session's (self.raster_layer): if the panel was
                # closed/reopened, an old 'Working DEM' — possibly already
                # damaged by a previous operation — can remain in the QGIS
                # project and show up as if it were just another raw tile.
                # Selecting it by mistake together with clean tiles
                # contaminates the mosaic from the very first load (real
                # bug encountered: DEM already skewed/split right after
                # loading).
                continue
            item = QListWidgetItem(lyr.name())
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if lyr.id() in checked_ids else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, lyr.id())
            self.list_layers.addItem(item)

    def select_all_layers(self):
        for i in range(self.list_layers.count()):
            self.list_layers.item(i).setCheckState(Qt.CheckState.Checked)

    def deselect_all_layers(self):
        for i in range(self.list_layers.count()):
            self.list_layers.item(i).setCheckState(Qt.CheckState.Unchecked)

    def use_selected_layers(self):
        selected_ids = [
            self.list_layers.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.list_layers.count())
            if self.list_layers.item(i).checkState() == Qt.CheckState.Checked
        ]
        if not selected_ids:
            QMessageBox.warning(self, "Warning",
                                 "Check at least one raster layer in the list.")
            return

        layers = [QgsProject.instance().mapLayer(lid) for lid in selected_ids]
        # layer.source() for local GDAL rasters is the file path; to be
        # safe we discard any '|...' suffix (e.g. band selection)
        paths = [lyr.source().split("|")[0] for lyr in layers]

        source_path = os.path.join(self._work_dir, "mosaic_input.tif")
        try:
            # reprojects ONLY if actually needed (tiles with different
            # CRSs from each other, or not already projected/metric): if
            # they're all already in the same metric CRS (the common case
            # of TINITALY-only tiles), no reprojection is applied —
            # applying it when not needed used to cause a slightly tilted
            # mosaic (real bug encountered)
            epsg = mosaic_tiles_auto_utm(paths, source_path)
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                  f"Mosaic/reprojection failed ({len(paths)} layers):\n{e}")
            return

        label = (f"{len(paths)} layers mosaicked" if len(paths) > 1 else layers[0].name())
        label += f" [EPSG:{epsg}]" if epsg else " [native CRS, no reprojection needed]"
        self._load_dem_from_path(source_path, label=label)

    def load_dem(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load DEM", "", "GeoTIFF (*.tif *.tiff)")
        if not path:
            return
        # even for a file loaded directly from disk (not from QGIS's
        # layer list), it goes through the same metric-CRS normalization
        # (here, with a single file, reprojection only kicks in if its
        # CRS isn't already projected — e.g. a tile downloaded in degrees)
        source_path = os.path.join(self._work_dir, "single_input.tif")
        try:
            epsg = mosaic_tiles_auto_utm([path], source_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load the DEM:\n{e}")
            return
        label = os.path.basename(path)
        label += f" [EPSG:{epsg}]" if epsg else " [native CRS]"
        self._load_dem_from_path(source_path, label=label)

    def _load_dem_from_path(self, path: str, label: str):
        nodata_mode = self.combo_nodata_mode.currentData()
        try:
            self.raster = RasterHandle(path, nodata_mode=nodata_mode,
                                        fill_max_search_dist=self.spin_fill_search_dist.value())
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load the DEM:\n{e}")
            return

        if nodata_mode == "interpolate" and self.raster.n_nodata_filled > 0:
            label += f" ({self.raster.n_nodata_filled} no-data px filled by interpolation"
            if self.raster.n_nodata_left_as_sea > 0:
                label += f", {self.raster.n_nodata_left_as_sea} px beyond the search " \
                          f"radius left at sea level"
            label += ")"

        self._undo_stack.clear()
        self._redo_stack.clear()
        self._write_working_layer(self.raster.array)
        self.canvas.setExtent(self.raster_layer.extent())
        self.canvas.refresh()
        self.lbl_dem.setText(f"Loaded: {label} ({self.raster.width}x{self.raster.height} px)")
        self.refresh_layer_list()

    # ------------------------------------------------------------------
    # 2. Blob detection (gdal.Polygonize) + native manual lasso
    # ------------------------------------------------------------------
    def detect_land_blobs(self):
        if self.raster is None:
            QMessageBox.warning(self, "Warning", "Load a DEM first.")
            return

        sea_level = self.spin_sea_level.value()
        land_mask = terrain_ops.build_land_mask(self.raster.array, sea_level)

        mem_ds = array_to_mem_dataset(land_mask, self.raster.geotransform, self.raster.projection)
        band = mem_ds.GetRasterBand(1)

        drv = ogr.GetDriverByName("Memory")
        ogr_ds = drv.CreateDataSource("blobs")
        srs = None
        try:
            from osgeo import osr
            srs = osr.SpatialReference()
            srs.ImportFromWkt(self.raster.projection)
        except Exception:
            pass
        layer = ogr_ds.CreateLayer("blobs", srs=srs, geom_type=ogr.wkbPolygon)
        layer.CreateField(ogr.FieldDefn("value", ogr.OFTInteger))

        # 8-connected: two diagonally adjacent land pixels stay connected
        # (more natural for islands/"staircase" coastlines)
        gdal.Polygonize(band, band, layer, 0, ["8CONNECTED=8"], callback=None)

        # keep only LAND blobs (value=1), discard the big sea polygon (value=0)
        vl = QgsVectorLayer("Polygon?crs=" + (self.raster_layer.crs().authid() if self.raster_layer else ""),
                             "land_blob", "memory")
        vl.dataProvider().addAttributes([QgsField("value", QVariant.Int)])
        vl.updateFields()

        feats_to_add = []
        layer.ResetReading()
        for ogr_feat in layer:
            if ogr_feat.GetField("value") != 1:
                continue
            geom_wkt = ogr_feat.GetGeometryRef().ExportToWkt()
            f = QgsFeature(vl.fields())
            f.setGeometry(QgsGeometry.fromWkt(geom_wkt))
            f.setAttribute("value", 1)
            feats_to_add.append(f)

        with edit(vl):
            vl.dataProvider().addFeatures(feats_to_add)

        # style: semi-transparent fill with a crisp outline, different
        # colors per blob so they're easy to tell apart at a glance
        symbol = QgsFillSymbol.createSimple({"color": "255,0,0,90", "outline_color": "255,0,0,255"})
        vl.renderer().setSymbol(symbol) if vl.renderer() else None

        if self.blob_layer is not None:
            QgsProject.instance().removeMapLayer(self.blob_layer.id())
        self.blob_layer = vl
        QgsProject.instance().addMapLayer(vl)
        self.iface.setActiveLayer(vl)
        self.iface.actionSelect().trigger()  # activates the native "Select Features" tool

        # any polygons drawn by hand but never consumed shouldn't linger
        # around and end up wrongly included in the next 'Apply
        # selection' (same family of bug as the blob/lasso priority
        # issue, just the other direction)
        self._clear_manual_layer()

        self.lbl_mask.setText(
            f"{len(feats_to_add)} land blobs detected: use QGIS's "
            f"selection tool to choose which ones to include, then press "
            f"'Apply selection'."
        )

    def start_manual_lasso(self):
        if self.raster_layer is None:
            QMessageBox.warning(self, "Warning", "Load a DEM first.")
            return
        if self.manual_layer is None:
            crs = self.raster_layer.crs().authid()
            self.manual_layer = QgsVectorLayer(f"Polygon?crs={crs}", "manual_selection", "memory")
            QgsProject.instance().addMapLayer(self.manual_layer)
        else:
            # ALWAYS clear the layer at the start of a new lasso session:
            # if a polygon from a previous selection were left there, it
            # would be included again on the next 'Apply selection' along
            # with the new one (e.g. an already-deleted area 'reappearing'
            # moved together with the next selection — real bug
            # encountered)
            self._clear_manual_layer()

        # 'Apply selection' ALWAYS gives priority to selected blobs, if
        # there are any: if a blob selection from a previous session had
        # been left active, the freshly drawn lasso would be silently
        # ignored (exactly the reported bug: 'the lasso doesn't work'
        # after using the blobs). Clearing the blob selection here means
        # switching to the lasso always takes effect.
        if self.blob_layer is not None:
            self.blob_layer.removeSelection()

        self.iface.setActiveLayer(self.manual_layer)
        self.manual_layer.startEditing()
        QMessageBox.information(
            self, "Manual lasso",
            "The 'manual_selection' layer is now cleared and in edit "
            "mode (any blob selection has also been cleared, so this "
            "lasso will always take priority). In QGIS's digitizing "
            "toolbar click 'Add Polygon Feature', then press the R key "
            "(Digitize Freehand) and draw while holding the left mouse "
            "button down. Release to close the polygon, save the edits, "
            "then come back here and press 'Apply selection'."
        )

    def _clear_manual_layer(self):
        if self.manual_layer is None:
            return
        ids = [f.id() for f in self.manual_layer.getFeatures()]
        if not ids:
            return
        was_editing = self.manual_layer.isEditable()
        if not was_editing:
            self.manual_layer.startEditing()
        self.manual_layer.deleteFeatures(ids)
        self.manual_layer.commitChanges()

    # ------------------------------------------------------------------
    # 4. Apply selection (selected blobs or manual polygon) -> mask
    # ------------------------------------------------------------------
    def apply_selection_to_mask(self):
        if self.raster is None:
            return
        # priority: selected blobs > manually drawn polygon(s).
        # start_manual_lasso() and detect_land_blobs() clear each other's
        # leftover selection/polygons, so this priority no longer causes
        # silent ambiguity between the two selection modes.
        geoms = []
        used_manual = False
        if self.blob_layer is not None and self.blob_layer.selectedFeatureCount() > 0:
            geoms = [f.geometry() for f in self.blob_layer.selectedFeatures()]
        elif self.manual_layer is not None and self.manual_layer.featureCount() > 0:
            geoms = [f.geometry() for f in self.manual_layer.getFeatures()]
            used_manual = True

        if not geoms:
            QMessageBox.warning(self, "Warning",
                                 "No selection found (neither selected blobs nor a drawn lasso).")
            return

        mask = self._rasterize_geometries(geoms)

        # ALWAYS intersect with the current state of the terrain: a
        # polygon (especially a blob) can be a snapshot that's fallen
        # behind deletions/moves done in the meantime — without this
        # check, mistakenly selecting an 'old' blob that used to
        # represent an already-deleted island would still include that
        # area in the mask, even though there's no land there anymore
        current_land = terrain_ops.build_land_mask(self.raster.array, self.spin_sea_level.value())
        pixels_requested = int(mask.sum())
        mask = mask & current_land
        pixels_actual = int(mask.sum())

        self.current_mask = mask
        rows, cols = np.where(mask)
        if rows.size:
            self._selection_anchor_rc = (int(rows.mean()), int(cols.mean()))
        if pixels_actual < pixels_requested:
            discarded = pixels_requested - pixels_actual
            self.lbl_mask.setText(
                f"Mask applied: {pixels_actual} pixels selected "
                f"({discarded} pixels discarded because they're no "
                f"longer land in the current DEM — probably an area "
                f"that was already deleted or moved)."
            )
        else:
            self.lbl_mask.setText(f"Mask applied: {pixels_actual} pixels selected.")

        if used_manual:
            # once consumed, the manual polygon must be removed: it's
            # meant to be 'single-use' for this mask, it shouldn't stay
            # around to be mistakenly included in the next operation
            self._clear_manual_layer()

    def _rasterize_geometries(self, geometries) -> np.ndarray:
        """Rasterizes a list of QgsGeometry onto the current DEM's grid."""
        drv = ogr.GetDriverByName("Memory")
        ds = drv.CreateDataSource("sel")
        layer = ds.CreateLayer("sel", geom_type=ogr.wkbPolygon)
        for g in geometries:
            f = ogr.Feature(layer.GetLayerDefn())
            f.SetGeometry(ogr.CreateGeometryFromWkt(g.asWkt()))
            layer.CreateFeature(f)

        target_ds = gdal.GetDriverByName("MEM").Create(
            "", self.raster.width, self.raster.height, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform(self.raster.geotransform)
        target_ds.SetProjection(self.raster.projection)
        gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[1])
        arr = target_ds.GetRasterBand(1).ReadAsArray()
        return arr.astype(bool)

    # ------------------------------------------------------------------
    # 3. Move / Delete
    # ------------------------------------------------------------------
    def start_move(self):
        if self.current_mask is None:
            QMessageBox.warning(self, "Warning", "Apply a selection first.")
            return
        self._pending_move = True
        self.canvas.setMapTool(self._point_tool)
        rotation = self.spin_rotation.value()
        msg = "Click on the canvas the point where you want to move the selection."
        if abs(rotation) > 1e-6:
            msg += f"\nIt will also be rotated by {rotation}° (as set above)."
        QMessageBox.information(self, "Move", msg)

    def _on_destination_picked(self, point):
        if not self._pending_move or self.current_mask is None:
            return
        self._pending_move = False

        # The clicked point arrives in the PROJECT/canvas CRS, which can
        # differ from the working raster's CRS — especially now that the
        # DEM is always automatically reprojected to a specific UTM zone.
        # Without an explicit conversion, the computed destination point
        # is wrong (often outside the canvas) and the move has no visible
        # effect, with no error.
        canvas_crs = self.canvas.mapSettings().destinationCrs()
        raster_crs = self.raster_layer.crs()
        if canvas_crs != raster_crs:
            transform = QgsCoordinateTransform(canvas_crs, raster_crs, QgsProject.instance())
            point = transform.transform(point)

        dst_col, dst_row = self.raster.geo_to_pixel(point.x(), point.y())
        anchor_row, anchor_col = self._selection_anchor_rc
        dr = int(round(dst_row - anchor_row))
        dc = int(round(dst_col - anchor_col))
        feather_px = self.spin_feather.value()
        rotation = self.spin_rotation.value()

        # compute FIRST which boxes (source + destination) will be
        # touched, so only that small portion can be saved for undo
        # instead of the whole canvas — if rotating, the source box needs
        # to be widened just enough not to clip the corners (see
        # rotation_safe_bbox)
        src_bbox = terrain_ops.rotation_safe_bbox(self.current_mask, feather_px,
                                                    self.raster.array.shape, rotation)
        bboxes = []
        dst_bbox = None
        if src_bbox is not None:
            bboxes.append(src_bbox)
            dst_bbox = terrain_ops.dst_bbox_from_src(src_bbox, dr, dc, self.raster.array.shape)
            if dst_bbox is not None:
                bboxes.append(dst_bbox)

        if dst_bbox is None:
            # nothing to move would end up inside the canvas: it used to
            # fail silently (no error, no effect) — now it's reported
            # explicitly instead of everything just disappearing without
            # explanation
            QMessageBox.warning(
                self, "Destination outside the canvas",
                "The clicked point, converted to the working raster's "
                "CRS, falls completely outside the loaded DEM's bounds: "
                "no move was applied. Try again by clicking a point "
                "closer to the DEM's content (use 'Zoom to Layer Extent' "
                "if you're not sure where it is)."
            )
            self.current_mask = None
            return

        self._push_undo_patch(bboxes)

        new_array = terrain_ops.move_region(
            self.raster.array, self.current_mask,
            dst_row_offset=dr, dst_col_offset=dc,
            feather_px=feather_px,
            sea_level=self.spin_sea_level.value(),
            rotation_degrees=rotation,
        )
        self._commit_array(new_array)
        moved_pts = self._move_linked_vector_points(dr, dc, rotation)
        self.current_mask = None
        msg = "Move applied." if abs(rotation) < 1e-6 else f"Move + rotation of {rotation}° applied."
        if moved_pts:
            msg += f" ({moved_pts} linked point(s) moved too.)"
        self.lbl_mask.setText(msg)

    def _move_linked_vector_points(self, dst_row_offset: int, dst_col_offset: int,
                                    rotation_degrees: float) -> int:
        """Moves/rotates any point feature from the layer selected in
        'Linked vector layer' that currently falls inside self.current_mask,
        by the EXACT same transform just applied to the raster (see
        terrain_ops.transform_points_with_move). Features outside the
        selection are left untouched. Returns how many features moved."""
        layer = self.combo_linked_vector_layer.currentLayer()
        if layer is None or self.current_mask is None:
            return 0

        raster_crs = self.raster_layer.crs()
        layer_crs = layer.crs()
        to_raster = (QgsCoordinateTransform(layer_crs, raster_crs, QgsProject.instance())
                     if layer_crs != raster_crs else None)
        to_layer = (QgsCoordinateTransform(raster_crs, layer_crs, QgsProject.instance())
                    if layer_crs != raster_crs else None)

        feat_ids, points_rc = [], []
        for feat in layer.getFeatures():
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            pt = geom.centroid().asPoint()
            if to_raster is not None:
                try:
                    pt = to_raster.transform(pt)
                except Exception:
                    continue
            col, row = self.raster.geo_to_pixel(pt.x(), pt.y())
            feat_ids.append(feat.id())
            points_rc.append((row, col))

        if not points_rc:
            return 0

        new_points = terrain_ops.transform_points_with_move(
            points_rc, self.current_mask, dst_row_offset, dst_col_offset, rotation_degrees)

        was_editing = layer.isEditable()
        if not was_editing:
            layer.startEditing()

        moved_count = 0
        for fid, old_pt, new_pt in zip(feat_ids, points_rc, new_points):
            if new_pt == old_pt:
                continue  # unchanged: this feature wasn't inside the selection
            x, y = self.raster.pixel_to_geo(new_pt[1], new_pt[0])
            map_point = QgsPointXY(x, y)
            if to_layer is not None:
                map_point = to_layer.transform(map_point)
            layer.changeGeometry(fid, QgsGeometry.fromPointXY(map_point))
            moved_count += 1

        if not was_editing:
            layer.commitChanges()
        else:
            layer.triggerRepaint()
        return moved_count

    def rotate_selection(self):
        if self.current_mask is None:
            QMessageBox.warning(self, "Warning", "Apply a selection first.")
            return
        rotation = self.spin_rotation.value()
        if abs(rotation) < 1e-6:
            QMessageBox.warning(self, "Warning",
                                 "Set a rotation angle different from 0.")
            return
        feather_px = self.spin_feather.value()
        bbox = terrain_ops.rotation_safe_bbox(self.current_mask, feather_px,
                                               self.raster.array.shape, rotation)
        self._push_undo_patch([bbox] if bbox is not None else [])

        new_array = terrain_ops.move_region(
            self.raster.array, self.current_mask,
            dst_row_offset=0, dst_col_offset=0,
            feather_px=feather_px,
            sea_level=self.spin_sea_level.value(),
            rotation_degrees=rotation,
        )
        self._commit_array(new_array)
        moved_pts = self._move_linked_vector_points(0, 0, rotation)
        self.current_mask = None
        msg = f"Rotation of {rotation}° applied (in place)."
        if moved_pts:
            msg += f" ({moved_pts} linked point(s) rotated too.)"
        self.lbl_mask.setText(msg)

    def delete_selection(self):
        if self.current_mask is None:
            QMessageBox.warning(self, "Warning", "Apply a selection first.")
            return
        feather_px = self.spin_feather.value()
        bbox = terrain_ops.selection_bbox(self.current_mask, feather_px=feather_px,
                                           shape=self.raster.array.shape)
        self._push_undo_patch([bbox] if bbox is not None else [])

        new_array = terrain_ops.delete_region(self.raster.array, self.current_mask,
                                               feather_px=feather_px)
        self._commit_array(new_array)
        self.current_mask = None
        self.lbl_mask.setText("Deletion applied.")

    # ------------------------------------------------------------------
    # Undo / redo — stack of "patches" (only the modified box, not the
    # whole canvas: on a mosaic of tens of millions of pixels, saving
    # full copies on every operation would exhaust RAM very quickly)
    # ------------------------------------------------------------------
    def _push_undo_patch(self, bboxes):
        patch = [(b, self.raster.array[b[0]:b[1], b[2]:b[3]].copy()) for b in bboxes]
        self._undo_stack.append(patch)
        if len(self._undo_stack) > MAX_UNDO_STATES:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _commit_array(self, new_array):
        self.raster.array = new_array
        self._write_working_layer(new_array)

    def _write_working_layer(self, array):
        """Writes 'array' to a NEW GeoTIFF file (incremental name) and
        replaces the working raster layer with one pointing to the new
        file, then removes the previous layer/file.

        Necessary on Windows: GDAL cannot rewrite/overwrite a file that
        QGIS still holds open as an active layer's source ('Permission
        denied' error on Create/Delete) — on Linux/Mac this usually isn't
        an issue, but always writing a new file is the more robust
        approach regardless of operating system."""
        self._commit_counter = getattr(self, "_commit_counter", 0) + 1
        new_path = os.path.join(self._work_dir, f"working_dem_{self._commit_counter:04d}.tif")
        self.raster.save_as(new_path)

        old_layer = self.raster_layer
        old_path = self.working_path

        self.working_path = new_path
        self.raster_layer = QgsRasterLayer(new_path, "Working DEM (SC4 Terrain Composer)")
        QgsProject.instance().addMapLayer(self.raster_layer)

        if old_layer is not None:
            QgsProject.instance().removeMapLayer(old_layer.id())
        self.canvas.refresh()

        if old_path and old_path != new_path:
            try:
                os.remove(old_path)  # best-effort: if still locked, no
            except OSError:          # matter, it just stays in the temp folder
                pass

    def undo(self):
        if not self._undo_stack:
            return
        patch = self._undo_stack.pop()
        # before restoring, capture the CURRENT state at the same
        # coordinates: needed to be able to "redo" afterwards
        redo_patch = [(b, self.raster.array[b[0]:b[1], b[2]:b[3]].copy()) for b, _ in patch]
        self._redo_stack.append(redo_patch)
        self._apply_patch(patch)

    def redo(self):
        if not self._redo_stack:
            return
        patch = self._redo_stack.pop()
        undo_patch = [(b, self.raster.array[b[0]:b[1], b[2]:b[3]].copy()) for b, _ in patch]
        self._undo_stack.append(undo_patch)
        self._apply_patch(patch)

    def _apply_patch(self, patch):
        """Rewrites only the boxes contained in 'patch' inside
        self.raster.array (in-place modification: no full-canvas copy),
        then refreshes the layer displayed in QGIS."""
        for (r0, r1, c0, c1), data in patch:
            self.raster.array[r0:r1, c0:c1] = data
        self._write_working_layer(self.raster.array)

    # ------------------------------------------------------------------
    # 4. SC4 export
    # ------------------------------------------------------------------
    def _prepare_export_array(self, target_w: int = None, target_h: int = None, points_rc=None):
        """Thin wrapper around terrain_ops.prepare_export: supplies the
        current UI settings (sea level, rotation, autocrop) and the
        GDAL-backed resample function it needs for the pre-rotation
        downsample step. Kept as a wrapper — not duplicated logic — so
        the image export and the towns-JSON export always go through the
        EXACT same crop/rotate/crop pipeline; see prepare_export's
        docstring for why that matters.

        Returns (ready_array, mapped_points, info_text)."""
        sea_level = self.spin_sea_level.value()
        rotation = self.spin_export_rotation.value()
        do_autocrop = self.chk_autocrop.isChecked()

        return terrain_ops.prepare_export(
            self.raster.array, sea_level=sea_level, rotation_degrees=rotation,
            do_autocrop=do_autocrop, target_w=target_w, target_h=target_h,
            points_rc=points_rc, resample_fn=resample_array,
        )

    def _resolve_sc4_scale(self) -> float:
        """Converts the scale-factor combo's text into a numeric value,
        using the SAME fallback logic as SC4Mapper (GetImageFactor in
        app.py): if the text is one of the known presets it uses that
        value, otherwise it tries to parse it as a hand-typed number,
        otherwise it falls back to 'Default factor' (3.0)."""
        text = self.combo_sc4_scale.currentText().strip()
        if text in terrain_ops.SC4MAPPER_SCALE_FACTORS:
            return terrain_ops.SC4MAPPER_SCALE_FACTORS[text]
        try:
            return float(text)
        except ValueError:
            return terrain_ops.SC4MAPPER_SCALE_FACTORS["Default factor"]

    def export_sc4(self):
        if self.raster is None:
            QMessageBox.warning(self, "Warning", "Load a DEM first.")
            return

        out_path, _ = QFileDialog.getSaveFileName(self, "Export SC4 heightmap", "", "Bitmap (*.bmp)")
        if not out_path:
            return

        n_x = self.spin_n_citta_x.value()
        n_y = self.spin_n_citta_y.value()
        celle = self.spin_celle_citta.value()
        target_w = celle * n_x + 1
        target_h = celle * n_y + 1

        array, _, prep_info = self._prepare_export_array(target_w, target_h)
        elev_max = float(np.nanmax(array))
        scale_factor = self._resolve_sc4_scale()
        coastal_offset = self.spin_sc4_coastal_offset.value()
        suggested = terrain_ops.suggest_scale(elev_max, scale_factor, coastal_offset)
        if suggested is not None:
            scale_factor = suggested
            QMessageBox.information(
                self, "Scale factor adjusted automatically",
                f"The maximum elevation ({elev_max:.0f} m) plus the "
                f"{coastal_offset:.0f} m coastal offset exceeds the "
                f"representable ceiling with the chosen factor. Using "
                f"{scale_factor} as scale factor: type EXACTLY this "
                f"number into SC4Mapper's 'Scale factor' field (it's an "
                f"editable field, it also accepts values not in the "
                f"preset list)."
            )

        resampled = resample_array(array, target_w, target_h)
        gray8 = terrain_ops.elevation_to_sc4_gray(resampled, scale_factor, coastal_offset)
        save_uint8_bmp(gray8, out_path)

        QMessageBox.information(
            self, "Export complete",
            f"Saved: {out_path}"
        )

    def export_openttd(self):
        if self.raster is None:
            QMessageBox.warning(self, "Warning", "Load a DEM first.")
            return

        out_path, _ = QFileDialog.getSaveFileName(self, "Export OpenTTD heightmap", "", "PNG (*.png)")
        if not out_path:
            return

        target_w = int(self.combo_ottd_w.currentText())
        target_h = int(self.combo_ottd_h.currentText())
        sea_level = self.spin_sea_level.value()

        array, _, prep_info = self._prepare_export_array(target_w, target_h)
        elev_max_dem = float(np.nanmax(array))
        max_elev_setting = self.spin_ottd_max_elev.value()
        max_elev = elev_max_dem if max_elev_setting <= 0 else max_elev_setting
        if max_elev <= sea_level:
            QMessageBox.warning(
                self, "Warning",
                "The maximum elevation must be greater than sea level."
            )
            return

        resampled = resample_array(array, target_w, target_h)
        gray8 = terrain_ops.elevation_to_openttd_gray(resampled, sea_level, max_elev)
        save_uint8_png(gray8, out_path)

        QMessageBox.information(
            self, "Export complete",
            f"Saved: {out_path}\n"
            f"Pre-processing: {prep_info}\n"
            f"Final size: {target_w}x{target_h} px (power of 2, as "
            f"required by OpenTTD)\n"
            f"Sea level: {sea_level} m -> black (0)\n"
            f"Maximum elevation: {max_elev:.0f} m -> white (255)\n\n"
            f"In OpenTTD: Scenario Editor → Load Heightmap, choose "
            f"{target_w}x{target_h} (or an equivalent ratio) as the map "
            f"size for a 1:1 match."
        )

    def _refresh_towns_field_combo(self, layer):
        self.combo_towns_field.clear()
        self.combo_towns_pop_field.clear()
        self.combo_towns_pop_field.addItem("(none)")
        if layer is None:
            return
        field_names = [f.name() for f in layer.fields()]
        self.combo_towns_field.addItems(field_names)
        self.combo_towns_pop_field.addItems(field_names)
        for guess in ("name", "NAME", "Name", "town", "place", "TOWN", "settlement"):
            if guess in field_names:
                self.combo_towns_field.setCurrentText(guess)
                break
        for guess in ("population", "POPULATION", "Population", "pop", "POP"):
            if guess in field_names:
                self.combo_towns_pop_field.setCurrentText(guess)
                break

    def export_openttd_towns_json(self):
        if self.raster is None:
            QMessageBox.warning(self, "Warning", "Load a DEM first.")
            return
        layer = self.combo_towns_layer.currentLayer()
        if layer is None:
            QMessageBox.warning(self, "Warning", "Select a towns layer first.")
            return
        name_field = self.combo_towns_field.currentText()
        if not name_field:
            QMessageBox.warning(self, "Warning", "Select a name field first.")
            return
        pop_field = self.combo_towns_pop_field.currentText()
        if pop_field == "(none)":
            pop_field = None
        default_pop = self.spin_towns_default_pop.value()
        city_threshold = self.spin_towns_city_threshold.value()

        out_path, _ = QFileDialog.getSaveFileName(self, "Export towns JSON", "", "JSON (*.json)")
        if not out_path:
            return

        target_w = int(self.combo_ottd_w.currentText())
        target_h = int(self.combo_ottd_h.currentText())

        # collect every point (as centroid, so polygons/multipoints work
        # too) in the raster's own pixel coordinates
        raster_crs = self.raster_layer.crs()
        layer_crs = layer.crs()
        transform = None
        if layer_crs != raster_crs:
            transform = QgsCoordinateTransform(layer_crs, raster_crs, QgsProject.instance())

        names, populations, points_rc = [], [], []
        for feat in layer.getFeatures():
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            pt = geom.centroid().asPoint()
            if transform is not None:
                try:
                    pt = transform.transform(pt)
                except Exception:
                    continue
            col, row = self.raster.geo_to_pixel(pt.x(), pt.y())
            name_val = feat[name_field]
            names.append("" if name_val is None else str(name_val))
            pop_val = feat[pop_field] if pop_field else None
            try:
                populations.append(float(pop_val) if pop_val not in (None, "") else default_pop)
            except (TypeError, ValueError):
                populations.append(default_pop)
            points_rc.append((row, col))

        if not points_rc:
            QMessageBox.warning(self, "Warning", "No point features found in the selected layer.")
            return

        # runs the EXACT same crop/rotation pipeline as the PNG export
        # above (see terrain_ops.prepare_export), so town positions stay
        # correct even with cropping/rotation applied
        array, mapped_points, prep_info = self._prepare_export_array(
            target_w, target_h, points_rc=points_rc)

        scale_y = target_h / array.shape[0]
        scale_x = target_w / array.shape[1]

        towns = []
        for name, pop, pt in zip(names, populations, mapped_points):
            if pt is None:
                continue
            final_row = pt[0] * scale_y
            final_col = pt[1] * scale_x
            if not (0 <= final_row < target_h and 0 <= final_col < target_w):
                continue
            # OpenTTD's official town-data format (docs/importing_town_data.md):
            # x/y are PROPORTIONS (0-1) of the map size, and — this is the
            # part that silently misplaces every town if missed — X and Y
            # are SWAPPED relative to normal image coordinates ("In OpenTTD,
            # X and Y axis are swapped compared to most image editing
            # programs... swap them before importing or towns won't line
            # up with your heightmap"). Since points are tracked here as
            # (row, col), that swap falls out naturally: OpenTTD 'x' comes
            # from the row (vertical) position, 'y' from the column
            # (horizontal) one.
            towns.append({
                "name": name,
                "population": round(pop, 2),
                "city": pop >= city_threshold,
                "x": final_row / target_h,
                "y": final_col / target_w,
            })

        if not towns:
            QMessageBox.warning(
                self, "Warning",
                "None of the towns in the selected layer fall inside the exported area."
            )
            return

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(towns, f, indent=4, ensure_ascii=False)

        QMessageBox.information(
            self, "Export complete",
            f"Saved {len(towns)} town(s) (out of {len(points_rc)} in the "
            f"source layer) to:\n{out_path}\n\n"
            f"This is OpenTTD's official town-data format: in the "
            f"Scenario Editor, after loading the matching {target_w}x{target_h} "
            f"heightmap (clockwise rotation), open Town Generation → "
            f"Load from file and pick this JSON."
        )
