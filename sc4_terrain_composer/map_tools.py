# -*- coding: utf-8 -*-
"""
map_tools.py
================================================================================
A single custom map tool: captures a click on the canvas and passes it
back as map coordinates, used at two points in the workflow:
  - to mark the ANCHOR POINT of the selection to be moved
  - to mark the DESTINATION where it should be moved to
Everything else in the interaction (selecting blobs, drawing the lasso)
deliberately reuses QGIS's native tools (click-select on the attribute
table/canvas, "Add Freehand Feature" from the digitizing toolbar) instead
of reinventing them: less custom code, fewer bugs, behavior QGIS users
already know.
================================================================================
"""

from qgis.gui import QgsMapToolEmitPoint
from qgis.PyQt.QtCore import pyqtSignal


class PointPickTool(QgsMapToolEmitPoint):
    """Minimal map tool: on every click, emits the point (in map
    coordinates, i.e. the project's CRS) via the point_picked signal."""

    point_picked = pyqtSignal(object)  # QgsPointXY

    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas

    def canvasReleaseEvent(self, event):
        point = self.toMapCoordinates(event.pos())
        self.point_picked.emit(point)
