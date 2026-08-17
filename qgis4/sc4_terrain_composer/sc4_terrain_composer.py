# -*- coding: utf-8 -*-
"""
sc4_terrain_composer.py
================================================================================
Main plugin class: only responsible for registering/removing the action
in QGIS's menu and toolbar, and for opening/closing the dock widget
(dock_widget.py) where all the logic lives.
================================================================================
"""

import os
from qgis.PyQt.QtWidgets import QAction
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import Qt
from qgis.core import Qgis


class Sc4TerrainComposerPlugin:

    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dock = None

    def initGui(self):
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
        self.action = QAction(icon, "SC4 Terrain Composer", self.iface.mainWindow())
        self.action.triggered.connect(self.toggle_dock)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&SC4 Terrain Composer", self.action)

    def unload(self):
        self.iface.removePluginMenu("&SC4 Terrain Composer", self.action)
        self.iface.removeToolBarIcon(self.action)
        if self.dock is not None:
            self.iface.removeDockWidget(self.dock)
            self.dock = None

    def toggle_dock(self):
        if self.dock is None:
            from .dock_widget import Sc4TerrainComposerDock
            self.dock = Sc4TerrainComposerDock(self.iface)
            self.iface.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        else:
            self.dock.setVisible(not self.dock.isVisible())
