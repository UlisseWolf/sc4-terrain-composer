# -*- coding: utf-8 -*-
def classFactory(iface):
    from .sc4_terrain_composer import Sc4TerrainComposerPlugin
    return Sc4TerrainComposerPlugin(iface)
