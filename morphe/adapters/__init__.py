"""
Morphe CAD Adapters

This package contains adapters for various CAD applications:
- common: Shared client/server infrastructure
- freecad: FreeCAD adapter
- fusion: Autodesk Fusion 360 adapter
- inventor: Autodesk Inventor adapter
- solidworks: SolidWorks adapter

CAD-specific adapters are not imported by default to avoid
requiring CAD dependencies. Import them directly:
    from morphe.adapters.freecad import FreeCADClient
    from morphe.adapters.fusion import FusionClient
"""

from . import common

__all__ = ["common"]
