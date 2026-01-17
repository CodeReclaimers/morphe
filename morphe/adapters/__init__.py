"""
Morphe CAD Adapters

This package contains adapters for various CAD applications:
- common: Shared client/server infrastructure
- freecad: FreeCAD adapter
- fusion: Autodesk Fusion 360 adapter
- inventor: Autodesk Inventor adapter
- solidworks: SolidWorks adapter
"""

from . import common, freecad, fusion, inventor, solidworks

__all__ = ["common", "freecad", "fusion", "inventor", "solidworks"]
