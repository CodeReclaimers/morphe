"""
FreeCAD Sketch Adapter

Adapter for creating and manipulating sketches in FreeCAD using the
canonical sketch schema.

Usage:
    from sketch_adapter_freecad import FreeCADAdapter

    # With FreeCAD available
    adapter = FreeCADAdapter()
    adapter.create_sketch("MySketch")
    adapter.load_sketch(canonical_sketch)

Note: This adapter requires FreeCAD to be installed and importable.
When FreeCAD is not available, a MockFreeCADAdapter is provided for testing.
"""

from .adapter import FREECAD_AVAILABLE, FreeCADAdapter
from .vertex_map import VertexMap, get_vertex_index

__all__ = [
    "FreeCADAdapter",
    "FREECAD_AVAILABLE",
    "VertexMap",
    "get_vertex_index",
]
