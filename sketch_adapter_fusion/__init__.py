"""Fusion 360 adapter for canonical sketch representation.

This module provides the FusionAdapter class for translating between
the canonical sketch representation and Autodesk Fusion 360's native
sketch API.

Example usage (within Fusion 360):

    from sketch_adapter_fusion import FusionAdapter
    from sketch_canonical.document import SketchDocument
    from sketch_canonical.primitives import Line
    from sketch_canonical.types import Point2D

    # Create adapter (requires running within Fusion 360)
    adapter = FusionAdapter()

    # Create a new sketch
    adapter.create_sketch("MySketch", plane="XY")

    # Add geometry
    line = Line(start=Point2D(0, 0), end=Point2D(100, 0))
    adapter.add_primitive(line)

    # Or load an entire SketchDocument
    doc = SketchDocument(name="ImportedSketch")
    # ... add primitives and constraints to doc ...
    adapter.load_sketch(doc)

    # Export back to canonical format
    exported_doc = adapter.export_sketch()

Note: This adapter must be run within Fusion 360's Python environment
where the 'adsk' module is available.
"""

from .adapter import FusionAdapter
from .vertex_map import VertexMap, get_point_from_sketch_entity

__all__ = [
    "FusionAdapter",
    "VertexMap",
    "get_point_from_sketch_entity",
]
