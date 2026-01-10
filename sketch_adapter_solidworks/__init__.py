"""SolidWorks adapter for canonical sketch representation.

This module provides the SolidWorksAdapter class for translating between
the canonical sketch representation and SolidWorks's native sketch API
via COM automation.

Example usage (on Windows with SolidWorks installed):

    from sketch_adapter_solidworks import SolidWorksAdapter
    from sketch_canonical import SketchDocument, Line, Point2D

    # Create adapter (connects to running SolidWorks instance)
    adapter = SolidWorksAdapter()

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

Requirements:
    - Windows operating system
    - SolidWorks installed
    - pywin32 package (pip install pywin32)

Note: This adapter must be run on Windows with SolidWorks installed.
The adapter will attempt to connect to a running SolidWorks instance,
or start a new one if none is available.
"""

from .adapter import SOLIDWORKS_AVAILABLE, SolidWorksAdapter, get_solidworks_application
from .vertex_map import (
    get_point_type_for_sketch_point,
    get_sketch_point_from_entity,
    get_valid_point_types,
)

__all__ = [
    "SolidWorksAdapter",
    "SOLIDWORKS_AVAILABLE",
    "get_solidworks_application",
    "get_sketch_point_from_entity",
    "get_point_type_for_sketch_point",
    "get_valid_point_types",
]
