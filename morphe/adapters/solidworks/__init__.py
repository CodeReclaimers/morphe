"""SolidWorks adapter for canonical sketch representation.

This module provides the SolidWorksAdapter class for translating between
the canonical sketch representation and SolidWorks's native sketch API
via COM automation.

Example usage (on Windows with SolidWorks installed):

    from morphe.adapters.solidworks import SolidWorksAdapter
    from morphe import SketchDocument, Line, Point2D

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

RPC Server (run as external process with SolidWorks running):
    from morphe.adapters.solidworks import start_server
    start_server()  # Starts on localhost:9878

RPC Client (connect to server):
    from morphe.adapters.solidworks import SolidWorksClient
    client = SolidWorksClient()
    if client.connect():
        sketches = client.list_sketches()

Requirements:
    - Windows operating system
    - SolidWorks installed
    - pywin32 package (pip install pywin32)

Note: This adapter must be run on Windows with SolidWorks installed.
The adapter will attempt to connect to a running SolidWorks instance,
or start a new one if none is available.
"""

from .adapter import SOLIDWORKS_AVAILABLE, SolidWorksAdapter, get_solidworks_application
from .client import SolidWorksClient, check_server, quick_export, quick_import
from .server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    is_server_running,
    start_server,
    stop_server,
    toggle_server,
)
from .vertex_map import (
    get_point_type_for_sketch_point,
    get_sketch_point_from_entity,
    get_valid_point_types,
)

__all__ = [
    # Adapter
    "SolidWorksAdapter",
    "SOLIDWORKS_AVAILABLE",
    "get_solidworks_application",
    "get_sketch_point_from_entity",
    "get_point_type_for_sketch_point",
    "get_valid_point_types",
    # Server (run as external process)
    "start_server",
    "stop_server",
    "toggle_server",
    "is_server_running",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    # Client (connect to server)
    "SolidWorksClient",
    "check_server",
    "quick_export",
    "quick_import",
]
