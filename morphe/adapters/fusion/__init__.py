"""Fusion 360 adapter for canonical sketch representation.

This module provides the FusionAdapter class for translating between
the canonical sketch representation and Autodesk Fusion 360's native
sketch API.

Example usage (within Fusion 360):

    from morphe.adapters.fusion import FusionAdapter
    from morphe import SketchDocument, Line, Point2D

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

RPC Server (run inside Fusion 360 as script/add-in):
    from morphe.adapters.fusion import start_server
    start_server()  # Starts on localhost:9879

RPC Client (connect from external Python):
    from morphe.adapters.fusion import FusionClient
    client = FusionClient()
    if client.connect():
        sketches = client.list_sketches()

Note: This adapter must be run within Fusion 360's Python environment
where the 'adsk' module is available.
"""

from .adapter import FusionAdapter
from .client import FusionClient, check_server, quick_export, quick_import
from .server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    is_server_running,
    start_server,
    stop_server,
    toggle_server,
)
from .vertex_map import VertexMap, get_point_from_sketch_entity

__all__ = [
    # Adapter
    "FusionAdapter",
    "VertexMap",
    "get_point_from_sketch_entity",
    # Server (run inside Fusion 360)
    "start_server",
    "stop_server",
    "toggle_server",
    "is_server_running",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    # Client (connect from external Python)
    "FusionClient",
    "check_server",
    "quick_export",
    "quick_import",
]
