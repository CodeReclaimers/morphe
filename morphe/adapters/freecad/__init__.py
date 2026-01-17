"""
FreeCAD Sketch Adapter

Adapter for creating and manipulating sketches in FreeCAD using the
Morphe sketch schema.

Usage:
    from morphe.adapters.freecad import FreeCADAdapter

    # With FreeCAD available
    adapter = FreeCADAdapter()
    adapter.create_sketch("MySketch")
    adapter.load_sketch(sketch_doc)

RPC Server (run inside FreeCAD):
    from morphe.adapters.freecad import start_server
    start_server()  # Starts on localhost:9876

RPC Client (run outside FreeCAD):
    from morphe.adapters.freecad import FreeCADClient
    client = FreeCADClient()
    if client.connect():
        sketches = client.list_sketches()

Note: This adapter requires FreeCAD to be installed and importable.
When FreeCAD is not available, a MockFreeCADAdapter is provided for testing.
"""

from .adapter import FREECAD_AVAILABLE, FreeCADAdapter
from .client import FreeCADClient, check_server, quick_export, quick_import
from .server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    is_server_running,
    start_server,
    stop_server,
    toggle_server,
)
from .vertex_map import VertexMap, get_vertex_index

__all__ = [
    # Adapter
    "FreeCADAdapter",
    "FREECAD_AVAILABLE",
    "VertexMap",
    "get_vertex_index",
    # Server (run inside FreeCAD)
    "start_server",
    "stop_server",
    "toggle_server",
    "is_server_running",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    # Client (run outside FreeCAD)
    "FreeCADClient",
    "check_server",
    "quick_export",
    "quick_import",
]
