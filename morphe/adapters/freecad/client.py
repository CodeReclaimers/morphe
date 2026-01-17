"""
Client for connecting to the FreeCAD RPC server.

This module provides a client class for communicating with a FreeCAD instance
running the sketch RPC server.

Usage:
    from adapter_freecad.client import FreeCADClient

    client = FreeCADClient()
    if client.connect():
        # List sketches
        for sketch in client.list_sketches():
            print(f"{sketch['name']}: {sketch['geometry_count']} geometries")

        # Export a sketch
        doc = client.export_sketch("Sketch")
        print(f"Exported: {len(doc.primitives)} primitives")

        # Import a sketch
        client.import_sketch(doc, name="ImportedSketch")
"""

from __future__ import annotations

from morphe import SketchDocument
from morphe.adapters.common import BaseCADClient

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9876


class FreeCADClient(BaseCADClient):
    """Client for communicating with the FreeCAD RPC server.

    This client is thread-safe - concurrent calls from multiple threads
    are serialized using an internal lock.
    """

    CAD_NAME = "FreeCAD"
    DEFAULT_PORT = 9876


def check_server(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 1.0
) -> bool:
    """
    Quick check if a FreeCAD server is running.

    Args:
        host: Server host
        port: Server port
        timeout: Connection timeout in seconds

    Returns:
        True if server is responding, False otherwise
    """
    client = FreeCADClient(host, port)
    return client.connect(timeout)


def quick_export(
    sketch_name: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> SketchDocument:
    """
    Quick helper to export a sketch from FreeCAD.

    Args:
        sketch_name: Name of the sketch in FreeCAD
        host: Server host
        port: Server port

    Returns:
        SketchDocument with the exported sketch

    Raises:
        ConnectionError: If cannot connect to server
    """
    client = FreeCADClient(host, port)
    if not client.connect():
        raise ConnectionError(f"Cannot connect to FreeCAD server at {host}:{port}")
    return client.export_sketch(sketch_name)


def quick_import(
    sketch: SketchDocument,
    name: str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> str:
    """
    Quick helper to import a sketch into FreeCAD.

    Args:
        sketch: SketchDocument to import
        name: Optional name override
        host: Server host
        port: Server port

    Returns:
        Name of the created sketch object in FreeCAD

    Raises:
        ConnectionError: If cannot connect to server
    """
    client = FreeCADClient(host, port)
    if not client.connect():
        raise ConnectionError(f"Cannot connect to FreeCAD server at {host}:{port}")
    return client.import_sketch(sketch, name)
