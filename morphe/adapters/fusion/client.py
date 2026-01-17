"""
Client for connecting to the Fusion 360 RPC server.

This module provides a client class for communicating with a Fusion 360 instance
running the sketch RPC server.

Usage:
    from adapter_fusion.client import FusionClient

    client = FusionClient()
    if client.connect():
        # List sketches
        for sketch in client.list_sketches():
            print(f"{sketch['name']}: {sketch['geometry_count']} geometries")

        # Export a sketch
        doc = client.export_sketch("Sketch1")
        print(f"Exported: {len(doc.primitives)} primitives")

        # Import a sketch
        client.import_sketch(doc, name="ImportedSketch")
"""

from __future__ import annotations

from morphe import SketchDocument
from morphe.adapters.common import BaseCADClient

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9879


class FusionClient(BaseCADClient):
    """Client for communicating with the Fusion 360 RPC server.

    This client is thread-safe - concurrent calls from multiple threads
    are serialized using an internal lock.
    """

    CAD_NAME = "Fusion 360"
    DEFAULT_PORT = 9879


def check_server(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 1.0
) -> bool:
    """
    Quick check if a Fusion 360 server is running.

    Args:
        host: Server host
        port: Server port
        timeout: Connection timeout in seconds

    Returns:
        True if server is responding, False otherwise
    """
    client = FusionClient(host, port)
    return client.connect(timeout)


def quick_export(
    sketch_name: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> SketchDocument:
    """
    Quick helper to export a sketch from Fusion 360.

    Args:
        sketch_name: Name of the sketch in Fusion 360
        host: Server host
        port: Server port

    Returns:
        SketchDocument with the exported sketch

    Raises:
        ConnectionError: If cannot connect to server
    """
    client = FusionClient(host, port)
    if not client.connect():
        raise ConnectionError(f"Cannot connect to Fusion 360 server at {host}:{port}")
    return client.export_sketch(sketch_name)


def quick_import(
    sketch: SketchDocument,
    name: str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> str:
    """
    Quick helper to import a sketch into Fusion 360.

    Args:
        sketch: SketchDocument to import
        name: Optional name override
        host: Server host
        port: Server port

    Returns:
        Name of the created sketch object in Fusion 360

    Raises:
        ConnectionError: If cannot connect to server
    """
    client = FusionClient(host, port)
    if not client.connect():
        raise ConnectionError(f"Cannot connect to Fusion 360 server at {host}:{port}")
    return client.import_sketch(sketch, name)
