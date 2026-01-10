"""
Client for connecting to the Fusion 360 RPC server.

This module provides a client class for communicating with a Fusion 360 instance
running the sketch RPC server.

Usage:
    from sketch_adapter_fusion.client import FusionClient

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

import socket
import xmlrpc.client

from sketch_canonical import SketchDocument, sketch_from_json, sketch_to_json

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9879
DEFAULT_TIMEOUT = 30.0  # Longer timeout for sketch operations


class FusionClient:
    """Client for communicating with the Fusion 360 RPC server."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        """
        Initialize the client.

        Args:
            host: Server host (default: localhost)
            port: Server port (default: 9879)
        """
        self.host = host
        self.port = port
        self._proxy: xmlrpc.client.ServerProxy | None = None
        self._timeout = DEFAULT_TIMEOUT

    @property
    def url(self) -> str:
        """Get the server URL."""
        return f"http://{self.host}:{self.port}"

    def connect(self, timeout: float = DEFAULT_TIMEOUT) -> bool:
        """
        Connect to the Fusion 360 server.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if connection successful, False otherwise
        """
        self._timeout = timeout
        try:
            # Create proxy with timeout
            transport = TimeoutTransport(timeout)
            self._proxy = xmlrpc.client.ServerProxy(
                self.url,
                transport=transport,
                allow_none=True,
            )
            # Test connection by calling get_status
            self._proxy.get_status()
            return True
        except Exception:
            self._proxy = None
            return False

    def disconnect(self) -> None:
        """Disconnect from the server."""
        self._proxy = None

    def is_connected(self) -> bool:
        """
        Check if connected to the server.

        This actually tests the connection by calling get_status.
        """
        if self._proxy is None:
            return False
        try:
            self._proxy.get_status()
            return True
        except Exception:
            return False

    def _ensure_connected(self) -> None:
        """Raise ConnectionError if not connected."""
        if self._proxy is None:
            raise ConnectionError("Not connected to Fusion 360 server")

    def get_status(self) -> dict:
        """
        Get server and Fusion 360 status.

        Returns:
            Dict with server_version, fusion_version, active_document, sketch_count
        """
        self._ensure_connected()
        return self._proxy.get_status()  # type: ignore[union-attr, return-value]

    def list_sketches(self) -> list[dict]:
        """
        List all sketches in the active Fusion 360 document.

        Returns:
            List of dicts with keys: name, constraint_count, geometry_count
        """
        self._ensure_connected()
        return self._proxy.list_sketches()  # type: ignore[union-attr, return-value]

    def export_sketch(self, sketch_name: str) -> SketchDocument:
        """
        Export a sketch from Fusion 360.

        Args:
            sketch_name: Name of the sketch in Fusion 360

        Returns:
            SketchDocument with the exported sketch
        """
        self._ensure_connected()
        json_str = self._proxy.export_sketch(sketch_name)  # type: ignore[union-attr]
        return sketch_from_json(json_str)

    def export_sketch_json(self, sketch_name: str) -> str:
        """
        Export a sketch from Fusion 360 as JSON string.

        Args:
            sketch_name: Name of the sketch in Fusion 360

        Returns:
            JSON string of the canonical sketch
        """
        self._ensure_connected()
        return self._proxy.export_sketch(sketch_name)  # type: ignore[union-attr]

    def import_sketch(
        self, sketch: SketchDocument, name: str | None = None
    ) -> str:
        """
        Import a sketch into Fusion 360.

        Args:
            sketch: SketchDocument to import
            name: Optional name override (uses sketch.name if not provided)

        Returns:
            Name of the created sketch object in Fusion 360
        """
        self._ensure_connected()
        json_str = sketch_to_json(sketch)
        return self._proxy.import_sketch(json_str, name)  # type: ignore[union-attr]

    def import_sketch_json(self, json_str: str, name: str | None = None) -> str:
        """
        Import a sketch into Fusion 360 from JSON string.

        Args:
            json_str: JSON string of the canonical sketch
            name: Optional name override

        Returns:
            Name of the created sketch object in Fusion 360
        """
        self._ensure_connected()
        return self._proxy.import_sketch(json_str, name)  # type: ignore[union-attr]

    def get_solver_status(self, sketch_name: str) -> tuple[str, int]:
        """
        Get solver status for a sketch.

        Args:
            sketch_name: Name of the sketch in Fusion 360

        Returns:
            Tuple of (status_name, degrees_of_freedom)
        """
        self._ensure_connected()
        result = self._proxy.get_solver_status(sketch_name)  # type: ignore[union-attr]
        return result["status"], result["dof"]

    def open_sketch(self, sketch_name: str) -> bool:
        """
        Open a sketch in edit mode.

        This activates sketch edit mode in Fusion 360.

        Args:
            sketch_name: Name of the sketch in Fusion 360

        Returns:
            True if successful
        """
        self._ensure_connected()
        return self._proxy.open_sketch_in_sketcher(sketch_name)  # type: ignore[union-attr]


class TimeoutTransport(xmlrpc.client.Transport):
    """XML-RPC transport with configurable timeout."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        super().__init__()
        self.timeout = timeout

    def make_connection(self, host: tuple | str) -> socket.socket:
        conn = super().make_connection(host)
        conn.timeout = self.timeout  # type: ignore[union-attr]
        return conn


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
