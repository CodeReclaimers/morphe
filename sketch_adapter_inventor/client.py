"""
Client for connecting to the Inventor RPC server.

This module provides a client class for communicating with an Inventor instance
running the sketch RPC server.

Usage:
    from sketch_adapter_inventor.client import InventorClient

    client = InventorClient()
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
DEFAULT_PORT = 9877
DEFAULT_TIMEOUT = 30.0  # Longer timeout for sketch operations


class InventorClient:
    """Client for communicating with the Inventor RPC server."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        """
        Initialize the client.

        Args:
            host: Server host (default: localhost)
            port: Server port (default: 9877)
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
        Connect to the Inventor server.

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
            raise ConnectionError("Not connected to Inventor server")

    def get_status(self) -> dict:
        """
        Get server and Inventor status.

        Returns:
            Dict with server_version, inventor_version, active_document, sketch_count
        """
        self._ensure_connected()
        return self._proxy.get_status()  # type: ignore[union-attr, return-value]

    def list_sketches(self) -> list[dict]:
        """
        List all sketches in the active Inventor document.

        Returns:
            List of dicts with keys: name, label, constraint_count, geometry_count
        """
        self._ensure_connected()
        return self._proxy.list_sketches()  # type: ignore[union-attr, return-value]

    def export_sketch(self, sketch_name: str) -> SketchDocument:
        """
        Export a sketch from Inventor.

        Args:
            sketch_name: Name of the sketch in Inventor

        Returns:
            SketchDocument with the exported sketch
        """
        self._ensure_connected()
        json_str = self._proxy.export_sketch(sketch_name)  # type: ignore[union-attr]
        return sketch_from_json(json_str)

    def export_sketch_json(self, sketch_name: str) -> str:
        """
        Export a sketch from Inventor as JSON string.

        Args:
            sketch_name: Name of the sketch in Inventor

        Returns:
            JSON string of the canonical sketch
        """
        self._ensure_connected()
        return self._proxy.export_sketch(sketch_name)  # type: ignore[union-attr]

    def import_sketch(
        self, sketch: SketchDocument, name: str | None = None
    ) -> str:
        """
        Import a sketch into Inventor.

        Args:
            sketch: SketchDocument to import
            name: Optional name override (uses sketch.name if not provided)

        Returns:
            Name of the created sketch object in Inventor
        """
        self._ensure_connected()
        json_str = sketch_to_json(sketch)
        return self._proxy.import_sketch(json_str, name)  # type: ignore[union-attr]

    def import_sketch_json(self, json_str: str, name: str | None = None) -> str:
        """
        Import a sketch into Inventor from JSON string.

        Args:
            json_str: JSON string of the canonical sketch
            name: Optional name override

        Returns:
            Name of the created sketch object in Inventor
        """
        self._ensure_connected()
        return self._proxy.import_sketch(json_str, name)  # type: ignore[union-attr]

    def get_solver_status(self, sketch_name: str) -> tuple[str, int]:
        """
        Get solver status for a sketch.

        Args:
            sketch_name: Name of the sketch in Inventor

        Returns:
            Tuple of (status_name, degrees_of_freedom)
        """
        self._ensure_connected()
        result = self._proxy.get_solver_status(sketch_name)  # type: ignore[union-attr]
        return result["status"], result["dof"]

    def open_sketch(self, sketch_name: str) -> bool:
        """
        Open a sketch in edit mode.

        This enters sketch edit mode in Inventor.

        Args:
            sketch_name: Name of the sketch in Inventor

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
    Quick check if an Inventor server is running.

    Args:
        host: Server host
        port: Server port
        timeout: Connection timeout in seconds

    Returns:
        True if server is responding, False otherwise
    """
    client = InventorClient(host, port)
    return client.connect(timeout)


def quick_export(
    sketch_name: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> SketchDocument:
    """
    Quick helper to export a sketch from Inventor.

    Args:
        sketch_name: Name of the sketch in Inventor
        host: Server host
        port: Server port

    Returns:
        SketchDocument with the exported sketch

    Raises:
        ConnectionError: If cannot connect to server
    """
    client = InventorClient(host, port)
    if not client.connect():
        raise ConnectionError(f"Cannot connect to Inventor server at {host}:{port}")
    return client.export_sketch(sketch_name)


def quick_import(
    sketch: SketchDocument,
    name: str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> str:
    """
    Quick helper to import a sketch into Inventor.

    Args:
        sketch: SketchDocument to import
        name: Optional name override
        host: Server host
        port: Server port

    Returns:
        Name of the created sketch object in Inventor

    Raises:
        ConnectionError: If cannot connect to server
    """
    client = InventorClient(host, port)
    if not client.connect():
        raise ConnectionError(f"Cannot connect to Inventor server at {host}:{port}")
    return client.import_sketch(sketch, name)
