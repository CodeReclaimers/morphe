"""
Base client for CAD RPC servers.

This module provides the base client class that all CAD-specific clients inherit from.
It handles connection management, thread safety, and common RPC operations.
"""

from __future__ import annotations

import http.client
import threading
import xmlrpc.client
from typing import Any, ClassVar

from morphe import SketchDocument, sketch_from_json, sketch_to_json

DEFAULT_HOST = "localhost"
DEFAULT_TIMEOUT = 30.0  # Longer timeout for sketch operations


class TimeoutTransport(xmlrpc.client.Transport):
    """XML-RPC transport with configurable timeout."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        super().__init__()
        self.timeout = timeout

    def make_connection(self, host: tuple | str) -> http.client.HTTPConnection:  # type: ignore[override]
        conn = super().make_connection(host)
        conn.timeout = self.timeout
        return conn


class BaseCADClient:
    """Base client for communicating with CAD RPC servers.

    This client is thread-safe - concurrent calls from multiple threads
    are serialized using an internal lock.

    Subclasses should define:
        CAD_NAME: Human-readable name of the CAD application (e.g., "Fusion 360")
        DEFAULT_PORT: Default port for the RPC server
    """

    CAD_NAME: ClassVar[str] = "CAD"
    DEFAULT_PORT: ClassVar[int] = 9876

    def __init__(self, host: str = DEFAULT_HOST, port: int | None = None):
        """
        Initialize the client.

        Args:
            host: Server host (default: localhost)
            port: Server port (default: class-specific DEFAULT_PORT)
        """
        self.host = host
        self.port = port if port is not None else self.DEFAULT_PORT
        self._proxy: xmlrpc.client.ServerProxy | None = None
        self._timeout = DEFAULT_TIMEOUT
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        """Get the server URL."""
        return f"http://{self.host}:{self.port}"

    def connect(self, timeout: float = DEFAULT_TIMEOUT) -> bool:
        """
        Connect to the server.

        Args:
            timeout: Connection test timeout in seconds (operations use DEFAULT_TIMEOUT)

        Returns:
            True if connection successful, False otherwise
        """
        with self._lock:
            try:
                # Test connection with the specified timeout
                test_transport = TimeoutTransport(timeout)
                test_proxy = xmlrpc.client.ServerProxy(
                    self.url,
                    transport=test_transport,
                    allow_none=True,
                )
                test_proxy.ping()

                # Connection successful - create the real proxy with default timeout
                # This ensures operations don't timeout prematurely
                transport = TimeoutTransport(DEFAULT_TIMEOUT)
                self._proxy = xmlrpc.client.ServerProxy(
                    self.url,
                    transport=transport,
                    allow_none=True,
                )
                return True
            except Exception:
                self._proxy = None
                return False

    def disconnect(self) -> None:
        """Disconnect from the server."""
        with self._lock:
            self._proxy = None

    def is_connected(self) -> bool:
        """
        Check if connected to the server.

        This uses the lightweight ping endpoint that doesn't require
        heavy operations on the CAD side.
        """
        with self._lock:
            if self._proxy is None:
                return False
            try:
                self._proxy.ping()
                return True
            except Exception:
                return False

    def _ensure_connected(self) -> None:
        """Raise ConnectionError if not connected."""
        if self._proxy is None:
            raise ConnectionError(f"Not connected to {self.CAD_NAME} server")

    def get_status(self) -> dict:
        """
        Get server and CAD application status.

        Returns:
            Dict with server_version, cad_version, active_document, sketch_count
        """
        with self._lock:
            self._ensure_connected()
            return self._proxy.get_status()  # type: ignore[union-attr, return-value]

    def ping(self) -> dict:
        """
        Lightweight health check.

        This doesn't require heavy operations on the CAD side, making it
        faster and more reliable for connection polling.

        Returns:
            Dict with server_version, cad_available, status
        """
        with self._lock:
            self._ensure_connected()
            return self._proxy.ping()  # type: ignore[union-attr, return-value]

    def list_sketches(self) -> list[dict]:
        """
        List all sketches in the active document.

        Returns:
            List of dicts with sketch info (keys vary by CAD application)
        """
        with self._lock:
            self._ensure_connected()
            return self._proxy.list_sketches()  # type: ignore[union-attr, return-value]

    def list_planes(self) -> list[dict]:
        """
        List available planes for sketch creation.

        Returns:
            List of dicts with keys: id, name, type
        """
        with self._lock:
            self._ensure_connected()
            return self._proxy.list_planes()  # type: ignore[union-attr, return-value]

    def export_sketch(self, sketch_name: str) -> SketchDocument:
        """
        Export a sketch from the CAD application.

        Args:
            sketch_name: Name of the sketch

        Returns:
            SketchDocument with the exported sketch
        """
        with self._lock:
            self._ensure_connected()
            json_str: str = self._proxy.export_sketch(sketch_name)  # type: ignore[union-attr, assignment]
        return sketch_from_json(json_str)

    def export_sketch_json(self, sketch_name: str) -> str:
        """
        Export a sketch as JSON string.

        Args:
            sketch_name: Name of the sketch

        Returns:
            JSON string of the canonical sketch
        """
        with self._lock:
            self._ensure_connected()
            result: str = self._proxy.export_sketch(sketch_name)  # type: ignore[union-attr, assignment]
            return result

    def import_sketch(
        self, sketch: SketchDocument, name: str | None = None, plane: str | None = None
    ) -> str:
        """
        Import a sketch into the CAD application.

        Args:
            sketch: SketchDocument to import
            name: Optional name override (uses sketch.name if not provided)
            plane: Optional plane ID (from list_planes). Defaults to "XY".

        Returns:
            Name of the created sketch object
        """
        json_str = sketch_to_json(sketch)
        with self._lock:
            self._ensure_connected()
            return self._proxy.import_sketch(json_str, name, plane)  # type: ignore[union-attr, return-value]

    def import_sketch_json(
        self, json_str: str, name: str | None = None, plane: str | None = None
    ) -> str:
        """
        Import a sketch from JSON string.

        Args:
            json_str: JSON string of the canonical sketch
            name: Optional name override
            plane: Optional plane ID (from list_planes). Defaults to "XY".

        Returns:
            Name of the created sketch object
        """
        with self._lock:
            self._ensure_connected()
            return self._proxy.import_sketch(json_str, name, plane)  # type: ignore[union-attr, return-value]

    def get_solver_status(self, sketch_name: str) -> tuple[str, int]:
        """
        Get solver status for a sketch.

        Args:
            sketch_name: Name of the sketch

        Returns:
            Tuple of (status_name, degrees_of_freedom)
        """
        with self._lock:
            self._ensure_connected()
            result: dict[str, Any] = self._proxy.get_solver_status(sketch_name)  # type: ignore[union-attr, assignment]
        return result["status"], result["dof"]

    def open_sketch(self, sketch_name: str) -> bool:
        """
        Open a sketch in edit mode.

        Args:
            sketch_name: Name of the sketch

        Returns:
            True if successful
        """
        with self._lock:
            self._ensure_connected()
            return self._proxy.open_sketch_in_sketcher(sketch_name)  # type: ignore[union-attr, return-value]
