"""
RPC server for Inventor sketch adapter.

This module provides a simple XML-RPC server that exposes the sketch adapter
functionality over the network. Unlike FreeCAD, this server runs as an
external process that connects to Inventor via COM.

Usage:
    Run this as a standalone Python script (requires Windows with Inventor):

    >>> from adapter_inventor.server import start_server
    >>> start_server()

    Or run directly:
    $ python -m adapter_inventor.server

    The server runs in blocking mode by default. For background mode:

    >>> start_server(blocking=False)

    To stop the server:

    >>> from adapter_inventor.server import stop_server
    >>> stop_server()
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any
from xmlrpc.server import SimpleXMLRPCServer

from morphe.adapters.common import QuietRequestHandler

from .adapter import INVENTOR_AVAILABLE, InventorAdapter

if TYPE_CHECKING:
    pass

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9877
SERVER_VERSION = "1.0.0"

# Global server instance
_server: SimpleXMLRPCServer | None = None
_server_thread: threading.Thread | None = None

# Global adapter instance (reused for connection persistence)
_adapter: InventorAdapter | None = None


def _init_com() -> None:
    """Initialize COM for the current thread."""
    if INVENTOR_AVAILABLE:
        try:
            import pythoncom

            pythoncom.CoInitialize()
        except Exception:
            pass


def _uninit_com() -> None:
    """Uninitialize COM for the current thread."""
    if INVENTOR_AVAILABLE:
        try:
            import pythoncom

            pythoncom.CoUninitialize()
        except Exception:
            pass


def _get_adapter() -> InventorAdapter:
    """Get or create the Inventor adapter instance."""
    global _adapter
    if _adapter is None:
        _adapter = InventorAdapter()
    return _adapter


def _get_sketch_to_json() -> Any:
    """Get sketch_to_json function."""
    from morphe import sketch_to_json

    return sketch_to_json


def _get_sketch_from_json() -> Any:
    """Get sketch_from_json function."""
    from morphe import sketch_from_json

    return sketch_from_json


def list_sketches() -> list[dict]:
    """
    List all sketches in the active Inventor document.

    Returns:
        List of dicts with sketch info:
        [{"name": str, "label": str, "constraint_count": int, "geometry_count": int}]
    """
    if not INVENTOR_AVAILABLE:
        raise RuntimeError("Inventor is not available")

    _init_com()
    try:
        adapter = _get_adapter()
        if adapter._document is None:
            adapter._ensure_document()

        doc = adapter._document
        if doc is None:
            return []

        sketches = []
        try:
            part_def = doc.ComponentDefinition
            for sketch in part_def.Sketches:
                sketches.append(
                    {
                        "name": sketch.Name,
                        "label": sketch.Name,
                        "constraint_count": sketch.GeometricConstraints.Count
                        + sketch.DimensionConstraints.Count,
                        "geometry_count": sketch.SketchLines.Count
                        + sketch.SketchCircles.Count
                        + sketch.SketchArcs.Count
                        + sketch.SketchPoints.Count,
                    }
                )
        except Exception:
            pass

        return sketches
    finally:
        _uninit_com()


def export_sketch(sketch_name: str) -> str:
    """
    Export a sketch to canonical JSON format.

    Args:
        sketch_name: Name of the sketch object in Inventor

    Returns:
        JSON string of the canonical sketch
    """
    if not INVENTOR_AVAILABLE:
        raise RuntimeError("Inventor is not available")

    _init_com()
    try:
        adapter = _get_adapter()
        if adapter._document is None:
            adapter._ensure_document()

        doc = adapter._document
        if doc is None:
            raise RuntimeError("No active document")

        # Find the sketch by name
        part_def = doc.ComponentDefinition
        sketch_obj = None
        for sketch in part_def.Sketches:
            if sketch.Name == sketch_name:
                sketch_obj = sketch
                break

        if sketch_obj is None:
            raise ValueError(f"Sketch '{sketch_name}' not found")

        # Set up the adapter with this sketch
        adapter._sketch = sketch_obj
        adapter._sketch_def = sketch_obj

        exported = adapter.export_sketch()
        sketch_to_json = _get_sketch_to_json()
        return sketch_to_json(exported)
    finally:
        _uninit_com()


def list_planes() -> list[dict]:
    """
    List available planes for sketch creation.

    Returns:
        List of dicts with plane info:
        [{"id": str, "name": str, "type": str}]
    """
    if not INVENTOR_AVAILABLE:
        raise RuntimeError("Inventor is not available")

    _init_com()
    try:
        # Standard work planes
        planes = [
            {"id": "XY", "name": "XY Plane", "type": "construction"},
            {"id": "XZ", "name": "XZ Plane", "type": "construction"},
            {"id": "YZ", "name": "YZ Plane", "type": "construction"},
        ]

        adapter = _get_adapter()
        if adapter._document is None:
            adapter._ensure_document()

        doc = adapter._document
        if doc:
            try:
                part_def = doc.ComponentDefinition
                # Add work planes
                for i in range(1, part_def.WorkPlanes.Count + 1):
                    wp = part_def.WorkPlanes.Item(i)
                    planes.append({
                        "id": f"WorkPlane:{wp.Name}",
                        "name": wp.Name,
                        "type": "workplane",
                    })
            except Exception:
                pass

        return planes
    finally:
        _uninit_com()


def import_sketch(
    json_str: str, sketch_name: str | None = None, plane: str | None = None
) -> str:
    """
    Import a sketch from canonical JSON format.

    Creates a new sketch in the active document (or creates a new document
    if none exists).

    Args:
        json_str: JSON string of the canonical sketch
        sketch_name: Optional name for the new sketch (uses name from JSON if not provided)
        plane: Optional plane ID (from list_planes). Defaults to "XY".

    Returns:
        Name of the created sketch object
    """
    if not INVENTOR_AVAILABLE:
        raise RuntimeError("Inventor is not available")

    _init_com()
    try:
        sketch_from_json = _get_sketch_from_json()
        sketch_doc = sketch_from_json(json_str)

        if sketch_name:
            sketch_doc.name = sketch_name

        adapter = _get_adapter()

        # Resolve plane
        plane_to_use = plane or "XY"
        if plane and plane.startswith("WorkPlane:"):
            # Resolve work plane reference
            try:
                wp_name = plane.split(":", 1)[1]
                part_def = adapter._document.ComponentDefinition
                for i in range(1, part_def.WorkPlanes.Count + 1):
                    wp = part_def.WorkPlanes.Item(i)
                    if wp.Name == wp_name:
                        plane_to_use = wp
                        break
            except Exception:
                plane_to_use = "XY"

        adapter.create_sketch(sketch_doc.name, plane=plane_to_use)
        adapter.load_sketch(sketch_doc)

        # Return the sketch name
        if adapter._sketch is not None:
            return adapter._sketch.Name
        return sketch_doc.name
    finally:
        _uninit_com()


def get_solver_status(sketch_name: str) -> dict:
    """
    Get the solver status for a sketch.

    Args:
        sketch_name: Name of the sketch object

    Returns:
        Dict with "status" and "dof" keys
    """
    if not INVENTOR_AVAILABLE:
        raise RuntimeError("Inventor is not available")

    _init_com()
    try:
        adapter = _get_adapter()
        if adapter._document is None:
            adapter._ensure_document()

        doc = adapter._document
        if doc is None:
            raise RuntimeError("No active document")

        # Find the sketch by name
        part_def = doc.ComponentDefinition
        sketch_obj = None
        for sketch in part_def.Sketches:
            if sketch.Name == sketch_name:
                sketch_obj = sketch
                break

        if sketch_obj is None:
            raise ValueError(f"Sketch '{sketch_name}' not found")

        adapter._sketch = sketch_obj
        adapter._sketch_def = sketch_obj
        status, dof = adapter.get_solver_status()
        return {"status": status.name, "dof": dof}
    finally:
        _uninit_com()


def get_status() -> dict:
    """
    Get server and Inventor status.

    Returns:
        Dict with version and document info
    """
    result: dict = {
        "server_version": SERVER_VERSION,
        "inventor_available": INVENTOR_AVAILABLE,
    }

    if INVENTOR_AVAILABLE:
        _init_com()
        try:
            from .adapter import get_inventor_application

            app = get_inventor_application()
            try:
                result["inventor_version"] = app.SoftwareVersion.DisplayVersion
            except Exception:
                result["inventor_version"] = "unknown"

            doc = app.ActiveDocument
            if doc:
                result["active_document"] = doc.DisplayName
                # Count sketches
                try:
                    part_def = doc.ComponentDefinition
                    result["sketch_count"] = part_def.Sketches.Count
                except Exception:
                    result["sketch_count"] = 0
            else:
                result["active_document"] = None
                result["sketch_count"] = 0
        except Exception as e:
            result["error"] = str(e)
        finally:
            _uninit_com()

    return result


def ping() -> dict:
    """
    Lightweight health check that doesn't require COM operations.

    This is useful for connection polling as it responds immediately
    without initializing COM or talking to Inventor.

    Returns:
        Dict with server_version, inventor_available, status
    """
    return {
        "server_version": SERVER_VERSION,
        "inventor_available": INVENTOR_AVAILABLE,
        "status": "ok",
    }


def open_sketch_in_edit_mode(sketch_name: str) -> bool:
    """
    Open a sketch in edit mode for editing.

    Args:
        sketch_name: Name of the sketch object

    Returns:
        True if successful
    """
    if not INVENTOR_AVAILABLE:
        raise RuntimeError("Inventor is not available")

    _init_com()
    try:
        adapter = _get_adapter()
        if adapter._document is None:
            adapter._ensure_document()

        doc = adapter._document
        if doc is None:
            raise RuntimeError("No active document")

        # Find the sketch by name
        part_def = doc.ComponentDefinition
        sketch_obj = None
        for sketch in part_def.Sketches:
            if sketch.Name == sketch_name:
                sketch_obj = sketch
                break

        if sketch_obj is None:
            raise ValueError(f"Sketch '{sketch_name}' not found")

        # Edit the sketch
        sketch_obj.Edit()
        return True
    finally:
        _uninit_com()


def start_server(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, blocking: bool = True
) -> bool:
    """
    Start the RPC server.

    Args:
        host: Host to bind to (default: localhost)
        port: Port to bind to (default: 9877)
        blocking: If True, block the main thread. If False, run in background thread.

    Returns:
        True if server started successfully
    """
    global _server, _server_thread

    if _server is not None:
        print(f"Server already running on {host}:{port}")
        return True

    try:
        _server = SimpleXMLRPCServer(
            (host, port), requestHandler=QuietRequestHandler, allow_none=True
        )
    except OSError as e:
        print(f"Failed to start server: {e}")
        return False

    # Register functions
    _server.register_function(list_sketches, "list_sketches")
    _server.register_function(list_planes, "list_planes")
    _server.register_function(export_sketch, "export_sketch")
    _server.register_function(import_sketch, "import_sketch")
    _server.register_function(get_solver_status, "get_solver_status")
    _server.register_function(get_status, "get_status")
    _server.register_function(ping, "ping")
    _server.register_function(open_sketch_in_edit_mode, "open_sketch_in_sketcher")

    print(f"Inventor sketch server started on {host}:{port}")

    if blocking:
        try:
            _server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped by user")
            stop_server()
    else:
        _server_thread = threading.Thread(target=_server.serve_forever, daemon=True)
        _server_thread.start()
        print("Server running in background thread")

    return True


def stop_server() -> None:
    """Stop the RPC server."""
    global _server, _server_thread, _adapter

    if _server is not None:
        _server.shutdown()
        _server = None
        _server_thread = None
        _adapter = None
        print("Server stopped")
    else:
        print("Server not running")


def is_server_running() -> bool:
    """Check if the server is currently running."""
    return _server is not None


def toggle_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Toggle the server on/off."""
    if is_server_running():
        stop_server()
    else:
        start_server(host, port)


# Allow running as a script
if __name__ == "__main__":
    start_server(blocking=True)
