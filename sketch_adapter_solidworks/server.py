"""
RPC server for SolidWorks sketch adapter.

This module provides a simple XML-RPC server that exposes the sketch adapter
functionality over the network. Unlike FreeCAD, this server runs as an
external process that connects to SolidWorks via COM.

Usage:
    Run this as a standalone Python script (requires Windows with SolidWorks):

    >>> from sketch_adapter_solidworks.server import start_server
    >>> start_server()

    Or run directly:
    $ python -m sketch_adapter_solidworks.server

    The server runs in blocking mode by default. For background mode:

    >>> start_server(blocking=False)

    To stop the server:

    >>> from sketch_adapter_solidworks.server import stop_server
    >>> stop_server()
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any
from xmlrpc.server import SimpleXMLRPCServer

from sketch_adapter_common import QuietRequestHandler

from .adapter import SOLIDWORKS_AVAILABLE, SolidWorksAdapter

if TYPE_CHECKING:
    pass

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9878
SERVER_VERSION = "1.0.0"

# Global server instance
_server: SimpleXMLRPCServer | None = None
_server_thread: threading.Thread | None = None

# Global adapter instance (reused for connection persistence)
_adapter: SolidWorksAdapter | None = None


def _init_com() -> None:
    """Initialize COM for the current thread."""
    if SOLIDWORKS_AVAILABLE:
        try:
            import pythoncom

            pythoncom.CoInitialize()
        except Exception:
            pass


def _uninit_com() -> None:
    """Uninitialize COM for the current thread."""
    if SOLIDWORKS_AVAILABLE:
        try:
            import pythoncom

            pythoncom.CoUninitialize()
        except Exception:
            pass


def _get_adapter() -> SolidWorksAdapter:
    """Get or create the SolidWorks adapter instance."""
    global _adapter
    if _adapter is None:
        _adapter = SolidWorksAdapter()
    return _adapter


def _get_sketch_to_json() -> Any:
    """Get sketch_to_json function."""
    from sketch_canonical import sketch_to_json

    return sketch_to_json


def _get_sketch_from_json() -> Any:
    """Get sketch_from_json function."""
    from sketch_canonical import sketch_from_json

    return sketch_from_json


def _iterate_features(doc: Any) -> list[Any]:
    """
    Iterate through all features in a document.

    Uses FeatureByPositionReverse since FirstFeature/GetNextFeature
    don't work with late-bound COM.

    Args:
        doc: SolidWorks document object

    Returns:
        List of feature objects
    """
    features = []
    try:
        fm = doc.FeatureManager
        feature_count = fm.GetFeatureCount(True)
        for i in range(feature_count):
            try:
                feat = doc.FeatureByPositionReverse(i)
                if feat is not None:
                    features.append(feat)
            except Exception:
                pass
    except Exception:
        pass
    return features


def _find_feature_by_name(doc: Any, name: str) -> Any | None:
    """
    Find a feature by name in a document.

    Args:
        doc: SolidWorks document object
        name: Feature name to find

    Returns:
        Feature object or None if not found
    """
    for feat in _iterate_features(doc):
        try:
            if feat.Name == name:
                return feat
        except Exception:
            pass
    return None


def _get_feature_type(feat: Any) -> str:
    """Get the type name of a feature, handling property/method ambiguity."""
    try:
        feat_type = feat.GetTypeName2
        if callable(feat_type):
            return feat_type()
        return feat_type
    except Exception:
        return ""


def list_sketches() -> list[dict]:
    """
    List all sketches in the active SolidWorks document.

    Returns:
        List of dicts with sketch info:
        [{"name": str, "feature_name": str, "geometry_count": int}]
    """
    if not SOLIDWORKS_AVAILABLE:
        raise RuntimeError("SolidWorks is not available")

    _init_com()
    try:
        from .adapter import get_solidworks_application

        app = get_solidworks_application()
        doc = app.ActiveDoc
        if doc is None:
            return []

        # Get feature count for iteration
        try:
            fm = doc.FeatureManager
            feature_count = fm.GetFeatureCount(True)
        except Exception:
            return []

        sketches = []
        # Use FeatureByPositionReverse with index since FirstFeature/GetNextFeature
        # don't work with late-bound COM
        for i in range(feature_count):
            try:
                feat = doc.FeatureByPositionReverse(i)
                if feat is None:
                    continue

                feat_type = feat.GetTypeName2
                if callable(feat_type):
                    feat_type = feat_type()

                feat_name = feat.Name

                if feat_type == "ProfileFeature" or "Sketch" in feat_name:
                    geom_count = 0
                    try:
                        # In late-bound COM, feat.GetSpecificFeature2 (without calling)
                        # returns a wrapper with sketch methods
                        sketch_obj = feat.GetSpecificFeature2
                        if sketch_obj is not None:
                            segments = getattr(sketch_obj, "GetSketchSegments", None)
                            if segments is not None:
                                if callable(segments):
                                    segments = segments()
                                if segments:
                                    geom_count = len(segments)
                    except Exception:
                        pass

                    sketches.append(
                        {
                            "name": feat_name,
                            "feature_name": feat_name,
                            "geometry_count": geom_count,
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
        sketch_name: Name of the sketch feature in SolidWorks

    Returns:
        JSON string of the canonical sketch
    """
    if not SOLIDWORKS_AVAILABLE:
        raise RuntimeError("SolidWorks is not available")

    _init_com()
    try:
        from .adapter import get_solidworks_application

        app = get_solidworks_application()
        doc = app.ActiveDoc
        if doc is None:
            raise RuntimeError("No active document")

        # Find the sketch by iterating features directly
        feat = None
        fm = doc.FeatureManager
        feature_count = fm.GetFeatureCount(True)
        for i in range(feature_count):
            f = doc.FeatureByPositionReverse(i)
            if f is not None and f.Name == sketch_name:
                feat = f
                break

        if feat is None:
            raise ValueError(f"Sketch '{sketch_name}' not found")

        # In late-bound COM, accessing feat.GetSpecificFeature2 (without calling it)
        # returns a CDispatch wrapper that provides access to sketch methods
        sketch_obj = feat.GetSpecificFeature2

        # Set up the adapter with this sketch
        adapter = _get_adapter()
        adapter._document = doc
        adapter._sketch = sketch_obj
        adapter._sketch_manager = doc.SketchManager

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
    if not SOLIDWORKS_AVAILABLE:
        raise RuntimeError("SolidWorks is not available")

    _init_com()
    try:
        from .adapter import get_solidworks_application

        # Standard reference planes always available
        planes = [
            {"id": "XY", "name": "Front Plane", "type": "construction"},
            {"id": "XZ", "name": "Top Plane", "type": "construction"},
            {"id": "YZ", "name": "Right Plane", "type": "construction"},
        ]

        app = get_solidworks_application()
        doc = app.ActiveDoc
        if doc:
            # Get reference planes from feature tree
            for feat in _iterate_features(doc):
                try:
                    feat_type = _get_feature_type(feat)
                    if feat_type == "RefPlane":
                        # Skip the standard planes we already added
                        name = feat.Name
                        if name not in ("Front Plane", "Top Plane", "Right Plane"):
                            planes.append({
                                "id": f"RefPlane:{name}",
                                "name": name,
                                "type": "reference",
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
        plane: Optional plane ID (from list_planes). Defaults to "XY" (Front Plane).

    Returns:
        Name of the created sketch feature
    """
    if not SOLIDWORKS_AVAILABLE:
        raise RuntimeError("SolidWorks is not available")

    _init_com()
    try:
        from .adapter import get_solidworks_application

        sketch_from_json = _get_sketch_from_json()
        sketch_doc = sketch_from_json(json_str)

        if sketch_name:
            sketch_doc.name = sketch_name

        adapter = _get_adapter()

        # Resolve plane
        plane_to_use = plane or "XY"
        if plane and plane.startswith("RefPlane:"):
            # Resolve reference plane
            try:
                plane_name = plane.split(":", 1)[1]
                app = get_solidworks_application()
                doc = app.ActiveDoc
                if doc:
                    for feat in _iterate_features(doc):
                        feat_type = _get_feature_type(feat)
                        if feat_type == "RefPlane" and feat.Name == plane_name:
                            plane_to_use = feat
                            break
            except Exception:
                plane_to_use = "XY"

        adapter.create_sketch(sketch_doc.name, plane=plane_to_use)
        adapter.load_sketch(sketch_doc)

        # Exit sketch edit mode to commit the sketch
        if adapter._sketch_manager is not None:
            adapter._sketch_manager.InsertSketch(True)

        # Return the sketch name
        if adapter._sketch is not None:
            try:
                return adapter._sketch.Name
            except Exception:
                pass
        return sketch_doc.name
    finally:
        _uninit_com()


def get_solver_status(sketch_name: str) -> dict:
    """
    Get the solver status for a sketch.

    Args:
        sketch_name: Name of the sketch feature

    Returns:
        Dict with "status" and "dof" keys
    """
    if not SOLIDWORKS_AVAILABLE:
        raise RuntimeError("SolidWorks is not available")

    _init_com()
    try:
        from .adapter import get_solidworks_application

        app = get_solidworks_application()
        doc = app.ActiveDoc
        if doc is None:
            raise RuntimeError("No active document")

        # Find the sketch by iterating features directly
        feat = None
        fm = doc.FeatureManager
        feature_count = fm.GetFeatureCount(True)
        for i in range(feature_count):
            f = doc.FeatureByPositionReverse(i)
            if f is not None and f.Name == sketch_name:
                feat = f
                break

        if feat is None:
            raise ValueError(f"Sketch '{sketch_name}' not found")

        # In late-bound COM, feat.GetSpecificFeature2 (without calling) provides sketch access
        sketch_obj = feat.GetSpecificFeature2

        adapter = _get_adapter()
        adapter._document = doc
        adapter._sketch = sketch_obj
        status, dof = adapter.get_solver_status()
        return {"status": status.name, "dof": dof}
    finally:
        _uninit_com()


def get_status() -> dict:
    """
    Get server and SolidWorks status.

    Returns:
        Dict with version and document info
    """
    result: dict = {
        "server_version": SERVER_VERSION,
        "solidworks_available": SOLIDWORKS_AVAILABLE,
    }

    if SOLIDWORKS_AVAILABLE:
        _init_com()
        try:
            from .adapter import get_solidworks_application

            app = get_solidworks_application()
            try:
                version = app.RevisionNumber
                result["solidworks_version"] = version
            except Exception:
                result["solidworks_version"] = "unknown"

            doc = app.ActiveDoc
            if doc:
                # GetPathName and GetTitle may be properties or methods depending on binding
                try:
                    path = doc.GetPathName
                    if callable(path):
                        path = path()
                    title = doc.GetTitle
                    if callable(title):
                        title = title()
                    result["active_document"] = path or title
                except Exception:
                    result["active_document"] = "unknown"
                # Count sketches using the helper function
                sketch_count = 0
                for feat in _iterate_features(doc):
                    try:
                        feat_type = _get_feature_type(feat)
                        feat_name = feat.Name
                        if feat_type == "ProfileFeature" or "Sketch" in feat_name:
                            sketch_count += 1
                    except Exception:
                        pass
                result["sketch_count"] = sketch_count
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
    without initializing COM or talking to SolidWorks.

    Returns:
        Dict with server_version, solidworks_available, status
    """
    return {
        "server_version": SERVER_VERSION,
        "solidworks_available": SOLIDWORKS_AVAILABLE,
        "status": "ok",
    }


def open_sketch_in_edit_mode(sketch_name: str) -> bool:
    """
    Open a sketch in edit mode for editing.

    Args:
        sketch_name: Name of the sketch feature

    Returns:
        True if successful
    """
    if not SOLIDWORKS_AVAILABLE:
        raise RuntimeError("SolidWorks is not available")

    _init_com()
    try:
        from .adapter import get_solidworks_application

        app = get_solidworks_application()
        doc = app.ActiveDoc
        if doc is None:
            raise RuntimeError("No active document")

        # Find the sketch by name
        feat = _find_feature_by_name(doc, sketch_name)
        if feat is None:
            raise ValueError(f"Sketch '{sketch_name}' not found")

        # Select the feature
        select_method = feat.Select2
        if callable(select_method):
            select_method(False, 0)

        # Edit the sketch
        edit_method = doc.EditSketch
        if callable(edit_method):
            edit_method()

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
        port: Port to bind to (default: 9878)
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

    print(f"SolidWorks sketch server started on {host}:{port}")

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
