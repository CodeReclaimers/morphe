"""
RPC server for FreeCAD sketch adapter.

This module provides a simple XML-RPC server that runs inside FreeCAD
and exposes the sketch adapter functionality over the network.

Usage:
    Run this as a FreeCAD macro or from the FreeCAD Python console:

    >>> from sketch_adapter_freecad.server import start_server
    >>> start_server()

    The server runs in a background thread, so FreeCAD remains responsive.
    To stop the server:

    >>> from sketch_adapter_freecad.server import stop_server
    >>> stop_server()

    For blocking mode (e.g., when running FreeCAD headless):

    >>> start_server(blocking=True)
"""

from __future__ import annotations

import queue
import threading
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer

# These imports only work inside FreeCAD
try:
    import FreeCAD

    FREECAD_AVAILABLE = True
except ImportError:
    FreeCAD = None  # type: ignore[assignment]
    FREECAD_AVAILABLE = False

# Try to import Qt for main thread execution
try:
    from PySide import QtCore

    QT_AVAILABLE = True
except ImportError:
    try:
        from PySide2 import QtCore  # type: ignore[import-not-found]

        QT_AVAILABLE = True
    except ImportError:
        QtCore = None  # type: ignore[assignment]
        QT_AVAILABLE = False

if TYPE_CHECKING:
    pass

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9876
SERVER_VERSION = "1.0.0"

# Global server instance
_server: SimpleXMLRPCServer | None = None
_server_thread: threading.Thread | None = None


# =============================================================================
# Thread-safe execution helpers
# =============================================================================

# Global invoker - must be created on main thread during start_server()
_invoker: Any = None  # Type is MainThreadInvoker when Qt is available


def _create_invoker_class() -> type | None:
    """Create the MainThreadInvoker class if Qt is available."""
    if not QT_AVAILABLE:
        return None

    class MainThreadInvoker(QtCore.QObject):
        """QObject that lives on the main thread and executes functions."""

        # Signal to request function execution
        execute_requested = QtCore.Signal(object, object)

        def __init__(self) -> None:
            super().__init__()
            # Connect signal to slot - this creates a queued connection
            # that will execute the slot on this object's thread (main thread)
            self.execute_requested.connect(
                self._do_execute, QtCore.Qt.QueuedConnection
            )

        def _do_execute(
            self, func: Callable[[], Any], result_queue: queue.Queue
        ) -> None:
            """Execute the function and put result in queue. Runs on main thread."""
            try:
                result = func()
                result_queue.put((True, result))
            except Exception as e:
                tb = traceback.format_exc()
                result_queue.put((False, (e, tb)))

    return MainThreadInvoker


# Create the class (will be None if Qt not available)
_MainThreadInvoker = _create_invoker_class()


def _init_invoker() -> None:
    """Initialize the invoker on the main thread. Called during start_server()."""
    global _invoker
    if _invoker is None and _MainThreadInvoker is not None:
        _invoker = _MainThreadInvoker()
        print("Main thread invoker initialized")


def _get_invoker() -> Any:
    """Get the main thread invoker."""
    global _invoker
    if _invoker is None:
        raise RuntimeError(
            "Invoker not initialized. Make sure start_server() was called "
            "from the main thread."
        )
    return _invoker


class MainThreadExecutor:
    """
    Helper to execute functions on the main GUI thread.

    FreeCAD document operations must run on the main thread. This class
    provides a way to queue functions from the RPC server thread and
    execute them safely on the main thread.
    """

    def __init__(self) -> None:
        self._result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue()

    def execute(self, func: Callable[[], Any], timeout: float = 30.0) -> Any:
        """
        Execute a function on the main thread and return the result.

        Args:
            func: Function to execute (no arguments)
            timeout: Maximum time to wait for result

        Returns:
            The function's return value

        Raises:
            RuntimeError: If execution fails or times out
        """
        if not QT_AVAILABLE:
            # No Qt available, just run directly (may crash, but worth trying)
            return func()

        # Clear any stale results
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break

        # Get the invoker and emit signal to execute on main thread
        invoker = _get_invoker()
        invoker.execute_requested.emit(func, self._result_queue)

        # Wait for result
        try:
            success, value = self._result_queue.get(timeout=timeout)
        except queue.Empty as e:
            raise RuntimeError(f"Operation timed out after {timeout}s") from e

        if success:
            return value
        else:
            exc, tb = value
            raise RuntimeError(f"Operation failed: {exc}\n{tb}")


# Global executor instance
_executor = MainThreadExecutor()


class QuietRequestHandler(SimpleXMLRPCRequestHandler):
    """Request handler that suppresses logging."""

    def log_message(self, format: str, *args: object) -> None:
        pass  # Suppress default logging


def _get_adapter() -> FreeCADAdapter:  # noqa: F821
    """Get a FreeCADAdapter instance. Import here to avoid circular imports."""
    from .adapter import FreeCADAdapter

    return FreeCADAdapter()


def _get_sketch_to_json() -> callable:
    """Get sketch_to_json function."""
    from sketch_canonical import sketch_to_json

    return sketch_to_json


def _get_sketch_from_json() -> callable:
    """Get sketch_from_json function."""
    from sketch_canonical import sketch_from_json

    return sketch_from_json


def list_sketches() -> list[dict]:
    """
    List all sketches in the active FreeCAD document.

    Returns:
        List of dicts with sketch info:
        [{"name": str, "label": str, "constraint_count": int, "geometry_count": int}]
    """
    if not FREECAD_AVAILABLE:
        raise RuntimeError("FreeCAD is not available")

    doc = FreeCAD.ActiveDocument
    if doc is None:
        return []

    sketches = []
    for obj in doc.Objects:
        if obj.TypeId == "Sketcher::SketchObject":
            sketches.append(
                {
                    "name": obj.Name,
                    "label": obj.Label,
                    "constraint_count": obj.ConstraintCount,
                    "geometry_count": obj.GeometryCount,
                }
            )

    return sketches


def export_sketch(sketch_name: str) -> str:
    """
    Export a sketch to canonical JSON format.

    Args:
        sketch_name: Name of the sketch object in FreeCAD

    Returns:
        JSON string of the canonical sketch
    """
    if not FREECAD_AVAILABLE:
        raise RuntimeError("FreeCAD is not available")

    doc = FreeCAD.ActiveDocument
    if doc is None:
        raise RuntimeError("No active document")

    sketch_obj = doc.getObject(sketch_name)
    if sketch_obj is None:
        raise ValueError(f"Sketch '{sketch_name}' not found")

    if sketch_obj.TypeId != "Sketcher::SketchObject":
        raise ValueError(f"Object '{sketch_name}' is not a sketch")

    adapter = _get_adapter()
    adapter._document = doc
    adapter._sketch = sketch_obj

    exported = adapter.export_sketch()
    sketch_to_json = _get_sketch_to_json()
    return sketch_to_json(exported)


def import_sketch(json_str: str, sketch_name: str | None = None) -> str:
    """
    Import a sketch from canonical JSON format.

    Creates a new sketch in the active document (or creates a new document
    if none exists).

    Args:
        json_str: JSON string of the canonical sketch
        sketch_name: Optional name for the new sketch (uses name from JSON if not provided)

    Returns:
        Name of the created sketch object
    """
    if not FREECAD_AVAILABLE:
        raise RuntimeError("FreeCAD is not available")

    sketch_from_json = _get_sketch_from_json()
    sketch_doc = sketch_from_json(json_str)

    if sketch_name:
        sketch_doc.name = sketch_name

    def do_import() -> str:
        adapter = _get_adapter()
        adapter.create_sketch(sketch_doc.name)
        adapter.load_sketch(sketch_doc)
        return adapter._sketch.Name

    # Execute on main thread to avoid crashes
    return _executor.execute(do_import)


def get_solver_status(sketch_name: str) -> dict:
    """
    Get the solver status for a sketch.

    Args:
        sketch_name: Name of the sketch object

    Returns:
        Dict with "status" and "dof" keys
    """
    if not FREECAD_AVAILABLE:
        raise RuntimeError("FreeCAD is not available")

    doc = FreeCAD.ActiveDocument
    if doc is None:
        raise RuntimeError("No active document")

    sketch_obj = doc.getObject(sketch_name)
    if sketch_obj is None:
        raise ValueError(f"Sketch '{sketch_name}' not found")

    adapter = _get_adapter()
    adapter._document = doc
    adapter._sketch = sketch_obj

    status, dof = adapter.get_solver_status()
    return {"status": status.name, "dof": dof}


def get_status() -> dict:
    """
    Get server and FreeCAD status.

    Returns:
        Dict with version and document info
    """
    result: dict = {
        "server_version": SERVER_VERSION,
        "freecad_available": FREECAD_AVAILABLE,
    }

    if FREECAD_AVAILABLE:
        version = FreeCAD.Version()
        result["freecad_version"] = f"{version[0]}.{version[1]}"
        doc = FreeCAD.ActiveDocument
        if doc:
            result["active_document"] = doc.Name
            result["sketch_count"] = len(
                [o for o in doc.Objects if o.TypeId == "Sketcher::SketchObject"]
            )
        else:
            result["active_document"] = None
            result["sketch_count"] = 0

    return result


def open_sketch_in_sketcher(sketch_name: str) -> bool:
    """
    Open a sketch in the Sketcher workbench for editing.

    Args:
        sketch_name: Name of the sketch object

    Returns:
        True if successful
    """
    if not FREECAD_AVAILABLE:
        raise RuntimeError("FreeCAD is not available")

    def do_open() -> bool:
        try:
            import FreeCADGui
        except ImportError as e:
            raise RuntimeError(
                "FreeCAD GUI is not available (running headless?)"
            ) from e

        doc = FreeCAD.ActiveDocument
        if doc is None:
            raise RuntimeError("No active document")

        sketch_obj = doc.getObject(sketch_name)
        if sketch_obj is None:
            raise ValueError(f"Sketch '{sketch_name}' not found")

        # Activate Sketcher workbench and open sketch for editing
        FreeCADGui.activateWorkbench("SketcherWorkbench")
        FreeCADGui.ActiveDocument.setEdit(sketch_obj.Name)

        # Zoom to fit after a short delay
        try:
            def zoom_fit() -> None:
                try:
                    FreeCADGui.SendMsgToActiveView("ViewFit")
                except Exception:
                    pass

            QtCore.QTimer.singleShot(200, zoom_fit)
        except Exception:
            pass

        return True

    # Execute on main thread
    return _executor.execute(do_open)


def start_server(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, blocking: bool = False
) -> bool:
    """
    Start the RPC server.

    Args:
        host: Host to bind to (default: localhost)
        port: Port to bind to (default: 9876)
        blocking: If True, block the main thread. If False, run in background thread.

    Returns:
        True if server started successfully
    """
    global _server, _server_thread

    if _server is not None:
        print(f"Server already running on {host}:{port}")
        return True

    # Initialize the main thread invoker BEFORE starting the background thread
    # This ensures the QObject lives on the main thread
    _init_invoker()

    try:
        _server = SimpleXMLRPCServer(
            (host, port), requestHandler=QuietRequestHandler, allow_none=True
        )
    except OSError as e:
        print(f"Failed to start server: {e}")
        return False

    # Register functions
    _server.register_function(list_sketches, "list_sketches")
    _server.register_function(export_sketch, "export_sketch")
    _server.register_function(import_sketch, "import_sketch")
    _server.register_function(get_solver_status, "get_solver_status")
    _server.register_function(get_status, "get_status")
    _server.register_function(open_sketch_in_sketcher, "open_sketch_in_sketcher")

    print(f"FreeCAD sketch server started on {host}:{port}")

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
    global _server, _server_thread

    if _server is not None:
        _server.shutdown()
        _server = None
        _server_thread = None
        print("Server stopped")
    else:
        print("Server not running")


def is_server_running() -> bool:
    """Check if the server is currently running."""
    return _server is not None


# Convenience function for FreeCAD macro usage
def toggle_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Toggle the server on/off. Useful as a FreeCAD macro."""
    if is_server_running():
        stop_server()
    else:
        start_server(host, port)


# Allow running as a script inside FreeCAD
if __name__ == "__main__":
    start_server(blocking=True)
