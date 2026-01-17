"""
RPC server for Fusion 360 sketch adapter.

This module provides a simple XML-RPC server that runs inside Fusion 360
and exposes the sketch adapter functionality over the network.

Usage:
    Run this as a Fusion 360 script or add-in:

    >>> from adapter_fusion.server import start_server
    >>> start_server()  # Starts on localhost:9879

    For add-in mode (non-blocking):

    >>> start_server(blocking=False)

    To stop the server:

    >>> from adapter_fusion.server import stop_server
    >>> stop_server()

Note: This server must run inside Fusion 360 where the 'adsk' module
is available. The server uses CustomEvents to safely execute operations
on the main UI thread.
"""

from __future__ import annotations

import queue
import threading
import traceback
from typing import TYPE_CHECKING, Any
from xmlrpc.server import SimpleXMLRPCServer

from morphe.adapters.common import QuietRequestHandler

if TYPE_CHECKING:
    pass

# Try to import Fusion 360 API
try:
    import adsk.core
    import adsk.fusion

    FUSION_AVAILABLE = True
except ImportError:
    adsk = None  # type: ignore[assignment]
    FUSION_AVAILABLE = False

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9879
SERVER_VERSION = "1.0.0"

# Global server instance
_server: SimpleXMLRPCServer | None = None
_server_thread: threading.Thread | None = None

# Custom event for thread-safe execution
_custom_event_id = "SketchAdapterRPCEvent"
_custom_event: Any = None
_event_handler: Any = None

# Queue for passing work to main thread
_work_queue: queue.Queue | None = None
_result_queue: queue.Queue | None = None


def _create_event_handler_class() -> type | None:
    """Create the CustomEventHandler class if Fusion is available."""
    if not FUSION_AVAILABLE:
        return None

    class RPCEventHandler(adsk.core.CustomEventHandler):
        """Handler for custom events that executes work on the main thread."""

        def __init__(self) -> None:
            super().__init__()

        def notify(self, args: adsk.core.CustomEventArgs) -> None:
            """Called on the main thread when the custom event fires."""
            global _work_queue, _result_queue

            if _work_queue is None or _result_queue is None:
                return

            try:
                # Get the work item from the queue
                func = _work_queue.get_nowait()

                try:
                    result = func()
                    _result_queue.put((True, result))
                except Exception as e:
                    tb = traceback.format_exc()
                    _result_queue.put((False, (e, tb)))

            except queue.Empty:
                pass

    return RPCEventHandler


# Create the handler class (will be None if Fusion not available)
_RPCEventHandler = _create_event_handler_class()


def _init_custom_event() -> bool:
    """Initialize the custom event for thread-safe execution."""
    global _custom_event, _event_handler, _work_queue, _result_queue

    if not FUSION_AVAILABLE or _RPCEventHandler is None:
        return False

    try:
        app = adsk.core.Application.get()
        if not app:
            return False

        # Create the custom event
        _custom_event = app.registerCustomEvent(_custom_event_id)
        if not _custom_event:
            return False

        # Create and connect the handler
        _event_handler = _RPCEventHandler()
        _custom_event.add(_event_handler)

        # Initialize queues
        _work_queue = queue.Queue()
        _result_queue = queue.Queue()

        return True

    except Exception:
        return False


def _cleanup_custom_event() -> None:
    """Clean up the custom event."""
    global _custom_event, _event_handler, _work_queue, _result_queue

    if FUSION_AVAILABLE and _custom_event is not None:
        try:
            app = adsk.core.Application.get()
            if app:
                app.unregisterCustomEvent(_custom_event_id)
        except Exception:
            pass

    _custom_event = None
    _event_handler = None
    _work_queue = None
    _result_queue = None


def _execute_on_main_thread(func: Any, timeout: float = 30.0) -> Any:
    """
    Execute a function on the main UI thread.

    Args:
        func: Function to execute (no arguments)
        timeout: Maximum time to wait for result

    Returns:
        The function's return value

    Raises:
        RuntimeError: If execution fails or times out
    """
    global _work_queue, _result_queue, _custom_event

    if not FUSION_AVAILABLE:
        # Not in Fusion, just run directly
        return func()

    if _work_queue is None or _result_queue is None or _custom_event is None:
        # Custom event not initialized, run directly (may crash)
        return func()

    # Clear any stale results
    while not _result_queue.empty():
        try:
            _result_queue.get_nowait()
        except queue.Empty:
            break

    # Put the work on the queue
    _work_queue.put(func)

    # Fire the custom event to trigger execution on main thread
    app = adsk.core.Application.get()
    if app:
        app.fireCustomEvent(_custom_event_id)

    # Wait for result
    try:
        success, value = _result_queue.get(timeout=timeout)
    except queue.Empty as e:
        raise RuntimeError(f"Operation timed out after {timeout}s") from e

    if success:
        return value
    else:
        exc, tb = value
        raise RuntimeError(f"Operation failed: {exc}\n{tb}")


def _get_adapter() -> Any:
    """Get a FusionAdapter instance. Import here to avoid circular imports."""
    from .adapter import FusionAdapter

    return FusionAdapter()


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
    List all sketches in the active Fusion 360 document.

    Returns:
        List of dicts with sketch info:
        [{"name": str, "constraint_count": int, "geometry_count": int}]
    """
    if not FUSION_AVAILABLE:
        raise RuntimeError("Fusion 360 is not available")

    def do_list() -> list[dict]:
        app = adsk.core.Application.get()
        if not app:
            return []

        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            return []

        sketches = []
        root_comp = design.rootComponent

        for i in range(root_comp.sketches.count):
            sketch = root_comp.sketches.item(i)

            # Count geometry
            geom_count = (
                sketch.sketchCurves.sketchLines.count
                + sketch.sketchCurves.sketchCircles.count
                + sketch.sketchCurves.sketchArcs.count
                + sketch.sketchPoints.count
            )

            # Count constraints
            constraint_count = (
                sketch.geometricConstraints.count + sketch.sketchDimensions.count
            )

            sketches.append(
                {
                    "name": sketch.name,
                    "constraint_count": constraint_count,
                    "geometry_count": geom_count,
                }
            )

        return sketches

    return _execute_on_main_thread(do_list)


def export_sketch(sketch_name: str) -> str:
    """
    Export a sketch to canonical JSON format.

    Args:
        sketch_name: Name of the sketch object in Fusion 360

    Returns:
        JSON string of the canonical sketch
    """
    if not FUSION_AVAILABLE:
        raise RuntimeError("Fusion 360 is not available")

    def do_export() -> str:
        app = adsk.core.Application.get()
        if not app:
            raise RuntimeError("Could not get Fusion 360 application")

        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            raise RuntimeError("No active design")

        # Find the sketch by name
        root_comp = design.rootComponent
        sketch_obj = None
        for i in range(root_comp.sketches.count):
            sketch = root_comp.sketches.item(i)
            if sketch.name == sketch_name:
                sketch_obj = sketch
                break

        if sketch_obj is None:
            raise ValueError(f"Sketch '{sketch_name}' not found")

        # Create adapter and export
        adapter = _get_adapter()
        adapter._sketch = sketch_obj

        exported = adapter.export_sketch()
        sketch_to_json = _get_sketch_to_json()
        return sketch_to_json(exported)

    return _execute_on_main_thread(do_export)


def list_planes() -> list[dict]:
    """
    List available planes for sketch creation.

    Returns:
        List of dicts with plane info:
        [{"id": str, "name": str, "type": str}]
    """
    if not FUSION_AVAILABLE:
        raise RuntimeError("Fusion 360 is not available")

    def do_list() -> list[dict]:
        app = adsk.core.Application.get()
        if not app:
            return []

        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            return []

        # Standard construction planes always available
        planes = [
            {"id": "XY", "name": "XY Plane", "type": "construction"},
            {"id": "XZ", "name": "XZ Plane", "type": "construction"},
            {"id": "YZ", "name": "YZ Plane", "type": "construction"},
        ]

        root_comp = design.rootComponent

        # Add construction planes from the component
        try:
            for i in range(root_comp.constructionPlanes.count):
                cp = root_comp.constructionPlanes.item(i)
                planes.append({
                    "id": f"ConstructionPlane:{cp.name}",
                    "name": cp.name,
                    "type": "construction",
                })
        except Exception:
            pass

        # Add planar faces from bodies
        try:
            for body_idx in range(root_comp.bRepBodies.count):
                body = root_comp.bRepBodies.item(body_idx)
                for face_idx in range(body.faces.count):
                    face = body.faces.item(face_idx)
                    # Check if face is planar
                    if face.geometry.objectType == adsk.core.Plane.classType():
                        planes.append({
                            "id": f"{body.name}:Face{face_idx + 1}",
                            "name": f"{body.name} - Face {face_idx + 1}",
                            "type": "face",
                        })
        except Exception:
            pass

        return planes

    return _execute_on_main_thread(do_list)


def import_sketch(
    json_str: str, sketch_name: str | None = None, plane: str | None = None
) -> str:
    """
    Import a sketch from canonical JSON format.

    Creates a new sketch in the active document.

    Args:
        json_str: JSON string of the canonical sketch
        sketch_name: Optional name for the new sketch (uses name from JSON if not provided)
        plane: Optional plane ID (from list_planes). Defaults to "XY".

    Returns:
        Name of the created sketch object
    """
    if not FUSION_AVAILABLE:
        raise RuntimeError("Fusion 360 is not available")

    def do_import() -> str:
        sketch_from_json = _get_sketch_from_json()
        sketch_doc = sketch_from_json(json_str)

        if sketch_name:
            sketch_doc.name = sketch_name

        adapter = _get_adapter()

        # Resolve plane
        plane_to_use = plane or "XY"
        if plane and ":" in plane:
            # Try to resolve plane reference
            try:
                app = adsk.core.Application.get()
                design = adsk.fusion.Design.cast(app.activeProduct)
                root_comp = design.rootComponent

                if plane.startswith("ConstructionPlane:"):
                    # Find construction plane by name
                    cp_name = plane.split(":", 1)[1]
                    for i in range(root_comp.constructionPlanes.count):
                        cp = root_comp.constructionPlanes.item(i)
                        if cp.name == cp_name:
                            plane_to_use = cp
                            break
                else:
                    # Face reference like "Body1:Face3"
                    parts = plane.split(":")
                    if len(parts) == 2:
                        body_name, face_ref = parts
                        face_idx = int(face_ref.replace("Face", "")) - 1
                        for body_idx in range(root_comp.bRepBodies.count):
                            body = root_comp.bRepBodies.item(body_idx)
                            if body.name == body_name:
                                if 0 <= face_idx < body.faces.count:
                                    plane_to_use = body.faces.item(face_idx)
                                break
            except Exception:
                plane_to_use = "XY"

        adapter.load_sketch(sketch_doc, plane=plane_to_use)

        return adapter._sketch.name

    return _execute_on_main_thread(do_import)


def get_solver_status(sketch_name: str) -> dict:
    """
    Get the solver status for a sketch.

    Args:
        sketch_name: Name of the sketch object

    Returns:
        Dict with "status" and "dof" keys
    """
    if not FUSION_AVAILABLE:
        raise RuntimeError("Fusion 360 is not available")

    def do_get_status() -> dict:
        app = adsk.core.Application.get()
        if not app:
            raise RuntimeError("Could not get Fusion 360 application")

        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            raise RuntimeError("No active design")

        # Find the sketch by name
        root_comp = design.rootComponent
        sketch_obj = None
        for i in range(root_comp.sketches.count):
            sketch = root_comp.sketches.item(i)
            if sketch.name == sketch_name:
                sketch_obj = sketch
                break

        if sketch_obj is None:
            raise ValueError(f"Sketch '{sketch_name}' not found")

        adapter = _get_adapter()
        adapter._sketch = sketch_obj

        status, dof = adapter.get_solver_status()
        return {"status": status.name, "dof": dof}

    return _execute_on_main_thread(do_get_status)


def ping() -> dict:
    """
    Lightweight health check that doesn't require main thread execution.

    This is useful for quick connection checks without blocking on Fusion's
    UI thread. Use this for polling/heartbeat instead of get_status().

    Returns:
        Dict with server_version and fusion_available
    """
    return {
        "server_version": SERVER_VERSION,
        "fusion_available": FUSION_AVAILABLE,
        "status": "ok",
    }


def get_status() -> dict:
    """
    Get server and Fusion 360 status.

    Returns:
        Dict with version and document info
    """
    result: dict = {
        "server_version": SERVER_VERSION,
        "fusion_available": FUSION_AVAILABLE,
    }

    if FUSION_AVAILABLE:

        def do_get_status() -> dict:
            app = adsk.core.Application.get()
            if not app:
                return {"error": "Could not get application"}

            try:
                result = {"fusion_version": app.version}

                doc = app.activeDocument
                if doc:
                    result["active_document"] = doc.name

                    design = adsk.fusion.Design.cast(app.activeProduct)
                    if design:
                        result["sketch_count"] = design.rootComponent.sketches.count
                    else:
                        result["sketch_count"] = 0
                else:
                    result["active_document"] = None
                    result["sketch_count"] = 0

                return result

            except Exception as e:
                return {"error": str(e)}

        try:
            status_info = _execute_on_main_thread(do_get_status)
            result.update(status_info)
        except Exception as e:
            result["error"] = str(e)

    return result


def open_sketch_in_edit_mode(sketch_name: str) -> bool:
    """
    Open a sketch in edit mode for editing.

    Args:
        sketch_name: Name of the sketch object

    Returns:
        True if successful
    """
    if not FUSION_AVAILABLE:
        raise RuntimeError("Fusion 360 is not available")

    def do_open() -> bool:
        app = adsk.core.Application.get()
        if not app:
            raise RuntimeError("Could not get Fusion 360 application")

        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            raise RuntimeError("No active design")

        # Find the sketch by name
        root_comp = design.rootComponent
        sketch_obj = None
        for i in range(root_comp.sketches.count):
            sketch = root_comp.sketches.item(i)
            if sketch.name == sketch_name:
                sketch_obj = sketch
                break

        if sketch_obj is None:
            raise ValueError(f"Sketch '{sketch_name}' not found")

        # Edit the sketch (this may require UI access)
        try:
            # Make the sketch visible and try to edit
            sketch_obj.isVisible = True

            # Try to activate edit mode via the UI
            ui = app.userInterface
            if ui:
                # Select the sketch and edit it
                ui.activeSelections.clear()
                ui.activeSelections.add(sketch_obj)
                ui.commandDefinitions.itemById("SketchActivate").execute()

            return True
        except Exception:
            # Fallback: just make it visible
            sketch_obj.isVisible = True
            return True

    return _execute_on_main_thread(do_open)


def probe_constraints(sketch_name: str) -> dict:
    """Probe a sketch to find what constraints exist.

    TODO: Remove on next cleanup pass - debug function for constraint export development.
    """
    if not FUSION_AVAILABLE:
        raise RuntimeError("Fusion 360 is not available")

    def do_probe() -> dict:
        app = adsk.core.Application.get()
        if not app:
            return {"error": "Could not get Fusion 360 application"}

        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            return {"error": "No active design"}

        # Find the sketch
        root_comp = design.rootComponent
        sketch_obj = None
        for i in range(root_comp.sketches.count):
            sketch = root_comp.sketches.item(i)
            if sketch.name == sketch_name:
                sketch_obj = sketch
                break

        if sketch_obj is None:
            return {"error": f"Sketch '{sketch_name}' not found"}

        result = {"sketch_name": sketch_name}

        # Count geometric constraints
        try:
            geo_constraints = sketch_obj.geometricConstraints
            result["geometric_constraint_count"] = geo_constraints.count

            # List constraint types
            constraint_types = []
            for i in range(geo_constraints.count):
                c = geo_constraints.item(i)
                constraint_types.append(c.objectType)
            result["geometric_constraints"] = constraint_types[:20]  # Limit
        except Exception as e:
            result["geometric_error"] = str(e)

        # Count sketch dimensions (dimensional constraints)
        try:
            dimensions = sketch_obj.sketchDimensions
            result["dimension_count"] = dimensions.count

            dim_types = []
            for i in range(dimensions.count):
                d = dimensions.item(i)
                dim_types.append(d.objectType)
            result["dimensions"] = dim_types[:20]
        except Exception as e:
            result["dimension_error"] = str(e)

        # Check entity mapping - do a full export to populate mappings
        try:
            adapter = _get_adapter()
            adapter._sketch = sketch_obj

            # Do the export to populate entity mappings
            exported = adapter.export_sketch()

            result["entity_to_id_count"] = len(adapter._entity_to_id)
            result["exported_primitive_count"] = len(exported.primitives)
            result["exported_constraint_count"] = len(exported.constraints)

            # Check if constraint entities can be found
            geo_constraints = sketch_obj.geometricConstraints
            constraint_entity_check = []
            for i in range(min(geo_constraints.count, 5)):
                c = geo_constraints.item(i)
                obj_type = c.objectType

                check = {"type": obj_type.split("::")[-1]}

                # Try to get the entity and its token
                if "HorizontalConstraint" in obj_type or "VerticalConstraint" in obj_type:
                    try:
                        line = c.line
                        token = line.entityToken
                        check["entity_token"] = token[:50] if token else "None"
                        check["found_in_mapping"] = token in adapter._entity_to_id
                    except Exception as e:
                        check["error"] = str(e)[:50]

                constraint_entity_check.append(check)

            result["constraint_entity_check"] = constraint_entity_check

            # Try to manually convert a constraint to see what fails
            if geo_constraints.count > 0:
                c = geo_constraints.item(0)

                # Check if _get_id_for_entity works
                try:
                    line = c.line
                    entity_id = adapter._get_id_for_entity(line)
                    result["get_id_result"] = entity_id if entity_id else "None returned"
                except Exception as e:
                    result["get_id_error"] = str(e)

                # Try direct horizontal conversion
                try:
                    from morphe import ConstraintType, SketchConstraint
                    line = c.line
                    entity_id = adapter._get_id_for_entity(line)
                    result["direct_entity_id"] = entity_id
                    if entity_id:
                        sc = SketchConstraint(
                            constraint_type=ConstraintType.HORIZONTAL,
                            references=[entity_id]
                        )
                        result["direct_constraint"] = str(sc)
                except Exception:
                    import traceback
                    result["direct_convert_error"] = traceback.format_exc()[-500:]

                # Try adapter's convert method with exception detail
                try:
                    converted = adapter._convert_horizontal(c)
                    result["convert_horizontal_result"] = str(converted) if converted else "None"
                except Exception:
                    import traceback
                    result["convert_horizontal_error"] = traceback.format_exc()[-500:]

            # Check offset dimensions specifically
            dims = sketch_obj.sketchDimensions
            offset_dim_info = []
            for i in range(dims.count):
                dim = dims.item(i)
                if "SketchOffsetDimension" in dim.objectType:
                    dim_info = {"index": i}
                    try:
                        # SketchOffsetDimension uses .line property
                        entity = dim.line
                        dim_info["entity_type"] = entity.objectType if entity else "None"
                        if entity:
                            token = getattr(entity, "entityToken", None)
                            dim_info["has_token"] = token is not None
                            dim_info["in_mapping"] = token in adapter._entity_to_id if token else False
                            entity_id = adapter._get_id_for_entity(entity)
                            dim_info["entity_id"] = entity_id if entity_id else "None"
                        dim_info["is_horizontal"] = getattr(dim, "isHorizontal", "N/A")
                    except Exception as e:
                        dim_info["error"] = str(e)[:100]
                    try:
                        dim_info["value"] = dim.parameter.value * 10  # cm to mm
                    except Exception:
                        pass
                    offset_dim_info.append(dim_info)
            result["offset_dimensions"] = offset_dim_info

        except Exception as e:
            result["adapter_error"] = str(e)

        return result

    return _execute_on_main_thread(do_probe)


def start_server(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, blocking: bool = True
) -> bool:
    """
    Start the RPC server.

    Args:
        host: Host to bind to (default: localhost)
        port: Port to bind to (default: 9879)
        blocking: If True, block the main thread. If False, run in background thread.

    Returns:
        True if server started successfully
    """
    global _server, _server_thread

    if _server is not None:
        print(f"Server already running on {host}:{port}")
        return True

    # Initialize the custom event for thread-safe execution
    if FUSION_AVAILABLE:
        if not _init_custom_event():
            print("Warning: Could not initialize custom event, thread safety limited")

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
    _server.register_function(probe_constraints, "probe_constraints")

    print(f"Fusion 360 sketch server started on {host}:{port}")

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
    """Stop the RPC server (non-blocking)."""
    global _server, _server_thread

    if _server is not None:
        server_to_stop = _server
        _server = None
        _server_thread = None
        _cleanup_custom_event()

        # Shutdown in background thread to avoid blocking
        def do_shutdown():
            try:
                server_to_stop.socket.settimeout(0.5)
            except Exception:
                pass
            try:
                server_to_stop.shutdown()
            except Exception:
                pass

        shutdown_thread = threading.Thread(target=do_shutdown, daemon=True)
        shutdown_thread.start()
        print("Server stopped")
    else:
        print("Server not running")


def is_server_running() -> bool:
    """Check if the server is currently running."""
    return _server is not None


def toggle_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Toggle the server on/off. Useful as a Fusion 360 script action."""
    if is_server_running():
        stop_server()
    else:
        start_server(host, port, blocking=False)


# Allow running as a Fusion 360 script
def run(context: dict) -> None:
    """Entry point when run as a Fusion 360 script."""
    start_server(blocking=False)


def stop(context: dict) -> None:
    """Called when the script is stopped."""
    stop_server()
