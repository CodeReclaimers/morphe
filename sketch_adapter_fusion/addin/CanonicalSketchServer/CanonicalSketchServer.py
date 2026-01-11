"""
Canonical Sketch Server Add-in for Fusion 360.

This add-in automatically starts the RPC server when Fusion 360 launches,
enabling external applications to interact with Fusion 360 sketches via
the canonical sketch format.

The server listens on localhost:9879 by default.

Installation:
    Use the setup_addin.py script in the parent directory:
        python setup_addin.py install

    This creates a symlink from Fusion's AddIns directory to this add-in,
    ensuring proper path resolution for imports.
"""

import os
import sys
import traceback


def _setup_import_paths() -> tuple[bool, str]:
    """
    Set up import paths to find sketch_adapter_fusion and sketch_canonical.

    This function resolves symlinks to find the actual add-in location,
    then adds the necessary paths for importing the sketch packages.

    Returns:
        Tuple of (success, error_message)
    """
    # Get the real path (resolving symlinks) to find the actual location
    addin_file = os.path.realpath(__file__)
    addin_dir = os.path.dirname(addin_file)

    # Expected structure:
    #   canonical_sketch/                    <- REPO_DIR
    #     sketch_adapter_fusion/             <- ADAPTER_DIR
    #       addin/
    #         CanonicalSketchServer/         <- addin_dir
    #           CanonicalSketchServer.py     <- this file

    # Navigate up to find the repository root
    addin_parent = os.path.dirname(addin_dir)  # addin/
    adapter_dir = os.path.dirname(addin_parent)  # sketch_adapter_fusion/
    repo_dir = os.path.dirname(adapter_dir)  # canonical_sketch/

    # Validate the directory structure
    expected_adapter_init = os.path.join(adapter_dir, "__init__.py")
    expected_canonical_dir = os.path.join(repo_dir, "sketch_canonical")

    if not os.path.exists(expected_adapter_init):
        return False, (
            f"Could not find sketch_adapter_fusion package.\n"
            f"Expected __init__.py at: {expected_adapter_init}\n"
            f"Add-in location: {addin_dir}\n\n"
            f"Please ensure the add-in is installed via setup_addin.py"
        )

    if not os.path.isdir(expected_canonical_dir):
        return False, (
            f"Could not find sketch_canonical package.\n"
            f"Expected directory at: {expected_canonical_dir}\n\n"
            f"Please ensure sketch_canonical is in the repository root."
        )

    # Add paths for imports
    for path in [repo_dir, adapter_dir]:
        if path not in sys.path:
            sys.path.insert(0, path)

    return True, ""


# Set up paths before any other imports
_paths_ok, _path_error = _setup_import_paths()

# Fusion 360 API imports
import adsk.core
import adsk.fusion

# Global references to keep handlers alive
_app: adsk.core.Application = None
_ui: adsk.core.UserInterface = None
_server_started: bool = False


def run(context: dict) -> None:
    """
    Entry point when the add-in starts.

    Called automatically by Fusion 360 when the add-in loads
    (either at startup or when manually started).
    """
    global _app, _ui, _server_started

    try:
        _app = adsk.core.Application.get()
        _ui = _app.userInterface

        # Check if paths were set up correctly
        if not _paths_ok:
            _ui.messageBox(
                f"Canonical Sketch Server failed to initialize:\n\n{_path_error}",
                "Canonical Sketch Server Error",
                adsk.core.MessageBoxButtonTypes.OKButtonType,
                adsk.core.MessageBoxIconTypes.CriticalIconType
            )
            return

        # Import and start the server
        from sketch_adapter_fusion.server import start_server, is_server_running

        if is_server_running():
            _ui.messageBox(
                "Canonical Sketch Server is already running.",
                "Canonical Sketch Server"
            )
            _server_started = True
            return

        # Start server in non-blocking mode
        success = start_server(blocking=False)

        if success:
            _server_started = True
            # Show a brief notification (palettes are less intrusive than messageBox)
            _app.log("Canonical Sketch Server started on localhost:9879")
        else:
            _ui.messageBox(
                "Failed to start Canonical Sketch Server.\n"
                "The port may already be in use.",
                "Canonical Sketch Server",
                adsk.core.MessageBoxButtonTypes.OKButtonType,
                adsk.core.MessageBoxIconTypes.WarningIconType
            )

    except Exception:
        if _ui:
            _ui.messageBox(
                f"Failed to start Canonical Sketch Server:\n\n{traceback.format_exc()}",
                "Canonical Sketch Server Error"
            )


def stop(context: dict) -> None:
    """
    Called when the add-in is stopped.

    Cleans up the RPC server and releases resources.
    """
    global _app, _ui, _server_started

    try:
        if _server_started and _paths_ok:
            from sketch_adapter_fusion.server import stop_server, is_server_running

            if is_server_running():
                stop_server()
                if _app:
                    _app.log("Canonical Sketch Server stopped")

            _server_started = False

    except Exception:
        # Log but don't show UI on shutdown
        if _app:
            _app.log(f"Error stopping Canonical Sketch Server: {traceback.format_exc()}")

    # Clear references
    _app = None
    _ui = None
