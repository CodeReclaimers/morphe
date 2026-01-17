"""
Morphe Server Add-in for Fusion 360.

This add-in automatically starts the RPC server when Fusion 360 launches,
enabling external applications to interact with Fusion 360 sketches via
the Morphe canonical sketch format.

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
    Set up import paths to find the morphe package.

    This function resolves symlinks to find the actual add-in location,
    then adds the necessary paths for importing the morphe packages.

    Returns:
        Tuple of (success, error_message)
    """
    # Get the real path (resolving symlinks) to find the actual location
    addin_file = os.path.realpath(__file__)
    addin_dir = os.path.dirname(addin_file)

    # Expected structure:
    #   morphe-repo/                         <- REPO_DIR
    #     morphe/                            <- MORPHE_PKG_DIR
    #       adapters/
    #         fusion/                        <- ADAPTER_DIR
    #           addin/
    #             MorpheServer/              <- addin_dir
    #               MorpheServer.py          <- this file

    # Navigate up to find the repository root
    addin_parent = os.path.dirname(addin_dir)  # addin/
    adapter_dir = os.path.dirname(addin_parent)  # fusion/
    adapters_dir = os.path.dirname(adapter_dir)  # adapters/
    morphe_pkg_dir = os.path.dirname(adapters_dir)  # morphe/
    repo_dir = os.path.dirname(morphe_pkg_dir)  # morphe-repo/

    # Validate the directory structure
    expected_morphe_init = os.path.join(morphe_pkg_dir, "__init__.py")

    if not os.path.exists(expected_morphe_init):
        return False, (
            f"Could not find morphe package.\n"
            f"Expected __init__.py at: {expected_morphe_init}\n"
            f"Add-in location: {addin_dir}\n\n"
            f"Please ensure the add-in is installed via setup_addin.py"
        )

    # Add repo root to path for imports
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

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
                f"Morphe Server failed to initialize:\n\n{_path_error}",
                "Morphe Server Error",
                adsk.core.MessageBoxButtonTypes.OKButtonType,
                adsk.core.MessageBoxIconTypes.CriticalIconType
            )
            return

        # Import and start the server
        from morphe.adapters.fusion.server import is_server_running, start_server

        if is_server_running():
            # Server already running (e.g., add-in reloaded within same session)
            _app.log("Morphe Server already running on localhost:9879")
            _server_started = True
            return

        # Start server in non-blocking mode
        success = start_server(blocking=False)

        if success:
            _server_started = True
            # Show a brief notification (palettes are less intrusive than messageBox)
            _app.log("Morphe Server started on localhost:9879")
        else:
            _ui.messageBox(
                "Failed to start Morphe Server.\n"
                "The port may already be in use.",
                "Morphe Server",
                adsk.core.MessageBoxButtonTypes.OKButtonType,
                adsk.core.MessageBoxIconTypes.WarningIconType
            )

    except Exception:
        if _ui:
            _ui.messageBox(
                f"Failed to start Morphe Server:\n\n{traceback.format_exc()}",
                "Morphe Server Error"
            )


def stop(context: dict) -> None:
    """
    Called when the add-in is stopped.

    Cleans up the RPC server and releases resources.
    """
    global _app, _ui, _server_started

    try:
        if _server_started and _paths_ok:
            from morphe.adapters.fusion.server import is_server_running, stop_server

            if is_server_running():
                stop_server()
                if _app:
                    _app.log("Morphe Server stopped")

            _server_started = False

    except Exception:
        # Log but don't show UI on shutdown
        if _app:
            _app.log(f"Error stopping Morphe Server: {traceback.format_exc()}")

    # Clear references
    _app = None
    _ui = None
