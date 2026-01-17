#!/usr/bin/env python3
"""
Setup script for the CanonicalSketchServer Fusion 360 add-in.

This script creates a symlink from Fusion 360's AddIns directory to the
add-in in this repository, allowing the add-in to properly find and import
the adapter_fusion package.

Usage:
    python setup_addin.py install    # Install the add-in (create symlink)
    python setup_addin.py uninstall  # Remove the add-in symlink
    python setup_addin.py status     # Check installation status
"""

import os
import platform
import sys
from pathlib import Path


def get_fusion_addins_dir() -> Path | None:
    """Get the Fusion 360 AddIns directory for the current platform."""
    system = platform.system()

    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        if appdata:
            # Check both possible locations
            paths = [
                Path(appdata) / "Autodesk" / "Autodesk Fusion 360" / "API" / "AddIns",
                Path(appdata) / "Autodesk" / "Autodesk Fusion" / "API" / "AddIns",
            ]
            for path in paths:
                if path.parent.parent.exists():
                    return path
            # Default to the standard path
            return paths[0]
        return None

    elif system == "Darwin":  # macOS
        home = Path.home()
        paths = [
            home / "Library" / "Application Support" / "Autodesk" / "Autodesk Fusion 360" / "API" / "AddIns",
            home / "Library" / "Application Support" / "Autodesk" / "Autodesk Fusion" / "API" / "AddIns",
        ]
        for path in paths:
            if path.parent.parent.exists():
                return path
        return paths[0]

    else:
        print(f"Unsupported platform: {system}")
        return None


def get_addin_source_dir() -> Path:
    """Get the source directory of the add-in."""
    return Path(__file__).parent / "CanonicalSketchServer"


def install() -> bool:
    """Install the add-in by creating a symlink."""
    addins_dir = get_fusion_addins_dir()
    if addins_dir is None:
        print("Error: Could not determine Fusion 360 AddIns directory.")
        return False

    source_dir = get_addin_source_dir()
    if not source_dir.exists():
        print(f"Error: Add-in source directory not found: {source_dir}")
        return False

    # Ensure the AddIns directory exists
    addins_dir.mkdir(parents=True, exist_ok=True)

    link_path = addins_dir / "CanonicalSketchServer"

    # Check if already installed
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink():
            current_target = link_path.resolve()
            if current_target == source_dir.resolve():
                print(f"Add-in already installed at: {link_path}")
                return True
            else:
                print(f"Existing symlink points to different location: {current_target}")
                print("Run 'uninstall' first to remove the existing installation.")
                return False
        else:
            print(f"A directory already exists at: {link_path}")
            print("Please remove it manually or run 'uninstall' first.")
            return False

    # Create the symlink
    try:
        # On Windows, we need to use directory symlink
        if platform.system() == "Windows":
            # Use os.symlink with target_is_directory=True
            os.symlink(source_dir, link_path, target_is_directory=True)
        else:
            link_path.symlink_to(source_dir)

        print("Successfully installed add-in!")
        print(f"  Source: {source_dir}")
        print(f"  Link:   {link_path}")
        print()
        print("Next steps:")
        print("  1. Restart Fusion 360 (or open Tools > Add-Ins)")
        print("  2. Find 'CanonicalSketchServer' in the Add-Ins tab")
        print("  3. Check 'Run on Startup' to auto-start the server")
        print("  4. Click 'Run' to start immediately")
        return True

    except OSError as e:
        if platform.system() == "Windows" and "privilege" in str(e).lower():
            print("Error: Creating symlinks on Windows requires administrator privileges.")
            print()
            print("Options:")
            print("  1. Run this script as Administrator")
            print("  2. Enable Developer Mode in Windows Settings > Update & Security > For developers")
            print("  3. Manually copy the add-in folder (not recommended, paths may break)")
        else:
            print(f"Error creating symlink: {e}")
        return False


def uninstall() -> bool:
    """Uninstall the add-in by removing the symlink."""
    addins_dir = get_fusion_addins_dir()
    if addins_dir is None:
        print("Error: Could not determine Fusion 360 AddIns directory.")
        return False

    link_path = addins_dir / "CanonicalSketchServer"

    if not link_path.exists() and not link_path.is_symlink():
        print("Add-in is not installed.")
        return True

    if link_path.is_symlink():
        try:
            link_path.unlink()
            print(f"Successfully uninstalled add-in from: {link_path}")
            return True
        except OSError as e:
            print(f"Error removing symlink: {e}")
            return False
    else:
        print(f"Warning: {link_path} is not a symlink.")
        print("This may be a manual installation. Please remove it manually if desired.")
        return False


def status() -> None:
    """Check the installation status."""
    addins_dir = get_fusion_addins_dir()
    source_dir = get_addin_source_dir()

    print("Canonical Sketch Server Add-in Status")
    print("=" * 40)
    print()

    print(f"Platform: {platform.system()}")
    print(f"Source directory: {source_dir}")
    print(f"  Exists: {source_dir.exists()}")
    print()

    if addins_dir is None:
        print("Fusion 360 AddIns directory: Could not determine")
        return

    print(f"Fusion 360 AddIns directory: {addins_dir}")
    print(f"  Exists: {addins_dir.exists()}")
    print()

    link_path = addins_dir / "CanonicalSketchServer"
    print(f"Installation path: {link_path}")

    if link_path.is_symlink():
        target = link_path.resolve()
        print("  Status: Installed (symlink)")
        print(f"  Points to: {target}")
        if target == source_dir.resolve():
            print("  Valid: Yes")
        else:
            print("  Valid: No (points to wrong location)")
    elif link_path.exists():
        print("  Status: Installed (directory, not symlink)")
        print("  Warning: Manual installation may have path issues")
    else:
        print("  Status: Not installed")


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    command = sys.argv[1].lower()

    if command == "install":
        return 0 if install() else 1
    elif command == "uninstall":
        return 0 if uninstall() else 1
    elif command == "status":
        status()
        return 0
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
