#!/usr/bin/env python3
"""
Demonstration script for the FreeCAD RPC server/client.

This script demonstrates how to use the RPC client to communicate with
a FreeCAD instance running the RPC server.

Prerequisites:
    1. Start FreeCAD
    2. In FreeCAD's Python console, run:
       >>> import sys
       >>> sys.path.insert(0, '/path/to/morphe')
       >>> from adapter_freecad import start_server
       >>> start_server()

    3. Then run this script from the command line:
       $ python demo_rpc.py

Usage:
    python demo_rpc.py                  # Show status and list sketches
    python demo_rpc.py --import-demo    # Import the demo mounting bracket
    python demo_rpc.py --export NAME    # Export sketch NAME to JSON
    python demo_rpc.py --roundtrip NAME # Export and re-import sketch NAME
"""

import argparse
import json
import sys
from pathlib import Path

from adapter_freecad import FreeCADClient, check_server


def print_status(client: FreeCADClient) -> None:
    """Print server and FreeCAD status."""
    status = client.get_status()
    print("\n=== FreeCAD RPC Server Status ===")
    print(f"Server version: {status['server_version']}")
    print(f"FreeCAD version: {status.get('freecad_version', 'N/A')}")
    print(f"Active document: {status.get('active_document', 'None')}")
    print(f"Sketch count: {status.get('sketch_count', 0)}")


def list_sketches(client: FreeCADClient) -> None:
    """List all sketches in FreeCAD."""
    sketches = client.list_sketches()
    print("\n=== Sketches ===")
    if not sketches:
        print("No sketches in active document")
        return

    for sketch in sketches:
        print(f"  {sketch['name']} ({sketch['label']})")
        print(f"    Geometry: {sketch['geometry_count']}")
        print(f"    Constraints: {sketch['constraint_count']}")


def import_demo(client: FreeCADClient) -> None:
    """Import the demo mounting bracket sketch."""
    # Import the demo module to create the sketch
    from demo import create_mounting_bracket

    print("\n=== Importing Demo Sketch ===")
    doc = create_mounting_bracket()
    print(f"Created: {doc.name}")
    print(f"  Primitives: {len(doc.primitives)}")
    print(f"  Constraints: {len(doc.constraints)}")

    # Import into FreeCAD
    sketch_name = client.import_sketch(doc)
    print(f"\nImported as: {sketch_name}")

    # Get solver status
    status, dof = client.get_solver_status(sketch_name)
    print(f"Solver status: {status}, DOF: {dof}")

    # Open in Sketcher
    print("\nOpening in Sketcher workbench...")
    client.open_sketch(sketch_name)


def export_sketch(client: FreeCADClient, sketch_name: str) -> None:
    """Export a sketch to JSON."""
    print(f"\n=== Exporting '{sketch_name}' ===")

    doc = client.export_sketch(sketch_name)
    print(f"Exported: {doc.name}")
    print(f"  Primitives: {len(doc.primitives)}")
    print(f"  Constraints: {len(doc.constraints)}")

    # Save to file
    from core import sketch_to_json

    output_file = Path(f"{sketch_name}.json")
    json_str = sketch_to_json(doc)
    with open(output_file, "w") as f:
        json.dump(json.loads(json_str), f, indent=2)
    print(f"\nSaved to: {output_file}")


def roundtrip_sketch(client: FreeCADClient, sketch_name: str) -> None:
    """Export and re-import a sketch to test roundtrip."""
    print(f"\n=== Roundtrip Test: '{sketch_name}' ===")

    # Export
    doc = client.export_sketch(sketch_name)
    print(f"Exported: {len(doc.primitives)} primitives, {len(doc.constraints)} constraints")

    # Re-import with new name
    new_name = f"{sketch_name}_roundtrip"
    created_name = client.import_sketch(doc, name=new_name)
    print(f"Re-imported as: {created_name}")

    # Compare solver status
    orig_status, orig_dof = client.get_solver_status(sketch_name)
    new_status, new_dof = client.get_solver_status(created_name)

    print(f"\nOriginal: {orig_status}, DOF: {orig_dof}")
    print(f"Roundtrip: {new_status}, DOF: {new_dof}")

    if orig_status == new_status and orig_dof == new_dof:
        print("\nRoundtrip successful - solver status matches!")
    else:
        print("\nWarning: Solver status differs after roundtrip")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo for FreeCAD RPC server/client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
To start the server in FreeCAD:
    >>> from adapter_freecad import start_server
    >>> start_server()
        """,
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9876,
        help="Server port (default: 9876)",
    )
    parser.add_argument(
        "--import-demo",
        action="store_true",
        help="Import the demo mounting bracket sketch",
    )
    parser.add_argument(
        "--export",
        metavar="NAME",
        help="Export sketch NAME to JSON file",
    )
    parser.add_argument(
        "--roundtrip",
        metavar="NAME",
        help="Export and re-import sketch NAME",
    )

    args = parser.parse_args()

    # Check if server is running
    print(f"Connecting to FreeCAD RPC server at {args.host}:{args.port}...")
    if not check_server(args.host, args.port, timeout=2.0):
        print("\nError: Cannot connect to FreeCAD RPC server.")
        print("\nMake sure FreeCAD is running and the server is started:")
        print("  >>> from adapter_freecad import start_server")
        print("  >>> start_server()")
        sys.exit(1)

    client = FreeCADClient(args.host, args.port)
    client.connect()
    print("Connected!")

    # Show status and sketches by default
    print_status(client)
    list_sketches(client)

    # Perform requested action
    if args.import_demo:
        import_demo(client)
    elif args.export:
        export_sketch(client, args.export)
    elif args.roundtrip:
        roundtrip_sketch(client, args.roundtrip)

    print("\nDone!")


if __name__ == "__main__":
    main()
