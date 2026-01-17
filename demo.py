#!/usr/bin/env python3
"""
Demonstration script for the canonical sketch library.

Creates a complex constrained mounting bracket sketch and exports it to
one of the supported CAD platforms.

Usage:
    python demo.py --freecad     # Export to FreeCAD
    python demo.py --inventor    # Export to Autodesk Inventor
    python demo.py --solidworks  # Export to SolidWorks
    python demo.py --fusion      # Export to Fusion 360
    python demo.py --json        # Export to JSON file
    python demo.py --show        # Just print the sketch summary

The bracket design features:
    - Rectangular outer profile with rounded corners (fillets)
    - Two mounting holes with equal radius constraint
    - A center slot for adjustment
    - Fully constrained with dimensions
    - Construction geometry for centerlines
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from core import (
    Arc,
    Circle,
    Distance,
    Equal,
    Fixed,
    Horizontal,
    Length,
    Line,
    Point,
    Point2D,
    PointRef,
    PointType,
    Radius,
    SketchDocument,
    Tangent,
    Vertical,
    sketch_to_json,
)


def create_mounting_bracket() -> SketchDocument:
    """
    Create a fully-constrained mounting bracket sketch.

    The bracket has:
    - Overall dimensions: 120mm x 60mm
    - Corner fillets: R5mm
    - Two mounting holes: Ø8mm, 15mm from edges
    - Center slot: 30mm x 10mm with rounded ends

    Returns:
        A fully-constrained SketchDocument
    """
    doc = SketchDocument(name="MountingBracket")

    # ==========================================================================
    # Parameters (all dimensions in mm)
    # ==========================================================================
    width = 120.0
    height = 60.0
    fillet_radius = 5.0
    hole_diameter = 8.0
    hole_inset = 15.0
    slot_length = 30.0
    slot_width = 10.0

    # Derived values
    hole_radius = hole_diameter / 2
    slot_radius = slot_width / 2

    # ==========================================================================
    # Construction geometry - centerlines
    # ==========================================================================

    # Horizontal centerline
    h_center_id = doc.add_primitive(Line(
        start=Point2D(-10, height / 2),
        end=Point2D(width + 10, height / 2),
        construction=True
    ))

    # Vertical centerline
    v_center_id = doc.add_primitive(Line(
        start=Point2D(width / 2, -10),
        end=Point2D(width / 2, height + 10),
        construction=True
    ))

    # Constrain centerlines
    doc.add_constraint(Horizontal(h_center_id))
    doc.add_constraint(Vertical(v_center_id))

    # Fix the intersection point (origin of our coordinate system)
    center_point_id = doc.add_primitive(Point(
        position=Point2D(width / 2, height / 2),
        construction=True
    ))
    doc.add_constraint(Fixed(center_point_id))

    # ==========================================================================
    # Outer profile - rectangle with filleted corners
    # ==========================================================================

    # Bottom edge (between fillets)
    bottom_id = doc.add_primitive(Line(
        start=Point2D(fillet_radius, 0),
        end=Point2D(width - fillet_radius, 0)
    ))

    # Right edge (between fillets)
    right_id = doc.add_primitive(Line(
        start=Point2D(width, fillet_radius),
        end=Point2D(width, height - fillet_radius)
    ))

    # Top edge (between fillets)
    top_id = doc.add_primitive(Line(
        start=Point2D(width - fillet_radius, height),
        end=Point2D(fillet_radius, height)
    ))

    # Left edge (between fillets)
    left_id = doc.add_primitive(Line(
        start=Point2D(0, height - fillet_radius),
        end=Point2D(0, fillet_radius)
    ))

    # Corner fillets (arcs)
    # Bottom-left fillet
    fillet_bl_id = doc.add_primitive(Arc(
        center=Point2D(fillet_radius, fillet_radius),
        start_point=Point2D(0, fillet_radius),
        end_point=Point2D(fillet_radius, 0),
        ccw=False
    ))

    # Bottom-right fillet
    fillet_br_id = doc.add_primitive(Arc(
        center=Point2D(width - fillet_radius, fillet_radius),
        start_point=Point2D(width - fillet_radius, 0),
        end_point=Point2D(width, fillet_radius),
        ccw=False
    ))

    # Top-right fillet
    fillet_tr_id = doc.add_primitive(Arc(
        center=Point2D(width - fillet_radius, height - fillet_radius),
        start_point=Point2D(width, height - fillet_radius),
        end_point=Point2D(width - fillet_radius, height),
        ccw=False
    ))

    # Top-left fillet
    fillet_tl_id = doc.add_primitive(Arc(
        center=Point2D(fillet_radius, height - fillet_radius),
        start_point=Point2D(fillet_radius, height),
        end_point=Point2D(0, height - fillet_radius),
        ccw=False
    ))

    # ==========================================================================
    # Outer profile constraints
    # ==========================================================================

    # Edges are horizontal/vertical
    doc.add_constraint(Horizontal(bottom_id))
    doc.add_constraint(Horizontal(top_id))
    doc.add_constraint(Vertical(left_id))
    doc.add_constraint(Vertical(right_id))

    # Tangent constraints between edges and fillets
    # (Tangent at shared endpoint implies coincident, so we only need tangent)
    # For a closed loop, we need n-1 tangent constraints (the last closes automatically)
    doc.add_constraint(Tangent(bottom_id, fillet_bl_id))
    doc.add_constraint(Tangent(bottom_id, fillet_br_id))
    doc.add_constraint(Tangent(right_id, fillet_br_id))
    doc.add_constraint(Tangent(right_id, fillet_tr_id))
    doc.add_constraint(Tangent(top_id, fillet_tr_id))
    doc.add_constraint(Tangent(top_id, fillet_tl_id))
    doc.add_constraint(Tangent(left_id, fillet_tl_id))
    # Last tangent (left_id, fillet_bl_id) omitted - closed loop closes automatically

    # All fillets have equal radius
    doc.add_constraint(Equal(fillet_bl_id, fillet_br_id))
    doc.add_constraint(Equal(fillet_br_id, fillet_tr_id))
    doc.add_constraint(Equal(fillet_tr_id, fillet_tl_id))

    # Dimension constraints for outer profile
    doc.add_constraint(Length(bottom_id, value=width - 2 * fillet_radius))
    doc.add_constraint(Length(right_id, value=height - 2 * fillet_radius))
    doc.add_constraint(Radius(fillet_bl_id, value=fillet_radius))

    # ==========================================================================
    # Mounting holes
    # ==========================================================================

    # Left mounting hole
    hole_left_id = doc.add_primitive(Circle(
        center=Point2D(hole_inset, height / 2),
        radius=hole_radius
    ))

    # Right mounting hole
    hole_right_id = doc.add_primitive(Circle(
        center=Point2D(width - hole_inset, height / 2),
        radius=hole_radius
    ))

    # Hole constraints
    # Both holes have equal radius
    doc.add_constraint(Equal(hole_left_id, hole_right_id))

    # Hole radius dimension
    doc.add_constraint(Radius(hole_left_id, value=hole_radius))

    # Distance from left edge corner to left hole center
    # This positions the left hole both horizontally and vertically
    doc.add_constraint(Distance(
        PointRef(left_id, PointType.START),
        PointRef(hole_left_id, PointType.CENTER),
        value=math.sqrt(hole_inset**2 + (height/2 - fillet_radius)**2)
    ))

    # Distance from right edge corner to right hole center (symmetric to left)
    doc.add_constraint(Distance(
        PointRef(right_id, PointType.END),
        PointRef(hole_right_id, PointType.CENTER),
        value=math.sqrt(hole_inset**2 + (height/2 - fillet_radius)**2)
    ))

    # ==========================================================================
    # Center slot (rounded rectangle / obround)
    # ==========================================================================

    slot_left_x = (width - slot_length) / 2
    slot_right_x = (width + slot_length) / 2
    slot_bottom_y = (height - slot_width) / 2
    slot_top_y = (height + slot_width) / 2

    # Slot straight edges
    slot_top_id = doc.add_primitive(Line(
        start=Point2D(slot_left_x + slot_radius, slot_top_y),
        end=Point2D(slot_right_x - slot_radius, slot_top_y)
    ))

    slot_bottom_id = doc.add_primitive(Line(
        start=Point2D(slot_right_x - slot_radius, slot_bottom_y),
        end=Point2D(slot_left_x + slot_radius, slot_bottom_y)
    ))

    # Slot end arcs (semicircles)
    slot_left_arc_id = doc.add_primitive(Arc(
        center=Point2D(slot_left_x + slot_radius, height / 2),
        start_point=Point2D(slot_left_x + slot_radius, slot_top_y),
        end_point=Point2D(slot_left_x + slot_radius, slot_bottom_y),
        ccw=True
    ))

    slot_right_arc_id = doc.add_primitive(Arc(
        center=Point2D(slot_right_x - slot_radius, height / 2),
        start_point=Point2D(slot_right_x - slot_radius, slot_bottom_y),
        end_point=Point2D(slot_right_x - slot_radius, slot_top_y),
        ccw=True
    ))

    # Slot constraints
    doc.add_constraint(Horizontal(slot_top_id))
    doc.add_constraint(Horizontal(slot_bottom_id))
    # Note: Equal length is implicit since both edges connect to equal-radius arcs
    # and both are horizontal

    # Tangent between slot edges and arcs
    # (Tangent at shared endpoint implies coincident, so we only need tangent)
    # For a closed loop, we need n-1 tangent constraints
    doc.add_constraint(Tangent(slot_top_id, slot_left_arc_id))
    doc.add_constraint(Tangent(slot_top_id, slot_right_arc_id))
    doc.add_constraint(Tangent(slot_bottom_id, slot_left_arc_id))
    # Last tangent (slot_bottom_id, slot_right_arc_id) omitted - closed loop closes automatically

    # Slot arcs are equal (same radius)
    doc.add_constraint(Equal(slot_left_arc_id, slot_right_arc_id))

    # Slot dimensions
    doc.add_constraint(Radius(slot_left_arc_id, value=slot_radius))
    doc.add_constraint(Length(slot_top_id, value=slot_length - slot_width))

    return doc


def print_sketch_summary(doc: SketchDocument) -> None:
    """Print a summary of the sketch contents."""
    print(f"\n{'='*60}")
    print(f"Sketch: {doc.name}")
    print(f"{'='*60}")

    # Count primitives by type
    prim_counts: dict[str, int] = {}
    for prim in doc.primitives.values():
        ptype = type(prim).__name__
        prim_counts[ptype] = prim_counts.get(ptype, 0) + 1

    print(f"\nPrimitives ({len(doc.primitives)} total):")
    for ptype, count in sorted(prim_counts.items()):
        print(f"  {ptype}: {count}")

    # Count constraints by type
    const_counts: dict[str, int] = {}
    for const in doc.constraints:
        ctype = const.constraint_type.name
        const_counts[ctype] = const_counts.get(ctype, 0) + 1

    print(f"\nConstraints ({len(doc.constraints)} total):")
    for ctype, count in sorted(const_counts.items()):
        print(f"  {ctype}: {count}")

    # Count construction vs regular geometry
    construction_count = sum(1 for p in doc.primitives.values() if p.construction)
    regular_count = len(doc.primitives) - construction_count

    print("\nGeometry breakdown:")
    print(f"  Regular geometry: {regular_count}")
    print(f"  Construction geometry: {construction_count}")

    print(f"\n{'='*60}\n")


def find_freecad_gui():
    """Find the FreeCAD GUI executable."""
    # Check for snap installation first (common on Ubuntu)
    if shutil.which("snap"):
        try:
            result = subprocess.run(
                ["snap", "run", "freecad", "--version"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and "FreeCAD" in result.stdout:
                return ["snap", "run", "freecad"]
        except Exception:
            pass

    # Check for freecad in PATH
    freecad = shutil.which("freecad") or shutil.which("FreeCAD")
    if freecad:
        return [freecad]

    return None


def export_to_freecad(doc: SketchDocument) -> None:
    """Export the sketch to FreeCAD GUI, leaving it open for interaction."""
    freecad_cmd = find_freecad_gui()

    if freecad_cmd is None:
        print("Error: FreeCAD executable not found.")
        print("Please install FreeCAD and ensure 'freecad' is in your PATH.")
        print("On Ubuntu with snap: snap install freecad")
        sys.exit(1)

    print("Launching FreeCAD...")

    # Get project root for imports
    project_root = Path(__file__).parent.absolute()

    # Serialize the sketch to JSON
    sketch_json = sketch_to_json(doc)

    # Create a script to run inside FreeCAD GUI
    # This script loads the sketch, activates Sketcher, and zooms to fit
    script = f'''
import sys
sys.path.insert(0, {repr(str(project_root))})

from core import sketch_from_json
from adapter_freecad import FreeCADAdapter
import FreeCADGui

# Load the sketch from JSON
sketch_json = {repr(sketch_json)}
doc = sketch_from_json(sketch_json)

# Create adapter and load sketch
adapter = FreeCADAdapter()
adapter.create_sketch(doc.name)
adapter.load_sketch(doc)

# Get solver status
status, dof = adapter.get_solver_status()
print(f"Solver status: {{status.name}}, DOF: {{dof}}")

# Get the sketch object for GUI operations
sketch_obj = adapter._sketch

# Activate the Sketcher workbench
FreeCADGui.activateWorkbench("SketcherWorkbench")

# Enter edit mode for the sketch (opens it in Sketcher)
FreeCADGui.ActiveDocument.setEdit(sketch_obj.Name)

# Use a timer to zoom to fit after the view is ready
from PySide import QtCore

def zoom_to_fit():
    try:
        FreeCADGui.SendMsgToActiveView("ViewFit")
    except Exception:
        pass

# Delay the zoom slightly to ensure the view is ready
QtCore.QTimer.singleShot(500, zoom_to_fit)

print("Sketch opened in Sketcher workbench.")
print(f"Loaded {{len(doc.primitives)}} primitives, {{len(doc.constraints)}} constraints")
'''

    # Handle snap vs regular FreeCAD - write script to appropriate location
    # Snap apps have restricted filesystem access, so we need to use their common dir
    snap_common = Path.home() / "snap" / "freecad" / "common"
    is_snap = "snap" in str(freecad_cmd[0]) or snap_common.exists()

    if is_snap and snap_common.exists():
        script_path = snap_common / "demo_script.py"
    else:
        # Use a file in /tmp for non-snap installations
        script_path = Path("/tmp/freecad_demo_script.py")

    script_path.write_text(script)

    # Build the command to run FreeCAD with the script
    # FreeCAD runs the script on startup when passed as an argument
    cmd = freecad_cmd + [str(script_path)]

    print(f"Running: {' '.join(cmd)}")

    # Use Popen to launch FreeCAD without waiting for it to exit
    subprocess.Popen(
        cmd,
        cwd=str(project_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print("FreeCAD launched. The sketch should open in the Sketcher workbench.")
    print(f"Script file: {script_path}")


def export_to_inventor(doc: SketchDocument) -> None:
    """Export the sketch to Autodesk Inventor."""
    try:
        from adapter_inventor import INVENTOR_AVAILABLE, InventorAdapter
    except ImportError:
        print("Error: adapter_inventor module not found.")
        print("Make sure the Inventor adapter is installed.")
        sys.exit(1)

    if not INVENTOR_AVAILABLE:
        print("Error: Autodesk Inventor is not available on this system.")
        print("Please install Inventor and ensure it's running (Windows only).")
        sys.exit(1)

    print("Exporting to Autodesk Inventor...")
    adapter = InventorAdapter()
    adapter.create_sketch(doc.name)
    adapter.load_sketch(doc)

    status, dof = adapter.get_solver_status()
    print(f"Solver status: {status.name}, DOF: {dof}")

    exported = adapter.export_sketch()
    print(f"Exported {len(exported.primitives)} primitives, {len(exported.constraints)} constraints")
    print("Sketch loaded successfully in Inventor!")


def export_to_solidworks(doc: SketchDocument) -> None:
    """Export the sketch to SolidWorks."""
    try:
        from adapter_solidworks import SOLIDWORKS_AVAILABLE, SolidWorksAdapter
    except ImportError:
        print("Error: adapter_solidworks module not found.")
        print("Make sure the SolidWorks adapter is installed.")
        sys.exit(1)

    if not SOLIDWORKS_AVAILABLE:
        print("Error: SolidWorks is not available on this system.")
        print("Please install SolidWorks and ensure it's running (Windows only).")
        sys.exit(1)

    print("Exporting to SolidWorks...")
    adapter = SolidWorksAdapter()
    adapter.create_sketch(doc.name)
    adapter.load_sketch(doc)

    status, dof = adapter.get_solver_status()
    print(f"Solver status: {status.name}, DOF: {dof}")

    exported = adapter.export_sketch()
    print(f"Exported {len(exported.primitives)} primitives, {len(exported.constraints)} constraints")
    print("Sketch loaded successfully in SolidWorks!")


def export_to_fusion(doc: SketchDocument) -> None:
    """Export the sketch to Fusion 360."""
    try:
        from adapter_fusion import FUSION_AVAILABLE, FusionAdapter
    except ImportError:
        print("Error: adapter_fusion module not found.")
        print("Make sure the Fusion 360 adapter is installed.")
        sys.exit(1)

    if not FUSION_AVAILABLE:
        print("Error: Fusion 360 is not available on this system.")
        print("Please run this script from within Fusion 360.")
        sys.exit(1)

    print("Exporting to Fusion 360...")
    adapter = FusionAdapter()
    adapter.create_sketch(doc.name)
    adapter.load_sketch(doc)

    status, dof = adapter.get_solver_status()
    print(f"Solver status: {status.name}, DOF: {dof}")

    exported = adapter.export_sketch()
    print(f"Exported {len(exported.primitives)} primitives, {len(exported.constraints)} constraints")
    print("Sketch loaded successfully in Fusion 360!")


def export_to_json(doc: SketchDocument, filename: str = "mounting_bracket.json") -> None:
    """Export the sketch to a JSON file."""
    print(f"Exporting to {filename}...")
    json_str = sketch_to_json(doc)

    with open(filename, 'w') as f:
        # Pretty print the JSON
        parsed = json.loads(json_str)
        json.dump(parsed, f, indent=2)

    print(f"Sketch exported to {filename}")
    print(f"File size: {len(json_str)} bytes")


def main():
    parser = argparse.ArgumentParser(
        description="Create and export a demo mounting bracket sketch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo.py --show              # Display sketch summary
    python demo.py --json              # Export to JSON file
    python demo.py --freecad           # Load into FreeCAD
    python demo.py --inventor          # Load into Inventor
    python demo.py --solidworks        # Load into SolidWorks
    python demo.py --json --freecad    # Export to JSON and FreeCAD
        """
    )

    parser.add_argument(
        '--show', action='store_true',
        help='Display a summary of the sketch'
    )
    parser.add_argument(
        '--json', action='store_true',
        help='Export to JSON file (mounting_bracket.json)'
    )
    parser.add_argument(
        '--freecad', action='store_true',
        help='Export to FreeCAD'
    )
    parser.add_argument(
        '--inventor', action='store_true',
        help='Export to Autodesk Inventor'
    )
    parser.add_argument(
        '--solidworks', action='store_true',
        help='Export to SolidWorks'
    )
    parser.add_argument(
        '--fusion', action='store_true',
        help='Export to Fusion 360'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='mounting_bracket.json',
        help='Output filename for JSON export (default: mounting_bracket.json)'
    )

    args = parser.parse_args()

    # If no arguments provided, show help
    if not any([args.show, args.json, args.freecad, args.inventor,
                args.solidworks, args.fusion]):
        parser.print_help()
        print("\nNo export target specified. Use --show to see the sketch summary.")
        sys.exit(0)

    # Create the sketch
    print("Creating mounting bracket sketch...")
    doc = create_mounting_bracket()
    print(f"Created sketch with {len(doc.primitives)} primitives and {len(doc.constraints)} constraints")

    # Show summary if requested
    if args.show:
        print_sketch_summary(doc)

    # Export to requested targets
    if args.json:
        export_to_json(doc, args.output)

    if args.freecad:
        export_to_freecad(doc)

    if args.inventor:
        export_to_inventor(doc)

    if args.solidworks:
        export_to_solidworks(doc)

    if args.fusion:
        export_to_fusion(doc)

    print("\nDone!")


if __name__ == "__main__":
    main()
