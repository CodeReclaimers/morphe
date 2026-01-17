"""
Round-trip tests for FreeCAD adapter.

These tests verify that sketches can be loaded into FreeCAD and exported back
without loss of essential information. Tests are skipped if FreeCAD is not
available on the system.
"""

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from morphe import (
    Angle,
    Arc,
    Circle,
    Coincident,
    Collinear,
    Concentric,
    Diameter,
    Distance,
    DistanceX,
    DistanceY,
    Ellipse,
    EllipticalArc,
    Equal,
    Fixed,
    Horizontal,
    Length,
    Line,
    MidpointConstraint,
    Parallel,
    Perpendicular,
    Point,
    Point2D,
    PointRef,
    PointType,
    Radius,
    SketchDocument,
    Spline,
    Tangent,
    Vertical,
    sketch_to_json,
)


def find_freecad_cmd():
    """Find the FreeCAD command-line executable."""
    # Check for snap installation first (common on Ubuntu)
    if shutil.which("snap"):
        result = subprocess.run(
            ["snap", "run", "freecad.cmd", "--version"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and "FreeCAD" in result.stdout:
            return ["snap", "run", "freecad.cmd"]

    # Check for freecadcmd in PATH
    freecadcmd = shutil.which("freecadcmd") or shutil.which("FreeCADCmd")
    if freecadcmd:
        return [freecadcmd]

    # Check for freecad with -c flag
    freecad = shutil.which("freecad") or shutil.which("FreeCAD")
    if freecad:
        # Verify it supports -c flag
        try:
            result = subprocess.run(
                [freecad, "-c", "print(1)"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return [freecad, "-c"]
        except Exception:
            pass

    return None


FREECAD_CMD = find_freecad_cmd()
FREECAD_AVAILABLE = FREECAD_CMD is not None

# Skip all tests in this module if FreeCAD is not available
pytestmark = pytest.mark.skipif(
    not FREECAD_AVAILABLE,
    reason="FreeCAD is not installed or not accessible"
)


# Path to the project root for imports within FreeCAD
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


def get_coord(point, coord):
    """Extract x or y coordinate from point (handles list or dict format)."""
    if isinstance(point, list):
        return point[0] if coord == "x" else point[1]
    return point[coord]


def run_in_freecad(script: str, timeout: int = 60) -> dict:
    """
    Run a Python script inside FreeCAD and return the result.

    The script should print a JSON object as its last output line.
    Returns the parsed JSON result or raises an exception on failure.
    """
    # Create a wrapper script that sets up the path and runs the user script
    wrapper_lines = [
        "import sys",
        "import json",
        "",
        f"sys.path.insert(0, {repr(str(PROJECT_ROOT))})",
        "",
        "try:",
    ]

    # Indent the user script
    for line in script.split('\n'):
        wrapper_lines.append("    " + line)

    wrapper_lines.extend([
        "except Exception as e:",
        "    import traceback",
        "    print(json.dumps({'error': str(e), 'traceback': traceback.format_exc()}))",
        "    sys.exit(1)",
    ])

    wrapper = '\n'.join(wrapper_lines)

    # For snap, we need to use a location snap can access
    # The snap can access ~/snap/freecad/common/
    snap_common = Path.home() / "snap" / "freecad" / "common"

    if FREECAD_CMD[0] == "snap" and snap_common.exists():
        # Use snap-accessible location
        script_path = snap_common / "roundtrip_test_script.py"
        script_path.write_text(wrapper)
        cmd = FREECAD_CMD + ["-c", f"exec(open({repr(str(script_path))}).read())"]
    else:
        # Use temp file for non-snap
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(wrapper)
            script_path = Path(f.name)

        if len(FREECAD_CMD) > 1 and FREECAD_CMD[-1] == "-c":
            cmd = FREECAD_CMD[:-1] + ["-c", f"exec(open({repr(str(script_path))}).read())"]
        else:
            cmd = FREECAD_CMD + [str(script_path)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT)
        )

        # Filter out FreeCAD startup messages and find JSON output
        output_lines = result.stdout.strip().split('\n')
        json_line = None
        for line in reversed(output_lines):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                json_line = line
                break

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise RuntimeError(f"FreeCAD script failed: {error_msg}")

        if json_line is None:
            raise RuntimeError(f"No JSON output found. stdout: {result.stdout}, stderr: {result.stderr}")

        return json.loads(json_line)

    finally:
        if script_path.exists():
            script_path.unlink()


class TestFreeCADRoundTripBasic:
    """Basic round-trip tests for simple geometries."""

    def test_single_line(self):
        """Test round-trip of a single line."""
        # Create source sketch
        sketch = SketchDocument(name="LineTest")
        sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 50)
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

# Load input sketch
input_json = {repr(input_json)}
sketch = sketch_from_json(input_json)

# Create adapter and load into FreeCAD
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)

# Export back
exported = adapter.export_sketch()
output_json = sketch_to_json(exported)

# Return result
result = {{
    "success": True,
    "primitive_count": len(exported.primitives),
    "output": json.loads(output_json)
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["primitive_count"] == 1

        # Verify line geometry
        exported = result["output"]
        prims = exported["primitives"]  # This is a list
        assert len(prims) == 1
        assert prims[0]["type"].lower() == "line"

        # Check coordinates
        assert abs(get_coord(prims[0]["start"], "x") - 0) < 1e-6
        assert abs(get_coord(prims[0]["start"], "y") - 0) < 1e-6
        assert abs(get_coord(prims[0]["end"], "x") - 100) < 1e-6
        assert abs(get_coord(prims[0]["end"], "y") - 50) < 1e-6

    def test_single_circle(self):
        """Test round-trip of a single circle."""
        sketch = SketchDocument(name="CircleTest")
        sketch.add_primitive(Circle(
            center=Point2D(50, 50),
            radius=25
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

result = {{
    "success": True,
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]

        exported = result["output"]
        prims = exported["primitives"]
        assert len(prims) == 1
        assert prims[0]["type"].lower() == "circle"
        assert abs(get_coord(prims[0]["center"], "x") - 50) < 1e-6
        assert abs(get_coord(prims[0]["center"], "y") - 50) < 1e-6
        assert abs(prims[0]["radius"] - 25) < 1e-6

    def test_single_arc(self):
        """Test round-trip of a single arc."""
        sketch = SketchDocument(name="ArcTest")
        sketch.add_primitive(Arc(
            center=Point2D(0, 0),
            start_point=Point2D(50, 0),
            end_point=Point2D(0, 50),
            ccw=True
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

result = {{
    "success": True,
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]

        exported = result["output"]
        prims = exported["primitives"]
        assert len(prims) == 1
        assert prims[0]["type"].lower() == "arc"

        # Verify arc geometry (center should be preserved)
        assert abs(get_coord(prims[0]["center"], "x") - 0) < 1e-6
        assert abs(get_coord(prims[0]["center"], "y") - 0) < 1e-6
        # Start and end points should be preserved (radius ~50)
        start = prims[0]["start_point"]
        end = prims[0]["end_point"]
        start_x = get_coord(start, "x")
        start_y = get_coord(start, "y")
        end_x = get_coord(end, "x")
        end_y = get_coord(end, "y")
        start_radius = math.sqrt(start_x**2 + start_y**2)
        end_radius = math.sqrt(end_x**2 + end_y**2)
        assert abs(start_radius - 50) < 1e-6
        assert abs(end_radius - 50) < 1e-6

    def test_single_point(self):
        """Test round-trip of a single point."""
        sketch = SketchDocument(name="PointTest")
        sketch.add_primitive(Point(position=Point2D(25, 75)))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

result = {{
    "success": True,
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]

        exported = result["output"]
        prims = exported["primitives"]
        assert len(prims) == 1
        assert prims[0]["type"].lower() == "point"
        assert abs(get_coord(prims[0]["position"], "x") - 25) < 1e-6
        assert abs(get_coord(prims[0]["position"], "y") - 75) < 1e-6

    def test_single_ellipse(self):
        """Test round-trip of a single ellipse."""
        sketch = SketchDocument(name="EllipseTest")
        sketch.add_primitive(Ellipse(
            center=Point2D(50, 50),
            major_radius=30,
            minor_radius=20,
            rotation=0.0
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

result = {{
    "success": True,
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]

        exported = result["output"]
        prims = exported["primitives"]
        assert len(prims) == 1
        assert prims[0]["type"].lower() == "ellipse"
        assert abs(get_coord(prims[0]["center"], "x") - 50) < 1e-6
        assert abs(get_coord(prims[0]["center"], "y") - 50) < 1e-6
        assert abs(prims[0]["major_radius"] - 30) < 1e-6
        assert abs(prims[0]["minor_radius"] - 20) < 1e-6

    def test_ellipse_rotated(self):
        """Test round-trip of a rotated ellipse."""
        sketch = SketchDocument(name="RotatedEllipseTest")
        sketch.add_primitive(Ellipse(
            center=Point2D(100, 100),
            major_radius=40,
            minor_radius=25,
            rotation=math.pi / 4  # 45 degrees
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

result = {{
    "success": True,
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]

        exported = result["output"]
        prims = exported["primitives"]
        assert len(prims) == 1
        assert prims[0]["type"].lower() == "ellipse"
        assert abs(get_coord(prims[0]["center"], "x") - 100) < 1e-6
        assert abs(get_coord(prims[0]["center"], "y") - 100) < 1e-6
        assert abs(prims[0]["major_radius"] - 40) < 1e-6
        assert abs(prims[0]["minor_radius"] - 25) < 1e-6
        # Rotation should be preserved (allow some tolerance)
        assert abs(prims[0]["rotation"] - math.pi / 4) < 0.01

    def test_single_elliptical_arc(self):
        """Test round-trip of a single elliptical arc."""
        sketch = SketchDocument(name="EllipticalArcTest")
        sketch.add_primitive(EllipticalArc(
            center=Point2D(50, 50),
            major_radius=30,
            minor_radius=20,
            rotation=0.0,
            start_param=0.0,
            end_param=math.pi / 2,  # Quarter ellipse
            ccw=True
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

result = {{
    "success": True,
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]

        exported = result["output"]
        prims = exported["primitives"]
        assert len(prims) == 1
        assert prims[0]["type"].lower() == "ellipticalarc"
        assert abs(get_coord(prims[0]["center"], "x") - 50) < 1e-6
        assert abs(get_coord(prims[0]["center"], "y") - 50) < 1e-6
        assert abs(prims[0]["major_radius"] - 30) < 1e-6
        assert abs(prims[0]["minor_radius"] - 20) < 1e-6


class TestFreeCADRoundTripComplex:
    """Round-trip tests for complex sketches with multiple geometries."""

    def test_rectangle(self):
        """Test round-trip of a rectangle (4 lines)."""
        sketch = SketchDocument(name="RectangleTest")

        # Create rectangle
        sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(100, 0)))
        sketch.add_primitive(Line(start=Point2D(100, 0), end=Point2D(100, 50)))
        sketch.add_primitive(Line(start=Point2D(100, 50), end=Point2D(0, 50)))
        sketch.add_primitive(Line(start=Point2D(0, 50), end=Point2D(0, 0)))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

result = {{
    "success": True,
    "primitive_count": len(exported.primitives),
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["primitive_count"] == 4

        # Verify all 4 lines were preserved
        exported = result["output"]
        prims = exported["primitives"]
        assert len(prims) == 4
        assert all(p["type"].lower() == "line" for p in prims)

    def test_mixed_geometry(self):
        """Test round-trip of mixed geometry types."""
        sketch = SketchDocument(name="MixedTest")

        # Add various geometry types
        sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(50, 0)))
        sketch.add_primitive(Arc(
            center=Point2D(50, 25),
            start_point=Point2D(50, 0),
            end_point=Point2D(75, 25),
            ccw=True
        ))
        sketch.add_primitive(Circle(center=Point2D(100, 50), radius=20))
        sketch.add_primitive(Point(position=Point2D(0, 50)))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

# Count types
types = [type(p).__name__ for p in exported.primitives.values()]
result = {{
    "success": True,
    "primitive_count": len(exported.primitives),
    "types": types,
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["primitive_count"] == 4

        types = result["types"]
        assert "Line" in types
        assert "Arc" in types
        assert "Circle" in types
        assert "Point" in types

    def test_construction_geometry(self):
        """Test that construction flag is preserved."""
        sketch = SketchDocument(name="ConstructionTest")
        sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 100),
            construction=True
        ))
        sketch.add_primitive(Circle(
            center=Point2D(50, 50),
            radius=30,
            construction=False
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

result = {{
    "success": True,
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]

        exported = result["output"]
        prims = exported["primitives"]

        # Find line and circle by type
        line = next(p for p in prims if p["type"].lower() == "line")
        circle = next(p for p in prims if p["type"].lower() == "circle")

        assert line["construction"] is True
        assert circle["construction"] is False


class TestFreeCADRoundTripConstraints:
    """Round-trip tests for constraints."""

    def test_horizontal_constraint(self):
        """Test horizontal constraint is applied."""
        sketch = SketchDocument(name="HorizontalTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(0, 10),  # Not horizontal initially
            end=Point2D(100, 20)
        ))
        sketch.add_constraint(Horizontal(line_id))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json, SolverStatus
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)

# Get solver status
status, dof = adapter.get_solver_status()

exported = adapter.export_sketch()

# Check the exported line is horizontal
line = list(exported.primitives.values())[0]
is_horizontal = abs(line.start.y - line.end.y) < 1e-6

result = {{
    "success": True,
    "is_horizontal": is_horizontal,
    "start_y": line.start.y,
    "end_y": line.end.y,
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        # The line should now be horizontal after constraint solving
        assert result["is_horizontal"], f"Line not horizontal: start_y={result['start_y']}, end_y={result['end_y']}"

    def test_vertical_constraint(self):
        """Test vertical constraint is applied."""
        sketch = SketchDocument(name="VerticalTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(10, 0),  # Not vertical initially
            end=Point2D(20, 100)
        ))
        sketch.add_constraint(Vertical(line_id))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

line = list(exported.primitives.values())[0]
is_vertical = abs(line.start.x - line.end.x) < 1e-6

result = {{
    "success": True,
    "is_vertical": is_vertical,
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["is_vertical"]

    def test_radius_constraint(self):
        """Test radius constraint is applied."""
        sketch = SketchDocument(name="RadiusTest")
        circle_id = sketch.add_primitive(Circle(
            center=Point2D(50, 50),
            radius=30  # Initial radius
        ))
        sketch.add_constraint(Radius(circle_id, 50))  # Constrain to radius 50

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

circle = list(exported.primitives.values())[0]

result = {{
    "success": True,
    "radius": circle.radius,
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        # Radius should be constrained to 50
        assert abs(result["radius"] - 50) < 1e-6

    def test_coincident_constraint(self):
        """Test coincident constraint connects line endpoints."""
        sketch = SketchDocument(name="CoincidentTest")
        l1 = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(50, 0)
        ))
        l2 = sketch.add_primitive(Line(
            start=Point2D(50, 5),  # Slightly off from l1 end
            end=Point2D(100, 50)
        ))
        sketch.add_constraint(Coincident(
            PointRef(l1, PointType.END),
            PointRef(l2, PointType.START)
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

prims = list(exported.primitives.values())
l1_end = prims[0].end
l2_start = prims[1].start

# Check if points are coincident
distance = ((l1_end.x - l2_start.x)**2 + (l1_end.y - l2_start.y)**2)**0.5

result = {{
    "success": True,
    "distance": distance,
    "l1_end": {{"x": l1_end.x, "y": l1_end.y}},
    "l2_start": {{"x": l2_start.x, "y": l2_start.y}},
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        # Points should now be coincident
        assert result["distance"] < 1e-6, f"Points not coincident: distance={result['distance']}"


class TestFreeCADRoundTripSpline:
    """Round-trip tests for B-splines."""

    def test_simple_bspline(self):
        """Test round-trip of a simple B-spline."""
        sketch = SketchDocument(name="SplineTest")

        # Create a degree-3 B-spline with 4 control points
        spline = Spline.create_uniform_bspline(
            control_points=[
                Point2D(0, 0),
                Point2D(30, 50),
                Point2D(70, 50),
                Point2D(100, 0)
            ],
            degree=3
        )
        sketch.add_primitive(spline)

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

spline = list(exported.primitives.values())[0]

result = {{
    "success": True,
    "type": type(spline).__name__,
    "degree": spline.degree,
    "control_point_count": len(spline.control_points),
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["type"] == "Spline"
        assert result["degree"] == 3
        assert result["control_point_count"] == 4


class TestFreeCADSolverStatus:
    """Tests for solver status reporting."""

    def test_fully_constrained_with_block(self):
        """Test detection of fully constrained sketch using Block constraint."""
        sketch = SketchDocument(name="FullyConstrainedTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 0)
        ))
        # Fix the line completely with Block constraint
        sketch.add_constraint(Fixed(line_id))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, SolverStatus
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)

status, dof = adapter.get_solver_status()

result = {{
    "success": True,
    "status": status.name,
    "dof": dof
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        # With Block constraint, should be fully constrained
        assert result["status"] == "FULLY_CONSTRAINED"
        assert result["dof"] == 0

    def test_solver_returns_status(self):
        """Test that solver returns a valid status for unconstrained geometry.

        Note: FreeCAD 1.0 returns solve()=0 even for unconstrained sketches,
        which differs from earlier versions. We just verify the adapter
        returns a valid status without crashing.
        """
        sketch = SketchDocument(name="UnconstrainedTest")
        # Just a line with no constraints
        sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 0)
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, SolverStatus
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)

status, dof = adapter.get_solver_status()

result = {{
    "success": True,
    "status": status.name,
    "dof": dof,
    "valid_status": status in [SolverStatus.FULLY_CONSTRAINED, SolverStatus.UNDER_CONSTRAINED]
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        # Just verify it's a valid status (FreeCAD 1.0 behavior varies)
        assert result["valid_status"]


class TestFreeCADRoundTripConstraintsExtended:
    """Extended constraint tests (backported from Fusion test suite)."""

    def test_parallel_constraint(self):
        """Test parallel constraint between two lines."""
        sketch = SketchDocument(name="ParallelTest")
        l1 = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 0)
        ))
        l2 = sketch.add_primitive(Line(
            start=Point2D(0, 50),
            end=Point2D(100, 60)  # Slightly not parallel
        ))
        sketch.add_constraint(Parallel(l1, l2))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
import math
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

prims = list(exported.primitives.values())
line1, line2 = prims[0], prims[1]

# Calculate direction vectors
dir1 = (line1.end.x - line1.start.x, line1.end.y - line1.start.y)
dir2 = (line2.end.x - line2.start.x, line2.end.y - line2.start.y)

# Normalize
len1 = math.sqrt(dir1[0]**2 + dir1[1]**2)
len2 = math.sqrt(dir2[0]**2 + dir2[1]**2)
dir1 = (dir1[0]/len1, dir1[1]/len1)
dir2 = (dir2[0]/len2, dir2[1]/len2)

# Cross product should be ~0 for parallel lines
cross = abs(dir1[0]*dir2[1] - dir1[1]*dir2[0])

result = {{
    "success": True,
    "cross_product": cross,
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["cross_product"] < 0.01, f"Lines not parallel: cross={result['cross_product']}"

    def test_perpendicular_constraint(self):
        """Test perpendicular constraint between two lines."""
        sketch = SketchDocument(name="PerpendicularTest")
        l1 = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 0)
        ))
        l2 = sketch.add_primitive(Line(
            start=Point2D(50, 0),
            end=Point2D(60, 100)  # Slightly not perpendicular
        ))
        sketch.add_constraint(Perpendicular(l1, l2))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
import math
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

prims = list(exported.primitives.values())
line1, line2 = prims[0], prims[1]

# Calculate direction vectors
dir1 = (line1.end.x - line1.start.x, line1.end.y - line1.start.y)
dir2 = (line2.end.x - line2.start.x, line2.end.y - line2.start.y)

# Dot product should be ~0 for perpendicular lines
dot = abs(dir1[0]*dir2[0] + dir1[1]*dir2[1])
len1 = math.sqrt(dir1[0]**2 + dir1[1]**2)
len2 = math.sqrt(dir2[0]**2 + dir2[1]**2)
dot_normalized = dot / (len1 * len2) if len1 * len2 > 0 else 0

result = {{
    "success": True,
    "dot_normalized": dot_normalized,
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["dot_normalized"] < 0.01, f"Lines not perpendicular: dot={result['dot_normalized']}"

    def test_equal_constraint(self):
        """Test equal constraint between two lines."""
        sketch = SketchDocument(name="EqualTest")
        l1 = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 0)
        ))
        l2 = sketch.add_primitive(Line(
            start=Point2D(0, 50),
            end=Point2D(80, 50)  # Different length initially
        ))
        sketch.add_constraint(Equal(l1, l2))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
import math
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

prims = list(exported.primitives.values())
line1, line2 = prims[0], prims[1]

len1 = math.sqrt((line1.end.x - line1.start.x)**2 + (line1.end.y - line1.start.y)**2)
len2 = math.sqrt((line2.end.x - line2.start.x)**2 + (line2.end.y - line2.start.y)**2)

result = {{
    "success": True,
    "len1": len1,
    "len2": len2,
    "diff": abs(len1 - len2),
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["diff"] < 0.1, f"Lines not equal: {result['len1']} vs {result['len2']}"

    def test_concentric_constraint(self):
        """Test concentric constraint between two circles."""
        sketch = SketchDocument(name="ConcentricTest")
        c1 = sketch.add_primitive(Circle(
            center=Point2D(50, 50),
            radius=30
        ))
        c2 = sketch.add_primitive(Circle(
            center=Point2D(55, 55),  # Slightly off center
            radius=50
        ))
        sketch.add_constraint(Concentric(c1, c2))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
import math
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

prims = list(exported.primitives.values())
circle1, circle2 = prims[0], prims[1]

center_distance = math.sqrt(
    (circle1.center.x - circle2.center.x)**2 +
    (circle1.center.y - circle2.center.y)**2
)

result = {{
    "success": True,
    "center_distance": center_distance,
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["center_distance"] < 0.01, f"Not concentric: distance={result['center_distance']}"

    def test_diameter_constraint(self):
        """Test diameter constraint is applied."""
        sketch = SketchDocument(name="DiameterTest")
        circle_id = sketch.add_primitive(Circle(
            center=Point2D(50, 50),
            radius=25  # Initial radius
        ))
        sketch.add_constraint(Diameter(circle_id, 80))  # Constrain to diameter 80 (radius 40)

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

circle = list(exported.primitives.values())[0]

result = {{
    "success": True,
    "radius": circle.radius,
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert abs(result["radius"] - 40) < 0.01, f"Radius mismatch: {result['radius']} (expected 40)"

    def test_angle_constraint(self):
        """Test angle constraint between two lines."""
        sketch = SketchDocument(name="AngleTest")
        l1 = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 0)
        ))
        l2 = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(70, 50)  # Some angle
        ))
        sketch.add_constraint(Angle(l1, l2, 45))  # Constrain to 45 degrees

        input_json = sketch_to_json(sketch)

        script = f'''
import json
import math
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

prims = list(exported.primitives.values())
line1, line2 = prims[0], prims[1]

# Calculate angle between lines
dir1 = (line1.end.x - line1.start.x, line1.end.y - line1.start.y)
dir2 = (line2.end.x - line2.start.x, line2.end.y - line2.start.y)

len1 = math.sqrt(dir1[0]**2 + dir1[1]**2)
len2 = math.sqrt(dir2[0]**2 + dir2[1]**2)

angle_deg = 0
if len1 > 0 and len2 > 0:
    dot = dir1[0]*dir2[0] + dir1[1]*dir2[1]
    cos_angle = dot / (len1 * len2)
    cos_angle = max(-1, min(1, cos_angle))
    angle_deg = math.degrees(math.acos(abs(cos_angle)))

result = {{
    "success": True,
    "angle": angle_deg,
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert abs(result["angle"] - 45) < 1, f"Angle mismatch: {result['angle']}"

    def test_quadratic_bspline(self):
        """Test round-trip of a degree-2 B-spline."""
        sketch = SketchDocument(name="QuadSplineTest")

        spline = Spline.create_uniform_bspline(
            control_points=[
                Point2D(0, 0),
                Point2D(50, 100),
                Point2D(100, 0)
            ],
            degree=2
        )
        sketch.add_primitive(spline)

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

spline = list(exported.primitives.values())[0]

result = {{
    "success": True,
    "type": type(spline).__name__,
    "degree": spline.degree,
    "control_point_count": len(spline.control_points),
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["type"] == "Spline"
        assert result["degree"] == 2
        assert result["control_point_count"] == 3

    def test_fully_constrained_rectangle(self):
        """Test a fully constrained rectangle with multiple constraints."""
        sketch = SketchDocument(name="FullRectTest")

        # Create rectangle
        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(100, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(100, 0), end=Point2D(100, 50)))
        l3 = sketch.add_primitive(Line(start=Point2D(100, 50), end=Point2D(0, 50)))
        l4 = sketch.add_primitive(Line(start=Point2D(0, 50), end=Point2D(0, 0)))

        # Add constraints to make it a proper rectangle
        sketch.add_constraint(Horizontal(l1))
        sketch.add_constraint(Horizontal(l3))
        sketch.add_constraint(Vertical(l2))
        sketch.add_constraint(Vertical(l4))

        # Connect corners
        sketch.add_constraint(Coincident(PointRef(l1, PointType.END), PointRef(l2, PointType.START)))
        sketch.add_constraint(Coincident(PointRef(l2, PointType.END), PointRef(l3, PointType.START)))
        sketch.add_constraint(Coincident(PointRef(l3, PointType.END), PointRef(l4, PointType.START)))
        sketch.add_constraint(Coincident(PointRef(l4, PointType.END), PointRef(l1, PointType.START)))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

lines = list(exported.primitives.values())

# Check horizontal lines are horizontal
horizontal_count = sum(1 for l in lines if abs(l.start.y - l.end.y) < 0.01)
vertical_count = sum(1 for l in lines if abs(l.start.x - l.end.x) < 0.01)

result = {{
    "success": True,
    "primitive_count": len(lines),
    "horizontal_count": horizontal_count,
    "vertical_count": vertical_count,
    "output": json.loads(sketch_to_json(exported))
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["primitive_count"] == 4
        assert result["horizontal_count"] == 2, "Should have 2 horizontal lines"
        assert result["vertical_count"] == 2, "Should have 2 vertical lines"


class TestFreeCADRoundTripGeometryEdgeCases:
    """Tests for geometry edge cases and coordinate variations."""

    def test_diagonal_line(self):
        """Test line at 45-degree angle."""
        sketch = SketchDocument(name="DiagonalTest")
        sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 100)
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

line = list(exported.primitives.values())[0]
dx = line.end.x - line.start.x
dy = line.end.y - line.start.y

result = {{
    "success": True,
    "dx": dx,
    "dy": dy,
    "is_45_deg": abs(dx - dy) < 0.01
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["is_45_deg"], "Line should be at 45 degrees"

    def test_negative_coordinates(self):
        """Test geometry in negative coordinate space."""
        sketch = SketchDocument(name="NegativeTest")
        sketch.add_primitive(Line(
            start=Point2D(-100, -50),
            end=Point2D(-20, -80)
        ))
        sketch.add_primitive(Circle(center=Point2D(-50, -50), radius=30))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json, Line, Circle
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

line = next(p for p in exported.primitives.values() if isinstance(p, Line))
circle = next(p for p in exported.primitives.values() if isinstance(p, Circle))

result = {{
    "success": True,
    "line_start_x": line.start.x,
    "line_start_y": line.start.y,
    "circle_center_x": circle.center.x,
    "circle_center_y": circle.center.y
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert abs(result["line_start_x"] - (-100)) < 0.01
        assert abs(result["line_start_y"] - (-50)) < 0.01
        assert abs(result["circle_center_x"] - (-50)) < 0.01
        assert abs(result["circle_center_y"] - (-50)) < 0.01

    def test_geometry_at_origin(self):
        """Test geometry centered at origin."""
        sketch = SketchDocument(name="OriginTest")
        sketch.add_primitive(Circle(center=Point2D(0, 0), radius=25))
        sketch.add_primitive(Line(start=Point2D(-50, 0), end=Point2D(50, 0)))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json, Circle
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

circle = next(p for p in exported.primitives.values() if isinstance(p, Circle))

result = {{
    "success": True,
    "center_x": circle.center.x,
    "center_y": circle.center.y,
    "primitive_count": len(exported.primitives)
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["primitive_count"] == 2
        assert abs(result["center_x"]) < 0.01
        assert abs(result["center_y"]) < 0.01

    def test_small_geometry(self):
        """Test very small geometry (precision test)."""
        sketch = SketchDocument(name="SmallTest")
        sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(0.1, 0.05)
        ))
        sketch.add_primitive(Circle(center=Point2D(1, 1), radius=0.05))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json, Line, Circle
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

line = next(p for p in exported.primitives.values() if isinstance(p, Line))
circle = next(p for p in exported.primitives.values() if isinstance(p, Circle))

result = {{
    "success": True,
    "line_end_x": line.end.x,
    "circle_radius": circle.radius
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert abs(result["line_end_x"] - 0.1) < 0.001
        assert abs(result["circle_radius"] - 0.05) < 0.001

    def test_large_geometry(self):
        """Test large geometry (1000mm scale)."""
        sketch = SketchDocument(name="LargeTest")
        sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(1000, 500)
        ))
        sketch.add_primitive(Circle(center=Point2D(500, 500), radius=250))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json, Line, Circle
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

line = next(p for p in exported.primitives.values() if isinstance(p, Line))
circle = next(p for p in exported.primitives.values() if isinstance(p, Circle))

result = {{
    "success": True,
    "line_end_x": line.end.x,
    "circle_radius": circle.radius
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert abs(result["line_end_x"] - 1000) < 0.1
        assert abs(result["circle_radius"] - 250) < 0.1

    def test_arc_clockwise(self):
        """Test round-trip of a clockwise arc."""
        sketch = SketchDocument(name="CWArcTest")
        sketch.add_primitive(Arc(
            center=Point2D(0, 0),
            start_point=Point2D(50, 0),
            end_point=Point2D(0, 50),
            ccw=False
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
import math
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

arc = list(exported.primitives.values())[0]
start_radius = math.sqrt(arc.start_point.x**2 + arc.start_point.y**2)

result = {{
    "success": True,
    "start_radius": start_radius,
    "primitive_count": len(exported.primitives)
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["primitive_count"] == 1
        assert abs(result["start_radius"] - 50) < 0.1

    def test_arc_large_angle(self):
        """Test round-trip of a large arc (> 180 degrees)."""
        sketch = SketchDocument(name="LargeArcTest")
        sketch.add_primitive(Arc(
            center=Point2D(0, 0),
            start_point=Point2D(50, 0),
            end_point=Point2D(0, -50),
            ccw=True
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
import math
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

arc = list(exported.primitives.values())[0]
start_radius = math.sqrt(arc.start_point.x**2 + arc.start_point.y**2)

result = {{
    "success": True,
    "start_radius": start_radius,
    "primitive_count": len(exported.primitives)
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert abs(result["start_radius"] - 50) < 0.1

    def test_construction_arc(self):
        """Test construction arc flag is preserved."""
        sketch = SketchDocument(name="ConstructionArcTest")
        sketch.add_primitive(Arc(
            center=Point2D(50, 50),
            start_point=Point2D(80, 50),
            end_point=Point2D(50, 80),
            ccw=True,
            construction=True
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

arc = list(exported.primitives.values())[0]

result = {{
    "success": True,
    "is_construction": arc.construction
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["is_construction"] is True

    def test_empty_sketch(self):
        """Test that empty sketch exports correctly."""
        sketch = SketchDocument(name="EmptySketchTest")

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

result = {{
    "success": True,
    "primitive_count": len(exported.primitives)
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["primitive_count"] == 0


class TestFreeCADRoundTripConstraintsAdvanced:
    """Advanced constraint tests including tangent, collinear, distance constraints."""

    def test_tangent_line_circle(self):
        """Test tangent constraint between line and circle."""
        sketch = SketchDocument(name="TangentTest")
        circle_id = sketch.add_primitive(Circle(
            center=Point2D(50, 50),
            radius=30
        ))
        line_id = sketch.add_primitive(Line(
            start=Point2D(0, 85),
            end=Point2D(100, 80)
        ))
        sketch.add_constraint(Tangent(line_id, circle_id))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
import math
from morphe import sketch_from_json, sketch_to_json, Circle, Line
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

circle = next(p for p in exported.primitives.values() if isinstance(p, Circle))
line = next(p for p in exported.primitives.values() if isinstance(p, Line))

# Calculate distance from circle center to line
x1, y1 = line.start.x, line.start.y
x2, y2 = line.end.x, line.end.y
px, py = circle.center.x, circle.center.y

line_len = math.sqrt((x2-x1)**2 + (y2-y1)**2)
dist = abs((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1) / line_len if line_len > 0 else 0

result = {{
    "success": True,
    "distance": dist,
    "radius": circle.radius,
    "is_tangent": abs(dist - circle.radius) < 0.5
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["is_tangent"], f"Not tangent: distance={result['distance']}, radius={result['radius']}"

    def test_fixed_constraint(self):
        """Test fixed constraint locks geometry in place."""
        sketch = SketchDocument(name="FixedTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(10, 20),
            end=Point2D(50, 60)
        ))
        sketch.add_constraint(Fixed(line_id))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

line = list(exported.primitives.values())[0]

result = {{
    "success": True,
    "start_x": line.start.x,
    "start_y": line.start.y,
    "end_x": line.end.x,
    "end_y": line.end.y
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert abs(result["start_x"] - 10) < 0.01
        assert abs(result["start_y"] - 20) < 0.01
        assert abs(result["end_x"] - 50) < 0.01
        assert abs(result["end_y"] - 60) < 0.01

    def test_distance_constraint(self):
        """Test distance constraint between two points."""
        sketch = SketchDocument(name="DistanceTest")
        l1 = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(50, 0)
        ))
        l2 = sketch.add_primitive(Line(
            start=Point2D(60, 0),
            end=Point2D(100, 0)
        ))
        sketch.add_constraint(Distance(
            PointRef(l1, PointType.END),
            PointRef(l2, PointType.START),
            20
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
import math
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

prims = list(exported.primitives.values())
l1_end = prims[0].end
l2_start = prims[1].start

dist = math.sqrt((l2_start.x - l1_end.x)**2 + (l2_start.y - l1_end.y)**2)

result = {{
    "success": True,
    "distance": dist
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert abs(result["distance"] - 20) < 0.1

    def test_distance_x_constraint(self):
        """Test horizontal distance constraint."""
        sketch = SketchDocument(name="DistanceXTest")
        l1 = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(30, 0)
        ))
        l2 = sketch.add_primitive(Line(
            start=Point2D(50, 20),
            end=Point2D(80, 20)
        ))
        sketch.add_constraint(DistanceX(
            PointRef(l1, PointType.END),
            40,
            PointRef(l2, PointType.START)
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

prims = list(exported.primitives.values())
l1_end = prims[0].end
l2_start = prims[1].start

dx = abs(l2_start.x - l1_end.x)

result = {{
    "success": True,
    "dx": dx
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert abs(result["dx"] - 40) < 0.1

    def test_distance_y_constraint(self):
        """Test vertical distance constraint."""
        sketch = SketchDocument(name="DistanceYTest")
        l1 = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(50, 0)
        ))
        l2 = sketch.add_primitive(Line(
            start=Point2D(0, 30),
            end=Point2D(50, 30)
        ))
        sketch.add_constraint(DistanceY(
            PointRef(l1, PointType.START),
            50,
            PointRef(l2, PointType.START)
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

prims = list(exported.primitives.values())
l1_start = prims[0].start
l2_start = prims[1].start

dy = abs(l2_start.y - l1_start.y)

result = {{
    "success": True,
    "dy": dy
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert abs(result["dy"] - 50) < 0.1


class TestFreeCADRoundTripSplineAdvanced:
    """Advanced spline tests including higher degrees and special cases."""

    def test_higher_degree_spline(self):
        """Test round-trip of a degree-4 B-spline."""
        sketch = SketchDocument(name="Degree4SplineTest")

        spline = Spline.create_uniform_bspline(
            control_points=[
                Point2D(0, 0),
                Point2D(25, 50),
                Point2D(50, 0),
                Point2D(75, 50),
                Point2D(100, 0)
            ],
            degree=4
        )
        sketch.add_primitive(spline)

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

spline = list(exported.primitives.values())[0]

result = {{
    "success": True,
    "type": type(spline).__name__,
    "degree": spline.degree,
    "control_point_count": len(spline.control_points)
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["type"] == "Spline"
        assert result["degree"] == 4
        assert result["control_point_count"] == 5

    def test_many_control_points_spline(self):
        """Test spline with many control points."""
        sketch = SketchDocument(name="ManyPointsSplineTest")

        control_pts = [
            Point2D(i * 15, 30 * math.sin(i * 0.8))
            for i in range(8)
        ]
        spline = Spline.create_uniform_bspline(
            control_points=control_pts,
            degree=3
        )
        sketch.add_primitive(spline)

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

spline = list(exported.primitives.values())[0]

result = {{
    "success": True,
    "control_point_count": len(spline.control_points)
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["control_point_count"] == 8

    def test_weighted_spline(self):
        """Test spline with non-uniform weights (NURBS)."""
        sketch = SketchDocument(name="WeightedSplineTest")

        spline = Spline(
            control_points=[
                Point2D(0, 0),
                Point2D(50, 100),
                Point2D(100, 0)
            ],
            degree=2,
            knots=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            weights=[1.0, 2.0, 1.0]  # Higher weight pulls curve toward middle point
        )
        sketch.add_primitive(spline)

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

spline = list(exported.primitives.values())[0]

result = {{
    "success": True,
    "type": type(spline).__name__,
    "has_weights": spline.weights is not None and len(spline.weights) > 0
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["type"] == "Spline"


class TestFreeCADRoundTripComplexScenarios:
    """Tests for complex geometry scenarios."""

    def test_closed_profile(self):
        """Test a closed profile with connected lines (triangle)."""
        sketch = SketchDocument(name="ClosedProfileTest")

        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(100, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(100, 0), end=Point2D(50, 80)))
        l3 = sketch.add_primitive(Line(start=Point2D(50, 80), end=Point2D(0, 0)))

        sketch.add_constraint(Coincident(PointRef(l1, PointType.END), PointRef(l2, PointType.START)))
        sketch.add_constraint(Coincident(PointRef(l2, PointType.END), PointRef(l3, PointType.START)))
        sketch.add_constraint(Coincident(PointRef(l3, PointType.END), PointRef(l1, PointType.START)))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from collections import Counter
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

lines = list(exported.primitives.values())
endpoints = []
for line in lines:
    endpoints.append((round(line.start.x, 1), round(line.start.y, 1)))
    endpoints.append((round(line.end.x, 1), round(line.end.y, 1)))

counts = Counter(endpoints)
is_closed = all(c == 2 for c in counts.values())

result = {{
    "success": True,
    "primitive_count": len(lines),
    "is_closed": is_closed
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["primitive_count"] == 3
        assert result["is_closed"], "Profile not properly closed"

    def test_slot_profile(self):
        """Test a slot profile (two semicircles connected by lines)."""
        sketch = SketchDocument(name="SlotTest")

        sketch.add_primitive(Line(start=Point2D(20, 0), end=Point2D(80, 0)))
        sketch.add_primitive(Line(start=Point2D(80, 40), end=Point2D(20, 40)))
        sketch.add_primitive(Arc(
            center=Point2D(80, 20),
            start_point=Point2D(80, 0),
            end_point=Point2D(80, 40),
            ccw=True
        ))
        sketch.add_primitive(Arc(
            center=Point2D(20, 20),
            start_point=Point2D(20, 40),
            end_point=Point2D(20, 0),
            ccw=True
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json, Line, Arc
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
arcs = [p for p in exported.primitives.values() if isinstance(p, Arc)]

result = {{
    "success": True,
    "line_count": len(lines),
    "arc_count": len(arcs),
    "total": len(exported.primitives)
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["line_count"] == 2
        assert result["arc_count"] == 2
        assert result["total"] == 4

    def test_nested_geometry(self):
        """Test circle inside rectangle (common CAD pattern)."""
        sketch = SketchDocument(name="NestedTest")

        sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(100, 0)))
        sketch.add_primitive(Line(start=Point2D(100, 0), end=Point2D(100, 80)))
        sketch.add_primitive(Line(start=Point2D(100, 80), end=Point2D(0, 80)))
        sketch.add_primitive(Line(start=Point2D(0, 80), end=Point2D(0, 0)))
        sketch.add_primitive(Circle(center=Point2D(50, 40), radius=25))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json, Line, Circle
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
circles = [p for p in exported.primitives.values() if isinstance(p, Circle)]

result = {{
    "success": True,
    "line_count": len(lines),
    "circle_count": len(circles),
    "total": len(exported.primitives)
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["line_count"] == 4
        assert result["circle_count"] == 1
        assert result["total"] == 5

    def test_concentric_circles(self):
        """Test multiple concentric circles."""
        sketch = SketchDocument(name="ConcentricCirclesTest")

        c1 = sketch.add_primitive(Circle(center=Point2D(50, 50), radius=10))
        c2 = sketch.add_primitive(Circle(center=Point2D(52, 52), radius=25))
        c3 = sketch.add_primitive(Circle(center=Point2D(48, 48), radius=40))

        sketch.add_constraint(Concentric(c1, c2))
        sketch.add_constraint(Concentric(c2, c3))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
import math
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

circles = list(exported.primitives.values())

# Check all centers are the same
centers = [(c.center.x, c.center.y) for c in circles]
max_dist = 0
for i, c1 in enumerate(centers):
    for c2 in centers[i+1:]:
        dist = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
        max_dist = max(max_dist, dist)

result = {{
    "success": True,
    "circle_count": len(circles),
    "max_center_distance": max_dist,
    "are_concentric": max_dist < 0.1
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["circle_count"] == 3
        assert result["are_concentric"], f"Not concentric: max distance={result['max_center_distance']}"

    def test_equal_circles(self):
        """Test equal constraint on circles."""
        sketch = SketchDocument(name="EqualCirclesTest")

        c1 = sketch.add_primitive(Circle(center=Point2D(30, 50), radius=20))
        c2 = sketch.add_primitive(Circle(center=Point2D(80, 50), radius=30))

        sketch.add_constraint(Equal(c1, c2))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

circles = list(exported.primitives.values())

result = {{
    "success": True,
    "radius1": circles[0].radius,
    "radius2": circles[1].radius,
    "are_equal": abs(circles[0].radius - circles[1].radius) < 0.1
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["are_equal"], f"Radii not equal: {result['radius1']} vs {result['radius2']}"


class TestFreeCADRoundTripAdditional:
    """Additional tests backported from Fusion test suite."""

    def test_solver_status_fullyconstrained(self):
        """Test that fully constrained sketch reports zero DOF."""
        sketch = SketchDocument(name="FullyConstrainedTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 0)
        ))
        sketch.add_constraint(Fixed(line_id))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

result = {{
    "success": True,
    "dof": exported.degrees_of_freedom
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["dof"] == 0, f"Expected DOF = 0, got {result['dof']}"

    def test_multiple_points_standalone(self):
        """Test that multiple standalone points are exported correctly."""
        sketch = SketchDocument(name="MultiPointTest")
        sketch.add_primitive(Point(position=Point2D(10, 20)))
        sketch.add_primitive(Point(position=Point2D(50, 60)))
        sketch.add_primitive(Point(position=Point2D(90, 30)))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json, Point
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

points = [p for p in exported.primitives.values() if isinstance(p, Point)]

result = {{
    "success": True,
    "point_count": len(points)
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["point_count"] == 3

    def test_constraint_export_horizontal(self):
        """Test that horizontal constraint is exported back correctly."""
        sketch = SketchDocument(name="ConstraintExportHorizTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(10, 20),
            end=Point2D(80, 25)
        ))
        sketch.add_constraint(Horizontal(line_id))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

line = list(exported.primitives.values())[0]

result = {{
    "success": True,
    "start_y": line.start.y,
    "end_y": line.end.y,
    "is_horizontal": abs(line.start.y - line.end.y) < 0.01
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["is_horizontal"], f"Line not horizontal: start.y={result['start_y']}, end.y={result['end_y']}"

    def test_constraint_export_perpendicular(self):
        """Test that perpendicular constraint produces 90-degree angle."""
        sketch = SketchDocument(name="ConstraintExportPerpTest")
        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(50, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(25, 0), end=Point2D(30, 40)))
        sketch.add_constraint(Perpendicular(l1, l2))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
import math
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

lines = list(exported.primitives.values())
dx1, dy1 = lines[0].end.x - lines[0].start.x, lines[0].end.y - lines[0].start.y
dx2, dy2 = lines[1].end.x - lines[1].start.x, lines[1].end.y - lines[1].start.y
dot = dx1 * dx2 + dy1 * dy2
len1 = math.sqrt(dx1**2 + dy1**2)
len2 = math.sqrt(dx2**2 + dy2**2)
cos_angle = dot / (len1 * len2) if len1 > 0 and len2 > 0 else 0

result = {{
    "success": True,
    "cos_angle": cos_angle,
    "is_perpendicular": abs(cos_angle) < 0.01
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["is_perpendicular"], f"Lines not perpendicular, cos(angle)={result['cos_angle']}"

    def test_multiple_constraints_circle(self):
        """Test concentric + equal constraints on circles."""
        sketch = SketchDocument(name="MultiConstraintCircleTest")
        c1 = sketch.add_primitive(Circle(center=Point2D(50, 50), radius=20))
        c2 = sketch.add_primitive(Circle(center=Point2D(60, 55), radius=35))
        sketch.add_constraint(Concentric(c1, c2))
        sketch.add_constraint(Equal(c1, c2))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

circles = list(exported.primitives.values())

result = {{
    "success": True,
    "center1_x": circles[0].center.x,
    "center1_y": circles[0].center.y,
    "center2_x": circles[1].center.x,
    "center2_y": circles[1].center.y,
    "radius1": circles[0].radius,
    "radius2": circles[1].radius,
    "is_concentric": abs(circles[0].center.x - circles[1].center.x) < 0.01 and abs(circles[0].center.y - circles[1].center.y) < 0.01,
    "is_equal": abs(circles[0].radius - circles[1].radius) < 0.01
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["is_concentric"], "Circles should share center"
        assert result["is_equal"], "Circles should have equal radii"

    def test_coincident_point_to_line_endpoint(self):
        """Test point coincident with line endpoint."""
        sketch = SketchDocument(name="PointToLineEndTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(10, 10),
            end=Point2D(80, 50)
        ))
        point_id = sketch.add_primitive(Point(position=Point2D(50, 30)))
        sketch.add_constraint(Coincident(
            PointRef(point_id, PointType.CENTER),
            PointRef(line_id, PointType.END)
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json, Line, Point
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

line = next(p for p in exported.primitives.values() if isinstance(p, Line))
point = next(p for p in exported.primitives.values() if isinstance(p, Point))

result = {{
    "success": True,
    "point_x": point.position.x,
    "point_y": point.position.y,
    "line_end_x": line.end.x,
    "line_end_y": line.end.y,
    "is_coincident": abs(point.position.x - line.end.x) < 0.01 and abs(point.position.y - line.end.y) < 0.01
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["is_coincident"], "Point should be at line end"

    def test_coincident_point_to_circle_center(self):
        """Test point coincident with circle center."""
        sketch = SketchDocument(name="PointToCircleCenterTest")
        circle_id = sketch.add_primitive(Circle(center=Point2D(60, 40), radius=25))
        point_id = sketch.add_primitive(Point(position=Point2D(30, 20)))
        sketch.add_constraint(Coincident(
            PointRef(point_id, PointType.CENTER),
            PointRef(circle_id, PointType.CENTER)
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json, Circle, Point
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

circle = next(p for p in exported.primitives.values() if isinstance(p, Circle))
point = next(p for p in exported.primitives.values() if isinstance(p, Point))

result = {{
    "success": True,
    "point_x": point.position.x,
    "point_y": point.position.y,
    "circle_center_x": circle.center.x,
    "circle_center_y": circle.center.y,
    "is_coincident": abs(point.position.x - circle.center.x) < 0.01 and abs(point.position.y - circle.center.y) < 0.01
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["is_coincident"], "Point should be at circle center"

    def test_arc_90_degree(self):
        """Test 90-degree arc preserves angle precisely."""
        sketch = SketchDocument(name="Arc90Test")
        sketch.add_primitive(Arc(
            center=Point2D(50, 50),
            start_point=Point2D(80, 50),
            end_point=Point2D(50, 80),
            ccw=True
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
import math
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

arc = list(exported.primitives.values())[0]
start_angle = math.atan2(arc.start_point.y - arc.center.y, arc.start_point.x - arc.center.x)
end_angle = math.atan2(arc.end_point.y - arc.center.y, arc.end_point.x - arc.center.x)
sweep = end_angle - start_angle
if sweep < 0:
    sweep += 2 * math.pi
sweep_deg = math.degrees(sweep)

result = {{
    "success": True,
    "sweep_deg": sweep_deg
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert abs(result["sweep_deg"] - 90) < 1.0, f"Arc should be 90 degrees, got {result['sweep_deg']}"

    def test_arc_180_degree(self):
        """Test 180-degree arc (semicircle) preserves angle precisely."""
        sketch = SketchDocument(name="Arc180Test")
        sketch.add_primitive(Arc(
            center=Point2D(50, 50),
            start_point=Point2D(80, 50),
            end_point=Point2D(20, 50),
            ccw=True
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
import math
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

arc = list(exported.primitives.values())[0]
start_angle = math.atan2(arc.start_point.y - arc.center.y, arc.start_point.x - arc.center.x)
end_angle = math.atan2(arc.end_point.y - arc.center.y, arc.end_point.x - arc.center.x)
sweep = end_angle - start_angle
if sweep < 0:
    sweep += 2 * math.pi
sweep_deg = math.degrees(sweep)

result = {{
    "success": True,
    "sweep_deg": sweep_deg
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert abs(result["sweep_deg"] - 180) < 1.0, f"Arc should be 180 degrees, got {result['sweep_deg']}"

    def test_arc_45_degree(self):
        """Test 45-degree arc preserves angle precisely."""
        r = 30
        sketch = SketchDocument(name="Arc45Test")
        sketch.add_primitive(Arc(
            center=Point2D(50, 50),
            start_point=Point2D(50 + r, 50),
            end_point=Point2D(50 + r * math.cos(math.radians(45)),
                             50 + r * math.sin(math.radians(45))),
            ccw=True
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
import math
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

arc = list(exported.primitives.values())[0]
start_angle = math.atan2(arc.start_point.y - arc.center.y, arc.start_point.x - arc.center.x)
end_angle = math.atan2(arc.end_point.y - arc.center.y, arc.end_point.x - arc.center.x)
sweep = end_angle - start_angle
if sweep < 0:
    sweep += 2 * math.pi
sweep_deg = math.degrees(sweep)

result = {{
    "success": True,
    "sweep_deg": sweep_deg
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert abs(result["sweep_deg"] - 45) < 1.0, f"Arc should be 45 degrees, got {result['sweep_deg']}"

    def test_periodic_spline(self):
        """Test closed/periodic spline round-trip."""
        control_points = [
            Point2D(50, 0),
            Point2D(100, 25),
            Point2D(100, 75),
            Point2D(50, 100),
            Point2D(0, 75),
            Point2D(0, 25),
            Point2D(50, 0),
        ]
        knots = [0, 0, 0, 0, 0.33, 0.5, 0.67, 1, 1, 1, 1]

        sketch = SketchDocument(name="PeriodicSplineTest")
        sketch.add_primitive(Spline(
            control_points=control_points,
            degree=3,
            knots=knots,
            periodic=False
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
import math
from morphe import sketch_from_json, sketch_to_json, Spline
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

spline = list(exported.primitives.values())[0]
start = spline.control_points[0]
end = spline.control_points[-1]
dist = math.sqrt((end.x - start.x)**2 + (end.y - start.y)**2)

result = {{
    "success": True,
    "type": type(spline).__name__,
    "cp_count": len(spline.control_points),
    "start_end_dist": dist,
    "is_closed": dist < 1.0
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["type"] == "Spline"
        assert result["cp_count"] >= 6
        assert result["is_closed"], f"Spline should be closed, start-end distance={result['start_end_dist']}"

    def test_radius_precision(self):
        """Test radius constraint with high precision value."""
        sketch = SketchDocument(name="RadiusPrecisionTest")
        circle_id = sketch.add_primitive(Circle(
            center=Point2D(50, 50),
            radius=20
        ))
        precise_radius = 33.789012
        sketch.add_constraint(Radius(circle_id, precise_radius))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

circle = list(exported.primitives.values())[0]

result = {{
    "success": True,
    "radius": circle.radius,
    "expected": {precise_radius}
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert abs(result["radius"] - precise_radius) < 0.001, f"Radius should be {precise_radius}, got {result['radius']}"

    def test_angle_precision(self):
        """Test angle constraint with precise value."""
        sketch = SketchDocument(name="AnglePrecisionTest")
        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(50, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(40, 30)))
        precise_angle = 37.5
        sketch.add_constraint(Angle(l1, l2, precise_angle))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
import math
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

lines = list(exported.primitives.values())
dx1, dy1 = lines[0].end.x - lines[0].start.x, lines[0].end.y - lines[0].start.y
dx2, dy2 = lines[1].end.x - lines[1].start.x, lines[1].end.y - lines[1].start.y
dot = dx1 * dx2 + dy1 * dy2
cross = dx1 * dy2 - dy1 * dx2
angle_rad = math.atan2(abs(cross), dot)
angle_deg = math.degrees(angle_rad)

result = {{
    "success": True,
    "angle_deg": angle_deg,
    "expected": {precise_angle}
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert abs(result["angle_deg"] - precise_angle) < 0.5, f"Angle should be {precise_angle}, got {result['angle_deg']}"

    def test_equal_chain_three_lines(self):
        """Test equal constraint across three lines."""
        sketch = SketchDocument(name="EqualChain3LinesTest")
        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(30, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(0, 20), end=Point2D(50, 20)))
        l3 = sketch.add_primitive(Line(start=Point2D(0, 40), end=Point2D(70, 40)))
        sketch.add_constraint(Equal(l1, l2))
        sketch.add_constraint(Equal(l2, l3))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
import math
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

lines = list(exported.primitives.values())
lengths = [
    math.sqrt((ln.end.x - ln.start.x)**2 + (ln.end.y - ln.start.y)**2)
    for ln in lines
]

result = {{
    "success": True,
    "lengths": lengths,
    "all_equal": abs(lengths[0] - lengths[1]) < 0.1 and abs(lengths[1] - lengths[2]) < 0.1
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["all_equal"], f"Lines should have equal lengths: {result['lengths']}"

    def test_equal_chain_four_circles(self):
        """Test equal constraint across four circles."""
        sketch = SketchDocument(name="EqualChain4CirclesTest")
        c1 = sketch.add_primitive(Circle(center=Point2D(20, 20), radius=10))
        c2 = sketch.add_primitive(Circle(center=Point2D(60, 20), radius=15))
        c3 = sketch.add_primitive(Circle(center=Point2D(20, 60), radius=20))
        c4 = sketch.add_primitive(Circle(center=Point2D(60, 60), radius=25))
        sketch.add_constraint(Equal(c1, c2))
        sketch.add_constraint(Equal(c2, c3))
        sketch.add_constraint(Equal(c3, c4))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

circles = list(exported.primitives.values())
radii = [c.radius for c in circles]

all_equal = all(abs(radii[i] - radii[i+1]) < 0.1 for i in range(len(radii) - 1))

result = {{
    "success": True,
    "radii": radii,
    "all_equal": all_equal
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["all_equal"], f"Circles should have equal radii: {result['radii']}"

    def test_arc_tangent_to_two_lines(self):
        """Test arc tangent to two lines (fillet pattern)."""
        sketch = SketchDocument(name="FilletPatternTest")
        l1 = sketch.add_primitive(Line(start=Point2D(0, 50), end=Point2D(50, 50)))
        l2 = sketch.add_primitive(Line(start=Point2D(50, 50), end=Point2D(50, 0)))
        arc_id = sketch.add_primitive(Arc(
            center=Point2D(35, 35),
            start_point=Point2D(35, 50),
            end_point=Point2D(50, 35),
            ccw=False
        ))
        sketch.add_constraint(Perpendicular(l1, l2))
        sketch.add_constraint(Tangent(l1, arc_id))
        sketch.add_constraint(Tangent(arc_id, l2))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

result = {{
    "success": True,
    "primitive_count": len(exported.primitives)
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["primitive_count"] == 3

    def test_smooth_corner_profile(self):
        """Test L-shaped profile with rounded corner."""
        sketch = SketchDocument(name="SmoothCornerTest")
        l1 = sketch.add_primitive(Line(start=Point2D(0, 30), end=Point2D(30, 30)))
        arc_id = sketch.add_primitive(Arc(
            center=Point2D(30, 20),
            start_point=Point2D(30, 30),
            end_point=Point2D(40, 20),
            ccw=False
        ))
        l2 = sketch.add_primitive(Line(start=Point2D(40, 20), end=Point2D(40, 0)))

        sketch.add_constraint(Coincident(
            PointRef(l1, PointType.END),
            PointRef(arc_id, PointType.START)
        ))
        sketch.add_constraint(Coincident(
            PointRef(arc_id, PointType.END),
            PointRef(l2, PointType.START)
        ))
        sketch.add_constraint(Tangent(l1, arc_id))
        sketch.add_constraint(Tangent(arc_id, l2))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

result = {{
    "success": True,
    "primitive_count": len(exported.primitives)
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["primitive_count"] == 3

    def test_very_small_dimensions(self):
        """Test geometry with very small dimensions (micrometer scale)."""
        sketch = SketchDocument(name="MicroScaleTest")
        sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(0.01, 0.01)
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

line = list(exported.primitives.values())[0]

result = {{
    "success": True,
    "end_x": line.end.x
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert abs(result["end_x"] - 0.01) < 0.001

    def test_very_large_dimensions(self):
        """Test geometry with very large dimensions (meter scale in mm)."""
        sketch = SketchDocument(name="LargeScaleTest")
        sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(1000, 1000)
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

line = list(exported.primitives.values())[0]

result = {{
    "success": True,
    "end_x": line.end.x
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert abs(result["end_x"] - 1000) < 0.1

    def test_coincident_chain(self):
        """Test chain of coincident constraints forming connected path."""
        sketch = SketchDocument(name="CoincidentChainTest")
        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(30, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(35, 5), end=Point2D(60, 30)))
        l3 = sketch.add_primitive(Line(start=Point2D(65, 35), end=Point2D(30, 60)))

        sketch.add_constraint(Coincident(
            PointRef(l1, PointType.END),
            PointRef(l2, PointType.START)
        ))
        sketch.add_constraint(Coincident(
            PointRef(l2, PointType.END),
            PointRef(l3, PointType.START)
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

lines = list(exported.primitives.values())
l1, l2, l3 = lines[0], lines[1], lines[2]

chain_ok = (
    abs(l1.end.x - l2.start.x) < 0.01 and
    abs(l1.end.y - l2.start.y) < 0.01 and
    abs(l2.end.x - l3.start.x) < 0.01 and
    abs(l2.end.y - l3.start.y) < 0.01
)

result = {{
    "success": True,
    "chain_connected": chain_ok
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["chain_connected"], "Coincident chain should form connected path"

    def test_tangent_arc_line(self):
        """Test tangent constraint between arc and line."""
        sketch = SketchDocument(name="TangentArcTest")
        arc_id = sketch.add_primitive(Arc(
            center=Point2D(50, 50),
            start_point=Point2D(80, 50),
            end_point=Point2D(50, 80),
            ccw=True
        ))
        line_id = sketch.add_primitive(Line(
            start=Point2D(80, 50),
            end=Point2D(120, 55)
        ))
        sketch.add_constraint(Tangent(arc_id, line_id))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

result = {{
    "success": True,
    "primitive_count": len(exported.primitives)
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["primitive_count"] == 2

    def test_length_constraint(self):
        """Test length constraint on a line."""
        sketch = SketchDocument(name="LengthTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 0)
        ))
        sketch.add_constraint(Length(line_id, 75.0))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
import math
from morphe import sketch_from_json, sketch_to_json, Line
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
line = lines[0]
length = math.sqrt((line.end.x - line.start.x)**2 + (line.end.y - line.start.y)**2)

result = {{
    "success": True,
    "length": length
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert abs(result["length"] - 75.0) < 0.1, f"Expected length 75, got {result['length']}"

    def test_collinear_constraint(self):
        """Test collinear constraint between two lines."""
        sketch = SketchDocument(name="CollinearTest")
        l1 = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(50, 10)
        ))
        l2 = sketch.add_primitive(Line(
            start=Point2D(60, 15),
            end=Point2D(100, 25)
        ))
        sketch.add_constraint(Collinear(l1, l2))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
import math
from morphe import sketch_from_json, sketch_to_json, Line
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
l1, l2 = lines[0], lines[1]

# Check if lines are collinear by verifying all 4 points lie on same line
# Using cross product to check if points are collinear
def cross(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

p1 = (l1.start.x, l1.start.y)
p2 = (l1.end.x, l1.end.y)
p3 = (l2.start.x, l2.start.y)
p4 = (l2.end.x, l2.end.y)

# All cross products should be near zero if collinear
c1 = abs(cross(p1, p2, p3))
c2 = abs(cross(p1, p2, p4))

is_collinear = c1 < 1.0 and c2 < 1.0

result = {{
    "success": True,
    "is_collinear": is_collinear,
    "cross1": c1,
    "cross2": c2
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["is_collinear"], \
            f"Lines should be collinear, cross products: {result['cross1']}, {result['cross2']}"

    def test_midpoint_constraint(self):
        """Test midpoint constraint (point at midpoint of line)."""
        sketch = SketchDocument(name="MidpointTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 0)
        ))
        point_id = sketch.add_primitive(Point(
            position=Point2D(40, 10)  # Not at midpoint initially
        ))
        sketch.add_constraint(MidpointConstraint(
            PointRef(point_id, PointType.CENTER),
            line_id
        ))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json, Line, Point
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
points = [p for p in exported.primitives.values() if isinstance(p, Point)]

line = lines[0]
point = points[0]

# Calculate expected midpoint
mid_x = (line.start.x + line.end.x) / 2
mid_y = (line.start.y + line.end.y) / 2

# Check if point is at midpoint
at_midpoint = abs(point.position.x - mid_x) < 0.1 and abs(point.position.y - mid_y) < 0.1

result = {{
    "success": True,
    "at_midpoint": at_midpoint,
    "point_x": point.position.x,
    "point_y": point.position.y,
    "expected_mid_x": mid_x,
    "expected_mid_y": mid_y
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["at_midpoint"], \
            f"Point should be at midpoint ({result['expected_mid_x']}, {result['expected_mid_y']}), " \
            f"but was at ({result['point_x']}, {result['point_y']})"

    def test_constraint_export_length(self):
        """Test that length constraint produces correct line length."""
        sketch = SketchDocument(name="ConstraintExportLengthTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(10, 10),
            end=Point2D(50, 10)
        ))
        sketch.add_constraint(Length(line_id, 75.0))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
import math
from morphe import sketch_from_json, sketch_to_json, Line
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

line = list(exported.primitives.values())[0]
actual_length = math.sqrt(
    (line.end.x - line.start.x)**2 + (line.end.y - line.start.y)**2
)

result = {{
    "success": True,
    "actual_length": actual_length
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert abs(result["actual_length"] - 75.0) < 0.1, \
            f"Line length should be 75, got {result['actual_length']}"

    def test_multiple_constraints_horizontal_length(self):
        """Test horizontal + length constraints on same line."""
        sketch = SketchDocument(name="MultiConstraintHLTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(0, 30),
            end=Point2D(40, 35)
        ))
        sketch.add_constraint(Horizontal(line_id))
        sketch.add_constraint(Length(line_id, 60.0))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json, Line
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

line = list(exported.primitives.values())[0]
is_horizontal = abs(line.start.y - line.end.y) < 0.01
actual_length = abs(line.end.x - line.start.x)

result = {{
    "success": True,
    "is_horizontal": is_horizontal,
    "actual_length": actual_length
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["is_horizontal"], "Line should be horizontal"
        assert abs(result["actual_length"] - 60.0) < 0.1, \
            f"Line length should be 60, got {result['actual_length']}"

    def test_multiple_constraints_vertical_length(self):
        """Test vertical + length constraints on same line."""
        sketch = SketchDocument(name="MultiConstraintVLTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(25, 10),
            end=Point2D(30, 50)
        ))
        sketch.add_constraint(Vertical(line_id))
        sketch.add_constraint(Length(line_id, 80.0))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json, Line
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

line = list(exported.primitives.values())[0]
is_vertical = abs(line.start.x - line.end.x) < 0.01
actual_length = abs(line.end.y - line.start.y)

result = {{
    "success": True,
    "is_vertical": is_vertical,
    "actual_length": actual_length
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert result["is_vertical"], "Line should be vertical"
        assert abs(result["actual_length"] - 80.0) < 0.1, \
            f"Line length should be 80, got {result['actual_length']}"

    def test_length_precision(self):
        """Test dimensional constraint with high precision value."""
        sketch = SketchDocument(name="LengthPrecisionTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(50, 0)
        ))
        precise_length = 47.123456
        sketch.add_constraint(Length(line_id, precise_length))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from morphe import sketch_from_json, sketch_to_json, Line
from morphe.adapters.freecad import FreeCADAdapter

sketch = sketch_from_json({repr(input_json)})
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)
exported = adapter.export_sketch()

line = list(exported.primitives.values())[0]
actual_length = abs(line.end.x - line.start.x)

result = {{
    "success": True,
    "actual_length": actual_length
}}
print(json.dumps(result))
'''

        result = run_in_freecad(script)
        assert result["success"]
        assert abs(result["actual_length"] - precise_length) < 0.001, \
            f"Length should be {precise_length}, got {result['actual_length']}"


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v"])
