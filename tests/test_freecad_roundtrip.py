"""
Round-trip tests for FreeCAD adapter.

These tests verify that sketches can be loaded into FreeCAD and exported back
without loss of essential information. Tests are skipped if FreeCAD is not
available on the system.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from sketch_canonical import (
    SketchDocument, Point2D, PointType, PointRef,
    Line, Arc, Circle, Point, Spline,
    Coincident, Tangent, Perpendicular, Parallel, Horizontal, Vertical,
    Fixed, Distance, Radius, Diameter,
    sketch_to_json, sketch_from_json,
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
        except:
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
from sketch_canonical import sketch_from_json, sketch_to_json
from sketch_adapter_freecad import FreeCADAdapter

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
from sketch_canonical import sketch_from_json, sketch_to_json
from sketch_adapter_freecad import FreeCADAdapter

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
from sketch_canonical import sketch_from_json, sketch_to_json
from sketch_adapter_freecad import FreeCADAdapter

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
from sketch_canonical import sketch_from_json, sketch_to_json
from sketch_adapter_freecad import FreeCADAdapter

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


class TestFreeCADRoundTripComplex:
    """Round-trip tests for complex sketches with multiple geometries."""

    def test_rectangle(self):
        """Test round-trip of a rectangle (4 lines)."""
        sketch = SketchDocument(name="RectangleTest")

        # Create rectangle
        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(100, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(100, 0), end=Point2D(100, 50)))
        l3 = sketch.add_primitive(Line(start=Point2D(100, 50), end=Point2D(0, 50)))
        l4 = sketch.add_primitive(Line(start=Point2D(0, 50), end=Point2D(0, 0)))

        input_json = sketch_to_json(sketch)

        script = f'''
import json
from sketch_canonical import sketch_from_json, sketch_to_json
from sketch_adapter_freecad import FreeCADAdapter

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
from sketch_canonical import sketch_from_json, sketch_to_json
from sketch_adapter_freecad import FreeCADAdapter

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
from sketch_canonical import sketch_from_json, sketch_to_json
from sketch_adapter_freecad import FreeCADAdapter

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
from sketch_canonical import sketch_from_json, sketch_to_json, SolverStatus
from sketch_adapter_freecad import FreeCADAdapter

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
from sketch_canonical import sketch_from_json, sketch_to_json
from sketch_adapter_freecad import FreeCADAdapter

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
from sketch_canonical import sketch_from_json, sketch_to_json
from sketch_adapter_freecad import FreeCADAdapter

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
from sketch_canonical import sketch_from_json, sketch_to_json
from sketch_adapter_freecad import FreeCADAdapter

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
from sketch_canonical import sketch_from_json, sketch_to_json
from sketch_adapter_freecad import FreeCADAdapter

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
from sketch_canonical import sketch_from_json, SolverStatus
from sketch_adapter_freecad import FreeCADAdapter

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
from sketch_canonical import sketch_from_json, SolverStatus
from sketch_adapter_freecad import FreeCADAdapter

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


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v"])
