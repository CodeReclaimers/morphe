"""
Round-trip tests for Fusion 360 adapter.

This script is designed to be run from inside Fusion 360 as an add-in script.
It tests that sketches can be loaded into Fusion 360 and exported back
without loss of essential information.

Usage:
    1. Open Fusion 360
    2. Go to Utilities > Add-Ins > Scripts
    3. Click the green '+' to add a new script
    4. Navigate to this file and run it

The script will create test sketches, verify the round-trip behavior,
and display results in a message box and text command palette.
"""

import math
import sys
import traceback
from pathlib import Path
from typing import List, Tuple, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Add the project root to path for imports
# Adjust this path based on where you've installed the canonical_sketch package
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import adsk.core
import adsk.fusion

# Import canonical sketch modules
from sketch_canonical import (
    SketchDocument,
    Point2D,
    PointType,
    PointRef,
    Line,
    Arc,
    Circle,
    Point,
    Spline,
    SolverStatus,
)
from sketch_canonical.constraints import (
    Coincident,
    Tangent,
    Perpendicular,
    Parallel,
    Horizontal,
    Vertical,
    Fixed,
    Distance,
    Radius,
    Diameter,
    Equal,
    Concentric,
    Length,
    Angle,
)
from sketch_canonical.serialization import sketch_to_json, sketch_from_json

from sketch_adapter_fusion import FusionAdapter


class TestStatus(Enum):
    """Test result status."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    status: TestStatus
    message: str = ""
    duration: float = 0.0


class FusionTestRunner:
    """Test runner for Fusion 360 round-trip tests."""

    def __init__(self):
        self.app = adsk.core.Application.get()
        self.ui = self.app.userInterface
        self.results: List[TestResult] = []
        self._test_doc = None
        self._adapter = None

    def setup(self):
        """Set up test environment - create a new document."""
        # Create a new document for testing
        doc_type = adsk.core.DocumentTypes.FusionDesignDocumentType
        self._test_doc = self.app.documents.add(doc_type)
        self._adapter = FusionAdapter()

    def teardown(self):
        """Clean up test environment."""
        if self._test_doc:
            try:
                self._test_doc.close(False)  # Close without saving
            except:
                pass
        self._test_doc = None
        self._adapter = None

    def run_test(self, name: str, test_func: Callable) -> TestResult:
        """Run a single test and capture the result."""
        import time
        start_time = time.time()

        try:
            # Create fresh adapter for each test
            self._adapter = FusionAdapter()
            test_func()
            duration = time.time() - start_time
            return TestResult(name, TestStatus.PASSED, duration=duration)
        except AssertionError as e:
            duration = time.time() - start_time
            return TestResult(name, TestStatus.FAILED, str(e), duration)
        except Exception as e:
            duration = time.time() - start_time
            tb = traceback.format_exc()
            return TestResult(name, TestStatus.ERROR, f"{e}\n{tb}", duration)

    def run_all_tests(self) -> List[TestResult]:
        """Run all registered tests."""
        self.results = []

        # Get all test methods
        test_methods = [
            (name, getattr(self, name))
            for name in dir(self)
            if name.startswith("test_") and callable(getattr(self, name))
        ]

        self.setup()
        try:
            for name, method in test_methods:
                result = self.run_test(name, method)
                self.results.append(result)
                self._log(f"  {result.status.value}: {name}")
        finally:
            self.teardown()

        return self.results

    def _log(self, message: str):
        """Log a message to the text commands palette."""
        palette = self.ui.palettes.itemById("TextCommands")
        if palette:
            palette.writeText(message)

    def report_results(self):
        """Display test results."""
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in self.results if r.status == TestStatus.ERROR)
        total = len(self.results)

        summary = f"Test Results: {passed}/{total} passed"
        if failed:
            summary += f", {failed} failed"
        if errors:
            summary += f", {errors} errors"

        # Build detailed report
        details = [summary, "=" * 50]

        for r in self.results:
            status_icon = {
                TestStatus.PASSED: "[OK]",
                TestStatus.FAILED: "[FAIL]",
                TestStatus.ERROR: "[ERR]",
                TestStatus.SKIPPED: "[SKIP]",
            }[r.status]

            details.append(f"{status_icon} {r.name} ({r.duration:.2f}s)")
            if r.message:
                # Indent message lines
                for line in r.message.split("\n")[:5]:  # Limit to first 5 lines
                    details.append(f"      {line}")

        details.append("=" * 50)
        full_report = "\n".join(details)

        # Log to text commands
        self._log(full_report)

        # Show summary in message box
        if failed or errors:
            self.ui.messageBox(
                f"{summary}\n\nSee Text Commands palette for details.",
                "Test Results - Some Tests Failed"
            )
        else:
            self.ui.messageBox(
                f"{summary}\n\nAll tests passed!",
                "Test Results - Success"
            )

    # =========================================================================
    # Basic Geometry Tests
    # =========================================================================

    def test_single_line(self):
        """Test round-trip of a single line."""
        # Create source sketch
        sketch = SketchDocument(name="LineTest")
        sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 50)
        ))

        # Load into Fusion
        self._adapter.create_sketch(sketch.name)
        self._adapter.load_sketch(sketch)

        # Export back
        exported = self._adapter.export_sketch()

        # Verify
        assert len(exported.primitives) == 1, f"Expected 1 primitive, got {len(exported.primitives)}"

        line = list(exported.primitives.values())[0]
        assert isinstance(line, Line), f"Expected Line, got {type(line)}"
        assert abs(line.start.x - 0) < 0.01, f"Start X mismatch: {line.start.x}"
        assert abs(line.start.y - 0) < 0.01, f"Start Y mismatch: {line.start.y}"
        assert abs(line.end.x - 100) < 0.01, f"End X mismatch: {line.end.x}"
        assert abs(line.end.y - 50) < 0.01, f"End Y mismatch: {line.end.y}"

    def test_single_circle(self):
        """Test round-trip of a single circle."""
        sketch = SketchDocument(name="CircleTest")
        sketch.add_primitive(Circle(
            center=Point2D(50, 50),
            radius=25
        ))

        self._adapter.create_sketch(sketch.name)
        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 1
        circle = list(exported.primitives.values())[0]
        assert isinstance(circle, Circle)
        assert abs(circle.center.x - 50) < 0.01
        assert abs(circle.center.y - 50) < 0.01
        assert abs(circle.radius - 25) < 0.01

    def test_single_arc(self):
        """Test round-trip of a single arc."""
        sketch = SketchDocument(name="ArcTest")
        sketch.add_primitive(Arc(
            center=Point2D(0, 0),
            start=Point2D(50, 0),
            end=Point2D(0, 50),
            ccw=True
        ))

        self._adapter.create_sketch(sketch.name)
        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 1
        arc = list(exported.primitives.values())[0]
        assert isinstance(arc, Arc)

        # Verify center is preserved
        assert abs(arc.center.x - 0) < 0.01
        assert abs(arc.center.y - 0) < 0.01

        # Verify radius (both start and end should be at radius 50)
        start_radius = math.sqrt(arc.start.x**2 + arc.start.y**2)
        end_radius = math.sqrt(arc.end.x**2 + arc.end.y**2)
        assert abs(start_radius - 50) < 0.1, f"Start radius: {start_radius}"
        assert abs(end_radius - 50) < 0.1, f"End radius: {end_radius}"

    def test_single_point(self):
        """Test round-trip of a single point."""
        sketch = SketchDocument(name="PointTest")
        sketch.add_primitive(Point(position=Point2D(25, 75)))

        self._adapter.create_sketch(sketch.name)
        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 1
        point = list(exported.primitives.values())[0]
        assert isinstance(point, Point)
        assert abs(point.position.x - 25) < 0.01
        assert abs(point.position.y - 75) < 0.01

    # =========================================================================
    # Complex Geometry Tests
    # =========================================================================

    def test_rectangle(self):
        """Test round-trip of a rectangle (4 lines)."""
        sketch = SketchDocument(name="RectangleTest")

        sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(100, 0)))
        sketch.add_primitive(Line(start=Point2D(100, 0), end=Point2D(100, 50)))
        sketch.add_primitive(Line(start=Point2D(100, 50), end=Point2D(0, 50)))
        sketch.add_primitive(Line(start=Point2D(0, 50), end=Point2D(0, 0)))

        self._adapter.create_sketch(sketch.name)
        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 4
        assert all(isinstance(p, Line) for p in exported.primitives.values())

    def test_mixed_geometry(self):
        """Test round-trip of mixed geometry types."""
        sketch = SketchDocument(name="MixedTest")

        sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(50, 0)))
        sketch.add_primitive(Arc(
            center=Point2D(50, 25),
            start=Point2D(50, 0),
            end=Point2D(75, 25),
            ccw=True
        ))
        sketch.add_primitive(Circle(center=Point2D(100, 50), radius=20))
        sketch.add_primitive(Point(position=Point2D(0, 50)))

        self._adapter.create_sketch(sketch.name)
        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 4

        types = {type(p).__name__ for p in exported.primitives.values()}
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

        self._adapter.create_sketch(sketch.name)
        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        # Find line and circle
        line = next(p for p in exported.primitives.values() if isinstance(p, Line))
        circle = next(p for p in exported.primitives.values() if isinstance(p, Circle))

        assert line.construction is True, "Line should be construction"
        assert circle.construction is False, "Circle should not be construction"

    # =========================================================================
    # Constraint Tests
    # =========================================================================

    def test_horizontal_constraint(self):
        """Test horizontal constraint is applied."""
        sketch = SketchDocument(name="HorizontalTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(0, 10),  # Not horizontal initially
            end=Point2D(100, 20)
        ))
        sketch.add_constraint(Horizontal(line_id))

        self._adapter.create_sketch(sketch.name)
        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        is_horizontal = abs(line.start.y - line.end.y) < 0.01
        assert is_horizontal, f"Line not horizontal: start.y={line.start.y}, end.y={line.end.y}"

    def test_vertical_constraint(self):
        """Test vertical constraint is applied."""
        sketch = SketchDocument(name="VerticalTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(10, 0),  # Not vertical initially
            end=Point2D(20, 100)
        ))
        sketch.add_constraint(Vertical(line_id))

        self._adapter.create_sketch(sketch.name)
        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        is_vertical = abs(line.start.x - line.end.x) < 0.01
        assert is_vertical, f"Line not vertical: start.x={line.start.x}, end.x={line.end.x}"

    def test_radius_constraint(self):
        """Test radius constraint is applied."""
        sketch = SketchDocument(name="RadiusTest")
        circle_id = sketch.add_primitive(Circle(
            center=Point2D(50, 50),
            radius=30  # Initial radius
        ))
        sketch.add_constraint(Radius(circle_id, 50))  # Constrain to radius 50

        self._adapter.create_sketch(sketch.name)
        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        circle = list(exported.primitives.values())[0]
        assert abs(circle.radius - 50) < 0.01, f"Radius mismatch: {circle.radius}"

    def test_diameter_constraint(self):
        """Test diameter constraint is applied."""
        sketch = SketchDocument(name="DiameterTest")
        circle_id = sketch.add_primitive(Circle(
            center=Point2D(50, 50),
            radius=25  # Initial radius
        ))
        sketch.add_constraint(Diameter(circle_id, 80))  # Constrain to diameter 80 (radius 40)

        self._adapter.create_sketch(sketch.name)
        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        circle = list(exported.primitives.values())[0]
        assert abs(circle.radius - 40) < 0.01, f"Radius mismatch: {circle.radius} (expected 40)"

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

        self._adapter.create_sketch(sketch.name)
        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        prims = list(exported.primitives.values())
        l1_end = prims[0].end
        l2_start = prims[1].start

        distance = math.sqrt((l1_end.x - l2_start.x)**2 + (l1_end.y - l2_start.y)**2)
        assert distance < 0.01, f"Points not coincident: distance={distance}"

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

        self._adapter.create_sketch(sketch.name)
        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

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
        assert cross < 0.01, f"Lines not parallel: cross product = {cross}"

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

        self._adapter.create_sketch(sketch.name)
        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        prims = list(exported.primitives.values())
        line1, line2 = prims[0], prims[1]

        # Calculate direction vectors
        dir1 = (line1.end.x - line1.start.x, line1.end.y - line1.start.y)
        dir2 = (line2.end.x - line2.start.x, line2.end.y - line2.start.y)

        # Dot product should be ~0 for perpendicular lines
        dot = abs(dir1[0]*dir2[0] + dir1[1]*dir2[1])
        # Normalize by lengths
        len1 = math.sqrt(dir1[0]**2 + dir1[1]**2)
        len2 = math.sqrt(dir2[0]**2 + dir2[1]**2)
        dot_normalized = dot / (len1 * len2) if len1 * len2 > 0 else 0

        assert dot_normalized < 0.01, f"Lines not perpendicular: dot product = {dot_normalized}"

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

        self._adapter.create_sketch(sketch.name)
        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        prims = list(exported.primitives.values())
        line1, line2 = prims[0], prims[1]

        len1 = math.sqrt((line1.end.x - line1.start.x)**2 + (line1.end.y - line1.start.y)**2)
        len2 = math.sqrt((line2.end.x - line2.start.x)**2 + (line2.end.y - line2.start.y)**2)

        assert abs(len1 - len2) < 0.1, f"Lines not equal length: {len1} vs {len2}"

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

        self._adapter.create_sketch(sketch.name)
        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        prims = list(exported.primitives.values())
        circle1, circle2 = prims[0], prims[1]

        center_distance = math.sqrt(
            (circle1.center.x - circle2.center.x)**2 +
            (circle1.center.y - circle2.center.y)**2
        )
        assert center_distance < 0.01, f"Circles not concentric: distance = {center_distance}"

    def test_length_constraint(self):
        """Test length constraint on a line."""
        sketch = SketchDocument(name="LengthTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(80, 0)  # Initial length 80
        ))
        sketch.add_constraint(Length(line_id, 100))  # Constrain to length 100

        self._adapter.create_sketch(sketch.name)
        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        length = math.sqrt((line.end.x - line.start.x)**2 + (line.end.y - line.start.y)**2)
        assert abs(length - 100) < 0.1, f"Length mismatch: {length}"

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

        self._adapter.create_sketch(sketch.name)
        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        prims = list(exported.primitives.values())
        line1, line2 = prims[0], prims[1]

        # Calculate angle between lines
        dir1 = (line1.end.x - line1.start.x, line1.end.y - line1.start.y)
        dir2 = (line2.end.x - line2.start.x, line2.end.y - line2.start.y)

        len1 = math.sqrt(dir1[0]**2 + dir1[1]**2)
        len2 = math.sqrt(dir2[0]**2 + dir2[1]**2)

        if len1 > 0 and len2 > 0:
            dot = dir1[0]*dir2[0] + dir1[1]*dir2[1]
            cos_angle = dot / (len1 * len2)
            cos_angle = max(-1, min(1, cos_angle))  # Clamp for numerical stability
            angle_deg = math.degrees(math.acos(abs(cos_angle)))
            assert abs(angle_deg - 45) < 1, f"Angle mismatch: {angle_deg}"

    # =========================================================================
    # Spline Tests
    # =========================================================================

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

        self._adapter.create_sketch(sketch.name)
        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 1
        exported_spline = list(exported.primitives.values())[0]
        assert isinstance(exported_spline, Spline), f"Expected Spline, got {type(exported_spline)}"
        assert exported_spline.degree == 3
        assert len(exported_spline.poles) == 4

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

        self._adapter.create_sketch(sketch.name)
        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 1
        exported_spline = list(exported.primitives.values())[0]
        assert isinstance(exported_spline, Spline)
        assert exported_spline.degree == 2
        assert len(exported_spline.poles) == 3

    # =========================================================================
    # Multiple Constraints Tests
    # =========================================================================

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

        self._adapter.create_sketch(sketch.name)
        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 4

        # Verify it's a proper rectangle
        lines = list(exported.primitives.values())

        # Check horizontal lines are horizontal
        horizontal_lines = [l for l in lines if abs(l.start.y - l.end.y) < 0.01]
        vertical_lines = [l for l in lines if abs(l.start.x - l.end.x) < 0.01]

        assert len(horizontal_lines) == 2, "Should have 2 horizontal lines"
        assert len(vertical_lines) == 2, "Should have 2 vertical lines"


def run(context):
    """Main entry point for Fusion 360 script."""
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface

        # Show the text commands palette for output
        palette = ui.palettes.itemById("TextCommands")
        if palette:
            palette.isVisible = True

        ui.messageBox(
            "Starting Fusion 360 Round-Trip Tests.\n\n"
            "Results will appear in a message box and the Text Commands palette.",
            "Round-Trip Tests"
        )

        runner = FusionTestRunner()
        runner.run_all_tests()
        runner.report_results()

    except Exception as e:
        if ui:
            ui.messageBox(f"Test run failed:\n{traceback.format_exc()}")


# Allow running as script
if __name__ == "__main__":
    run(None)
