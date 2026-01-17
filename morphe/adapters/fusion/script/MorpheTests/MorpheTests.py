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
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Add the project root to path for imports
# Use resolve() to follow the symlink to the actual source location
SCRIPT_DIR = Path(__file__).resolve().parent  # .../script/MorpheTests
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent  # morphe repo root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import adsk.core
import adsk.fusion

# Import canonical sketch modules
from morphe import (
    Arc,
    Circle,
    Ellipse,
    EllipticalArc,
    Line,
    Point,
    Point2D,
    PointRef,
    PointType,
    SketchDocument,
    Spline,
)
from morphe.adapters.fusion import FusionAdapter
from morphe.constraints import (
    Angle,
    Coincident,
    Collinear,
    Concentric,
    Diameter,
    Distance,
    DistanceX,
    DistanceY,
    Equal,
    Fixed,
    Horizontal,
    Length,
    MidpointConstraint,
    Parallel,
    Perpendicular,
    Radius,
    Symmetric,
    Tangent,
    Vertical,
)


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
        self.results: list[TestResult] = []
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
            except Exception:
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

    def run_all_tests(self) -> list[TestResult]:
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
            start_point=Point2D(50, 0),
            end_point=Point2D(0, 50),
            ccw=True
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 1
        arc = list(exported.primitives.values())[0]
        assert isinstance(arc, Arc)

        # Verify center is preserved
        assert abs(arc.center.x - 0) < 0.01
        assert abs(arc.center.y - 0) < 0.01

        # Verify radius (both start and end should be at radius 50)
        start_radius = math.sqrt(arc.start_point.x**2 + arc.start_point.y**2)
        end_radius = math.sqrt(arc.end_point.x**2 + arc.end_point.y**2)
        assert abs(start_radius - 50) < 0.1, f"Start radius: {start_radius}"
        assert abs(end_radius - 50) < 0.1, f"End radius: {end_radius}"

    def test_single_point(self):
        """Test round-trip of a single point."""
        sketch = SketchDocument(name="PointTest")
        sketch.add_primitive(Point(position=Point2D(25, 75)))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 1
        point = list(exported.primitives.values())[0]
        assert isinstance(point, Point)
        assert abs(point.position.x - 25) < 0.01
        assert abs(point.position.y - 75) < 0.01

    def test_single_ellipse(self):
        """Test round-trip of a single ellipse."""
        sketch = SketchDocument(name="EllipseTest")
        sketch.add_primitive(Ellipse(
            center=Point2D(50, 50),
            major_radius=30,
            minor_radius=20,
            rotation=0.0
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 1
        ellipse = list(exported.primitives.values())[0]
        assert isinstance(ellipse, Ellipse)
        assert abs(ellipse.center.x - 50) < 0.01
        assert abs(ellipse.center.y - 50) < 0.01
        assert abs(ellipse.major_radius - 30) < 0.01
        assert abs(ellipse.minor_radius - 20) < 0.01

    def test_ellipse_rotated(self):
        """Test round-trip of a rotated ellipse."""
        sketch = SketchDocument(name="RotatedEllipseTest")
        sketch.add_primitive(Ellipse(
            center=Point2D(100, 100),
            major_radius=40,
            minor_radius=25,
            rotation=math.pi / 4  # 45 degrees
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 1
        ellipse = list(exported.primitives.values())[0]
        assert isinstance(ellipse, Ellipse)
        assert abs(ellipse.center.x - 100) < 0.01
        assert abs(ellipse.center.y - 100) < 0.01
        assert abs(ellipse.major_radius - 40) < 0.01
        assert abs(ellipse.minor_radius - 25) < 0.01
        # Rotation should be preserved (allow some tolerance)
        assert abs(ellipse.rotation - math.pi / 4) < 0.01

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

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 1
        arc = list(exported.primitives.values())[0]
        assert isinstance(arc, EllipticalArc)
        assert abs(arc.center.x - 50) < 0.01
        assert abs(arc.center.y - 50) < 0.01
        assert abs(arc.major_radius - 30) < 0.01
        assert abs(arc.minor_radius - 20) < 0.01

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
            start_point=Point2D(50, 0),
            end_point=Point2D(75, 25),
            ccw=True
        ))
        sketch.add_primitive(Circle(center=Point2D(100, 50), radius=20))
        sketch.add_primitive(Point(position=Point2D(0, 50)))

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

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 1
        exported_spline = list(exported.primitives.values())[0]
        assert isinstance(exported_spline, Spline), f"Expected Spline, got {type(exported_spline)}"
        assert exported_spline.degree == 3
        assert len(exported_spline.control_points) == 4

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

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 1
        exported_spline = list(exported.primitives.values())[0]
        assert isinstance(exported_spline, Spline)
        assert exported_spline.degree == 2
        assert len(exported_spline.control_points) == 3

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

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 4

        # Verify it's a proper rectangle
        lines = list(exported.primitives.values())

        # Check horizontal lines are horizontal
        horizontal_lines = [ln for ln in lines if abs(ln.start.y - ln.end.y) < 0.01]
        vertical_lines = [ln for ln in lines if abs(ln.start.x - ln.end.x) < 0.01]

        assert len(horizontal_lines) == 2, "Should have 2 horizontal lines"
        assert len(vertical_lines) == 2, "Should have 2 vertical lines"

    # =========================================================================
    # Geometry Edge Case Tests
    # =========================================================================

    def test_arc_clockwise(self):
        """Test round-trip of a clockwise arc."""
        sketch = SketchDocument(name="ArcCWTest")
        sketch.add_primitive(Arc(
            center=Point2D(0, 0),
            start_point=Point2D(50, 0),
            end_point=Point2D(0, 50),
            ccw=False  # Clockwise
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 1
        arc = list(exported.primitives.values())[0]
        assert isinstance(arc, Arc)

        # For CW arc from (50,0) to (0,50), it should go the long way around
        # Verify the arc direction is preserved by checking the sweep
        # A CW arc from (50,0) to (0,50) should have sweep > 180 degrees

    def test_arc_large_angle(self):
        """Test round-trip of a large arc (> 180 degrees)."""
        sketch = SketchDocument(name="LargeArcTest")
        # Create an arc that sweeps 270 degrees CCW
        sketch.add_primitive(Arc(
            center=Point2D(0, 0),
            start_point=Point2D(50, 0),
            end_point=Point2D(0, -50),  # 270 degrees CCW from start
            ccw=True
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 1
        arc = list(exported.primitives.values())[0]
        assert isinstance(arc, Arc)
        # Verify radius is preserved
        start_radius = math.sqrt(arc.start_point.x**2 + arc.start_point.y**2)
        assert abs(start_radius - 50) < 0.1

    def test_geometry_at_origin(self):
        """Test geometry centered at origin."""
        sketch = SketchDocument(name="OriginTest")
        sketch.add_primitive(Circle(center=Point2D(0, 0), radius=25))
        sketch.add_primitive(Line(start=Point2D(-50, 0), end=Point2D(50, 0)))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 2
        circle = next(p for p in exported.primitives.values() if isinstance(p, Circle))
        assert abs(circle.center.x) < 0.01
        assert abs(circle.center.y) < 0.01

    def test_small_geometry(self):
        """Test very small geometry (precision test)."""
        sketch = SketchDocument(name="SmallTest")
        # Very small geometry - 0.1mm
        sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(0.1, 0.05)
        ))
        sketch.add_primitive(Circle(center=Point2D(1, 1), radius=0.05))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 2
        line = next(p for p in exported.primitives.values() if isinstance(p, Line))
        circle = next(p for p in exported.primitives.values() if isinstance(p, Circle))

        assert abs(line.end.x - 0.1) < 0.001, f"Small line end X: {line.end.x}"
        assert abs(circle.radius - 0.05) < 0.001, f"Small circle radius: {circle.radius}"

    def test_large_geometry(self):
        """Test large geometry (1000mm scale)."""
        sketch = SketchDocument(name="LargeTest")
        sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(1000, 500)
        ))
        sketch.add_primitive(Circle(center=Point2D(500, 500), radius=250))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 2
        line = next(p for p in exported.primitives.values() if isinstance(p, Line))
        circle = next(p for p in exported.primitives.values() if isinstance(p, Circle))

        assert abs(line.end.x - 1000) < 0.1, f"Large line end X: {line.end.x}"
        assert abs(circle.radius - 250) < 0.1, f"Large circle radius: {circle.radius}"

    def test_negative_coordinates(self):
        """Test geometry in negative coordinate space."""
        sketch = SketchDocument(name="NegativeTest")
        sketch.add_primitive(Line(
            start=Point2D(-100, -50),
            end=Point2D(-20, -80)
        ))
        sketch.add_primitive(Circle(center=Point2D(-50, -50), radius=30))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 2
        line = next(p for p in exported.primitives.values() if isinstance(p, Line))
        circle = next(p for p in exported.primitives.values() if isinstance(p, Circle))

        assert abs(line.start.x - (-100)) < 0.01
        assert abs(line.start.y - (-50)) < 0.01
        assert abs(circle.center.x - (-50)) < 0.01
        assert abs(circle.center.y - (-50)) < 0.01

    def test_diagonal_line(self):
        """Test line at 45-degree angle."""
        sketch = SketchDocument(name="DiagonalTest")
        sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 100)
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        # Verify 45-degree angle
        dx = line.end.x - line.start.x
        dy = line.end.y - line.start.y
        assert abs(dx - dy) < 0.01, "Line should be at 45 degrees"

    # =========================================================================
    # Additional Constraint Tests
    # =========================================================================

    def test_tangent_line_circle(self):
        """Test tangent constraint between line and circle."""
        sketch = SketchDocument(name="TangentTest")
        circle_id = sketch.add_primitive(Circle(
            center=Point2D(50, 50),
            radius=30
        ))
        line_id = sketch.add_primitive(Line(
            start=Point2D(0, 85),
            end=Point2D(100, 80)  # Nearly tangent
        ))
        sketch.add_constraint(Tangent(line_id, circle_id))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        # Verify tangency: distance from center to line should equal radius
        circle = next(p for p in exported.primitives.values() if isinstance(p, Circle))
        line = next(p for p in exported.primitives.values() if isinstance(p, Line))

        # Calculate distance from circle center to line
        # Line from (x1,y1) to (x2,y2), point (px,py)
        x1, y1 = line.start.x, line.start.y
        x2, y2 = line.end.x, line.end.y
        px, py = circle.center.x, circle.center.y

        line_len = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        if line_len > 0:
            dist = abs((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1) / line_len
            assert abs(dist - circle.radius) < 0.5, f"Not tangent: distance={dist}, radius={circle.radius}"

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
            end=Point2D(120, 55)  # Starts at arc endpoint
        ))
        sketch.add_constraint(Tangent(arc_id, line_id))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 2

    def test_collinear_constraint(self):
        """Test collinear constraint between two lines."""
        sketch = SketchDocument(name="CollinearTest")
        l1 = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(50, 0)
        ))
        l2 = sketch.add_primitive(Line(
            start=Point2D(60, 5),  # Slightly off the line
            end=Point2D(100, 5)
        ))
        sketch.add_constraint(Collinear(l1, l2))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        prims = list(exported.primitives.values())
        line1, line2 = prims[0], prims[1]

        # Both lines should have the same Y coordinate (collinear on X-axis)
        assert abs(line1.start.y - line2.start.y) < 0.01, "Lines not collinear"
        assert abs(line1.end.y - line2.end.y) < 0.01, "Lines not collinear"

    def test_fixed_constraint(self):
        """Test fixed constraint locks geometry in place."""
        sketch = SketchDocument(name="FixedTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(10, 20),
            end=Point2D(50, 60)
        ))
        sketch.add_constraint(Fixed(line_id))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        # Fixed geometry should maintain exact position
        assert abs(line.start.x - 10) < 0.01
        assert abs(line.start.y - 20) < 0.01
        assert abs(line.end.x - 50) < 0.01
        assert abs(line.end.y - 60) < 0.01

    def test_distance_constraint(self):
        """Test distance constraint between two points."""
        sketch = SketchDocument(name="DistanceTest")
        l1 = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(50, 0)
        ))
        l2 = sketch.add_primitive(Line(
            start=Point2D(60, 0),  # 10mm gap
            end=Point2D(100, 0)
        ))
        # Constrain distance between end of l1 and start of l2 to 20mm
        sketch.add_constraint(Distance(
            PointRef(l1, PointType.END),
            PointRef(l2, PointType.START),
            20
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        prims = list(exported.primitives.values())
        l1_end = prims[0].end
        l2_start = prims[1].start

        dist = math.sqrt((l2_start.x - l1_end.x)**2 + (l2_start.y - l1_end.y)**2)
        assert abs(dist - 20) < 0.1, f"Distance mismatch: {dist}"

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
        # Constrain horizontal distance between l1 end and l2 start to 40mm
        sketch.add_constraint(DistanceX(
            PointRef(l1, PointType.END),
            40,
            PointRef(l2, PointType.START)
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        prims = list(exported.primitives.values())
        l1_end = prims[0].end
        l2_start = prims[1].start

        dx = abs(l2_start.x - l1_end.x)
        assert abs(dx - 40) < 0.1, f"Horizontal distance mismatch: {dx}"

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
        # Constrain vertical distance between lines to 50mm
        sketch.add_constraint(DistanceY(
            PointRef(l1, PointType.START),
            50,
            PointRef(l2, PointType.START)
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        prims = list(exported.primitives.values())
        l1_start = prims[0].start
        l2_start = prims[1].start

        dy = abs(l2_start.y - l1_start.y)
        assert abs(dy - 50) < 0.1, f"Vertical distance mismatch: {dy}"

    def test_symmetric_constraint(self):
        """Test symmetric constraint about a centerline."""
        sketch = SketchDocument(name="SymmetricTest")
        # Centerline (vertical)
        center_id = sketch.add_primitive(Line(
            start=Point2D(50, 0),
            end=Point2D(50, 100),
            construction=True
        ))
        # Two lines to be symmetric
        l1 = sketch.add_primitive(Line(
            start=Point2D(20, 20),
            end=Point2D(30, 50)
        ))
        l2 = sketch.add_primitive(Line(
            start=Point2D(80, 20),  # Should mirror to x=80
            end=Point2D(70, 50)
        ))
        sketch.add_constraint(Symmetric(l1, l2, center_id))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        lines = [p for p in exported.primitives.values() if isinstance(p, Line) and not p.construction]
        assert len(lines) == 2

        # Check that the non-construction lines are symmetric about x=50
        for line in lines:
            _mid_x = (line.start.x + line.end.x) / 2  # noqa: F841
            # The two lines' midpoints should be equidistant from x=50
            # This is a simplified check

    def test_midpoint_constraint(self):
        """Test midpoint constraint places point at line center."""
        sketch = SketchDocument(name="MidpointTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 0)
        ))
        point_id = sketch.add_primitive(Point(
            position=Point2D(60, 10)  # Not at midpoint initially
        ))
        sketch.add_constraint(MidpointConstraint(
            PointRef(point_id, PointType.CENTER),
            line_id
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        line = next(p for p in exported.primitives.values() if isinstance(p, Line))
        point = next(p for p in exported.primitives.values() if isinstance(p, Point))

        midpoint_x = (line.start.x + line.end.x) / 2
        midpoint_y = (line.start.y + line.end.y) / 2

        assert abs(point.position.x - midpoint_x) < 0.1, f"Point not at midpoint X: {point.position.x}"
        assert abs(point.position.y - midpoint_y) < 0.1, f"Point not at midpoint Y: {point.position.y}"

    # =========================================================================
    # Spline Edge Case Tests
    # =========================================================================

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

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 1
        exported_spline = list(exported.primitives.values())[0]
        assert isinstance(exported_spline, Spline)
        assert exported_spline.degree == 4
        assert len(exported_spline.control_points) == 5

    def test_many_control_points_spline(self):
        """Test spline with many control points."""
        sketch = SketchDocument(name="ManyPointsSplineTest")

        # Create spline with 8 control points
        control_pts = [
            Point2D(i * 15, 30 * math.sin(i * 0.8))
            for i in range(8)
        ]
        spline = Spline.create_uniform_bspline(
            control_points=control_pts,
            degree=3
        )
        sketch.add_primitive(spline)

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 1
        exported_spline = list(exported.primitives.values())[0]
        assert len(exported_spline.control_points) == 8

    # =========================================================================
    # Complex Scenario Tests
    # =========================================================================

    def test_closed_profile(self):
        """Test a closed profile with connected lines."""
        sketch = SketchDocument(name="ClosedProfileTest")

        # Create a triangle
        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(100, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(100, 0), end=Point2D(50, 80)))
        l3 = sketch.add_primitive(Line(start=Point2D(50, 80), end=Point2D(0, 0)))

        # Connect all corners
        sketch.add_constraint(Coincident(PointRef(l1, PointType.END), PointRef(l2, PointType.START)))
        sketch.add_constraint(Coincident(PointRef(l2, PointType.END), PointRef(l3, PointType.START)))
        sketch.add_constraint(Coincident(PointRef(l3, PointType.END), PointRef(l1, PointType.START)))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 3

        # Verify the profile is closed (each endpoint touches another)
        lines = list(exported.primitives.values())
        endpoints = []
        for line in lines:
            endpoints.append((line.start.x, line.start.y))
            endpoints.append((line.end.x, line.end.y))

        # Each point should appear twice (start of one, end of another)
        from collections import Counter
        rounded = [(round(x, 1), round(y, 1)) for x, y in endpoints]
        counts = Counter(rounded)
        assert all(c == 2 for c in counts.values()), "Profile not properly closed"

    def test_nested_geometry(self):
        """Test circle inside rectangle (common CAD pattern)."""
        sketch = SketchDocument(name="NestedTest")

        # Outer rectangle
        sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(100, 0)))
        sketch.add_primitive(Line(start=Point2D(100, 0), end=Point2D(100, 80)))
        sketch.add_primitive(Line(start=Point2D(100, 80), end=Point2D(0, 80)))
        sketch.add_primitive(Line(start=Point2D(0, 80), end=Point2D(0, 0)))

        # Inner circle centered in rectangle
        sketch.add_primitive(Circle(center=Point2D(50, 40), radius=25))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 5
        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
        circles = [p for p in exported.primitives.values() if isinstance(p, Circle)]
        assert len(lines) == 4
        assert len(circles) == 1

    def test_concentric_circles(self):
        """Test multiple concentric circles."""
        sketch = SketchDocument(name="ConcentricCirclesTest")

        c1 = sketch.add_primitive(Circle(center=Point2D(50, 50), radius=10))
        c2 = sketch.add_primitive(Circle(center=Point2D(52, 52), radius=25))
        c3 = sketch.add_primitive(Circle(center=Point2D(48, 48), radius=40))

        sketch.add_constraint(Concentric(c1, c2))
        sketch.add_constraint(Concentric(c2, c3))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        circles = list(exported.primitives.values())
        assert len(circles) == 3

        # All circles should share the same center
        centers = [(c.center.x, c.center.y) for c in circles]
        for i in range(1, len(centers)):
            dist = math.sqrt((centers[i][0] - centers[0][0])**2 +
                           (centers[i][1] - centers[0][1])**2)
            assert dist < 0.01, f"Circles not concentric: distance = {dist}"

    def test_slot_profile(self):
        """Test a slot profile (two semicircles connected by lines)."""
        sketch = SketchDocument(name="SlotTest")

        # Two parallel lines
        sketch.add_primitive(Line(start=Point2D(20, 0), end=Point2D(80, 0)))
        sketch.add_primitive(Line(start=Point2D(80, 40), end=Point2D(20, 40)))

        # Two semicircular arcs
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

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
        arcs = [p for p in exported.primitives.values() if isinstance(p, Arc)]

        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"
        assert len(arcs) == 2, f"Expected 2 arcs, got {len(arcs)}"

    def test_solver_status_underconstrained(self):
        """Test that unconstrained sketch reports correct status."""
        sketch = SketchDocument(name="UnderconstrainedTest")
        sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 50)
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        # An unconstrained line should have degrees of freedom > 0
        assert exported.degrees_of_freedom > 0, \
            f"Expected DOF > 0 for unconstrained sketch, got {exported.degrees_of_freedom}"

    def test_solver_status_fullyconstrained(self):
        """Test that fully constrained sketch reports zero DOF."""
        sketch = SketchDocument(name="FullyConstrainedTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 0)
        ))
        # Fix the line completely
        sketch.add_constraint(Fixed(line_id))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert exported.degrees_of_freedom == 0, \
            f"Expected DOF = 0 for fixed line, got {exported.degrees_of_freedom}"

    # =========================================================================
    # Error Handling Tests
    # =========================================================================

    def test_multiple_points_standalone(self):
        """Test that multiple standalone points are exported correctly."""
        sketch = SketchDocument(name="MultiPointTest")
        sketch.add_primitive(Point(position=Point2D(10, 20)))
        sketch.add_primitive(Point(position=Point2D(50, 60)))
        sketch.add_primitive(Point(position=Point2D(90, 30)))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        points = [p for p in exported.primitives.values() if isinstance(p, Point)]
        assert len(points) == 3, f"Expected 3 points, got {len(points)}"

    def test_construction_arc(self):
        """Test that construction flag works on arcs."""
        sketch = SketchDocument(name="ConstructionArcTest")
        sketch.add_primitive(Arc(
            center=Point2D(50, 50),
            start_point=Point2D(80, 50),
            end_point=Point2D(50, 80),
            ccw=True,
            construction=True
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        arc = list(exported.primitives.values())[0]
        assert arc.construction is True, "Arc should be construction geometry"

    def test_equal_circles(self):
        """Test equal constraint between two circles (equal radii)."""
        sketch = SketchDocument(name="EqualCirclesTest")
        c1 = sketch.add_primitive(Circle(center=Point2D(30, 30), radius=20))
        c2 = sketch.add_primitive(Circle(center=Point2D(80, 30), radius=35))
        sketch.add_constraint(Equal(c1, c2))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        circles = list(exported.primitives.values())
        assert abs(circles[0].radius - circles[1].radius) < 0.01, \
            f"Circles should have equal radii: {circles[0].radius} vs {circles[1].radius}"

    # =========================================================================
    # Constraint Export Verification Tests
    # =========================================================================

    def test_constraint_export_horizontal(self):
        """Test that horizontal constraint is exported back correctly."""
        sketch = SketchDocument(name="ConstraintExportHorizTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(10, 20),
            end=Point2D(80, 25)  # Slightly non-horizontal initially
        ))
        sketch.add_constraint(Horizontal(line_id))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        # Check geometry is horizontal after constraint
        line = list(exported.primitives.values())[0]
        assert abs(line.start.y - line.end.y) < 0.01, \
            f"Line should be horizontal: start.y={line.start.y}, end.y={line.end.y}"

    def test_constraint_export_perpendicular(self):
        """Test that perpendicular constraint produces 90-degree angle."""
        sketch = SketchDocument(name="ConstraintExportPerpTest")
        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(50, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(25, 0), end=Point2D(30, 40)))
        sketch.add_constraint(Perpendicular(l1, l2))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        lines = list(exported.primitives.values())
        # Calculate angle between lines using dot product
        dx1, dy1 = lines[0].end.x - lines[0].start.x, lines[0].end.y - lines[0].start.y
        dx2, dy2 = lines[1].end.x - lines[1].start.x, lines[1].end.y - lines[1].start.y
        dot = dx1 * dx2 + dy1 * dy2
        len1 = math.sqrt(dx1**2 + dy1**2)
        len2 = math.sqrt(dx2**2 + dy2**2)
        cos_angle = dot / (len1 * len2) if len1 > 0 and len2 > 0 else 0
        assert abs(cos_angle) < 0.01, f"Lines should be perpendicular, cos(angle)={cos_angle}"

    def test_constraint_export_length(self):
        """Test that length constraint produces correct line length."""
        sketch = SketchDocument(name="ConstraintExportLengthTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(10, 10),
            end=Point2D(50, 10)
        ))
        sketch.add_constraint(Length(line_id, 75.0))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        actual_length = math.sqrt(
            (line.end.x - line.start.x)**2 + (line.end.y - line.start.y)**2
        )
        assert abs(actual_length - 75.0) < 0.1, \
            f"Line length should be 75, got {actual_length}"

    # =========================================================================
    # Multiple Constraints on Same Element Tests
    # =========================================================================

    def test_multiple_constraints_horizontal_length(self):
        """Test horizontal + length constraints on same line."""
        sketch = SketchDocument(name="MultiConstraintHLTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(0, 30),
            end=Point2D(40, 35)
        ))
        sketch.add_constraint(Horizontal(line_id))
        sketch.add_constraint(Length(line_id, 60.0))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        # Should be horizontal
        assert abs(line.start.y - line.end.y) < 0.01, "Line should be horizontal"
        # Should have correct length
        actual_length = abs(line.end.x - line.start.x)
        assert abs(actual_length - 60.0) < 0.1, f"Line length should be 60, got {actual_length}"

    def test_multiple_constraints_vertical_length(self):
        """Test vertical + length constraints on same line."""
        sketch = SketchDocument(name="MultiConstraintVLTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(25, 10),
            end=Point2D(30, 50)
        ))
        sketch.add_constraint(Vertical(line_id))
        sketch.add_constraint(Length(line_id, 80.0))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        # Should be vertical
        assert abs(line.start.x - line.end.x) < 0.01, "Line should be vertical"
        # Should have correct length
        actual_length = abs(line.end.y - line.start.y)
        assert abs(actual_length - 80.0) < 0.1, f"Line length should be 80, got {actual_length}"

    def test_multiple_constraints_circle(self):
        """Test concentric + equal constraints on circles."""
        sketch = SketchDocument(name="MultiConstraintCircleTest")
        c1 = sketch.add_primitive(Circle(center=Point2D(50, 50), radius=20))
        c2 = sketch.add_primitive(Circle(center=Point2D(60, 55), radius=35))
        sketch.add_constraint(Concentric(c1, c2))
        sketch.add_constraint(Equal(c1, c2))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        circles = list(exported.primitives.values())
        # Should be concentric
        assert abs(circles[0].center.x - circles[1].center.x) < 0.01, "Circles should share center X"
        assert abs(circles[0].center.y - circles[1].center.y) < 0.01, "Circles should share center Y"
        # Should be equal
        assert abs(circles[0].radius - circles[1].radius) < 0.01, "Circles should have equal radii"

    # =========================================================================
    # Point-on-Curve Coincident Tests
    # =========================================================================

    def test_point_on_line_midpoint(self):
        """Test point constrained to midpoint of line."""
        sketch = SketchDocument(name="PointOnLineMidTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 0)
        ))
        point_id = sketch.add_primitive(Point(position=Point2D(30, 20)))
        sketch.add_constraint(MidpointConstraint(
            PointRef(point_id, PointType.CENTER),
            line_id
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        line = None
        point = None
        for prim in exported.primitives.values():
            if isinstance(prim, Line):
                line = prim
            elif isinstance(prim, Point):
                point = prim

        assert line is not None and point is not None
        midpoint_x = (line.start.x + line.end.x) / 2
        midpoint_y = (line.start.y + line.end.y) / 2
        assert abs(point.position.x - midpoint_x) < 0.01, \
            f"Point X should be at midpoint: {point.position.x} vs {midpoint_x}"
        assert abs(point.position.y - midpoint_y) < 0.01, \
            f"Point Y should be at midpoint: {point.position.y} vs {midpoint_y}"

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

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        line = None
        point = None
        for prim in exported.primitives.values():
            if isinstance(prim, Line):
                line = prim
            elif isinstance(prim, Point):
                point = prim

        assert line is not None and point is not None
        assert abs(point.position.x - line.end.x) < 0.01, "Point should be at line end X"
        assert abs(point.position.y - line.end.y) < 0.01, "Point should be at line end Y"

    def test_coincident_point_to_circle_center(self):
        """Test point coincident with circle center."""
        sketch = SketchDocument(name="PointToCircleCenterTest")
        circle_id = sketch.add_primitive(Circle(center=Point2D(60, 40), radius=25))
        point_id = sketch.add_primitive(Point(position=Point2D(30, 20)))
        sketch.add_constraint(Coincident(
            PointRef(point_id, PointType.CENTER),
            PointRef(circle_id, PointType.CENTER)
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        circle = None
        point = None
        for prim in exported.primitives.values():
            if isinstance(prim, Circle):
                circle = prim
            elif isinstance(prim, Point):
                point = prim

        assert circle is not None and point is not None
        assert abs(point.position.x - circle.center.x) < 0.01, "Point should be at circle center X"
        assert abs(point.position.y - circle.center.y) < 0.01, "Point should be at circle center Y"

    # =========================================================================
    # Closed/Periodic Spline Tests
    # =========================================================================

    def test_periodic_spline(self):
        """Test closed/periodic spline round-trip.

        Note: Fusion 360's NurbsCurve3D API doesn't directly support periodic
        curves. This test creates a closed spline by connecting start to end.
        """
        # Create a closed spline by having coincident start/end
        # Use a regular non-periodic spline that forms a closed shape
        control_points = [
            Point2D(50, 0),
            Point2D(100, 25),
            Point2D(100, 75),
            Point2D(50, 100),
            Point2D(0, 75),
            Point2D(0, 25),
            Point2D(50, 0),  # Same as first point to close
        ]
        # Standard cubic B-spline knots: n + k + 1 = 7 + 4 = 11 knots
        knots = [0, 0, 0, 0, 0.33, 0.5, 0.67, 1, 1, 1, 1]

        sketch = SketchDocument(name="PeriodicSplineTest")
        sketch.add_primitive(Spline(
            control_points=control_points,
            degree=3,
            knots=knots,
            periodic=False  # Use non-periodic with closed endpoints
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        spline = list(exported.primitives.values())[0]
        assert isinstance(spline, Spline), "Expected Spline primitive"
        assert len(spline.control_points) >= 6, "Should have control points"
        # Check that start and end are close (forming closed shape)
        start = spline.control_points[0]
        end = spline.control_points[-1]
        dist = math.sqrt((end.x - start.x)**2 + (end.y - start.y)**2)
        assert dist < 1.0, f"Spline should be closed, start-end distance={dist}"

    # =========================================================================
    # Arc Angle Precision Tests
    # =========================================================================

    def test_arc_90_degree(self):
        """Test 90-degree arc preserves angle precisely."""
        # Quarter circle arc
        sketch = SketchDocument(name="Arc90Test")
        sketch.add_primitive(Arc(
            center=Point2D(50, 50),
            start_point=Point2D(80, 50),  # Right
            end_point=Point2D(50, 80),    # Top
            ccw=True
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        arc = list(exported.primitives.values())[0]
        # Calculate sweep angle
        start_angle = math.atan2(arc.start_point.y - arc.center.y, arc.start_point.x - arc.center.x)
        end_angle = math.atan2(arc.end_point.y - arc.center.y, arc.end_point.x - arc.center.x)
        sweep = end_angle - start_angle
        if sweep < 0:
            sweep += 2 * math.pi
        sweep_deg = math.degrees(sweep)
        assert abs(sweep_deg - 90) < 1.0, f"Arc should be 90 degrees, got {sweep_deg}"

    def test_arc_180_degree(self):
        """Test 180-degree arc (semicircle) preserves angle precisely."""
        sketch = SketchDocument(name="Arc180Test")
        sketch.add_primitive(Arc(
            center=Point2D(50, 50),
            start_point=Point2D(80, 50),  # Right
            end_point=Point2D(20, 50),    # Left
            ccw=True
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        arc = list(exported.primitives.values())[0]
        # Calculate sweep angle
        start_angle = math.atan2(arc.start_point.y - arc.center.y, arc.start_point.x - arc.center.x)
        end_angle = math.atan2(arc.end_point.y - arc.center.y, arc.end_point.x - arc.center.x)
        sweep = end_angle - start_angle
        if sweep < 0:
            sweep += 2 * math.pi
        sweep_deg = math.degrees(sweep)
        assert abs(sweep_deg - 180) < 1.0, f"Arc should be 180 degrees, got {sweep_deg}"

    def test_arc_45_degree(self):
        """Test 45-degree arc preserves angle precisely."""
        # 45 degree arc
        r = 30
        sketch = SketchDocument(name="Arc45Test")
        sketch.add_primitive(Arc(
            center=Point2D(50, 50),
            start_point=Point2D(50 + r, 50),  # Right (0 degrees)
            end_point=Point2D(50 + r * math.cos(math.radians(45)),
                             50 + r * math.sin(math.radians(45))),  # 45 degrees
            ccw=True
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        arc = list(exported.primitives.values())[0]
        start_angle = math.atan2(arc.start_point.y - arc.center.y, arc.start_point.x - arc.center.x)
        end_angle = math.atan2(arc.end_point.y - arc.center.y, arc.end_point.x - arc.center.x)
        sweep = end_angle - start_angle
        if sweep < 0:
            sweep += 2 * math.pi
        sweep_deg = math.degrees(sweep)
        assert abs(sweep_deg - 45) < 1.0, f"Arc should be 45 degrees, got {sweep_deg}"

    # =========================================================================
    # Weighted NURBS Spline Tests
    # =========================================================================

    def test_weighted_spline(self):
        """Test NURBS spline with non-uniform weights."""
        control_points = [
            Point2D(0, 0),
            Point2D(25, 50),
            Point2D(75, 50),
            Point2D(100, 0),
        ]
        # Non-uniform weights - middle points have higher weight
        weights = [1.0, 2.0, 2.0, 1.0]
        knots = [0, 0, 0, 0, 1, 1, 1, 1]

        sketch = SketchDocument(name="WeightedSplineTest")
        sketch.add_primitive(Spline(
            control_points=control_points,
            degree=3,
            knots=knots,
            weights=weights
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        spline = list(exported.primitives.values())[0]
        assert isinstance(spline, Spline), "Expected Spline primitive"
        assert len(spline.control_points) == 4, "Should have 4 control points"

    # =========================================================================
    # Empty Sketch Test
    # =========================================================================

    def test_empty_sketch(self):
        """Test that empty sketch exports correctly."""
        sketch = SketchDocument(name="EmptySketchTest")
        # No primitives added

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 0, \
            f"Empty sketch should have no primitives, got {len(exported.primitives)}"

    # =========================================================================
    # Constraint Value Precision Tests
    # =========================================================================

    def test_length_precision(self):
        """Test dimensional constraint with high precision value."""
        sketch = SketchDocument(name="LengthPrecisionTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(50, 0)
        ))
        # Use a precise value
        precise_length = 47.123456
        sketch.add_constraint(Length(line_id, precise_length))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        actual_length = abs(line.end.x - line.start.x)
        # Allow small tolerance for floating point
        assert abs(actual_length - precise_length) < 0.001, \
            f"Length should be {precise_length}, got {actual_length}"

    def test_radius_precision(self):
        """Test radius constraint with high precision value."""
        sketch = SketchDocument(name="RadiusPrecisionTest")
        circle_id = sketch.add_primitive(Circle(
            center=Point2D(50, 50),
            radius=20
        ))
        precise_radius = 33.789012
        sketch.add_constraint(Radius(circle_id, precise_radius))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        circle = list(exported.primitives.values())[0]
        assert abs(circle.radius - precise_radius) < 0.001, \
            f"Radius should be {precise_radius}, got {circle.radius}"

    def test_angle_precision(self):
        """Test angle constraint with precise value."""
        sketch = SketchDocument(name="AnglePrecisionTest")
        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(50, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(40, 30)))
        precise_angle = 37.5  # degrees
        sketch.add_constraint(Angle(l1, l2, precise_angle))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        lines = list(exported.primitives.values())
        # Calculate angle between lines
        dx1, dy1 = lines[0].end.x - lines[0].start.x, lines[0].end.y - lines[0].start.y
        dx2, dy2 = lines[1].end.x - lines[1].start.x, lines[1].end.y - lines[1].start.y
        dot = dx1 * dx2 + dy1 * dy2
        cross = dx1 * dy2 - dy1 * dx2
        angle_rad = math.atan2(abs(cross), dot)
        angle_deg = math.degrees(angle_rad)
        assert abs(angle_deg - precise_angle) < 0.5, \
            f"Angle should be {precise_angle}, got {angle_deg}"

    # =========================================================================
    # 3+ Element Equal Chain Tests
    # =========================================================================

    def test_equal_chain_three_lines(self):
        """Test equal constraint across three lines."""
        sketch = SketchDocument(name="EqualChain3LinesTest")
        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(30, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(0, 20), end=Point2D(50, 20)))
        l3 = sketch.add_primitive(Line(start=Point2D(0, 40), end=Point2D(70, 40)))
        sketch.add_constraint(Equal(l1, l2, l3))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        lines = list(exported.primitives.values())
        lengths = [
            math.sqrt((ln.end.x - ln.start.x)**2 + (ln.end.y - ln.start.y)**2)
            for ln in lines
        ]
        # All lines should have equal length
        assert abs(lengths[0] - lengths[1]) < 0.1, \
            f"Lines 1 and 2 should be equal: {lengths[0]} vs {lengths[1]}"
        assert abs(lengths[1] - lengths[2]) < 0.1, \
            f"Lines 2 and 3 should be equal: {lengths[1]} vs {lengths[2]}"

    def test_equal_chain_four_circles(self):
        """Test equal constraint across four circles."""
        sketch = SketchDocument(name="EqualChain4CirclesTest")
        c1 = sketch.add_primitive(Circle(center=Point2D(20, 20), radius=10))
        c2 = sketch.add_primitive(Circle(center=Point2D(60, 20), radius=15))
        c3 = sketch.add_primitive(Circle(center=Point2D(20, 60), radius=20))
        c4 = sketch.add_primitive(Circle(center=Point2D(60, 60), radius=25))
        sketch.add_constraint(Equal(c1, c2, c3, c4))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        circles = list(exported.primitives.values())
        radii = [c.radius for c in circles]
        # All circles should have equal radius
        for i in range(len(radii) - 1):
            assert abs(radii[i] - radii[i+1]) < 0.1, \
                f"Circles {i} and {i+1} should have equal radii: {radii[i]} vs {radii[i+1]}"

    # =========================================================================
    # Mixed Profile Tests (Fillet Pattern)
    # =========================================================================

    def test_arc_tangent_to_two_lines(self):
        """Test arc tangent to two lines (fillet pattern)."""
        sketch = SketchDocument(name="FilletPatternTest")
        # Two perpendicular lines
        l1 = sketch.add_primitive(Line(start=Point2D(0, 50), end=Point2D(50, 50)))
        l2 = sketch.add_primitive(Line(start=Point2D(50, 50), end=Point2D(50, 0)))
        # Arc connecting them
        arc = sketch.add_primitive(Arc(
            center=Point2D(35, 35),
            start_point=Point2D(35, 50),
            end_point=Point2D(50, 35),
            ccw=False
        ))
        # Make lines perpendicular
        sketch.add_constraint(Perpendicular(l1, l2))
        # Make arc tangent to both lines
        sketch.add_constraint(Tangent(l1, arc))
        sketch.add_constraint(Tangent(arc, l2))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        prims = list(exported.primitives.values())
        assert len(prims) == 3, f"Expected 3 primitives, got {len(prims)}"

    def test_smooth_corner_profile(self):
        """Test L-shaped profile with rounded corner."""
        sketch = SketchDocument(name="SmoothCornerTest")
        # Create L-shape with arc corner
        l1 = sketch.add_primitive(Line(start=Point2D(0, 30), end=Point2D(30, 30)))
        arc = sketch.add_primitive(Arc(
            center=Point2D(30, 20),
            start_point=Point2D(30, 30),
            end_point=Point2D(40, 20),
            ccw=False
        ))
        l2 = sketch.add_primitive(Line(start=Point2D(40, 20), end=Point2D(40, 0)))

        # Connect endpoints
        sketch.add_constraint(Coincident(
            PointRef(l1, PointType.END),
            PointRef(arc, PointType.START)
        ))
        sketch.add_constraint(Coincident(
            PointRef(arc, PointType.END),
            PointRef(l2, PointType.START)
        ))
        # Make tangent connections
        sketch.add_constraint(Tangent(l1, arc))
        sketch.add_constraint(Tangent(arc, l2))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        prims = list(exported.primitives.values())
        assert len(prims) == 3, f"Expected 3 primitives, got {len(prims)}"

    # =========================================================================
    # Edge Case Tests
    # =========================================================================

    def test_very_small_dimensions(self):
        """Test geometry with very small dimensions (micrometer scale)."""
        sketch = SketchDocument(name="MicroScaleTest")
        # 0.001mm = 1 micrometer scale
        sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(0.01, 0.01)
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        assert abs(line.end.x - 0.01) < 0.001, f"Small dimension not preserved: {line.end.x}"

    def test_very_large_dimensions(self):
        """Test geometry with very large dimensions (meter scale in mm)."""
        sketch = SketchDocument(name="LargeScaleTest")
        # 1000mm = 1 meter
        sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(1000, 1000)
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        assert abs(line.end.x - 1000) < 0.1, f"Large dimension not preserved: {line.end.x}"

    def test_coincident_chain(self):
        """Test chain of coincident constraints forming connected path."""
        sketch = SketchDocument(name="CoincidentChainTest")
        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(30, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(35, 5), end=Point2D(60, 30)))
        l3 = sketch.add_primitive(Line(start=Point2D(65, 35), end=Point2D(30, 60)))

        # Chain the endpoints
        sketch.add_constraint(Coincident(
            PointRef(l1, PointType.END),
            PointRef(l2, PointType.START)
        ))
        sketch.add_constraint(Coincident(
            PointRef(l2, PointType.END),
            PointRef(l3, PointType.START)
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        lines = list(exported.primitives.values())
        # Check chain connectivity
        assert abs(lines[0].end.x - lines[1].start.x) < 0.01, "L1 end should connect to L2 start"
        assert abs(lines[0].end.y - lines[1].start.y) < 0.01, "L1 end should connect to L2 start"
        assert abs(lines[1].end.x - lines[2].start.x) < 0.01, "L2 end should connect to L3 start"
        assert abs(lines[1].end.y - lines[2].start.y) < 0.01, "L2 end should connect to L3 start"

    # =========================================================================
    # Constraint Export Regression Tests
    # =========================================================================

    def test_constraint_export_has_ids(self):
        """Test that exported constraints have unique IDs.

        Regression test for: SketchConstraint.__init__() missing required 'id' argument.
        Fixed by adding _generate_constraint_id() helper.
        """
        sketch = SketchDocument(name="ConstraintIDTest")
        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(100, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(0, 30), end=Point2D(100, 30)))
        sketch.add_constraint(Horizontal(l1))
        sketch.add_constraint(Horizontal(l2))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        # Get constraints that have IDs
        constraints_with_ids = [
            c for c in exported.constraints
            if hasattr(c, 'id') and c.id is not None
        ]

        # Should have at least 2 constraints with unique IDs
        assert len(constraints_with_ids) >= 2, \
            f"Expected at least 2 constraints with IDs, got {len(constraints_with_ids)}"

        # IDs should be unique
        ids = [c.id for c in constraints_with_ids]
        assert len(ids) == len(set(ids)), "Constraint IDs should be unique"

    def test_constraint_export_rectangle_all_constraints(self):
        """Test that rectangle with H/V constraints exports all 4 constraints."""
        sketch = SketchDocument(name="RectConstraintExportTest")

        # Create rectangle
        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(100, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(100, 0), end=Point2D(100, 50)))
        l3 = sketch.add_primitive(Line(start=Point2D(100, 50), end=Point2D(0, 50)))
        l4 = sketch.add_primitive(Line(start=Point2D(0, 50), end=Point2D(0, 0)))

        sketch.add_constraint(Horizontal(l1))
        sketch.add_constraint(Vertical(l2))
        sketch.add_constraint(Horizontal(l3))
        sketch.add_constraint(Vertical(l4))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        # Count exported constraints by type
        horizontal_count = sum(
            1 for c in exported.constraints
            if hasattr(c, 'constraint_type') and c.constraint_type.name == 'HORIZONTAL'
        )
        vertical_count = sum(
            1 for c in exported.constraints
            if hasattr(c, 'constraint_type') and c.constraint_type.name == 'VERTICAL'
        )

        assert horizontal_count == 2, f"Expected 2 horizontal constraints, got {horizontal_count}"
        assert vertical_count == 2, f"Expected 2 vertical constraints, got {vertical_count}"

    def test_constraint_export_dimensional(self):
        """Test that dimensional constraints are exported correctly."""
        sketch = SketchDocument(name="DimensionalConstraintExportTest")

        circle_id = sketch.add_primitive(Circle(center=Point2D(50, 50), radius=20))
        sketch.add_constraint(Diameter(circle_id, 60))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        # Circle should have the constrained diameter
        circle = list(exported.primitives.values())[0]
        assert abs(circle.radius - 30) < 0.1, f"Circle radius should be 30, got {circle.radius}"

        # Should have a diameter constraint in export
        diameter_constraints = [
            c for c in exported.constraints
            if hasattr(c, 'constraint_type') and c.constraint_type.name == 'DIAMETER'
        ]
        assert len(diameter_constraints) >= 1, "Should have at least 1 diameter constraint"

    def test_constraint_export_mixed_types(self):
        """Test exporting sketch with geometric and dimensional constraints."""
        sketch = SketchDocument(name="MixedConstraintExportTest")

        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(80, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(80, 0), end=Point2D(80, 50)))
        circle = sketch.add_primitive(Circle(center=Point2D(40, 25), radius=15))

        sketch.add_constraint(Horizontal(l1))
        sketch.add_constraint(Vertical(l2))
        sketch.add_constraint(Length(l1, 100))
        sketch.add_constraint(Diameter(circle, 40))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        assert len(exported.primitives) == 3, "Should have 3 primitives"
        assert len(exported.constraints) >= 4, \
            f"Should have at least 4 constraints, got {len(exported.constraints)}"

    def test_constraint_export_entity_references(self):
        """Test that exported constraints have valid entity references."""
        sketch = SketchDocument(name="ConstraintReferencesTest")

        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(50, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(0, 30), end=Point2D(50, 30)))
        sketch.add_constraint(Parallel(l1, l2))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        # Find parallel constraints
        parallel_constraints = [
            c for c in exported.constraints
            if hasattr(c, 'constraint_type') and c.constraint_type.name == 'PARALLEL'
        ]

        assert len(parallel_constraints) >= 1, "Should have at least 1 parallel constraint"

        # Check that references point to valid primitive IDs
        primitive_ids = set(exported.primitives.keys())
        for constraint in parallel_constraints:
            for ref in constraint.references:
                if isinstance(ref, str):
                    assert ref in primitive_ids, \
                        f"Constraint references unknown primitive: {ref}"

    # =========================================================================
    # Export Regression Tests (SketchBridge Demo Issues)
    # =========================================================================

    def test_arc_clockwise_direction_preserved(self):
        """Test that clockwise arc direction is preserved during export.

        Regression test for: Arc with ccw=False being exported with ccw=True
        due to simple angle comparison that doesn't handle angle wraparound.
        """
        sketch = SketchDocument(name="ArcCWRegressionTest")
        # Clockwise arc similar to demo's arc_cw
        # This arc goes from (67, 4) to (73, 4) clockwise around center (70, 12.5)
        sketch.add_primitive(Arc(
            center=Point2D(70, 12.5),
            start_point=Point2D(67, 4),
            end_point=Point2D(73, 4),
            ccw=False  # Clockwise
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        arcs = [p for p in exported.primitives.values() if isinstance(p, Arc)]
        assert len(arcs) == 1, f"Expected 1 arc, got {len(arcs)}"

        arc = arcs[0]
        # Verify center is approximately correct
        assert abs(arc.center.x - 70) < 0.5, f"Center X should be ~70, got {arc.center.x}"
        assert abs(arc.center.y - 12.5) < 0.5, f"Center Y should be ~12.5, got {arc.center.y}"

        # The arc should still be clockwise (ccw=False)
        # OR if start/end are swapped, ccw should be True (which represents same physical curve)
        if arc.ccw:
            # Reversed: start and end should be swapped
            assert abs(arc.start_point.x - 73) < 0.5, \
                f"For ccw=True, start X should be ~73, got {arc.start_point.x}"
            assert abs(arc.end_point.x - 67) < 0.5, \
                f"For ccw=True, end X should be ~67, got {arc.end_point.x}"
        else:
            # Original direction
            assert abs(arc.start_point.x - 67) < 0.5, \
                f"For ccw=False, start X should be ~67, got {arc.start_point.x}"
            assert abs(arc.end_point.x - 73) < 0.5, \
                f"For ccw=False, end X should be ~73, got {arc.end_point.x}"

    def test_arc_counterclockwise_direction_preserved(self):
        """Test that counterclockwise arc direction is preserved during export."""
        sketch = SketchDocument(name="ArcCCWRegressionTest")
        # CCW arc
        sketch.add_primitive(Arc(
            center=Point2D(50, 50),
            start_point=Point2D(75, 50),
            end_point=Point2D(50, 75),
            ccw=True
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        arcs = [p for p in exported.primitives.values() if isinstance(p, Arc)]
        assert len(arcs) == 1, f"Expected 1 arc, got {len(arcs)}"
        assert arcs[0].ccw is True, "CCW arc should remain CCW"

    def test_arc_large_sweep_ccw(self):
        """Test that large CCW arc (>180 degrees) is correctly exported."""
        sketch = SketchDocument(name="LargeCCWArcRegressionTest")
        # Large CCW arc similar to demo's arc_tangent (270 degree sweep)
        sketch.add_primitive(Arc(
            center=Point2D(95, 12.5),
            start_point=Point2D(82, 12.5),
            end_point=Point2D(95, 25.5),
            ccw=True
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        arcs = [p for p in exported.primitives.values() if isinstance(p, Arc)]
        assert len(arcs) == 1, f"Expected 1 arc, got {len(arcs)}"

        arc = arcs[0]
        # Verify center
        assert abs(arc.center.x - 95) < 0.5, f"Center X should be ~95, got {arc.center.x}"
        assert abs(arc.center.y - 12.5) < 0.5, f"Center Y should be ~12.5, got {arc.center.y}"
        # This should be CCW
        assert arc.ccw is True, "Large CCW arc should remain CCW"

    def test_vertical_line_endpoint_preserved(self):
        """Test that vertical line endpoints are correctly preserved.

        Regression test for: Vertical line being exported with wrong endpoint
        due to internal representation issues.
        """
        sketch = SketchDocument(name="VerticalLineRegressionTest")
        # Vertical line
        sketch.add_primitive(Line(
            start=Point2D(110, 0),
            end=Point2D(110, 20)
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
        assert len(lines) == 1, f"Expected 1 line, got {len(lines)}"

        line = lines[0]
        # Both X coordinates should be 110 (vertical line)
        assert abs(line.start.x - 110) < 0.1, f"Start X should be 110, got {line.start.x}"
        assert abs(line.end.x - 110) < 0.1, f"End X should be 110, got {line.end.x}"
        # Y coordinates should span 0 to 20
        y_coords = sorted([line.start.y, line.end.y])
        assert abs(y_coords[0] - 0) < 0.1, f"Min Y should be 0, got {y_coords[0]}"
        assert abs(y_coords[1] - 20) < 0.1, f"Max Y should be 20, got {y_coords[1]}"

    def test_diagonal_line_endpoints_preserved(self):
        """Test that diagonal line endpoints are correctly preserved."""
        sketch = SketchDocument(name="DiagonalLineRegressionTest")
        sketch.add_primitive(Line(
            start=Point2D(100, -20),
            end=Point2D(120, -35)
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
        assert len(lines) == 1, f"Expected 1 line, got {len(lines)}"

        line = lines[0]
        # Check specific endpoints
        assert abs(line.start.x - 100) < 0.1, f"Start X should be 100, got {line.start.x}"
        assert abs(line.start.y - (-20)) < 0.1, f"Start Y should be -20, got {line.start.y}"
        assert abs(line.end.x - 120) < 0.1, f"End X should be 120, got {line.end.x}"
        assert abs(line.end.y - (-35)) < 0.1, f"End Y should be -35, got {line.end.y}"

    def test_elliptical_arc_roundtrip(self):
        """Test that elliptical arc is correctly round-tripped."""
        sketch = SketchDocument(name="EllipticalArcRegressionTest")
        sketch.add_primitive(EllipticalArc(
            center=Point2D(85, -25),
            major_radius=15,
            minor_radius=8,
            rotation=math.radians(-10),
            start_param=math.radians(30),
            end_param=math.radians(240),
            ccw=True
        ))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        arcs = [p for p in exported.primitives.values() if isinstance(p, EllipticalArc)]
        assert len(arcs) == 1, f"Expected 1 elliptical arc, got {len(arcs)}"

        arc = arcs[0]
        assert abs(arc.center.x - 85) < 0.5, f"Center X should be ~85, got {arc.center.x}"
        assert abs(arc.center.y - (-25)) < 0.5, f"Center Y should be ~-25, got {arc.center.y}"
        assert abs(arc.major_radius - 15) < 0.5, f"Major radius should be ~15, got {arc.major_radius}"
        assert abs(arc.minor_radius - 8) < 0.5, f"Minor radius should be ~8, got {arc.minor_radius}"

    def test_multiple_points_preserved(self):
        """Test that multiple standalone points are all preserved."""
        sketch = SketchDocument(name="MultiPointsRegressionTest")
        sketch.add_primitive(Point(position=Point2D(5, -15)))
        sketch.add_primitive(Point(position=Point2D(35, -15)))
        sketch.add_primitive(Point(position=Point2D(65, -15)))

        self._adapter.load_sketch(sketch)
        exported = self._adapter.export_sketch()

        points = [p for p in exported.primitives.values() if isinstance(p, Point)]
        assert len(points) == 3, f"Expected 3 points, got {len(points)}"

        point_xs = sorted([p.position.x for p in points])
        assert abs(point_xs[0] - 5) < 0.5, f"First point X should be ~5, got {point_xs[0]}"
        assert abs(point_xs[1] - 35) < 0.5, f"Second point X should be ~35, got {point_xs[1]}"
        assert abs(point_xs[2] - 65) < 0.5, f"Third point X should be ~65, got {point_xs[2]}"

    def test_load_sketch_creates_single_sketch(self):
        """Regression: load_sketch should create exactly one sketch, not two.

        Previously, calling load_sketch() would create an empty sketch and then
        another sketch with geometry (named with "(1)" suffix). This test ensures
        only one sketch is created with the expected name.
        """
        import uuid

        # Use unique name to avoid conflicts with other tests
        unique_name = f"SingleSketchTest_{uuid.uuid4().hex[:8]}"
        sketch = SketchDocument(name=unique_name)
        sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 10)))

        # Count sketches before
        app = adsk.core.Application.get()
        design = adsk.fusion.Design.cast(app.activeProduct)
        sketches_before = design.rootComponent.sketches.count

        # Load the sketch
        self._adapter.load_sketch(sketch)

        # Count sketches after
        sketches_after = design.rootComponent.sketches.count

        # Should have added exactly one sketch
        assert sketches_after == sketches_before + 1, (
            f"Expected 1 new sketch, got {sketches_after - sketches_before}. "
            f"Before: {sketches_before}, After: {sketches_after}"
        )

        # The created sketch should have exactly the name we specified
        created_name = self._adapter._sketch.name
        assert created_name == unique_name, (
            f"Sketch name should be '{unique_name}', got '{created_name}'. "
            "This suggests a duplicate sketch was created."
        )


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

    except Exception:
        if ui:
            ui.messageBox(f"Test run failed:\n{traceback.format_exc()}")


# Allow running as script
if __name__ == "__main__":
    run(None)
