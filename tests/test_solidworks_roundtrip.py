"""
Round-trip tests for SolidWorks adapter.

These tests verify that sketches can be loaded into SolidWorks and exported back
without loss of essential information. Tests are skipped if SolidWorks is not
available on the system (requires Windows with SolidWorks installed).
"""

import math

import pytest

from sketch_canonical import (
    Angle,
    Arc,
    Circle,
    Coincident,
    Collinear,
    Concentric,
    Diameter,
    Distance,
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
    SolverStatus,
    Spline,
    Tangent,
    Vertical,
)

# Try to import the SolidWorks adapter
try:
    from sketch_adapter_solidworks import SOLIDWORKS_AVAILABLE, SolidWorksAdapter
except ImportError:
    SOLIDWORKS_AVAILABLE = False
    SolidWorksAdapter = None  # type: ignore[misc,assignment]

# Skip all tests in this module if SolidWorks is not available
pytestmark = pytest.mark.skipif(
    not SOLIDWORKS_AVAILABLE,
    reason="SolidWorks is not installed or not accessible (requires Windows)"
)


@pytest.fixture
def adapter():
    """Create a fresh SolidWorksAdapter for each test."""
    if not SOLIDWORKS_AVAILABLE:
        pytest.skip("SolidWorks not available")
    adapter = SolidWorksAdapter()
    yield adapter
    # Cleanup: close the document without saving
    try:
        if adapter._document is not None:
            adapter._document.Close(False)  # False = don't save
    except Exception:
        pass


class TestSolidWorksRoundTripBasic:
    """Basic round-trip tests for simple geometries."""

    def test_single_line(self, adapter):
        """Test round-trip of a single line."""
        sketch = SketchDocument(name="LineTest")
        sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 50)
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        assert len(exported.primitives) == 1
        line = list(exported.primitives.values())[0]
        assert isinstance(line, Line)
        assert abs(line.start.x - 0) < 1e-6
        assert abs(line.start.y - 0) < 1e-6
        assert abs(line.end.x - 100) < 1e-6
        assert abs(line.end.y - 50) < 1e-6

    def test_single_circle(self, adapter):
        """Test round-trip of a single circle."""
        sketch = SketchDocument(name="CircleTest")
        sketch.add_primitive(Circle(
            center=Point2D(50, 50),
            radius=25
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        assert len(exported.primitives) == 1
        circle = list(exported.primitives.values())[0]
        assert isinstance(circle, Circle)
        assert abs(circle.center.x - 50) < 1e-6
        assert abs(circle.center.y - 50) < 1e-6
        assert abs(circle.radius - 25) < 1e-6

    def test_single_arc(self, adapter):
        """Test round-trip of a single arc."""
        sketch = SketchDocument(name="ArcTest")
        sketch.add_primitive(Arc(
            center=Point2D(0, 0),
            start_point=Point2D(50, 0),
            end_point=Point2D(0, 50),
            ccw=True
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        assert len(exported.primitives) == 1
        arc = list(exported.primitives.values())[0]
        assert isinstance(arc, Arc)
        assert abs(arc.center.x - 0) < 1e-6
        assert abs(arc.center.y - 0) < 1e-6
        # Radius should be 50
        radius = math.sqrt(arc.start_point.x**2 + arc.start_point.y**2)
        assert abs(radius - 50) < 1e-6

    def test_single_point(self, adapter):
        """Test round-trip of a single point."""
        sketch = SketchDocument(name="PointTest")
        sketch.add_primitive(Point(position=Point2D(75, 25)))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        assert len(exported.primitives) == 1
        point = list(exported.primitives.values())[0]
        assert isinstance(point, Point)
        assert abs(point.position.x - 75) < 1e-6
        assert abs(point.position.y - 25) < 1e-6


class TestSolidWorksRoundTripComplex:
    """Round-trip tests for more complex geometries."""

    def test_rectangle(self, adapter):
        """Test round-trip of a rectangle (4 lines)."""
        sketch = SketchDocument(name="RectangleTest")
        sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(100, 0)))
        sketch.add_primitive(Line(start=Point2D(100, 0), end=Point2D(100, 50)))
        sketch.add_primitive(Line(start=Point2D(100, 50), end=Point2D(0, 50)))
        sketch.add_primitive(Line(start=Point2D(0, 50), end=Point2D(0, 0)))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        assert len(exported.primitives) == 4
        assert all(isinstance(p, Line) for p in exported.primitives.values())

    def test_mixed_geometry(self, adapter):
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

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        assert len(exported.primitives) == 4
        types = [type(p).__name__ for p in exported.primitives.values()]
        assert "Line" in types
        assert "Arc" in types
        assert "Circle" in types
        assert "Point" in types

    def test_construction_geometry(self, adapter):
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

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        prims = list(exported.primitives.values())
        line = next(p for p in prims if isinstance(p, Line))
        circle = next(p for p in prims if isinstance(p, Circle))

        assert line.construction is True
        assert circle.construction is False


class TestSolidWorksRoundTripConstraints:
    """Round-trip tests for constraints."""

    def test_horizontal_constraint(self, adapter):
        """Test horizontal constraint is applied."""
        sketch = SketchDocument(name="HorizontalTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(0, 10),
            end=Point2D(100, 20)
        ))
        sketch.add_constraint(Horizontal(line_id))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        is_horizontal = abs(line.start.y - line.end.y) < 1e-6
        assert is_horizontal, f"Line not horizontal: start_y={line.start.y}, end_y={line.end.y}"

    def test_vertical_constraint(self, adapter):
        """Test vertical constraint is applied."""
        sketch = SketchDocument(name="VerticalTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(10, 0),
            end=Point2D(20, 100)
        ))
        sketch.add_constraint(Vertical(line_id))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        is_vertical = abs(line.start.x - line.end.x) < 1e-6
        assert is_vertical, f"Line not vertical: start_x={line.start.x}, end_x={line.end.x}"

    def test_radius_constraint(self, adapter):
        """Test radius constraint is applied."""
        sketch = SketchDocument(name="RadiusTest")
        circle_id = sketch.add_primitive(Circle(
            center=Point2D(50, 50),
            radius=20
        ))
        sketch.add_constraint(Radius(circle_id, value=35))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        circle = list(exported.primitives.values())[0]
        assert abs(circle.radius - 35) < 1e-6

    def test_coincident_constraint(self, adapter):
        """Test coincident constraint between two lines."""
        sketch = SketchDocument(name="CoincidentTest")
        line1_id = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(50, 0)
        ))
        line2_id = sketch.add_primitive(Line(
            start=Point2D(55, 5),
            end=Point2D(100, 50)
        ))
        sketch.add_constraint(Coincident(
            PointRef(line1_id, PointType.END),
            PointRef(line2_id, PointType.START)
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        prims = list(exported.primitives.values())
        line1 = next(p for p in prims if abs(p.start.x) < 1)
        line2 = next(p for p in prims if p != line1)

        # The end of line1 should coincide with the start of line2
        dist = math.sqrt(
            (line1.end.x - line2.start.x)**2 +
            (line1.end.y - line2.start.y)**2
        )
        assert dist < 1e-6, f"Points not coincident, distance: {dist}"


class TestSolidWorksRoundTripSpline:
    """Round-trip tests for splines."""

    def test_simple_bspline(self, adapter):
        """Test round-trip of a simple B-spline."""
        sketch = SketchDocument(name="SplineTest")
        sketch.add_primitive(Spline(
            control_points=[
                Point2D(0, 0),
                Point2D(25, 50),
                Point2D(75, 50),
                Point2D(100, 0)
            ],
            degree=3
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        assert len(exported.primitives) == 1
        spline = list(exported.primitives.values())[0]
        assert isinstance(spline, Spline)
        assert len(spline.control_points) >= 4


class TestSolidWorksSolverStatus:
    """Tests for solver status reporting."""

    def test_fully_constrained_with_fixed(self, adapter):
        """Test that a fixed point reports as fully constrained."""
        sketch = SketchDocument(name="FixedTest")
        point_id = sketch.add_primitive(Point(position=Point2D(50, 50)))
        sketch.add_constraint(Fixed(point_id))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        status, dof = adapter.get_solver_status()

        # A single fixed point should be fully constrained
        assert status == SolverStatus.FULLY_CONSTRAINED or dof == 0

    def test_solver_returns_status(self, adapter):
        """Test that solver status is returned."""
        sketch = SketchDocument(name="StatusTest")
        sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 0)
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        status, dof = adapter.get_solver_status()

        assert status in [
            SolverStatus.FULLY_CONSTRAINED,
            SolverStatus.UNDER_CONSTRAINED,
            SolverStatus.OVER_CONSTRAINED,
            SolverStatus.INCONSISTENT,
            SolverStatus.DIRTY
        ]


class TestSolidWorksRoundTripConstraintsExtended:
    """Extended constraint tests."""

    def test_parallel_constraint(self, adapter):
        """Test parallel constraint between two lines."""
        sketch = SketchDocument(name="ParallelTest")
        line1_id = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 0)
        ))
        line2_id = sketch.add_primitive(Line(
            start=Point2D(0, 50),
            end=Point2D(100, 60)
        ))
        sketch.add_constraint(Parallel(line1_id, line2_id))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        prims = list(exported.primitives.values())
        line1 = prims[0]
        line2 = prims[1]

        # Calculate direction vectors
        dx1 = line1.end.x - line1.start.x
        dy1 = line1.end.y - line1.start.y
        dx2 = line2.end.x - line2.start.x
        dy2 = line2.end.y - line2.start.y

        # Cross product should be near zero for parallel lines
        cross = abs(dx1 * dy2 - dy1 * dx2)
        len1 = math.sqrt(dx1**2 + dy1**2)
        len2 = math.sqrt(dx2**2 + dy2**2)
        normalized_cross = cross / (len1 * len2) if len1 > 0 and len2 > 0 else 0

        assert normalized_cross < 1e-6, f"Lines not parallel, cross product: {normalized_cross}"

    def test_perpendicular_constraint(self, adapter):
        """Test perpendicular constraint between two lines."""
        sketch = SketchDocument(name="PerpendicularTest")
        line1_id = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 0)
        ))
        line2_id = sketch.add_primitive(Line(
            start=Point2D(50, 0),
            end=Point2D(50, 50)
        ))
        sketch.add_constraint(Perpendicular(line1_id, line2_id))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        prims = list(exported.primitives.values())
        line1 = prims[0]
        line2 = prims[1]

        # Calculate direction vectors
        dx1 = line1.end.x - line1.start.x
        dy1 = line1.end.y - line1.start.y
        dx2 = line2.end.x - line2.start.x
        dy2 = line2.end.y - line2.start.y

        # Dot product should be near zero for perpendicular lines
        dot = abs(dx1 * dx2 + dy1 * dy2)
        len1 = math.sqrt(dx1**2 + dy1**2)
        len2 = math.sqrt(dx2**2 + dy2**2)
        normalized_dot = dot / (len1 * len2) if len1 > 0 and len2 > 0 else 0

        assert normalized_dot < 1e-6, f"Lines not perpendicular, dot product: {normalized_dot}"

    def test_equal_constraint(self, adapter):
        """Test equal length constraint between two lines."""
        sketch = SketchDocument(name="EqualTest")
        line1_id = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(50, 0)
        ))
        line2_id = sketch.add_primitive(Line(
            start=Point2D(0, 50),
            end=Point2D(80, 50)
        ))
        sketch.add_constraint(Equal(line1_id, line2_id))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        prims = list(exported.primitives.values())
        line1 = prims[0]
        line2 = prims[1]

        len1 = math.sqrt(
            (line1.end.x - line1.start.x)**2 +
            (line1.end.y - line1.start.y)**2
        )
        len2 = math.sqrt(
            (line2.end.x - line2.start.x)**2 +
            (line2.end.y - line2.start.y)**2
        )

        assert abs(len1 - len2) < 1e-6, f"Lines not equal length: {len1} vs {len2}"

    def test_concentric_constraint(self, adapter):
        """Test concentric constraint between two circles."""
        sketch = SketchDocument(name="ConcentricTest")
        circle1_id = sketch.add_primitive(Circle(
            center=Point2D(50, 50),
            radius=30
        ))
        circle2_id = sketch.add_primitive(Circle(
            center=Point2D(55, 55),
            radius=20
        ))
        sketch.add_constraint(Concentric(circle1_id, circle2_id))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        prims = list(exported.primitives.values())
        circle1 = prims[0]
        circle2 = prims[1]

        dist = math.sqrt(
            (circle1.center.x - circle2.center.x)**2 +
            (circle1.center.y - circle2.center.y)**2
        )
        assert dist < 1e-6, f"Circles not concentric, distance: {dist}"

    def test_diameter_constraint(self, adapter):
        """Test diameter constraint on a circle."""
        sketch = SketchDocument(name="DiameterTest")
        circle_id = sketch.add_primitive(Circle(
            center=Point2D(50, 50),
            radius=20
        ))
        sketch.add_constraint(Diameter(circle_id, value=60))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        circle = list(exported.primitives.values())[0]
        diameter = circle.radius * 2
        assert abs(diameter - 60) < 1e-6, f"Diameter mismatch: {diameter}"

    def test_angle_constraint(self, adapter):
        """Test angle constraint between two lines."""
        sketch = SketchDocument(name="AngleTest")
        line1_id = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 0)
        ))
        line2_id = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 100)
        ))
        sketch.add_constraint(Angle(line1_id, line2_id, value=45))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        prims = list(exported.primitives.values())
        line1 = prims[0]
        line2 = prims[1]

        # Calculate angles
        angle1 = math.atan2(
            line1.end.y - line1.start.y,
            line1.end.x - line1.start.x
        )
        angle2 = math.atan2(
            line2.end.y - line2.start.y,
            line2.end.x - line2.start.x
        )
        angle_diff = abs(math.degrees(angle2 - angle1))
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        assert abs(angle_diff - 45) < 1, f"Angle mismatch: {angle_diff}"


class TestSolidWorksRoundTripGeometryEdgeCases:
    """Tests for geometry edge cases."""

    def test_diagonal_line(self, adapter):
        """Test a diagonal line at 45 degrees."""
        sketch = SketchDocument(name="DiagonalTest")
        sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 100)
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        assert abs(line.start.x - 0) < 1e-6
        assert abs(line.start.y - 0) < 1e-6
        assert abs(line.end.x - 100) < 1e-6
        assert abs(line.end.y - 100) < 1e-6

    def test_negative_coordinates(self, adapter):
        """Test geometry with negative coordinates."""
        sketch = SketchDocument(name="NegativeTest")
        sketch.add_primitive(Line(
            start=Point2D(-50, -25),
            end=Point2D(50, 25)
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        assert abs(line.start.x - (-50)) < 1e-6
        assert abs(line.start.y - (-25)) < 1e-6
        assert abs(line.end.x - 50) < 1e-6
        assert abs(line.end.y - 25) < 1e-6

    def test_geometry_at_origin(self, adapter):
        """Test geometry centered at origin."""
        sketch = SketchDocument(name="OriginTest")
        sketch.add_primitive(Circle(
            center=Point2D(0, 0),
            radius=50
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        circle = list(exported.primitives.values())[0]
        assert abs(circle.center.x) < 1e-6
        assert abs(circle.center.y) < 1e-6
        assert abs(circle.radius - 50) < 1e-6

    def test_small_geometry(self, adapter):
        """Test very small geometry (1mm scale)."""
        sketch = SketchDocument(name="SmallTest")
        sketch.add_primitive(Circle(
            center=Point2D(0.5, 0.5),
            radius=0.25
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        circle = list(exported.primitives.values())[0]
        assert abs(circle.center.x - 0.5) < 1e-6
        assert abs(circle.center.y - 0.5) < 1e-6
        assert abs(circle.radius - 0.25) < 1e-6

    def test_large_geometry(self, adapter):
        """Test large geometry (1000mm scale)."""
        sketch = SketchDocument(name="LargeTest")
        sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(1000, 500)
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        assert abs(line.end.x - 1000) < 1e-3
        assert abs(line.end.y - 500) < 1e-3

    def test_empty_sketch(self, adapter):
        """Test exporting an empty sketch."""
        sketch = SketchDocument(name="EmptyTest")

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        assert len(exported.primitives) == 0


class TestSolidWorksRoundTripConstraintsAdvanced:
    """Advanced constraint tests."""

    def test_tangent_line_circle(self, adapter):
        """Test tangent constraint between line and circle."""
        sketch = SketchDocument(name="TangentTest")
        circle_id = sketch.add_primitive(Circle(
            center=Point2D(50, 50),
            radius=30
        ))
        line_id = sketch.add_primitive(Line(
            start=Point2D(0, 80),
            end=Point2D(100, 80)
        ))
        sketch.add_constraint(Tangent(line_id, circle_id))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        prims = list(exported.primitives.values())
        circle = next(p for p in prims if isinstance(p, Circle))
        line = next(p for p in prims if isinstance(p, Line))

        # Distance from circle center to line should equal radius
        dx = line.end.x - line.start.x
        dy = line.end.y - line.start.y
        line_len = math.sqrt(dx**2 + dy**2)
        if line_len > 0:
            dist = abs(
                (line.end.y - line.start.y) * circle.center.x -
                (line.end.x - line.start.x) * circle.center.y +
                line.end.x * line.start.y - line.end.y * line.start.x
            ) / line_len
            assert abs(dist - circle.radius) < 1, f"Not tangent: distance={dist}, radius={circle.radius}"

    def test_fixed_constraint(self, adapter):
        """Test fixed constraint on a point."""
        sketch = SketchDocument(name="FixedPointTest")
        point_id = sketch.add_primitive(Point(position=Point2D(75, 25)))
        sketch.add_constraint(Fixed(point_id))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        point = list(exported.primitives.values())[0]
        assert abs(point.position.x - 75) < 1e-6
        assert abs(point.position.y - 25) < 1e-6

    def test_distance_constraint(self, adapter):
        """Test distance constraint between two points."""
        sketch = SketchDocument(name="DistanceTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(50, 0)
        ))
        sketch.add_constraint(Distance(
            PointRef(line_id, PointType.START),
            PointRef(line_id, PointType.END),
            value=75
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        length = math.sqrt(
            (line.end.x - line.start.x)**2 +
            (line.end.y - line.start.y)**2
        )
        assert abs(length - 75) < 1e-6, f"Distance mismatch: {length}"

    def test_length_constraint(self, adapter):
        """Test length constraint on a line."""
        sketch = SketchDocument(name="LengthTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(50, 0)
        ))
        sketch.add_constraint(Length(line_id, value=75))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        length = math.sqrt(
            (line.end.x - line.start.x)**2 +
            (line.end.y - line.start.y)**2
        )
        assert abs(length - 75) < 1e-6, f"Length mismatch: {length}"

    def test_collinear_constraint(self, adapter):
        """Test collinear constraint between two lines."""
        sketch = SketchDocument(name="CollinearTest")
        line1_id = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(50, 0)
        ))
        line2_id = sketch.add_primitive(Line(
            start=Point2D(60, 5),
            end=Point2D(100, 5)
        ))
        sketch.add_constraint(Collinear(line1_id, line2_id))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        prims = list(exported.primitives.values())
        line1 = prims[0]
        line2 = prims[1]

        # All 4 points should be collinear
        dx1 = line1.end.x - line1.start.x
        dy1 = line1.end.y - line1.start.y

        dx2 = line2.start.x - line1.start.x
        dy2 = line2.start.y - line1.start.y

        cross = abs(dx1 * dy2 - dy1 * dx2)
        len1 = math.sqrt(dx1**2 + dy1**2)
        if len1 > 0:
            normalized = cross / len1
            assert normalized < 1, f"Lines not collinear: {normalized}"

    def test_midpoint_constraint(self, adapter):
        """Test midpoint constraint."""
        sketch = SketchDocument(name="MidpointTest")
        line_id = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 0)
        ))
        point_id = sketch.add_primitive(Point(
            position=Point2D(40, 10)
        ))
        sketch.add_constraint(MidpointConstraint(
            PointRef(point_id, PointType.CENTER),
            line_id
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        prims = list(exported.primitives.values())
        point = next(p for p in prims if isinstance(p, Point))
        line = next(p for p in prims if isinstance(p, Line))

        midpoint_x = (line.start.x + line.end.x) / 2
        midpoint_y = (line.start.y + line.end.y) / 2

        dist = math.sqrt(
            (point.position.x - midpoint_x)**2 +
            (point.position.y - midpoint_y)**2
        )
        assert dist < 1, f"Point not at midpoint: distance={dist}"


class TestSolidWorksRoundTripComplexScenarios:
    """Complex scenario tests."""

    def test_closed_profile(self, adapter):
        """Test a closed triangular profile."""
        sketch = SketchDocument(name="TriangleTest")
        l1_id = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(100, 0)))
        l2_id = sketch.add_primitive(Line(start=Point2D(100, 0), end=Point2D(50, 86.6)))
        l3_id = sketch.add_primitive(Line(start=Point2D(50, 86.6), end=Point2D(0, 0)))

        sketch.add_constraint(Coincident(
            PointRef(l1_id, PointType.END),
            PointRef(l2_id, PointType.START)
        ))
        sketch.add_constraint(Coincident(
            PointRef(l2_id, PointType.END),
            PointRef(l3_id, PointType.START)
        ))
        sketch.add_constraint(Coincident(
            PointRef(l3_id, PointType.END),
            PointRef(l1_id, PointType.START)
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        assert len(exported.primitives) == 3

    def test_concentric_circles(self, adapter):
        """Test multiple concentric circles."""
        sketch = SketchDocument(name="ConcentricCirclesTest")
        c1_id = sketch.add_primitive(Circle(center=Point2D(50, 50), radius=10))
        c2_id = sketch.add_primitive(Circle(center=Point2D(52, 52), radius=20))
        c3_id = sketch.add_primitive(Circle(center=Point2D(48, 48), radius=30))

        sketch.add_constraint(Concentric(c1_id, c2_id))
        sketch.add_constraint(Concentric(c2_id, c3_id))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        circles = list(exported.primitives.values())
        centers = [(c.center.x, c.center.y) for c in circles]

        for i in range(1, len(centers)):
            dist = math.sqrt(
                (centers[i][0] - centers[0][0])**2 +
                (centers[i][1] - centers[0][1])**2
            )
            assert dist < 1e-6, f"Circles not concentric: {centers}"

    def test_equal_circles(self, adapter):
        """Test equal radius circles."""
        sketch = SketchDocument(name="EqualCirclesTest")
        c1_id = sketch.add_primitive(Circle(center=Point2D(25, 50), radius=15))
        c2_id = sketch.add_primitive(Circle(center=Point2D(75, 50), radius=25))

        sketch.add_constraint(Equal(c1_id, c2_id))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        circles = list(exported.primitives.values())
        assert abs(circles[0].radius - circles[1].radius) < 1e-6

    def test_equal_chain_three_lines(self, adapter):
        """Test equal constraint chain on three lines."""
        sketch = SketchDocument(name="EqualChainTest")
        l1_id = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(30, 0)))
        l2_id = sketch.add_primitive(Line(start=Point2D(0, 20), end=Point2D(50, 20)))
        l3_id = sketch.add_primitive(Line(start=Point2D(0, 40), end=Point2D(70, 40)))

        sketch.add_constraint(Equal(l1_id, l2_id))
        sketch.add_constraint(Equal(l2_id, l3_id))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        lines = list(exported.primitives.values())
        lengths = [
            math.sqrt((line.end.x - line.start.x)**2 + (line.end.y - line.start.y)**2)
            for line in lines
        ]

        assert abs(lengths[0] - lengths[1]) < 1e-6
        assert abs(lengths[1] - lengths[2]) < 1e-6
