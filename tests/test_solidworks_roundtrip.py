"""
Round-trip tests for SolidWorks adapter.

These tests verify that sketches can be loaded into SolidWorks and exported back
without loss of essential information. Tests are skipped if SolidWorks is not
available on the system (requires Windows with SolidWorks installed).
"""

import math

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
    SolverStatus,
    Spline,
    Symmetric,
    Tangent,
    Vertical,
)

# Try to import the SolidWorks adapter
try:
    from adapter_solidworks import SOLIDWORKS_AVAILABLE, SolidWorksAdapter
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

    def test_single_ellipse(self, adapter):
        """Test round-trip of a single ellipse."""
        sketch = SketchDocument(name="EllipseTest")
        sketch.add_primitive(Ellipse(
            center=Point2D(50, 50),
            major_radius=30,
            minor_radius=20,
            rotation=0.0
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        assert len(exported.primitives) == 1
        ellipse = list(exported.primitives.values())[0]
        assert isinstance(ellipse, Ellipse)
        assert abs(ellipse.center.x - 50) < 1e-6
        assert abs(ellipse.center.y - 50) < 1e-6
        assert abs(ellipse.major_radius - 30) < 1e-6
        assert abs(ellipse.minor_radius - 20) < 1e-6

    def test_ellipse_rotated(self, adapter):
        """Test round-trip of a rotated ellipse."""
        sketch = SketchDocument(name="RotatedEllipseTest")
        sketch.add_primitive(Ellipse(
            center=Point2D(100, 100),
            major_radius=40,
            minor_radius=25,
            rotation=math.pi / 4  # 45 degrees
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        assert len(exported.primitives) == 1
        ellipse = list(exported.primitives.values())[0]
        assert isinstance(ellipse, Ellipse)
        assert abs(ellipse.center.x - 100) < 1e-6
        assert abs(ellipse.center.y - 100) < 1e-6
        assert abs(ellipse.major_radius - 40) < 1e-6
        assert abs(ellipse.minor_radius - 25) < 1e-6
        # Rotation should be preserved (allow some tolerance)
        assert abs(ellipse.rotation - math.pi / 4) < 0.01

    def test_single_elliptical_arc(self, adapter):
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

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        assert len(exported.primitives) == 1
        arc = list(exported.primitives.values())[0]
        assert isinstance(arc, EllipticalArc)
        assert abs(arc.center.x - 50) < 1e-6
        assert abs(arc.center.y - 50) < 1e-6
        assert abs(arc.major_radius - 30) < 1e-6
        assert abs(arc.minor_radius - 20) < 1e-6


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


class TestSolidWorksRoundTripArcVariations:
    """Tests for various arc configurations."""

    def test_arc_clockwise(self, adapter):
        """Test round-trip of a clockwise arc."""
        sketch = SketchDocument(name="CWArcTest")
        sketch.add_primitive(Arc(
            center=Point2D(0, 0),
            start_point=Point2D(50, 0),
            end_point=Point2D(0, 50),
            ccw=False
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        assert len(exported.primitives) == 1
        arc = list(exported.primitives.values())[0]
        assert isinstance(arc, Arc)
        start_radius = math.sqrt(arc.start_point.x**2 + arc.start_point.y**2)
        assert abs(start_radius - 50) < 0.1

    def test_arc_large_angle(self, adapter):
        """Test round-trip of a large arc (> 180 degrees)."""
        sketch = SketchDocument(name="LargeArcTest")
        sketch.add_primitive(Arc(
            center=Point2D(0, 0),
            start_point=Point2D(50, 0),
            end_point=Point2D(0, -50),
            ccw=True
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        assert len(exported.primitives) == 1
        arc = list(exported.primitives.values())[0]
        assert isinstance(arc, Arc)
        start_radius = math.sqrt(arc.start_point.x**2 + arc.start_point.y**2)
        assert abs(start_radius - 50) < 0.1

    def test_arc_90_degree(self, adapter):
        """Test 90-degree arc preserves angle precisely."""
        sketch = SketchDocument(name="Arc90Test")
        sketch.add_primitive(Arc(
            center=Point2D(50, 50),
            start_point=Point2D(80, 50),
            end_point=Point2D(50, 80),
            ccw=True
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        assert len(exported.primitives) == 1
        arc = list(exported.primitives.values())[0]
        assert isinstance(arc, Arc)

        # Calculate sweep angle
        start_angle = math.atan2(arc.start_point.y - arc.center.y, arc.start_point.x - arc.center.x)
        end_angle = math.atan2(arc.end_point.y - arc.center.y, arc.end_point.x - arc.center.x)
        sweep = end_angle - start_angle
        if sweep < 0:
            sweep += 2 * math.pi
        sweep_deg = math.degrees(sweep)

        assert abs(sweep_deg - 90) < 1.0, f"Arc should be 90 degrees, got {sweep_deg}"

    def test_arc_180_degree(self, adapter):
        """Test 180-degree arc (semicircle) preserves angle precisely."""
        sketch = SketchDocument(name="Arc180Test")
        sketch.add_primitive(Arc(
            center=Point2D(50, 50),
            start_point=Point2D(80, 50),
            end_point=Point2D(20, 50),
            ccw=True
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        assert len(exported.primitives) == 1
        arc = list(exported.primitives.values())[0]
        assert isinstance(arc, Arc)

        # Calculate sweep angle
        start_angle = math.atan2(arc.start_point.y - arc.center.y, arc.start_point.x - arc.center.x)
        end_angle = math.atan2(arc.end_point.y - arc.center.y, arc.end_point.x - arc.center.x)
        sweep = end_angle - start_angle
        if sweep < 0:
            sweep += 2 * math.pi
        sweep_deg = math.degrees(sweep)

        assert abs(sweep_deg - 180) < 1.0, f"Arc should be 180 degrees, got {sweep_deg}"

    def test_arc_45_degree(self, adapter):
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

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        assert len(exported.primitives) == 1
        arc = list(exported.primitives.values())[0]
        assert isinstance(arc, Arc)

        # Calculate sweep angle
        start_angle = math.atan2(arc.start_point.y - arc.center.y, arc.start_point.x - arc.center.x)
        end_angle = math.atan2(arc.end_point.y - arc.center.y, arc.end_point.x - arc.center.x)
        sweep = end_angle - start_angle
        if sweep < 0:
            sweep += 2 * math.pi
        sweep_deg = math.degrees(sweep)

        assert abs(sweep_deg - 45) < 1.0, f"Arc should be 45 degrees, got {sweep_deg}"

    def test_construction_arc(self, adapter):
        """Test construction arc flag is preserved."""
        sketch = SketchDocument(name="ConstructionArcTest")
        sketch.add_primitive(Arc(
            center=Point2D(50, 50),
            start_point=Point2D(80, 50),
            end_point=Point2D(50, 80),
            ccw=True,
            construction=True
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        assert len(exported.primitives) == 1
        arc = list(exported.primitives.values())[0]
        assert isinstance(arc, Arc)
        assert arc.construction is True


class TestSolidWorksRoundTripCoincidentVariations:
    """Tests for coincident constraint variations."""

    def test_coincident_chain(self, adapter):
        """Test chain of coincident constraints connecting multiple lines."""
        sketch = SketchDocument(name="CoincidentChainTest")
        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(50, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(50, 0), end=Point2D(50, 50)))
        l3 = sketch.add_primitive(Line(start=Point2D(50, 50), end=Point2D(0, 50)))

        sketch.add_constraint(Coincident(PointRef(l1, PointType.END), PointRef(l2, PointType.START)))
        sketch.add_constraint(Coincident(PointRef(l2, PointType.END), PointRef(l3, PointType.START)))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
        assert len(lines) == 3

        # Check that endpoints are connected
        l1_exp, l2_exp, l3_exp = lines[0], lines[1], lines[2]

        # l1 end should be at l2 start
        dist1 = math.sqrt((l1_exp.end.x - l2_exp.start.x)**2 + (l1_exp.end.y - l2_exp.start.y)**2)
        assert dist1 < 1.0, f"l1 end should connect to l2 start, distance={dist1}"

        # l2 end should be at l3 start
        dist2 = math.sqrt((l2_exp.end.x - l3_exp.start.x)**2 + (l2_exp.end.y - l3_exp.start.y)**2)
        assert dist2 < 1.0, f"l2 end should connect to l3 start, distance={dist2}"

    def test_coincident_point_to_line_endpoint(self, adapter):
        """Test coincident constraint between a point and a line endpoint."""
        sketch = SketchDocument(name="CoincidentPointLineTest")
        line_id = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(100, 0)))
        point_id = sketch.add_primitive(Point(position=Point2D(90, 10)))

        sketch.add_constraint(Coincident(
            PointRef(point_id, PointType.CENTER),
            PointRef(line_id, PointType.END)
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        prims = list(exported.primitives.values())
        line = next(p for p in prims if isinstance(p, Line))
        point = next(p for p in prims if isinstance(p, Point))

        # Point should be at line's end
        dist = math.sqrt((point.position.x - line.end.x)**2 + (point.position.y - line.end.y)**2)
        assert dist < 1.0, f"Point should be at line end, distance={dist}"

    def test_coincident_point_to_circle_center(self, adapter):
        """Test coincident constraint between a point and a circle center."""
        sketch = SketchDocument(name="CoincidentPointCircleTest")
        circle_id = sketch.add_primitive(Circle(center=Point2D(50, 50), radius=25))
        point_id = sketch.add_primitive(Point(position=Point2D(40, 40)))

        sketch.add_constraint(Coincident(
            PointRef(point_id, PointType.CENTER),
            PointRef(circle_id, PointType.CENTER)
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        prims = list(exported.primitives.values())
        circle = next(p for p in prims if isinstance(p, Circle))
        point = next(p for p in prims if isinstance(p, Point))

        # Point should be at circle center
        dist = math.sqrt((point.position.x - circle.center.x)**2 + (point.position.y - circle.center.y)**2)
        assert dist < 1.0, f"Point should be at circle center, distance={dist}"


class TestSolidWorksRoundTripDistanceConstraints:
    """Tests for distance X and Y constraints."""

    def test_distance_x_constraint(self, adapter):
        """Test horizontal distance constraint."""
        sketch = SketchDocument(name="DistanceXTest")
        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(30, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(50, 20), end=Point2D(80, 20)))

        sketch.add_constraint(DistanceX(
            PointRef(l1, PointType.END),
            40,
            PointRef(l2, PointType.START)
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
        l1_end = lines[0].end
        l2_start = lines[1].start

        dx = abs(l2_start.x - l1_end.x)
        assert abs(dx - 40) < 1.0, f"Horizontal distance should be 40, got {dx}"

    def test_distance_y_constraint(self, adapter):
        """Test vertical distance constraint."""
        sketch = SketchDocument(name="DistanceYTest")
        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(50, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(0, 30), end=Point2D(50, 30)))

        sketch.add_constraint(DistanceY(
            PointRef(l1, PointType.START),
            50,
            PointRef(l2, PointType.START)
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
        l1_start = lines[0].start
        l2_start = lines[1].start

        dy = abs(l2_start.y - l1_start.y)
        assert abs(dy - 50) < 1.0, f"Vertical distance should be 50, got {dy}"


class TestSolidWorksRoundTripSplineAdvanced:
    """Advanced spline tests including higher degrees and special cases."""

    def test_higher_degree_spline(self, adapter):
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

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        splines = [p for p in exported.primitives.values() if isinstance(p, Spline)]
        assert len(splines) >= 1, "Should have at least 1 spline"

    def test_many_control_points_spline(self, adapter):
        """Test spline with many control points."""
        sketch = SketchDocument(name="ManyPointsSplineTest")

        # Create a wavy spline with 10 control points
        control_points = []
        for i in range(10):
            y = 25 * math.sin(i * math.pi / 3)
            control_points.append(Point2D(i * 20, 50 + y))

        spline = Spline.create_uniform_bspline(control_points=control_points, degree=3)
        sketch.add_primitive(spline)

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        splines = [p for p in exported.primitives.values() if isinstance(p, Spline)]
        assert len(splines) >= 1, "Should have at least 1 spline"
        assert len(splines[0].control_points) >= 5, "Spline should have multiple control points"

    def test_quadratic_bspline(self, adapter):
        """Test degree-2 (quadratic) B-spline."""
        sketch = SketchDocument(name="QuadraticSplineTest")

        spline = Spline.create_uniform_bspline(
            control_points=[
                Point2D(0, 0),
                Point2D(50, 100),
                Point2D(100, 0)
            ],
            degree=2
        )
        sketch.add_primitive(spline)

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        splines = [p for p in exported.primitives.values() if isinstance(p, Spline)]
        assert len(splines) >= 1, "Should have at least 1 spline"

    def test_periodic_spline(self, adapter):
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

        sketch = SketchDocument(name="PeriodicSplineTest")
        spline = Spline.create_uniform_bspline(control_points=control_points, degree=3)
        sketch.add_primitive(spline)

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        splines = [p for p in exported.primitives.values() if isinstance(p, Spline)]
        assert len(splines) >= 1, "Should have at least 1 spline"

    def test_weighted_spline(self, adapter):
        """Test spline with non-uniform weights (NURBS)."""
        sketch = SketchDocument(name="WeightedSplineTest")

        spline = Spline(
            control_points=[
                Point2D(0, 0),
                Point2D(50, 100),
                Point2D(100, 0)
            ],
            degree=2,
            knots=[0, 0, 0, 1, 1, 1],
            weights=[1.0, 2.0, 1.0]  # Higher weight on middle point
        )
        sketch.add_primitive(spline)

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        splines = [p for p in exported.primitives.values() if isinstance(p, Spline)]
        assert len(splines) >= 1, "Should have at least 1 spline"


class TestSolidWorksRoundTripPrecision:
    """Tests for precision of constraints and dimensions."""

    def test_angle_precision(self, adapter):
        """Test angle constraint precision."""
        sketch = SketchDocument(name="AnglePrecisionTest")
        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(100, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(50, 50)))

        sketch.add_constraint(Angle(l1, l2, 30))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]

        # Calculate angle between lines
        dx1 = lines[0].end.x - lines[0].start.x
        dy1 = lines[0].end.y - lines[0].start.y
        dx2 = lines[1].end.x - lines[1].start.x
        dy2 = lines[1].end.y - lines[1].start.y

        angle1 = math.atan2(dy1, dx1)
        angle2 = math.atan2(dy2, dx2)
        angle_diff = abs(math.degrees(angle2 - angle1))

        assert abs(angle_diff - 30) < 1.0, f"Angle should be 30 degrees, got {angle_diff}"

    def test_length_precision(self, adapter):
        """Test length constraint precision."""
        sketch = SketchDocument(name="LengthPrecisionTest")
        line_id = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(50, 0)))

        sketch.add_constraint(Length(line_id, 123.456))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        length = math.sqrt((line.end.x - line.start.x)**2 + (line.end.y - line.start.y)**2)

        assert abs(length - 123.456) < 0.01, f"Length should be 123.456, got {length}"

    def test_radius_precision(self, adapter):
        """Test radius constraint precision."""
        sketch = SketchDocument(name="RadiusPrecisionTest")
        circle_id = sketch.add_primitive(Circle(center=Point2D(50, 50), radius=20))

        sketch.add_constraint(Radius(PointRef(circle_id, PointType.CENTER), 37.5))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        circle = list(exported.primitives.values())[0]
        assert abs(circle.radius - 37.5) < 0.01, f"Radius should be 37.5, got {circle.radius}"


class TestSolidWorksRoundTripMultipleConstraints:
    """Tests for multiple constraints on single entities."""

    def test_multiple_constraints_circle(self, adapter):
        """Test circle with both radius and concentric constraints."""
        sketch = SketchDocument(name="MultiConstraintCircleTest")
        c1 = sketch.add_primitive(Circle(center=Point2D(50, 50), radius=20))
        c2 = sketch.add_primitive(Circle(center=Point2D(60, 60), radius=30))

        sketch.add_constraint(Concentric(c1, c2))
        sketch.add_constraint(Radius(PointRef(c2, PointType.CENTER), 40))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        circles = [p for p in exported.primitives.values() if isinstance(p, Circle)]
        assert len(circles) == 2

        # Check concentric (centers should match)
        dist = math.sqrt((circles[0].center.x - circles[1].center.x)**2 +
                        (circles[0].center.y - circles[1].center.y)**2)
        assert dist < 1.0, f"Circles should be concentric, center distance={dist}"

    def test_multiple_constraints_horizontal_length(self, adapter):
        """Test line with both horizontal and length constraints."""
        sketch = SketchDocument(name="HorizontalLengthTest")
        line_id = sketch.add_primitive(Line(start=Point2D(0, 10), end=Point2D(50, 20)))

        sketch.add_constraint(Horizontal(line_id))
        sketch.add_constraint(Length(line_id, 80))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        line = list(exported.primitives.values())[0]

        # Check horizontal
        assert abs(line.start.y - line.end.y) < 1.0, "Line should be horizontal"

        # Check length
        length = math.sqrt((line.end.x - line.start.x)**2 + (line.end.y - line.start.y)**2)
        assert abs(length - 80) < 1.0, f"Length should be 80, got {length}"

    def test_multiple_constraints_vertical_length(self, adapter):
        """Test line with both vertical and length constraints."""
        sketch = SketchDocument(name="VerticalLengthTest")
        line_id = sketch.add_primitive(Line(start=Point2D(10, 0), end=Point2D(20, 50)))

        sketch.add_constraint(Vertical(line_id))
        sketch.add_constraint(Length(line_id, 60))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        line = list(exported.primitives.values())[0]

        # Check vertical
        assert abs(line.start.x - line.end.x) < 1.0, "Line should be vertical"

        # Check length
        length = math.sqrt((line.end.x - line.start.x)**2 + (line.end.y - line.start.y)**2)
        assert abs(length - 60) < 1.0, f"Length should be 60, got {length}"


class TestSolidWorksRoundTripProfiles:
    """Tests for complex profile geometries."""

    def test_slot_profile(self, adapter):
        """Test slot profile (rectangle with semicircular ends)."""
        sketch = SketchDocument(name="SlotProfileTest")

        # Create a slot: two parallel lines connected by two semicircular arcs
        sketch.add_primitive(Line(start=Point2D(20, 0), end=Point2D(80, 0)))
        sketch.add_primitive(Line(start=Point2D(80, 40), end=Point2D(20, 40)))

        # Right semicircle
        sketch.add_primitive(Arc(
            center=Point2D(80, 20),
            start_point=Point2D(80, 0),
            end_point=Point2D(80, 40),
            ccw=False
        ))

        # Left semicircle
        sketch.add_primitive(Arc(
            center=Point2D(20, 20),
            start_point=Point2D(20, 40),
            end_point=Point2D(20, 0),
            ccw=False
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        # Should have 2 lines and 2 arcs
        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
        arcs = [p for p in exported.primitives.values() if isinstance(p, Arc)]

        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"
        assert len(arcs) == 2, f"Expected 2 arcs, got {len(arcs)}"

    def test_smooth_corner_profile(self, adapter):
        """Test profile with tangent arc corners."""
        sketch = SketchDocument(name="SmoothCornerTest")

        # Create an L-shape with a fillet
        sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(0, 50)))
        sketch.add_primitive(Arc(
            center=Point2D(10, 50),
            start_point=Point2D(0, 50),
            end_point=Point2D(10, 60),
            ccw=True
        ))
        sketch.add_primitive(Line(start=Point2D(10, 60), end=Point2D(60, 60)))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
        arcs = [p for p in exported.primitives.values() if isinstance(p, Arc)]

        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"
        assert len(arcs) == 1, f"Expected 1 arc, got {len(arcs)}"

    def test_nested_geometry(self, adapter):
        """Test nested shapes (circle inside rectangle)."""
        sketch = SketchDocument(name="NestedGeometryTest")

        # Outer rectangle
        sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(100, 0)))
        sketch.add_primitive(Line(start=Point2D(100, 0), end=Point2D(100, 80)))
        sketch.add_primitive(Line(start=Point2D(100, 80), end=Point2D(0, 80)))
        sketch.add_primitive(Line(start=Point2D(0, 80), end=Point2D(0, 0)))

        # Inner circle
        sketch.add_primitive(Circle(center=Point2D(50, 40), radius=20))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
        circles = [p for p in exported.primitives.values() if isinstance(p, Circle)]

        assert len(lines) == 4, f"Expected 4 lines, got {len(lines)}"
        assert len(circles) == 1, f"Expected 1 circle, got {len(circles)}"

    def test_multiple_points_standalone(self, adapter):
        """Test multiple standalone points."""
        sketch = SketchDocument(name="MultiplePointsTest")

        sketch.add_primitive(Point(position=Point2D(10, 10)))
        sketch.add_primitive(Point(position=Point2D(50, 50)))
        sketch.add_primitive(Point(position=Point2D(90, 10)))
        sketch.add_primitive(Point(position=Point2D(50, 90)))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        points = [p for p in exported.primitives.values() if isinstance(p, Point)]
        assert len(points) == 4, f"Expected 4 points, got {len(points)}"


class TestSolidWorksRoundTripConstraintExport:
    """Tests for constraint export functionality."""

    def test_constraint_export_horizontal(self, adapter):
        """Test that horizontal constraint is applied correctly."""
        sketch = SketchDocument(name="ExportHorizontalTest")
        line_id = sketch.add_primitive(Line(start=Point2D(0, 10), end=Point2D(100, 20)))
        sketch.add_constraint(Horizontal(line_id))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        # Line should be horizontal after constraint
        assert abs(line.start.y - line.end.y) < 1.0, "Line should be horizontal"

    def test_constraint_export_perpendicular(self, adapter):
        """Test that perpendicular constraint is applied correctly."""
        sketch = SketchDocument(name="ExportPerpTest")
        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(100, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(50, 0), end=Point2D(60, 40)))
        sketch.add_constraint(Perpendicular(l1, l2))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]

        # Calculate angle between lines
        dx1 = lines[0].end.x - lines[0].start.x
        dy1 = lines[0].end.y - lines[0].start.y
        dx2 = lines[1].end.x - lines[1].start.x
        dy2 = lines[1].end.y - lines[1].start.y

        # Dot product should be near zero for perpendicular lines
        dot = dx1 * dx2 + dy1 * dy2
        len1 = math.sqrt(dx1**2 + dy1**2)
        len2 = math.sqrt(dx2**2 + dy2**2)
        cos_angle = dot / (len1 * len2) if len1 > 0 and len2 > 0 else 0

        assert abs(cos_angle) < 0.1, f"Lines should be perpendicular, cos(angle)={cos_angle}"

    def test_constraint_export_length(self, adapter):
        """Test that length constraint is applied correctly."""
        sketch = SketchDocument(name="ExportLengthTest")
        line_id = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(30, 0)))
        sketch.add_constraint(Length(line_id, 75))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        length = math.sqrt((line.end.x - line.start.x)**2 + (line.end.y - line.start.y)**2)
        assert abs(length - 75) < 1.0, f"Length should be 75, got {length}"


class TestSolidWorksRoundTripAdvanced:
    """Additional advanced tests."""

    def test_fully_constrained_rectangle(self, adapter):
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

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
        assert len(lines) == 4

        # Check horizontal lines are horizontal
        horizontal_count = sum(1 for ln in lines if abs(ln.start.y - ln.end.y) < 0.5)
        vertical_count = sum(1 for ln in lines if abs(ln.start.x - ln.end.x) < 0.5)

        assert horizontal_count == 2, f"Should have 2 horizontal lines, got {horizontal_count}"
        assert vertical_count == 2, f"Should have 2 vertical lines, got {vertical_count}"

    def test_solver_status_fullyconstrained(self, adapter):
        """Test that solver reports fully constrained status."""
        sketch = SketchDocument(name="FullyConstrainedTest")
        point_id = sketch.add_primitive(Point(position=Point2D(50, 50)))
        sketch.add_constraint(Fixed(point_id))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        status, dof = adapter.get_solver_status()

        # Should be fully constrained or have 0 DOF
        assert status == SolverStatus.FULLY_CONSTRAINED or dof == 0

    def test_equal_chain_four_circles(self, adapter):
        """Test equal constraint chain on four circles."""
        sketch = SketchDocument(name="EqualChain4CirclesTest")
        c1 = sketch.add_primitive(Circle(center=Point2D(20, 50), radius=10))
        c2 = sketch.add_primitive(Circle(center=Point2D(50, 50), radius=15))
        c3 = sketch.add_primitive(Circle(center=Point2D(80, 50), radius=20))
        c4 = sketch.add_primitive(Circle(center=Point2D(110, 50), radius=25))

        sketch.add_constraint(Equal(c1, c2))
        sketch.add_constraint(Equal(c2, c3))
        sketch.add_constraint(Equal(c3, c4))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        circles = [p for p in exported.primitives.values() if isinstance(p, Circle)]
        assert len(circles) == 4

        # All radii should be equal
        radii = [c.radius for c in circles]
        for r in radii[1:]:
            assert abs(r - radii[0]) < 1.0, f"All radii should be equal: {radii}"

    def test_tangent_arc_line(self, adapter):
        """Test tangent constraint between arc and line."""
        sketch = SketchDocument(name="TangentArcLineTest")
        arc_id = sketch.add_primitive(Arc(
            center=Point2D(50, 50),
            start_point=Point2D(80, 50),
            end_point=Point2D(50, 80),
            ccw=True
        ))
        line_id = sketch.add_primitive(Line(start=Point2D(80, 50), end=Point2D(120, 50)))

        sketch.add_constraint(Tangent(arc_id, line_id))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        arcs = [p for p in exported.primitives.values() if isinstance(p, Arc)]
        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]

        assert len(arcs) >= 1, "Should have at least 1 arc"
        assert len(lines) >= 1, "Should have at least 1 line"

    def test_arc_tangent_to_two_lines(self, adapter):
        """Test arc tangent to two lines (fillet-like)."""
        sketch = SketchDocument(name="ArcTangent2LinesTest")

        sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(50, 0)))
        sketch.add_primitive(Line(start=Point2D(50, 50), end=Point2D(50, 0)))

        # Arc connecting the two lines
        sketch.add_primitive(Arc(
            center=Point2D(50, 0),
            start_point=Point2D(50, 0),
            end_point=Point2D(50, 0),
            ccw=True
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
        assert len(lines) == 2

    def test_very_large_dimensions(self, adapter):
        """Test geometry with very large dimensions."""
        sketch = SketchDocument(name="LargeDimensionsTest")
        sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(10000, 5000)
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        assert abs(line.end.x - 10000) < 1, f"X should be 10000, got {line.end.x}"
        assert abs(line.end.y - 5000) < 1, f"Y should be 5000, got {line.end.y}"

    def test_very_small_dimensions(self, adapter):
        """Test geometry with very small dimensions."""
        sketch = SketchDocument(name="SmallDimensionsTest")
        sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(0.1, 0.05)
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        line = list(exported.primitives.values())[0]
        assert abs(line.end.x - 0.1) < 0.01, f"X should be 0.1, got {line.end.x}"
        assert abs(line.end.y - 0.05) < 0.01, f"Y should be 0.05, got {line.end.y}"


class TestSolidWorksRoundTripSymmetric:
    """Tests for symmetric constraint in SolidWorks adapter."""

    @pytest.fixture
    def adapter(self):
        """Create a fresh SolidWorks adapter for each test."""
        if not SOLIDWORKS_AVAILABLE:
            pytest.skip("SolidWorks not available")
        adapter = SolidWorksAdapter()
        yield adapter

    def test_symmetric_points_about_vertical_line(self, adapter):
        """Test point symmetry about a vertical centerline."""
        sketch = SketchDocument(name="SymmetricPointsVerticalTest")

        # Create a vertical centerline
        centerline = sketch.add_primitive(Line(
            start=Point2D(50, 0),
            end=Point2D(50, 100),
            construction=True
        ))

        # Create two points that should be symmetric about the centerline
        p1 = sketch.add_primitive(Point(position=Point2D(30, 50)))
        p2 = sketch.add_primitive(Point(position=Point2D(70, 50)))

        # Add symmetric constraint
        sketch.add_constraint(Symmetric(
            PointRef(p1, PointType.CENTER),
            PointRef(p2, PointType.CENTER),
            centerline
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        # Should have 2 points and 1 line (centerline)
        points = [p for p in exported.primitives.values() if isinstance(p, Point)]
        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]

        assert len(points) >= 2, "Should have at least 2 points"
        assert len(lines) >= 1, "Should have at least 1 line (centerline)"

    def test_symmetric_points_about_horizontal_line(self, adapter):
        """Test point symmetry about a horizontal centerline."""
        sketch = SketchDocument(name="SymmetricPointsHorizontalTest")

        # Create a horizontal centerline
        centerline = sketch.add_primitive(Line(
            start=Point2D(0, 50),
            end=Point2D(100, 50),
            construction=True
        ))

        # Create two points that should be symmetric about the centerline
        p1 = sketch.add_primitive(Point(position=Point2D(50, 30)))
        p2 = sketch.add_primitive(Point(position=Point2D(50, 70)))

        # Add symmetric constraint
        sketch.add_constraint(Symmetric(
            PointRef(p1, PointType.CENTER),
            PointRef(p2, PointType.CENTER),
            centerline
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        points = [p for p in exported.primitives.values() if isinstance(p, Point)]
        assert len(points) >= 2, "Should have at least 2 points"

    def test_symmetric_lines_about_centerline(self, adapter):
        """Test line symmetry about a vertical centerline."""
        sketch = SketchDocument(name="SymmetricLinesTest")

        # Create a vertical centerline
        centerline = sketch.add_primitive(Line(
            start=Point2D(50, 0),
            end=Point2D(50, 100),
            construction=True
        ))

        # Create two lines that should be symmetric about the centerline
        line1 = sketch.add_primitive(Line(
            start=Point2D(20, 20),
            end=Point2D(40, 60)
        ))
        line2 = sketch.add_primitive(Line(
            start=Point2D(80, 20),
            end=Point2D(60, 60)
        ))

        # Add symmetric constraint
        sketch.add_constraint(Symmetric(line1, line2, centerline))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
        assert len(lines) >= 3, "Should have at least 3 lines (2 symmetric + centerline)"

    def test_symmetric_line_endpoints(self, adapter):
        """Test symmetry of line endpoints about a centerline."""
        sketch = SketchDocument(name="SymmetricEndpointsTest")

        # Create a vertical centerline
        centerline = sketch.add_primitive(Line(
            start=Point2D(50, 0),
            end=Point2D(50, 100),
            construction=True
        ))

        # Create two lines
        line1 = sketch.add_primitive(Line(
            start=Point2D(30, 40),
            end=Point2D(30, 80)
        ))
        line2 = sketch.add_primitive(Line(
            start=Point2D(70, 40),
            end=Point2D(70, 80)
        ))

        # Make start points symmetric
        sketch.add_constraint(Symmetric(
            PointRef(line1, PointType.START),
            PointRef(line2, PointType.START),
            centerline
        ))

        # Make end points symmetric
        sketch.add_constraint(Symmetric(
            PointRef(line1, PointType.END),
            PointRef(line2, PointType.END),
            centerline
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
        non_construction = [ln for ln in lines if not ln.construction]
        assert len(non_construction) >= 2, "Should have at least 2 non-construction lines"

    def test_symmetric_circles(self, adapter):
        """Test circle symmetry about a centerline."""
        sketch = SketchDocument(name="SymmetricCirclesTest")

        # Create a vertical centerline
        centerline = sketch.add_primitive(Line(
            start=Point2D(50, 0),
            end=Point2D(50, 100),
            construction=True
        ))

        # Create two circles that should be symmetric about the centerline
        circle1 = sketch.add_primitive(Circle(
            center=Point2D(30, 50),
            radius=10
        ))
        circle2 = sketch.add_primitive(Circle(
            center=Point2D(70, 50),
            radius=10
        ))

        # Add symmetric constraint for the circle centers
        sketch.add_constraint(Symmetric(
            PointRef(circle1, PointType.CENTER),
            PointRef(circle2, PointType.CENTER),
            centerline
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        circles = [p for p in exported.primitives.values() if isinstance(p, Circle)]
        assert len(circles) >= 2, "Should have at least 2 circles"

    def test_symmetric_arcs(self, adapter):
        """Test arc symmetry about a centerline."""
        sketch = SketchDocument(name="SymmetricArcsTest")

        # Create a vertical centerline
        centerline = sketch.add_primitive(Line(
            start=Point2D(50, 0),
            end=Point2D(50, 100),
            construction=True
        ))

        # Create two arcs that should be symmetric
        arc1 = sketch.add_primitive(Arc(
            center=Point2D(30, 50),
            start_point=Point2D(40, 50),
            end_point=Point2D(30, 60),
            ccw=True
        ))
        arc2 = sketch.add_primitive(Arc(
            center=Point2D(70, 50),
            start_point=Point2D(60, 50),
            end_point=Point2D(70, 60),
            ccw=False  # Mirror flips direction
        ))

        # Add symmetric constraint for the arc centers
        sketch.add_constraint(Symmetric(
            PointRef(arc1, PointType.CENTER),
            PointRef(arc2, PointType.CENTER),
            centerline
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        arcs = [p for p in exported.primitives.values() if isinstance(p, Arc)]
        assert len(arcs) >= 2, "Should have at least 2 arcs"

    def test_symmetric_with_diagonal_axis(self, adapter):
        """Test symmetry about a diagonal axis line."""
        sketch = SketchDocument(name="SymmetricDiagonalTest")

        # Create a diagonal centerline (45 degrees)
        centerline = sketch.add_primitive(Line(
            start=Point2D(0, 0),
            end=Point2D(100, 100),
            construction=True
        ))

        # Create two points symmetric about the diagonal
        p1 = sketch.add_primitive(Point(position=Point2D(20, 60)))
        p2 = sketch.add_primitive(Point(position=Point2D(60, 20)))

        # Add symmetric constraint
        sketch.add_constraint(Symmetric(
            PointRef(p1, PointType.CENTER),
            PointRef(p2, PointType.CENTER),
            centerline
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        points = [p for p in exported.primitives.values() if isinstance(p, Point)]
        assert len(points) >= 2, "Should have at least 2 points"

    def test_symmetric_rectangle_halves(self, adapter):
        """Test creating a symmetric rectangle using half and mirror."""
        sketch = SketchDocument(name="SymmetricRectangleTest")

        # Create a vertical centerline
        centerline = sketch.add_primitive(Line(
            start=Point2D(50, 0),
            end=Point2D(50, 100),
            construction=True
        ))

        # Create left half of rectangle
        sketch.add_primitive(Line(
            start=Point2D(0, 80),
            end=Point2D(50, 80)
        ))
        left_side = sketch.add_primitive(Line(
            start=Point2D(0, 20),
            end=Point2D(0, 80)
        ))
        sketch.add_primitive(Line(
            start=Point2D(0, 20),
            end=Point2D(50, 20)
        ))

        # Create right half of rectangle
        sketch.add_primitive(Line(
            start=Point2D(50, 80),
            end=Point2D(100, 80)
        ))
        right_side = sketch.add_primitive(Line(
            start=Point2D(100, 20),
            end=Point2D(100, 80)
        ))
        sketch.add_primitive(Line(
            start=Point2D(50, 20),
            end=Point2D(100, 20)
        ))

        # Make the sides symmetric
        sketch.add_constraint(Symmetric(left_side, right_side, centerline))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
        non_construction = [ln for ln in lines if not ln.construction]
        assert len(non_construction) >= 6, "Should have at least 6 non-construction lines"


class TestSolidWorksConstraintExportRegression:
    """Regression tests for constraint export edge cases.

    These tests verify fixes for COM late-binding quirks discovered during
    constraint export development. See COM_QUIRKS.md for details.
    """

    def test_rectangle_exports_all_constraints(self, adapter):
        """Test that rectangle with H/V constraints exports all 4 constraints.

        Regression test for: Same-length segments returning same ID due to
        length-based property matching. Fixed by using index-based segment ID lookup.
        """
        sketch = SketchDocument(name="RectConstraintExportTest")

        # Create rectangle - opposite sides have same length
        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(100, 0)))      # bottom (horizontal)
        l2 = sketch.add_primitive(Line(start=Point2D(100, 0), end=Point2D(100, 50)))   # right (vertical)
        l3 = sketch.add_primitive(Line(start=Point2D(100, 50), end=Point2D(0, 50)))    # top (horizontal)
        l4 = sketch.add_primitive(Line(start=Point2D(0, 50), end=Point2D(0, 0)))       # left (vertical)

        # Add horizontal/vertical constraints to all sides
        sketch.add_constraint(Horizontal(l1))
        sketch.add_constraint(Vertical(l2))
        sketch.add_constraint(Horizontal(l3))
        sketch.add_constraint(Vertical(l4))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        # Should export all 4 constraints, not just 2
        horizontal_constraints = [
            c for c in exported.constraints
            if hasattr(c, 'constraint_type') and c.constraint_type.name == 'HORIZONTAL'
        ]
        vertical_constraints = [
            c for c in exported.constraints
            if hasattr(c, 'constraint_type') and c.constraint_type.name == 'VERTICAL'
        ]

        assert len(horizontal_constraints) == 2, \
            f"Expected 2 horizontal constraints, got {len(horizontal_constraints)}"
        assert len(vertical_constraints) == 2, \
            f"Expected 2 vertical constraints, got {len(vertical_constraints)}"

    def test_single_entity_constraint_uses_source_segment(self, adapter):
        """Test that H/V constraints use source segment ID directly.

        Regression test for: GetEntities returning ambiguous COM objects
        that can't be reliably matched. Fixed by using source segment ID
        for single-entity constraint types.
        """
        sketch = SketchDocument(name="SingleEntityConstraintTest")

        # Create two lines with the same length
        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(50, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(0, 30), end=Point2D(50, 30)))

        # Add horizontal constraint to each
        sketch.add_constraint(Horizontal(l1))
        sketch.add_constraint(Horizontal(l2))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        # Should have 2 horizontal constraints on different lines
        horizontal_constraints = [
            c for c in exported.constraints
            if hasattr(c, 'constraint_type') and c.constraint_type.name == 'HORIZONTAL'
        ]

        assert len(horizontal_constraints) == 2, \
            f"Expected 2 horizontal constraints, got {len(horizontal_constraints)}"

        # Constraints should reference different entities
        refs = [tuple(c.references) for c in horizontal_constraints]
        assert refs[0] != refs[1], "Constraints should reference different lines"

    def test_constraint_deduplication(self, adapter):
        """Test that duplicate constraints from multiple segments are deduplicated.

        When iterating through segments, the same constraint may be found from
        multiple entry points. The adapter should deduplicate these.
        """
        sketch = SketchDocument(name="ConstraintDedupTest")

        line_id = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(100, 0)))
        sketch.add_constraint(Horizontal(line_id))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        # Should have exactly 1 horizontal constraint, not duplicates
        horizontal_constraints = [
            c for c in exported.constraints
            if hasattr(c, 'constraint_type') and c.constraint_type.name == 'HORIZONTAL'
        ]

        assert len(horizontal_constraints) == 1, \
            f"Expected exactly 1 horizontal constraint, got {len(horizontal_constraints)}"

    def test_circle_constraint_export(self, adapter):
        """Test that constraints on circles are exported correctly."""
        sketch = SketchDocument(name="CircleConstraintExportTest")

        c1 = sketch.add_primitive(Circle(center=Point2D(50, 50), radius=20))
        c2 = sketch.add_primitive(Circle(center=Point2D(100, 50), radius=30))
        sketch.add_constraint(Equal(c1, c2))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        # Circles should have equal radii
        circles = list(exported.primitives.values())
        assert abs(circles[0].radius - circles[1].radius) < 0.1, \
            "Circles should have equal radii after constraint"

    def test_mixed_constraint_types_export(self, adapter):
        """Test exporting sketch with multiple constraint types."""
        sketch = SketchDocument(name="MixedConstraintExportTest")

        l1 = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(80, 0)))
        l2 = sketch.add_primitive(Line(start=Point2D(80, 0), end=Point2D(80, 60)))
        sketch.add_primitive(Circle(center=Point2D(40, 30), radius=15))  # Just add, ID not needed

        sketch.add_constraint(Horizontal(l1))
        sketch.add_constraint(Vertical(l2))
        sketch.add_constraint(Perpendicular(l1, l2))
        sketch.add_constraint(Coincident(
            PointRef(l1, PointType.END),
            PointRef(l2, PointType.START)
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        assert len(exported.primitives) == 3, "Should have 3 primitives"
        # Verify geometry constraints were applied
        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
        horizontal_line = next((ln for ln in lines if abs(ln.start.y - ln.end.y) < 0.1), None)
        vertical_line = next((ln for ln in lines if abs(ln.start.x - ln.end.x) < 0.1), None)

        assert horizontal_line is not None, "Should have a horizontal line"
        assert vertical_line is not None, "Should have a vertical line"


class TestSolidWorksExportRegression:
    """Regression tests for export edge cases discovered via SketchBridge demo.

    These tests verify fixes for issues found when exporting the comprehensive
    SketchBridge demo sketch to SolidWorks.
    """

    def test_arc_270_degrees_not_circle(self, adapter):
        """Test that a 270-degree arc is exported as Arc, not Circle.

        Regression test for: Large arcs (>180°) being incorrectly identified
        as full circles due to arc length comparison tolerance issues.
        """
        sketch = SketchDocument(name="Arc270Test")
        # 270 degree CCW arc (from right going up and around to bottom)
        # This is similar to the tangent arc in the demo that was incorrectly exported
        sketch.add_primitive(Arc(
            center=Point2D(95, 12.5),
            start_point=Point2D(82, 12.5),  # Left of center (180°)
            end_point=Point2D(95, 25.5),    # Top of center (90°)
            ccw=True  # Going CCW from 180° to 90° = 270° sweep
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        assert len(exported.primitives) == 1, "Should have exactly 1 primitive"
        prim = list(exported.primitives.values())[0]
        assert isinstance(prim, Arc), f"Should be Arc, not {type(prim).__name__}"
        assert not isinstance(prim, Circle), "Should not be a Circle"

    def test_multiple_lines_correct_endpoints(self, adapter):
        """Test that multiple lines preserve their correct endpoints.

        Regression test for: Line export matching wrong point pairs when
        multiple lines exist in the sketch, especially with similar lengths.
        """
        sketch = SketchDocument(name="MultipleLinesTest")

        # Create lines similar to the demo - rectangle plus angled lines
        # Rectangle
        sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(40, 0)))
        sketch.add_primitive(Line(start=Point2D(40, 0), end=Point2D(40, 25)))
        sketch.add_primitive(Line(start=Point2D(40, 25), end=Point2D(0, 25)))
        sketch.add_primitive(Line(start=Point2D(0, 25), end=Point2D(0, 0)))

        # Angled line at 30 degrees (similar to demo's line_angled1)
        angle_rad = math.radians(30)
        end_x = 110 + 25 * math.cos(angle_rad)
        end_y = 25 * math.sin(angle_rad)
        sketch.add_primitive(Line(start=Point2D(110, 0), end=Point2D(end_x, end_y)))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
        assert len(lines) == 5, f"Expected 5 lines, got {len(lines)}"

        # Find the angled line (should have start at 110, 0)
        angled_line = None
        for ln in lines:
            if abs(ln.start.x - 110) < 0.1 and abs(ln.start.y - 0) < 0.1:
                angled_line = ln
                break

        assert angled_line is not None, "Should find angled line starting at (110, 0)"
        # End point should be near (131.65, 12.5), not (0, 25) or other wrong point
        assert angled_line.end.x > 120, \
            f"Angled line end X should be > 120, got {angled_line.end.x}"
        assert angled_line.end.y > 10, \
            f"Angled line end Y should be > 10, got {angled_line.end.y}"

    def test_no_degenerate_zero_length_lines(self, adapter):
        """Test that no zero-length (degenerate) lines are exported.

        Regression test for: SolidWorks creating internal degenerate segments
        that get incorrectly exported as zero-length lines.
        """
        sketch = SketchDocument(name="DegenerateLineTest")

        # Create a simple rectangle - should not produce any degenerate lines
        sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(50, 0)))
        sketch.add_primitive(Line(start=Point2D(50, 0), end=Point2D(50, 30)))
        sketch.add_primitive(Line(start=Point2D(50, 30), end=Point2D(0, 30)))
        sketch.add_primitive(Line(start=Point2D(0, 30), end=Point2D(0, 0)))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]

        for ln in lines:
            length = math.sqrt((ln.end.x - ln.start.x)**2 + (ln.end.y - ln.start.y)**2)
            assert length > 0.01, \
                f"Found degenerate line from ({ln.start.x}, {ln.start.y}) to ({ln.end.x}, {ln.end.y})"

    def test_ellipse_no_extra_standalone_points(self, adapter):
        """Test that ellipse export doesn't create extra standalone points.

        Regression test for: SolidWorks internal ellipse vertex points being
        incorrectly exported as standalone Point primitives.
        """
        sketch = SketchDocument(name="EllipsePointsTest")
        sketch.add_primitive(Ellipse(
            center=Point2D(30, -25),
            major_radius=18,
            minor_radius=10,
            rotation=math.radians(15)
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        # Should have only 1 primitive (the ellipse)
        ellipses = [p for p in exported.primitives.values() if isinstance(p, Ellipse)]
        points = [p for p in exported.primitives.values() if isinstance(p, Point)]

        assert len(ellipses) == 1, f"Expected 1 ellipse, got {len(ellipses)}"
        assert len(points) == 0, \
            f"Expected 0 standalone points, got {len(points)} (ellipse vertex points leaked)"

    def test_elliptical_arc_no_extra_standalone_points(self, adapter):
        """Test that elliptical arc export doesn't create extra standalone points.

        Regression test for: SolidWorks internal elliptical arc vertex points
        being incorrectly exported as standalone Point primitives.
        """
        sketch = SketchDocument(name="EllipticalArcPointsTest")
        sketch.add_primitive(EllipticalArc(
            center=Point2D(85, -25),
            major_radius=15,
            minor_radius=8,
            rotation=math.radians(-10),
            start_param=math.radians(30),
            end_param=math.radians(240),
            ccw=True
        ))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        # Should have only 1 primitive (the elliptical arc)
        arcs = [p for p in exported.primitives.values() if isinstance(p, EllipticalArc)]
        points = [p for p in exported.primitives.values() if isinstance(p, Point)]

        assert len(arcs) == 1, f"Expected 1 elliptical arc, got {len(arcs)}"
        assert len(points) == 0, \
            f"Expected 0 standalone points, got {len(points)} (arc vertex points leaked)"

    def test_line_with_negative_coordinates(self, adapter):
        """Test that lines with negative Y coordinates are correctly exported.

        Regression test for: Potential issues with negative coordinate handling
        in the line export logic.
        """
        sketch = SketchDocument(name="NegativeCoordLineTest")
        # Line in negative Y region (like the midpoint_line in demo)
        sketch.add_primitive(Line(start=Point2D(55, -15), end=Point2D(75, -15)))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        lines = [p for p in exported.primitives.values() if isinstance(p, Line)]
        assert len(lines) == 1, f"Expected 1 line, got {len(lines)}"

        ln = lines[0]
        # Check coordinates are preserved
        assert abs(ln.start.x - 55) < 0.1, f"Start X should be 55, got {ln.start.x}"
        assert abs(ln.start.y - (-15)) < 0.1, f"Start Y should be -15, got {ln.start.y}"
        assert abs(ln.end.x - 75) < 0.1, f"End X should be 75, got {ln.end.x}"
        assert abs(ln.end.y - (-15)) < 0.1, f"End Y should be -15, got {ln.end.y}"

    def test_multiple_standalone_points_preserved(self, adapter):
        """Test that multiple standalone points are all preserved.

        Regression test for: Some standalone points being lost during export
        when multiple points exist in the sketch.
        """
        sketch = SketchDocument(name="MultiplePointsTest")
        # Three points at different locations (like the demo's symmetric points)
        sketch.add_primitive(Point(position=Point2D(5, -15)))
        sketch.add_primitive(Point(position=Point2D(35, -15)))
        sketch.add_primitive(Point(position=Point2D(65, -15)))

        adapter.create_sketch(sketch.name)
        adapter.load_sketch(sketch)
        exported = adapter.export_sketch()

        points = [p for p in exported.primitives.values() if isinstance(p, Point)]
        assert len(points) == 3, f"Expected 3 points, got {len(points)}"

        # Verify the specific positions are preserved
        point_xs = sorted([p.position.x for p in points])
        assert abs(point_xs[0] - 5) < 0.1, f"First point X should be 5, got {point_xs[0]}"
        assert abs(point_xs[1] - 35) < 0.1, f"Second point X should be 35, got {point_xs[1]}"
        assert abs(point_xs[2] - 65) < 0.1, f"Third point X should be 65, got {point_xs[2]}"
