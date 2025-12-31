"""Tests for the canonical sketch schema."""

import pytest
import math
import json

from sketch_canonical import (
    # Types
    Point2D, Vector2D, ElementId, PointType, PointRef, ElementPrefix,
    # Primitives
    SketchPrimitive, Line, Arc, Circle, Point, Spline,
    # Constraints
    ConstraintType, ConstraintStatus, SketchConstraint, CONSTRAINT_RULES,
    Coincident, Tangent, Perpendicular, Parallel, Concentric,
    Equal, Collinear, Horizontal, Vertical, Fixed,
    Distance, DistanceX, DistanceY, Length, Radius, Diameter, Angle,
    Symmetric, MidpointConstraint,
    # Document
    SketchDocument, SolverStatus,
    # Validation
    validate_sketch, validate_primitive, validate_constraint,
    ValidationResult, ValidationSeverity, ValidationIssue, DEFAULT_TOLERANCE,
    # Serialization
    SketchEncoder,
    sketch_to_json, sketch_from_json, sketch_to_dict, dict_to_sketch,
    primitive_to_dict, dict_to_primitive,
    constraint_to_dict, dict_to_constraint,
    point_ref_to_dict, dict_to_point_ref,
    save_sketch, load_sketch,
)


# =============================================================================
# Types Tests
# =============================================================================

class TestPoint2D:
    def test_creation(self):
        p = Point2D(3.0, 4.0)
        assert p.x == 3.0
        assert p.y == 4.0

    def test_distance(self):
        p1 = Point2D(0, 0)
        p2 = Point2D(3, 4)
        assert p1.distance_to(p2) == 5.0

    def test_midpoint(self):
        p1 = Point2D(0, 0)
        p2 = Point2D(10, 20)
        mid = p1.midpoint(p2)
        assert mid.x == 5.0
        assert mid.y == 10.0

    def test_add_vector(self):
        p = Point2D(1, 2)
        v = Vector2D(3, 4)
        result = p + v
        assert result.x == 4.0
        assert result.y == 6.0

    def test_subtract_point(self):
        p1 = Point2D(5, 7)
        p2 = Point2D(2, 3)
        v = p1 - p2
        assert v.dx == 3.0
        assert v.dy == 4.0


class TestVector2D:
    def test_magnitude(self):
        v = Vector2D(3, 4)
        assert v.magnitude == 5.0

    def test_normalized(self):
        v = Vector2D(3, 4)
        n = v.normalized()
        assert abs(n.magnitude - 1.0) < 1e-10

    def test_normalized_zero_vector(self):
        v = Vector2D(0, 0)
        n = v.normalized()
        assert n.dx == 0 and n.dy == 0

    def test_dot_product(self):
        v1 = Vector2D(1, 0)
        v2 = Vector2D(0, 1)
        assert v1.dot(v2) == 0.0

    def test_cross_product(self):
        v1 = Vector2D(1, 0)
        v2 = Vector2D(0, 1)
        assert v1.cross(v2) == 1.0

    def test_multiply_scalar(self):
        v = Vector2D(2, 3)
        result = v * 2
        assert result.dx == 4.0
        assert result.dy == 6.0

    def test_rmultiply_scalar(self):
        v = Vector2D(2, 3)
        result = 2 * v
        assert result.dx == 4.0
        assert result.dy == 6.0

    def test_add_vectors(self):
        v1 = Vector2D(1, 2)
        v2 = Vector2D(3, 4)
        result = v1 + v2
        assert result.dx == 4.0
        assert result.dy == 6.0

    def test_negate(self):
        v = Vector2D(3, -4)
        neg = -v
        assert neg.dx == -3.0
        assert neg.dy == 4.0


class TestElementId:
    def test_str(self):
        eid = ElementId("L", 5)
        assert str(eid) == "L5"

    def test_parse(self):
        eid = ElementId.parse("A12")
        assert eid.prefix == "A"
        assert eid.index == 12

    def test_parse_invalid_empty(self):
        with pytest.raises(ValueError):
            ElementId.parse("")

    def test_parse_invalid_short(self):
        with pytest.raises(ValueError):
            ElementId.parse("L")

    def test_parse_invalid_index(self):
        with pytest.raises(ValueError):
            ElementId.parse("Labc")


class TestPointRef:
    def test_str_basic(self):
        ref = PointRef("L0", PointType.START)
        assert str(ref) == "L0.start"

    def test_str_control(self):
        ref = PointRef("S0", PointType.CONTROL, index=2)
        assert str(ref) == "S0.control[2]"

    def test_str_on_curve(self):
        ref = PointRef("S0", PointType.ON_CURVE, parameter=0.5)
        assert str(ref) == "S0.on_curve(0.5)"


# =============================================================================
# Primitives Tests
# =============================================================================

class TestLine:
    def test_length(self):
        line = Line(start=Point2D(0, 0), end=Point2D(3, 4))
        assert line.length == 5.0

    def test_direction(self):
        line = Line(start=Point2D(0, 0), end=Point2D(10, 0))
        d = line.direction
        assert d.dx == 10.0
        assert d.dy == 0.0

    def test_midpoint(self):
        line = Line(start=Point2D(0, 0), end=Point2D(10, 20))
        mid = line.midpoint
        assert mid.x == 5.0
        assert mid.y == 10.0

    def test_get_point(self):
        line = Line(start=Point2D(0, 0), end=Point2D(10, 20))
        assert line.get_point(PointType.START) == Point2D(0, 0)
        assert line.get_point(PointType.END) == Point2D(10, 20)
        assert line.get_point(PointType.MIDPOINT) == Point2D(5, 10)

    def test_get_valid_point_types(self):
        line = Line(start=Point2D(0, 0), end=Point2D(10, 0))
        types = line.get_valid_point_types()
        assert PointType.START in types
        assert PointType.END in types
        assert PointType.MIDPOINT in types

    def test_invalid_point_type(self):
        line = Line(start=Point2D(0, 0), end=Point2D(10, 0))
        with pytest.raises(ValueError):
            line.get_point(PointType.CENTER)

    def test_default_values(self):
        line = Line()
        assert line.start == Point2D(0, 0)
        assert line.end == Point2D(0, 0)


class TestArc:
    def test_radius(self):
        arc = Arc(
            center=Point2D(0, 0),
            start_point=Point2D(10, 0),
            end_point=Point2D(0, 10),
            ccw=True
        )
        assert arc.radius == 10.0

    def test_start_angle(self):
        arc = Arc(
            center=Point2D(0, 0),
            start_point=Point2D(10, 0),
            end_point=Point2D(0, 10),
            ccw=True
        )
        assert abs(arc.start_angle - 0) < 1e-10

    def test_end_angle(self):
        arc = Arc(
            center=Point2D(0, 0),
            start_point=Point2D(10, 0),
            end_point=Point2D(0, 10),
            ccw=True
        )
        assert abs(arc.end_angle - math.pi/2) < 1e-10

    def test_sweep_angle_ccw(self):
        arc = Arc(
            center=Point2D(0, 0),
            start_point=Point2D(10, 0),
            end_point=Point2D(0, 10),
            ccw=True
        )
        assert abs(arc.sweep_angle - math.pi/2) < 1e-10

    def test_sweep_angle_cw(self):
        arc = Arc(
            center=Point2D(0, 0),
            start_point=Point2D(10, 0),
            end_point=Point2D(0, 10),
            ccw=False
        )
        assert abs(arc.sweep_angle - (-3*math.pi/2)) < 1e-10

    def test_sweep_angle_ccw_negative_delta(self):
        # Arc where end_angle < start_angle but still CCW
        arc = Arc(
            center=Point2D(0, 0),
            start_point=Point2D(0, 10),
            end_point=Point2D(10, 0),
            ccw=True
        )
        assert arc.sweep_angle > 0

    def test_sweep_angle_cw_positive_delta(self):
        # Arc where end_angle > start_angle but still CW
        arc = Arc(
            center=Point2D(0, 0),
            start_point=Point2D(0, 10),
            end_point=Point2D(10, 0),
            ccw=False
        )
        assert arc.sweep_angle < 0

    def test_arc_length(self):
        arc = Arc(
            center=Point2D(0, 0),
            start_point=Point2D(10, 0),
            end_point=Point2D(0, 10),
            ccw=True
        )
        expected = (math.pi / 2) * 10  # quarter circle
        assert abs(arc.arc_length - expected) < 1e-10

    def test_midpoint(self):
        arc = Arc(
            center=Point2D(0, 0),
            start_point=Point2D(10, 0),
            end_point=Point2D(0, 10),
            ccw=True
        )
        mid = arc.midpoint
        expected_x = 10 * math.cos(math.pi/4)
        expected_y = 10 * math.sin(math.pi/4)
        assert abs(mid.x - expected_x) < 1e-10
        assert abs(mid.y - expected_y) < 1e-10

    def test_get_point(self):
        arc = Arc(
            center=Point2D(5, 5),
            start_point=Point2D(10, 5),
            end_point=Point2D(5, 10),
            ccw=True
        )
        assert arc.get_point(PointType.START) == Point2D(10, 5)
        assert arc.get_point(PointType.END) == Point2D(5, 10)
        assert arc.get_point(PointType.CENTER) == Point2D(5, 5)

    def test_get_valid_point_types(self):
        arc = Arc(center=Point2D(0, 0), start_point=Point2D(5, 0), end_point=Point2D(0, 5), ccw=True)
        types = arc.get_valid_point_types()
        assert PointType.START in types
        assert PointType.END in types
        assert PointType.CENTER in types
        assert PointType.MIDPOINT in types

    def test_invalid_point_type(self):
        arc = Arc(center=Point2D(0, 0), start_point=Point2D(5, 0), end_point=Point2D(0, 5), ccw=True)
        with pytest.raises(ValueError):
            arc.get_point(PointType.CONTROL)

    def test_to_three_point(self):
        arc = Arc(
            center=Point2D(0, 0),
            start_point=Point2D(10, 0),
            end_point=Point2D(0, 10),
            ccw=True
        )
        start, mid, end = arc.to_three_point()
        assert start == arc.start_point
        assert end == arc.end_point

    def test_point_at_angle(self):
        arc = Arc(
            center=Point2D(0, 0),
            start_point=Point2D(10, 0),
            end_point=Point2D(0, 10),
            ccw=True
        )
        p = arc.point_at_angle(math.pi/2)
        assert abs(p.x) < 1e-10
        assert abs(p.y - 10) < 1e-10


class TestCircle:
    def test_diameter(self):
        c = Circle(center=Point2D(0, 0), radius=5)
        assert c.diameter == 10.0

    def test_circumference(self):
        c = Circle(center=Point2D(0, 0), radius=1)
        assert abs(c.circumference - 2*math.pi) < 1e-10

    def test_area(self):
        c = Circle(center=Point2D(0, 0), radius=2)
        assert abs(c.area - 4*math.pi) < 1e-10

    def test_get_point(self):
        c = Circle(center=Point2D(5, 10), radius=3)
        assert c.get_point(PointType.CENTER) == Point2D(5, 10)

    def test_get_valid_point_types(self):
        c = Circle(center=Point2D(0, 0), radius=5)
        types = c.get_valid_point_types()
        assert PointType.CENTER in types

    def test_invalid_point_type(self):
        c = Circle(center=Point2D(0, 0), radius=5)
        with pytest.raises(ValueError):
            c.get_point(PointType.START)

    def test_point_at_angle(self):
        c = Circle(center=Point2D(0, 0), radius=10)
        p = c.point_at_angle(math.pi/2)
        assert abs(p.x) < 1e-10
        assert abs(p.y - 10) < 1e-10


class TestPoint:
    def test_get_point(self):
        p = Point(position=Point2D(3, 4))
        assert p.get_point(PointType.CENTER) == Point2D(3, 4)

    def test_get_valid_point_types(self):
        p = Point(position=Point2D(0, 0))
        types = p.get_valid_point_types()
        assert PointType.CENTER in types

    def test_invalid_point_type(self):
        p = Point(position=Point2D(0, 0))
        with pytest.raises(ValueError):
            p.get_point(PointType.START)


class TestSpline:
    def test_create_uniform_bspline(self):
        spline = Spline.create_uniform_bspline([
            Point2D(0, 0),
            Point2D(10, 20),
            Point2D(20, 10),
            Point2D(30, 30)
        ], degree=3)
        assert spline.degree == 3
        assert len(spline.control_points) == 4
        assert len(spline.knots) == 8

    def test_order(self):
        spline = Spline(degree=3, control_points=[], knots=[])
        assert spline.order == 4

    def test_is_rational(self):
        spline1 = Spline(degree=3, control_points=[], knots=[])
        assert not spline1.is_rational

        spline2 = Spline(degree=3, control_points=[], knots=[], weights=[1.0, 1.0])
        assert spline2.is_rational

    def test_num_control_points(self):
        spline = Spline.create_uniform_bspline([
            Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)
        ])
        assert spline.num_control_points == 4

    def test_evaluate_endpoints(self):
        spline = Spline.create_uniform_bspline([
            Point2D(0, 0),
            Point2D(10, 20),
            Point2D(20, 10),
            Point2D(30, 0)
        ], degree=3)
        start = spline.evaluate(0)
        end = spline.evaluate(1)
        assert abs(start.x) < 1e-10
        assert abs(start.y) < 1e-10
        assert abs(end.x - 30) < 1e-10
        assert abs(end.y) < 1e-10

    def test_evaluate_middle(self):
        spline = Spline.create_uniform_bspline([
            Point2D(0, 0),
            Point2D(10, 20),
            Point2D(20, 10),
            Point2D(30, 0)
        ], degree=3)
        mid = spline.evaluate(0.5)
        # Just check it's a valid point
        assert 0 <= mid.x <= 30
        assert 0 <= mid.y <= 20

    def test_evaluate_empty_spline(self):
        spline = Spline(degree=3, control_points=[], knots=[])
        with pytest.raises(ValueError):
            spline.evaluate(0.5)

    def test_get_point_start(self):
        spline = Spline.create_uniform_bspline([
            Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)
        ])
        assert spline.get_point(PointType.START) == Point2D(0, 0)

    def test_get_point_end(self):
        spline = Spline.create_uniform_bspline([
            Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)
        ])
        assert spline.get_point(PointType.END) == Point2D(30, 10)

    def test_get_point_control_raises(self):
        spline = Spline.create_uniform_bspline([
            Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)
        ])
        with pytest.raises(ValueError):
            spline.get_point(PointType.CONTROL)

    def test_get_point_invalid(self):
        spline = Spline.create_uniform_bspline([
            Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)
        ])
        with pytest.raises(ValueError):
            spline.get_point(PointType.CENTER)

    def test_get_point_empty_spline(self):
        spline = Spline(degree=3, control_points=[], knots=[])
        with pytest.raises(ValueError):
            spline.get_point(PointType.START)

    def test_get_control_point(self):
        spline = Spline.create_uniform_bspline([
            Point2D(0, 0), Point2D(10, 20), Point2D(20, 10), Point2D(30, 0)
        ])
        assert spline.get_control_point(1) == Point2D(10, 20)

    def test_get_control_point_out_of_range(self):
        spline = Spline.create_uniform_bspline([
            Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)
        ])
        with pytest.raises(IndexError):
            spline.get_control_point(10)

    def test_get_valid_point_types(self):
        spline = Spline.create_uniform_bspline([
            Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)
        ])
        types = spline.get_valid_point_types()
        assert PointType.START in types
        assert PointType.END in types
        assert PointType.CONTROL in types

    def test_validate_knot_vector(self):
        spline = Spline.create_uniform_bspline([
            Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)
        ])
        assert spline.validate_knot_vector()

    def test_validate_knot_vector_invalid(self):
        spline = Spline(
            degree=3,
            control_points=[Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)],
            knots=[0, 0, 0, 1, 1, 1]  # Wrong number of knots
        )
        assert not spline.validate_knot_vector()

    def test_insufficient_points(self):
        with pytest.raises(ValueError):
            Spline.create_uniform_bspline([Point2D(0, 0), Point2D(1, 1)], degree=3)

    def test_create_with_construction(self):
        spline = Spline.create_uniform_bspline([
            Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)
        ], construction=True)
        assert spline.construction


# =============================================================================
# Constraints Tests
# =============================================================================

class TestConstraintBuilders:
    def test_coincident(self):
        c = Coincident(
            PointRef("L0", PointType.END),
            PointRef("L1", PointType.START)
        )
        assert c.constraint_type == ConstraintType.COINCIDENT
        assert len(c.references) == 2

    def test_tangent(self):
        c = Tangent("L0", "A0")
        assert c.constraint_type == ConstraintType.TANGENT
        assert c.references == ["L0", "A0"]

    def test_tangent_with_connection_point(self):
        c = Tangent("L0", "A0", at=PointRef("A0", PointType.START))
        assert c.connection_point is not None

    def test_perpendicular(self):
        c = Perpendicular("L0", "L1")
        assert c.constraint_type == ConstraintType.PERPENDICULAR

    def test_parallel(self):
        c = Parallel("L0", "L1")
        assert c.constraint_type == ConstraintType.PARALLEL

    def test_concentric(self):
        c = Concentric("C0", "C1")
        assert c.constraint_type == ConstraintType.CONCENTRIC

    def test_equal(self):
        c = Equal("L0", "L1", "L2")
        assert c.constraint_type == ConstraintType.EQUAL
        assert len(c.references) == 3

    def test_equal_insufficient(self):
        with pytest.raises(ValueError):
            Equal("L0")

    def test_collinear(self):
        c = Collinear("L0", "L1")
        assert c.constraint_type == ConstraintType.COLLINEAR

    def test_collinear_insufficient(self):
        with pytest.raises(ValueError):
            Collinear("L0")

    def test_horizontal(self):
        c = Horizontal("L0")
        assert c.constraint_type == ConstraintType.HORIZONTAL

    def test_vertical(self):
        c = Vertical("L0")
        assert c.constraint_type == ConstraintType.VERTICAL

    def test_fixed(self):
        c = Fixed("L0")
        assert c.constraint_type == ConstraintType.FIXED

    def test_distance(self):
        c = Distance(
            PointRef("L0", PointType.START),
            PointRef("L1", PointType.END),
            100.0
        )
        assert c.constraint_type == ConstraintType.DISTANCE
        assert c.value == 100.0

    def test_distance_x(self):
        c = DistanceX(PointRef("L0", PointType.START), 50.0)
        assert c.constraint_type == ConstraintType.DISTANCE_X

    def test_distance_x_two_points(self):
        c = DistanceX(
            PointRef("L0", PointType.START),
            50.0,
            PointRef("L1", PointType.END)
        )
        assert len(c.references) == 2

    def test_distance_y(self):
        c = DistanceY(PointRef("L0", PointType.START), 50.0)
        assert c.constraint_type == ConstraintType.DISTANCE_Y

    def test_length(self):
        c = Length("L0", 100.0)
        assert c.constraint_type == ConstraintType.LENGTH
        assert c.value == 100.0

    def test_radius(self):
        c = Radius("C0", 10.0)
        assert c.constraint_type == ConstraintType.RADIUS
        assert c.value == 10.0

    def test_diameter(self):
        c = Diameter("C0", 20.0)
        assert c.constraint_type == ConstraintType.DIAMETER
        assert c.value == 20.0

    def test_angle(self):
        c = Angle("L0", "L1", 45.0)
        assert c.constraint_type == ConstraintType.ANGLE
        assert c.value == 45.0

    def test_symmetric(self):
        c = Symmetric("L0", "L1", "L2")
        assert c.constraint_type == ConstraintType.SYMMETRIC

    def test_midpoint_constraint(self):
        c = MidpointConstraint(PointRef("P0", PointType.CENTER), "L0")
        assert c.constraint_type == ConstraintType.MIDPOINT


class TestSketchConstraint:
    def test_str_without_value(self):
        c = Horizontal("L0", id="c1")
        assert "horizontal" in str(c)
        assert "L0" in str(c)

    def test_str_with_value(self):
        c = Length("L0", 100.0, id="c1")
        assert "length" in str(c)
        assert "100.0" in str(c)

    def test_get_element_ids(self):
        c = Coincident(
            PointRef("L0", PointType.END),
            PointRef("A1", PointType.START),
            id="c1"
        )
        ids = c.get_element_ids()
        assert "L0" in ids
        assert "A1" in ids

    def test_get_element_ids_with_connection_point(self):
        c = Tangent("L0", "A1", at=PointRef("A1", PointType.START), id="c1")
        ids = c.get_element_ids()
        assert "L0" in ids
        assert "A1" in ids


# =============================================================================
# Document Tests
# =============================================================================

class TestSketchDocument:
    def test_add_primitive(self):
        doc = SketchDocument(name="Test")
        line_id = doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        assert line_id == "L0"
        assert "L0" in doc.primitives

    def test_add_multiple_primitives(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        doc.add_primitive(Line(start=Point2D(10, 0), end=Point2D(10, 10)))
        doc.add_primitive(Arc(center=Point2D(5, 5), start_point=Point2D(5, 0), end_point=Point2D(0, 5), ccw=True))
        doc.add_primitive(Circle(center=Point2D(5, 5), radius=2))
        doc.add_primitive(Point(position=Point2D(7, 7)))
        doc.add_primitive(Spline.create_uniform_bspline([
            Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)
        ]))
        assert len(doc.primitives) == 6

    def test_add_primitive_with_id(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive_with_id(Line(start=Point2D(0, 0), end=Point2D(10, 0)), "L5")
        assert "L5" in doc.primitives
        # Check next_index was updated
        next_line_id = doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(5, 5)))
        assert next_line_id == "L6"

    def test_add_primitive_with_id_duplicate(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        with pytest.raises(ValueError):
            doc.add_primitive_with_id(Line(start=Point2D(0, 0), end=Point2D(5, 5)), "L0")

    def test_remove_primitive(self):
        doc = SketchDocument(name="Test")
        line_id = doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        assert doc.remove_primitive(line_id)
        assert line_id not in doc.primitives

    def test_remove_primitive_nonexistent(self):
        doc = SketchDocument(name="Test")
        assert not doc.remove_primitive("L99")

    def test_remove_primitive_removes_constraints(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        doc.add_constraint(Horizontal("L0"))
        doc.remove_primitive("L0")
        assert len(doc.constraints) == 0

    def test_get_primitive(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        prim = doc.get_primitive("L0")
        assert prim is not None
        assert isinstance(prim, Line)

    def test_get_primitive_nonexistent(self):
        doc = SketchDocument(name="Test")
        assert doc.get_primitive("L99") is None

    def test_get_point(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(5, 10), end=Point2D(15, 20)))
        pt = doc.get_point(PointRef("L0", PointType.START))
        assert pt == Point2D(5, 10)

    def test_get_point_nonexistent_element(self):
        doc = SketchDocument(name="Test")
        with pytest.raises(KeyError):
            doc.get_point(PointRef("L99", PointType.START))

    def test_get_point_spline_control(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Spline.create_uniform_bspline([
            Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)
        ]))
        pt = doc.get_point(PointRef("S0", PointType.CONTROL, index=1))
        assert pt == Point2D(10, 10)

    def test_add_constraint(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        doc.add_constraint(Horizontal("L0"))
        assert len(doc.constraints) == 1
        assert doc.solver_status == SolverStatus.DIRTY

    def test_add_constraint_invalid_reference(self):
        doc = SketchDocument(name="Test")
        with pytest.raises(KeyError):
            doc.add_constraint(Horizontal("L99"))

    def test_remove_constraint(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        c = Horizontal("L0", id="c1")
        doc.add_constraint(c)
        assert doc.remove_constraint("c1")
        assert len(doc.constraints) == 0

    def test_remove_constraint_nonexistent(self):
        doc = SketchDocument(name="Test")
        assert not doc.remove_constraint("c99")

    def test_get_constraint(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        doc.add_constraint(Horizontal("L0", id="c1"))
        c = doc.get_constraint("c1")
        assert c is not None
        assert c.constraint_type == ConstraintType.HORIZONTAL

    def test_get_constraint_nonexistent(self):
        doc = SketchDocument(name="Test")
        assert doc.get_constraint("c99") is None

    def test_get_constraints_for(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        doc.add_primitive(Line(start=Point2D(10, 0), end=Point2D(10, 10)))
        doc.add_constraint(Horizontal("L0"))
        doc.add_constraint(Vertical("L1"))
        constraints = doc.get_constraints_for("L0")
        assert len(constraints) == 1

    def test_get_primitives_by_type(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        doc.add_primitive(Circle(center=Point2D(5, 5), radius=2))
        doc.add_primitive(Line(start=Point2D(10, 0), end=Point2D(10, 10)))
        lines = doc.get_primitives_by_type(Line)
        assert len(lines) == 2

    def test_get_lines(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        doc.add_primitive(Circle(center=Point2D(5, 5), radius=2))
        assert len(doc.get_lines()) == 1

    def test_get_arcs(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Arc(center=Point2D(0, 0), start_point=Point2D(5, 0), end_point=Point2D(0, 5), ccw=True))
        assert len(doc.get_arcs()) == 1

    def test_get_circles(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Circle(center=Point2D(5, 5), radius=2))
        assert len(doc.get_circles()) == 1

    def test_get_splines(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Spline.create_uniform_bspline([
            Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)
        ]))
        assert len(doc.get_splines()) == 1

    def test_get_construction_geometry(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0), construction=True))
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(0, 10)))
        assert len(doc.get_construction_geometry()) == 1

    def test_get_profile_geometry(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0), construction=True))
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(0, 10)))
        assert len(doc.get_profile_geometry()) == 1

    def test_clear(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        doc.add_constraint(Horizontal("L0"))
        doc.clear()
        assert len(doc.primitives) == 0
        assert len(doc.constraints) == 0

    def test_to_text_description(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        doc.add_constraint(Horizontal("L0"))
        text = doc.to_text_description()
        assert "Test" in text
        assert "L0" in text
        assert "horizontal" in text

    def test_to_text_description_with_coords(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        doc.add_primitive(Spline.create_uniform_bspline([
            Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)
        ]))
        text = doc.to_text_description(include_point_coords=True)
        assert "start:" in text
        assert "control[" in text

    def test_to_text_description_all_primitives(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        doc.add_primitive(Arc(center=Point2D(10, 5), start_point=Point2D(10, 0), end_point=Point2D(15, 5), ccw=True))
        doc.add_primitive(Circle(center=Point2D(5, 5), radius=2))
        doc.add_primitive(Point(position=Point2D(7, 7)))
        doc.add_primitive(Spline.create_uniform_bspline([
            Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)
        ]))
        doc.add_primitive(Line(start=Point2D(0, 10), end=Point2D(10, 10), construction=True))
        text = doc.to_text_description()
        assert "Line" in text
        assert "Arc" in text
        assert "Circle" in text
        assert "Point" in text
        assert "Spline" in text
        assert "[C]" in text

    def test_repr(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        r = repr(doc)
        assert "Test" in r
        assert "primitives=1" in r


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidationResult:
    def test_add_error(self):
        result = ValidationResult()
        result.add_error("Test error", "ERR001", "L0")
        assert len(result.errors) == 1
        assert not result.is_valid

    def test_add_warning(self):
        result = ValidationResult()
        result.add_warning("Test warning", "WARN001")
        assert len(result.warnings) == 1
        assert result.is_valid
        assert result.has_warnings

    def test_add_info(self):
        result = ValidationResult()
        result.add_info("Test info", "INFO001")
        assert len(result.issues) == 1
        assert result.is_valid

    def test_str_no_issues(self):
        result = ValidationResult()
        assert "no issues" in str(result).lower()

    def test_str_with_issues(self):
        result = ValidationResult()
        result.add_error("Test error", "ERR001")
        s = str(result)
        assert "1 issue" in s

    def test_bool(self):
        result = ValidationResult()
        assert bool(result) is True
        result.add_error("Error", "ERR")
        assert bool(result) is False


class TestValidation:
    def test_valid_sketch(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        doc.add_primitive(Circle(center=Point2D(5, 5), radius=3))
        result = validate_sketch(doc)
        assert result.is_valid

    def test_empty_primitive_id(self):
        doc = SketchDocument(name="Test")
        line = Line(start=Point2D(0, 0), end=Point2D(10, 0))
        line.id = ""
        doc.primitives[""] = line
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_confidence_out_of_range(self):
        doc = SketchDocument(name="Test")
        line = Line(start=Point2D(0, 0), end=Point2D(10, 0))
        line.confidence = 1.5
        doc.add_primitive(line)
        result = validate_sketch(doc)
        assert result.has_warnings

    def test_zero_length_line(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(0, 0)))
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_line_nan_coords(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(float('nan'), 0), end=Point2D(10, 0)))
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_invalid_arc_radius(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Arc(
            center=Point2D(0, 0),
            start_point=Point2D(10, 0),
            end_point=Point2D(0, 20),
            ccw=True
        ))
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_arc_zero_radius(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Arc(
            center=Point2D(0, 0),
            start_point=Point2D(0, 0),
            end_point=Point2D(0, 0),
            ccw=True
        ))
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_arc_degenerate(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Arc(
            center=Point2D(0, 0),
            start_point=Point2D(10, 0),
            end_point=Point2D(10, 0),
            ccw=True
        ))
        result = validate_sketch(doc)
        assert result.has_warnings

    def test_arc_nan_coords(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Arc(
            center=Point2D(float('nan'), 0),
            start_point=Point2D(10, 0),
            end_point=Point2D(0, 10),
            ccw=True
        ))
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_negative_radius(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Circle(center=Point2D(0, 0), radius=-5))
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_tiny_radius(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Circle(center=Point2D(0, 0), radius=0.0001))
        result = validate_sketch(doc)
        assert result.has_warnings

    def test_circle_nan_values(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Circle(center=Point2D(0, 0), radius=float('nan')))
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_point_nan_coords(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Point(position=Point2D(float('inf'), 0)))
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_spline_invalid_degree(self):
        doc = SketchDocument(name="Test")
        spline = Spline(degree=0, control_points=[Point2D(0, 0)], knots=[])
        spline.id = "S0"
        doc.primitives["S0"] = spline
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_spline_insufficient_points(self):
        doc = SketchDocument(name="Test")
        spline = Spline(degree=3, control_points=[Point2D(0, 0), Point2D(1, 1)], knots=[])
        spline.id = "S0"
        doc.primitives["S0"] = spline
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_spline_nan_control_point(self):
        doc = SketchDocument(name="Test")
        spline = Spline.create_uniform_bspline([
            Point2D(float('nan'), 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)
        ])
        doc.add_primitive(spline)
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_spline_non_monotonic_knots(self):
        doc = SketchDocument(name="Test")
        spline = Spline(
            degree=3,
            control_points=[Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)],
            knots=[0, 0, 0, 0, 0.5, 0.3, 1, 1]  # Non-monotonic
        )
        spline.id = "S0"
        doc.primitives["S0"] = spline
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_spline_invalid_weights(self):
        doc = SketchDocument(name="Test")
        spline = Spline(
            degree=3,
            control_points=[Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)],
            knots=[0, 0, 0, 0, 1, 1, 1, 1],
            weights=[1, 1, -1, 1]  # Negative weight
        )
        spline.id = "S0"
        doc.primitives["S0"] = spline
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_spline_weights_count_mismatch(self):
        doc = SketchDocument(name="Test")
        spline = Spline(
            degree=3,
            control_points=[Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)],
            knots=[0, 0, 0, 0, 1, 1, 1, 1],
            weights=[1, 1, 1]  # Wrong count
        )
        spline.id = "S0"
        doc.primitives["S0"] = spline
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_spline_nan_weight(self):
        doc = SketchDocument(name="Test")
        spline = Spline(
            degree=3,
            control_points=[Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)],
            knots=[0, 0, 0, 0, 1, 1, 1, 1],
            weights=[1, 1, float('nan'), 1]
        )
        spline.id = "S0"
        doc.primitives["S0"] = spline
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_constraint_empty_id(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        c = Horizontal("L0", id="")
        doc.constraints.append(c)
        result = validate_sketch(doc)
        assert result.has_warnings

    def test_constraint_too_few_refs(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        c = SketchConstraint(
            id="c1",
            constraint_type=ConstraintType.COINCIDENT,
            references=[PointRef("L0", PointType.START)]  # Need 2
        )
        doc.constraints.append(c)
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_constraint_too_many_refs(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        c = SketchConstraint(
            id="c1",
            constraint_type=ConstraintType.HORIZONTAL,
            references=["L0", "L0"]  # Max 1
        )
        doc.constraints.append(c)
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_constraint_missing_value(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        c = SketchConstraint(
            id="c1",
            constraint_type=ConstraintType.LENGTH,
            references=["L0"],
            value=None  # Required
        )
        doc.constraints.append(c)
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_constraint_nan_value(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        c = SketchConstraint(
            id="c1",
            constraint_type=ConstraintType.LENGTH,
            references=["L0"],
            value=float('nan')
        )
        doc.constraints.append(c)
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_constraint_negative_length(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        c = Length("L0", -10.0)
        doc.constraints.append(c)
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_constraint_invalid_element_ref(self):
        doc = SketchDocument(name="Test")
        c = Horizontal("L99")
        doc.constraints.append(c)
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_constraint_invalid_point_ref(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        c = SketchConstraint(
            id="c1",
            constraint_type=ConstraintType.COINCIDENT,
            references=[
                PointRef("L0", PointType.CENTER),  # Invalid for Line
                PointRef("L0", PointType.START)
            ]
        )
        doc.constraints.append(c)
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_constraint_control_point_missing_index(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Spline.create_uniform_bspline([
            Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)
        ]))
        c = SketchConstraint(
            id="c1",
            constraint_type=ConstraintType.FIXED,
            references=[PointRef("S0", PointType.CONTROL)]  # Missing index
        )
        doc.constraints.append(c)
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_constraint_control_point_out_of_range(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Spline.create_uniform_bspline([
            Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)
        ]))
        c = SketchConstraint(
            id="c1",
            constraint_type=ConstraintType.FIXED,
            references=[PointRef("S0", PointType.CONTROL, index=99)]
        )
        doc.constraints.append(c)
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_constraint_invalid_connection_point(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        doc.add_primitive(Arc(center=Point2D(10, 5), start_point=Point2D(10, 0), end_point=Point2D(15, 5), ccw=True))
        c = Tangent("L0", "A0", at=PointRef("L99", PointType.START))
        doc.constraints.append(c)
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_constraint_invalid_connection_point_type(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        doc.add_primitive(Arc(center=Point2D(10, 5), start_point=Point2D(10, 0), end_point=Point2D(15, 5), ccw=True))
        c = Tangent("L0", "A0", at=PointRef("L0", PointType.CENTER))  # Invalid for Line
        doc.constraints.append(c)
        result = validate_sketch(doc)
        assert not result.is_valid

    def test_constraint_confidence_out_of_range(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        c = Horizontal("L0")
        c.confidence = 2.0
        doc.constraints.append(c)
        result = validate_sketch(doc)
        assert result.has_warnings

    def test_duplicate_constraint_ids(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(0, 10)))
        doc.constraints.append(Horizontal("L0", id="c1"))
        doc.constraints.append(Vertical("L1", id="c1"))
        result = validate_sketch(doc)
        assert result.has_warnings

    def test_validate_primitive_function(self):
        line = Line(start=Point2D(0, 0), end=Point2D(0, 0))
        errors = validate_primitive(line)
        assert len(errors) > 0

    def test_validate_constraint_function(self):
        doc = SketchDocument(name="Test")
        c = Horizontal("L99")
        errors = validate_constraint(c, doc)
        assert len(errors) > 0


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerialization:
    def test_line_round_trip(self):
        line = Line(id="L0", start=Point2D(1, 2), end=Point2D(3, 4), construction=True)
        d = primitive_to_dict(line)
        line2 = dict_to_primitive(d)
        assert line2.id == line.id
        assert line2.start == line.start
        assert line2.end == line.end
        assert line2.construction == line.construction

    def test_line_with_metadata(self):
        line = Line(id="L0", start=Point2D(1, 2), end=Point2D(3, 4))
        line.source = "fitted"
        line.confidence = 0.95
        d = primitive_to_dict(line)
        line2 = dict_to_primitive(d)
        assert line2.source == "fitted"
        assert line2.confidence == 0.95

    def test_arc_round_trip(self):
        arc = Arc(
            id="A0",
            center=Point2D(0, 0),
            start_point=Point2D(10, 0),
            end_point=Point2D(0, 10),
            ccw=True
        )
        d = primitive_to_dict(arc)
        arc2 = dict_to_primitive(d)
        assert arc2.center == arc.center
        assert arc2.start_point == arc.start_point
        assert arc2.end_point == arc.end_point
        assert arc2.ccw == arc.ccw

    def test_arc_ccw_false(self):
        arc = Arc(
            id="A0",
            center=Point2D(0, 0),
            start_point=Point2D(10, 0),
            end_point=Point2D(0, 10),
            ccw=False
        )
        d = primitive_to_dict(arc)
        arc2 = dict_to_primitive(d)
        assert arc2.ccw is False

    def test_circle_round_trip(self):
        circle = Circle(id="C0", center=Point2D(5, 5), radius=10)
        d = primitive_to_dict(circle)
        circle2 = dict_to_primitive(d)
        assert circle2.center == circle.center
        assert circle2.radius == circle.radius

    def test_point_round_trip(self):
        point = Point(id="P0", position=Point2D(7, 8))
        d = primitive_to_dict(point)
        point2 = dict_to_primitive(d)
        assert point2.position == point.position

    def test_spline_round_trip(self):
        spline = Spline.create_uniform_bspline([
            Point2D(0, 0),
            Point2D(10, 20),
            Point2D(20, 10),
            Point2D(30, 0)
        ])
        spline.id = "S0"
        d = primitive_to_dict(spline)
        spline2 = dict_to_primitive(d)
        assert spline2.degree == spline.degree
        assert len(spline2.control_points) == len(spline.control_points)
        assert spline2.knots == spline.knots
        assert spline2.periodic == spline.periodic
        assert spline2.is_fit_spline == spline.is_fit_spline

    def test_spline_with_weights(self):
        spline = Spline(
            id="S0",
            degree=3,
            control_points=[Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)],
            knots=[0, 0, 0, 0, 1, 1, 1, 1],
            weights=[1, 2, 1, 1]
        )
        d = primitive_to_dict(spline)
        spline2 = dict_to_primitive(d)
        assert spline2.weights == [1, 2, 1, 1]

    def test_unknown_primitive_type(self):
        with pytest.raises(ValueError):
            dict_to_primitive({"type": "unknown", "id": "X0"})

    def test_constraint_round_trip(self):
        c = Horizontal("L0", id="c1")
        d = constraint_to_dict(c)
        c2 = dict_to_constraint(d)
        assert c2.id == c.id
        assert c2.constraint_type == c.constraint_type

    def test_constraint_with_value(self):
        c = Length("L0", 100.0, id="c1")
        d = constraint_to_dict(c)
        c2 = dict_to_constraint(d)
        assert c2.value == 100.0

    def test_constraint_with_point_refs(self):
        c = Coincident(
            PointRef("L0", PointType.END),
            PointRef("L1", PointType.START),
            id="c1"
        )
        d = constraint_to_dict(c)
        c2 = dict_to_constraint(d)
        assert len(c2.references) == 2
        assert isinstance(c2.references[0], PointRef)

    def test_constraint_with_connection_point(self):
        c = Tangent("L0", "A0", at=PointRef("A0", PointType.START), id="c1")
        d = constraint_to_dict(c)
        c2 = dict_to_constraint(d)
        assert c2.connection_point is not None
        assert c2.connection_point.element_id == "A0"

    def test_constraint_with_metadata(self):
        c = Horizontal("L0", id="c1", inferred=True, confidence=0.9, source="ai")
        d = constraint_to_dict(c)
        c2 = dict_to_constraint(d)
        assert c2.inferred is True
        assert c2.confidence == 0.9
        assert c2.source == "ai"

    def test_constraint_with_status(self):
        c = Horizontal("L0", id="c1")
        c.status = ConstraintStatus.SATISFIED
        d = constraint_to_dict(c)
        c2 = dict_to_constraint(d)
        assert c2.status == ConstraintStatus.SATISFIED

    def test_point_ref_round_trip(self):
        ref = PointRef("L0", PointType.END)
        d = point_ref_to_dict(ref)
        ref2 = dict_to_point_ref(d)
        assert ref2.element_id == ref.element_id
        assert ref2.point_type == ref.point_type

    def test_point_ref_with_parameter(self):
        ref = PointRef("S0", PointType.ON_CURVE, parameter=0.5)
        d = point_ref_to_dict(ref)
        ref2 = dict_to_point_ref(d)
        assert ref2.parameter == 0.5

    def test_point_ref_with_index(self):
        ref = PointRef("S0", PointType.CONTROL, index=2)
        d = point_ref_to_dict(ref)
        ref2 = dict_to_point_ref(d)
        assert ref2.index == 2

    def test_sketch_round_trip(self):
        doc = SketchDocument(name="RoundTrip")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        doc.add_primitive(Arc(center=Point2D(10, 5), start_point=Point2D(10, 0), end_point=Point2D(15, 5), ccw=True))
        doc.add_constraint(Horizontal("L0"))
        doc.add_constraint(Tangent("L0", "A0"))

        json_str = sketch_to_json(doc)
        doc2 = sketch_from_json(json_str)

        assert doc2.name == doc.name
        assert len(doc2.primitives) == len(doc.primitives)
        assert len(doc2.constraints) == len(doc.constraints)

    def test_sketch_to_dict_and_back(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        doc.solver_status = SolverStatus.FULLY_CONSTRAINED
        doc.degrees_of_freedom = 0

        d = sketch_to_dict(doc)
        doc2 = dict_to_sketch(d)

        assert doc2.solver_status == SolverStatus.FULLY_CONSTRAINED
        assert doc2.degrees_of_freedom == 0

    def test_sketch_encoder(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        json_str = json.dumps(doc, cls=SketchEncoder)
        assert "Test" in json_str

    def test_sketch_encoder_primitives(self):
        line = Line(id="L0", start=Point2D(0, 0), end=Point2D(10, 0))
        json_str = json.dumps(line, cls=SketchEncoder)
        assert "L0" in json_str

    def test_sketch_encoder_constraint(self):
        c = Horizontal("L0", id="c1")
        json_str = json.dumps(c, cls=SketchEncoder)
        assert "horizontal" in json_str

    def test_sketch_encoder_point2d(self):
        p = Point2D(1, 2)
        json_str = json.dumps(p, cls=SketchEncoder)
        assert "[1, 2]" in json_str

    def test_sketch_encoder_point_ref(self):
        ref = PointRef("L0", PointType.START)
        json_str = json.dumps(ref, cls=SketchEncoder)
        assert "L0" in json_str

    def test_sketch_encoder_enums(self):
        json_str = json.dumps(ConstraintType.HORIZONTAL, cls=SketchEncoder)
        assert "horizontal" in json_str

    def test_save_and_load(self, tmp_path):
        doc = SketchDocument(name="SaveTest")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        doc.add_constraint(Horizontal("L0"))

        filepath = tmp_path / "test_sketch.json"
        save_sketch(doc, str(filepath))

        doc2 = load_sketch(str(filepath))
        assert doc2.name == doc.name
        assert len(doc2.primitives) == 1
        assert len(doc2.constraints) == 1

    def test_json_compact(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        json_compact = sketch_to_json(doc, indent=None)
        assert "\n" not in json_compact


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
