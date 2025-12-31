"""Tests for the canonical sketch schema."""

import pytest
import math

from sketch_canonical import (
    # Types
    Point2D, Vector2D, ElementId, PointType, PointRef, ElementPrefix,
    # Primitives
    Line, Arc, Circle, Point, Spline,
    # Constraints
    ConstraintType, SketchConstraint,
    Coincident, Tangent, Horizontal, Vertical, Radius, Length, Distance,
    # Document
    SketchDocument, SolverStatus,
    # Validation
    validate_sketch, validate_primitive, ValidationResult,
    # Serialization
    sketch_to_json, sketch_from_json, primitive_to_dict, dict_to_primitive,
)


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

    def test_dot_product(self):
        v1 = Vector2D(1, 0)
        v2 = Vector2D(0, 1)
        assert v1.dot(v2) == 0.0

    def test_cross_product(self):
        v1 = Vector2D(1, 0)
        v2 = Vector2D(0, 1)
        assert v1.cross(v2) == 1.0


class TestElementId:
    def test_str(self):
        eid = ElementId("L", 5)
        assert str(eid) == "L5"

    def test_parse(self):
        eid = ElementId.parse("A12")
        assert eid.prefix == "A"
        assert eid.index == 12

    def test_parse_invalid(self):
        with pytest.raises(ValueError):
            ElementId.parse("")


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

    def test_invalid_point_type(self):
        line = Line(start=Point2D(0, 0), end=Point2D(10, 0))
        with pytest.raises(ValueError):
            line.get_point(PointType.CENTER)


class TestArc:
    def test_radius(self):
        arc = Arc(
            center=Point2D(0, 0),
            start_point=Point2D(10, 0),
            end_point=Point2D(0, 10),
            ccw=True
        )
        assert arc.radius == 10.0

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


class TestCircle:
    def test_diameter(self):
        c = Circle(center=Point2D(0, 0), radius=5)
        assert c.diameter == 10.0

    def test_circumference(self):
        c = Circle(center=Point2D(0, 0), radius=1)
        assert abs(c.circumference - 2*math.pi) < 1e-10

    def test_point_at_angle(self):
        c = Circle(center=Point2D(0, 0), radius=10)
        p = c.point_at_angle(math.pi/2)
        assert abs(p.x) < 1e-10
        assert abs(p.y - 10) < 1e-10


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
        assert len(spline.knots) == 8  # n + k = 4 + 4

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

    def test_get_control_point(self):
        spline = Spline.create_uniform_bspline([
            Point2D(0, 0),
            Point2D(10, 20),
            Point2D(20, 10),
            Point2D(30, 0)
        ])
        assert spline.get_control_point(1) == Point2D(10, 20)

    def test_insufficient_points(self):
        with pytest.raises(ValueError):
            Spline.create_uniform_bspline([Point2D(0, 0), Point2D(1, 1)], degree=3)


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
        assert len(doc.primitives) == 3
        assert "L0" in doc.primitives
        assert "L1" in doc.primitives
        assert "A0" in doc.primitives

    def test_remove_primitive(self):
        doc = SketchDocument(name="Test")
        line_id = doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        assert doc.remove_primitive(line_id)
        assert line_id not in doc.primitives

    def test_get_point(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(5, 10), end=Point2D(15, 20)))
        pt = doc.get_point(PointRef("L0", PointType.START))
        assert pt == Point2D(5, 10)

    def test_add_constraint(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        doc.add_constraint(Horizontal("L0"))
        assert len(doc.constraints) == 1

    def test_constraint_references_invalid_element(self):
        doc = SketchDocument(name="Test")
        with pytest.raises(KeyError):
            doc.add_constraint(Horizontal("L99"))


class TestConstraints:
    def test_coincident(self):
        c = Coincident(
            PointRef("L0", PointType.END),
            PointRef("L1", PointType.START)
        )
        assert c.constraint_type == ConstraintType.COINCIDENT
        assert len(c.references) == 2

    def test_tangent_with_connection_point(self):
        c = Tangent("L0", "A0", at=PointRef("A0", PointType.START))
        assert c.constraint_type == ConstraintType.TANGENT
        assert c.connection_point is not None

    def test_radius(self):
        c = Radius("C0", 10.0)
        assert c.constraint_type == ConstraintType.RADIUS
        assert c.value == 10.0


class TestValidation:
    def test_valid_sketch(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        doc.add_primitive(Circle(center=Point2D(5, 5), radius=3))
        result = validate_sketch(doc)
        assert result.is_valid

    def test_zero_length_line(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Line(start=Point2D(0, 0), end=Point2D(0, 0)))
        result = validate_sketch(doc)
        assert not result.is_valid
        assert any("zero length" in str(e).lower() for e in result.errors)

    def test_invalid_arc_radius(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Arc(
            center=Point2D(0, 0),
            start_point=Point2D(10, 0),
            end_point=Point2D(0, 20),  # Different radius
            ccw=True
        ))
        result = validate_sketch(doc)
        assert not result.is_valid
        assert any("inconsistent" in str(e).lower() for e in result.errors)

    def test_negative_radius(self):
        doc = SketchDocument(name="Test")
        doc.add_primitive(Circle(center=Point2D(0, 0), radius=-5))
        result = validate_sketch(doc)
        assert not result.is_valid


class TestSerialization:
    def test_line_round_trip(self):
        line = Line(id="L0", start=Point2D(1, 2), end=Point2D(3, 4), construction=True)
        d = primitive_to_dict(line)
        line2 = dict_to_primitive(d)
        assert line2.id == line.id
        assert line2.start == line.start
        assert line2.end == line.end
        assert line2.construction == line.construction

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
