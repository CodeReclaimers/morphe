#include "doctest/doctest.h"

#include "morphe/morphe.hpp"

#include <algorithm>
#include <cmath>

namespace m = morphe;

namespace {

bool has_code(const m::ValidationResult& r, std::string_view code) {
    for (const auto& i : r.issues) {
        if (i.code == code) return true;
    }
    return false;
}

}  // namespace

TEST_CASE("Empty document validates clean") {
    m::SketchDocument d;
    const auto r = m::validate(d);
    CHECK(r.is_valid());
    CHECK(r.issues.empty());
}

TEST_CASE("Default tolerance is 1 micron") {
    CHECK(m::default_tolerance == 0.001);
}

// ---------- primitive validation ----------

TEST_CASE("Line: zero length is an error") {
    m::SketchDocument d;
    m::Line l;
    l.start = {0.0, 0.0};
    l.end   = {0.0, 0.0};
    d.add_primitive(l);
    const auto r = m::validate(d);
    CHECK_FALSE(r.is_valid());
    CHECK(has_code(r, "LINE_ZERO_LENGTH"));
}

TEST_CASE("Line: micro-length below tolerance is an error") {
    m::SketchDocument d;
    m::Line l;
    l.start = {0.0, 0.0};
    l.end   = {0.0005, 0.0};  // half a micron
    d.add_primitive(l);
    const auto r = m::validate(d);
    CHECK(has_code(r, "LINE_ZERO_LENGTH"));
}

TEST_CASE("Line: NaN coordinates flagged") {
    m::SketchDocument d;
    m::Line l;
    l.start = {std::nan(""), 0.0};
    l.end   = {1.0, 0.0};
    d.add_primitive(l);
    const auto r = m::validate(d);
    CHECK(has_code(r, "LINE_INVALID_COORDS"));
}

TEST_CASE("Arc: inconsistent radius is an error") {
    m::SketchDocument d;
    m::Arc a;
    a.center      = {0.0, 0.0};
    a.start_point = {5.0, 0.0};
    a.end_point   = {0.0, 7.0};  // distance 7, mismatched
    d.add_primitive(a);
    const auto r = m::validate(d);
    CHECK(has_code(r, "ARC_RADIUS_INCONSISTENT"));
}

TEST_CASE("Arc: zero radius flagged") {
    m::SketchDocument d;
    m::Arc a;
    a.center      = {0.0, 0.0};
    a.start_point = {0.0, 0.0};
    a.end_point   = {0.0, 0.0};
    d.add_primitive(a);
    const auto r = m::validate(d);
    CHECK(has_code(r, "ARC_ZERO_RADIUS"));
}

TEST_CASE("Arc: degenerate (start == end) is a warning, not error") {
    m::SketchDocument d;
    m::Arc a;
    a.center      = {0.0, 0.0};
    a.start_point = {5.0, 0.0};
    a.end_point   = {5.0, 0.0};
    d.add_primitive(a);
    const auto r = m::validate(d);
    CHECK(has_code(r, "ARC_DEGENERATE"));
    // Has a warning but should still be considered valid (no errors)
    CHECK(r.is_valid());
    CHECK(r.has_warnings());
}

TEST_CASE("Arc: valid arc passes") {
    m::SketchDocument d;
    m::Arc a;
    a.center      = {0.0, 0.0};
    a.start_point = {5.0, 0.0};
    a.end_point   = {0.0, 5.0};
    d.add_primitive(a);
    const auto r = m::validate(d);
    CHECK(r.is_valid());
    CHECK_FALSE(has_code(r, "ARC_RADIUS_INCONSISTENT"));
}

TEST_CASE("Circle: non-positive radius is an error") {
    m::SketchDocument d;
    m::Circle c;
    c.center = {0.0, 0.0};
    c.radius = 0.0;
    d.add_primitive(c);
    const auto r = m::validate(d);
    CHECK(has_code(r, "CIRCLE_INVALID_RADIUS"));

    m::SketchDocument d2;
    m::Circle c2;
    c2.center = {0.0, 0.0};
    c2.radius = -1.0;
    d2.add_primitive(c2);
    const auto r2 = m::validate(d2);
    CHECK(has_code(r2, "CIRCLE_INVALID_RADIUS"));
}

TEST_CASE("Circle: tiny but positive radius is a warning") {
    m::SketchDocument d;
    m::Circle c;
    c.center = {0.0, 0.0};
    c.radius = 1e-6;  // below 1 micron
    d.add_primitive(c);
    const auto r = m::validate(d);
    CHECK(has_code(r, "CIRCLE_TINY_RADIUS"));
    CHECK(r.is_valid());
    CHECK(r.has_warnings());
}

TEST_CASE("Spline: degree < 1 is an error") {
    m::SketchDocument d;
    m::Spline s;
    s.degree = 0;
    s.control_points = {{0,0},{1,1}};
    s.knots = {0,0,1,1};
    d.add_primitive(s);
    const auto r = m::validate(d);
    CHECK(has_code(r, "SPLINE_INVALID_DEGREE"));
}

TEST_CASE("Spline: insufficient control points is an error") {
    m::SketchDocument d;
    m::Spline s;
    s.degree = 3;
    s.control_points = {{0,0},{1,1}};  // need 4 for degree 3
    d.add_primitive(s);
    const auto r = m::validate(d);
    CHECK(has_code(r, "SPLINE_INSUFFICIENT_POINTS"));
}

TEST_CASE("Spline: non-monotonic knot vector is an error") {
    m::SketchDocument d;
    m::Spline s;
    s.degree = 3;
    s.control_points = {{0,0},{1,1},{2,0.5},{3,1.5}};
    s.knots = {0, 0, 1, 0, 1, 1, 1, 1};  // dip at index 3
    d.add_primitive(s);
    const auto r = m::validate(d);
    CHECK(has_code(r, "SPLINE_INVALID_KNOTS"));
}

TEST_CASE("Spline: knot length mismatch is a warning") {
    m::SketchDocument d;
    m::Spline s;
    s.degree = 3;
    s.control_points = {{0,0},{1,1},{2,0.5},{3,1.5}};
    s.knots = {0, 0, 0, 1, 1, 1};  // expected 8, got 6
    d.add_primitive(s);
    const auto r = m::validate(d);
    CHECK(has_code(r, "SPLINE_KNOT_LENGTH_MISMATCH"));
}

TEST_CASE("Spline: weight count mismatch is an error") {
    m::SketchDocument d;
    m::Spline s;
    s.degree = 3;
    s.control_points = {{0,0},{1,1},{2,0.5},{3,1.5}};
    s.knots = {0,0,0,0,1,1,1,1};
    s.weights = std::vector<double>{1.0, 1.0};  // count 2 != 4
    d.add_primitive(s);
    const auto r = m::validate(d);
    CHECK(has_code(r, "SPLINE_WEIGHT_COUNT_MISMATCH"));
}

TEST_CASE("Spline: non-positive weight is an error") {
    m::SketchDocument d;
    m::Spline s;
    s.degree = 3;
    s.control_points = {{0,0},{1,1},{2,0.5},{3,1.5}};
    s.knots = {0,0,0,0,1,1,1,1};
    s.weights = std::vector<double>{1.0, 1.0, 0.0, 1.0};
    d.add_primitive(s);
    const auto r = m::validate(d);
    CHECK(has_code(r, "SPLINE_INVALID_WEIGHT"));
}

TEST_CASE("Spline: valid degree-3 spline with 4 CPs and 8 knots passes") {
    m::SketchDocument d;
    m::Spline s;
    s.degree = 3;
    s.control_points = {{0,0},{1,1},{2,0.5},{3,1.5}};
    s.knots = {0,0,0,0,1,1,1,1};
    d.add_primitive(s);
    const auto r = m::validate(d);
    CHECK(r.is_valid());
    CHECK_FALSE(r.has_warnings());
}

TEST_CASE("Confidence outside [0,1] is a warning") {
    m::SketchDocument d;
    m::Line l;
    l.start = {0,0}; l.end = {1,0};
    l.meta.confidence = 1.5;
    d.add_primitive(l);
    const auto r = m::validate(d);
    CHECK(has_code(r, "PRIM_CONFIDENCE_RANGE"));
    CHECK(r.is_valid());  // warning, not error
}

TEST_CASE("Ellipse: valid default passes") {
    m::SketchDocument d;
    d.add_primitive(m::Ellipse{});
    const auto r = m::validate(d);
    CHECK(r.is_valid());
}

TEST_CASE("Ellipse: non-positive radii are errors") {
    m::SketchDocument d;
    m::Ellipse e1; e1.major_radius = 0.0;
    d.add_primitive(e1);
    m::Ellipse e2; e2.major_radius = 1.0; e2.minor_radius = -0.5;
    d.add_primitive(e2);
    const auto r = m::validate(d);
    CHECK(has_code(r, "ELLIPSE_INVALID_MAJOR_RADIUS"));
    CHECK(has_code(r, "ELLIPSE_INVALID_MINOR_RADIUS"));
}

TEST_CASE("Ellipse: major < minor is an error") {
    m::SketchDocument d;
    m::Ellipse e;
    e.major_radius = 1.0;
    e.minor_radius = 2.0;  // minor > major
    d.add_primitive(e);
    const auto r = m::validate(d);
    CHECK(has_code(r, "ELLIPSE_MAJOR_LESS_THAN_MINOR"));
}

TEST_CASE("Ellipse: non-finite center is an error") {
    m::SketchDocument d;
    m::Ellipse e;
    e.center = {std::nan(""), 0.0};
    d.add_primitive(e);
    const auto r = m::validate(d);
    CHECK(has_code(r, "ELLIPSE_INVALID_COORDS"));
}

TEST_CASE("EllipticalArc: valid default passes") {
    m::SketchDocument d;
    d.add_primitive(m::EllipticalArc{});  // default end_param = pi/2 != start_param = 0
    const auto r = m::validate(d);
    CHECK(r.is_valid());
}

TEST_CASE("EllipticalArc: non-positive radii are errors") {
    m::SketchDocument d;
    m::EllipticalArc a;
    a.major_radius = -1.0;
    a.minor_radius = 0.0;
    d.add_primitive(a);
    const auto r = m::validate(d);
    CHECK(has_code(r, "ELLIPTICAL_ARC_INVALID_MAJOR_RADIUS"));
    CHECK(has_code(r, "ELLIPTICAL_ARC_INVALID_MINOR_RADIUS"));
}

TEST_CASE("EllipticalArc: major < minor is an error") {
    m::SketchDocument d;
    m::EllipticalArc a;
    a.major_radius = 1.0;
    a.minor_radius = 2.0;
    d.add_primitive(a);
    const auto r = m::validate(d);
    CHECK(has_code(r, "ELLIPTICAL_ARC_MAJOR_LESS_THAN_MINOR"));
}

TEST_CASE("EllipticalArc: non-finite parametric angles are errors") {
    m::SketchDocument d;
    m::EllipticalArc a;
    a.start_param = std::nan("");
    d.add_primitive(a);
    const auto r = m::validate(d);
    CHECK(has_code(r, "ELLIPTICAL_ARC_INVALID_START_PARAM"));

    m::SketchDocument d2;
    m::EllipticalArc b;
    b.end_param = std::nan("");
    d2.add_primitive(b);
    const auto r2 = m::validate(d2);
    CHECK(has_code(r2, "ELLIPTICAL_ARC_INVALID_END_PARAM"));
}

TEST_CASE("EllipticalArc: zero sweep (start_param == end_param) is an error") {
    m::SketchDocument d;
    m::EllipticalArc a;
    a.start_param = 0.5;
    a.end_param = 0.5;
    d.add_primitive(a);
    const auto r = m::validate(d);
    CHECK(has_code(r, "ELLIPTICAL_ARC_ZERO_SWEEP"));
}

// ---------- constraint validation ----------

TEST_CASE("Constraint: too few references is an error") {
    m::SketchDocument d;
    d.add_primitive(m::Line{});
    m::SketchConstraint c;
    c.id = "c";
    c.constraint_type = m::ConstraintType::Coincident;  // needs 2
    c.references = {std::string{"L0"}};
    d.constraints.push_back(c);
    const auto r = m::validate(d);
    CHECK(has_code(r, "CONST_TOO_FEW_REFS"));
}

TEST_CASE("Constraint: too many references is an error for fixed-arity types") {
    m::SketchDocument d;
    d.add_primitive(m::Line{});
    m::SketchConstraint c;
    c.id = "c";
    c.constraint_type = m::ConstraintType::Horizontal;  // exactly 1
    c.references = {std::string{"L0"}, std::string{"L0"}};
    d.constraints.push_back(c);
    const auto r = m::validate(d);
    CHECK(has_code(r, "CONST_TOO_MANY_REFS"));
}

TEST_CASE("Constraint: chainable types accept many refs") {
    m::SketchDocument d;
    d.add_primitive(m::Line{});
    d.add_primitive(m::Line{});
    d.add_primitive(m::Line{});
    m::SketchConstraint c;
    c.id = "c";
    c.constraint_type = m::ConstraintType::Equal;  // chainable
    c.references = {std::string{"L0"}, std::string{"L1"}, std::string{"L2"}};
    d.constraints.push_back(c);
    const auto r = m::validate(d);
    CHECK_FALSE(has_code(r, "CONST_TOO_MANY_REFS"));
}

TEST_CASE("Constraint: missing required value for dimensional constraint") {
    m::SketchDocument d;
    d.add_primitive(m::Line{});
    m::SketchConstraint c;
    c.id = "c";
    c.constraint_type = m::ConstraintType::Length;
    c.references = {std::string{"L0"}};
    // value intentionally not set
    d.constraints.push_back(c);
    const auto r = m::validate(d);
    CHECK(has_code(r, "CONST_MISSING_VALUE"));
}

TEST_CASE("Constraint: non-finite value is an error") {
    m::SketchDocument d;
    d.add_primitive(m::Line{});
    m::SketchConstraint c;
    c.id = "c";
    c.constraint_type = m::ConstraintType::Length;
    c.references = {std::string{"L0"}};
    c.value = std::nan("");
    d.constraints.push_back(c);
    const auto r = m::validate(d);
    CHECK(has_code(r, "CONST_INVALID_VALUE"));
}

TEST_CASE("Constraint: negative dimensional value (Length/Radius/Diameter/Distance)") {
    auto run = [](m::ConstraintType t) {
        m::SketchDocument d;
        d.add_primitive(m::Line{});
        d.add_primitive(m::Line{});
        m::SketchConstraint c;
        c.id = "c";
        c.constraint_type = t;
        if (t == m::ConstraintType::Distance) {
            c.references = {
                m::PointRef{"L0", m::PointType::Start},
                m::PointRef{"L1", m::PointType::End},
            };
        } else {
            c.references = {std::string{"L0"}};
        }
        c.value = -3.0;
        d.constraints.push_back(c);
        return m::validate(d);
    };
    CHECK(has_code(run(m::ConstraintType::Length),   "CONST_NEGATIVE_VALUE"));
    CHECK(has_code(run(m::ConstraintType::Radius),   "CONST_NEGATIVE_VALUE"));
    CHECK(has_code(run(m::ConstraintType::Diameter), "CONST_NEGATIVE_VALUE"));
    CHECK(has_code(run(m::ConstraintType::Distance), "CONST_NEGATIVE_VALUE"));
}

TEST_CASE("Constraint: Angle accepts negative values (it's an angle, not a length)") {
    m::SketchDocument d;
    d.add_primitive(m::Line{});
    d.add_primitive(m::Line{});
    m::SketchConstraint c;
    c.id = "c";
    c.constraint_type = m::ConstraintType::Angle;
    c.references = {std::string{"L0"}, std::string{"L1"}};
    c.value = -45.0;
    d.constraints.push_back(c);
    const auto r = m::validate(d);
    CHECK_FALSE(has_code(r, "CONST_NEGATIVE_VALUE"));
}

TEST_CASE("Constraint: reference to non-existent element") {
    m::SketchDocument d;
    m::SketchConstraint c;
    c.id = "c";
    c.constraint_type = m::ConstraintType::Horizontal;
    c.references = {std::string{"L99"}};
    d.constraints.push_back(c);
    const auto r = m::validate(d);
    CHECK(has_code(r, "CONST_INVALID_REF"));
}

TEST_CASE("Constraint: invalid PointType for primitive") {
    m::SketchDocument d;
    d.add_primitive(m::Circle{});  // C0; valid points: CENTER only
    m::SketchConstraint c;
    c.id = "c";
    c.constraint_type = m::ConstraintType::Coincident;
    c.references = {
        m::PointRef{"C0", m::PointType::Start},  // invalid for Circle
        m::PointRef{"C0", m::PointType::Center},
    };
    d.constraints.push_back(c);
    const auto r = m::validate(d);
    CHECK(has_code(r, "CONST_INVALID_POINT_TYPE"));
}

TEST_CASE("Constraint: CONTROL without index is an error") {
    m::SketchDocument d;
    m::Spline s;
    s.degree = 3;
    s.control_points = {{0,0},{1,1},{2,0.5},{3,1.5}};
    s.knots = {0,0,0,0,1,1,1,1};
    d.add_primitive(s);
    d.add_primitive(m::Point{});

    m::SketchConstraint c;
    c.id = "c";
    c.constraint_type = m::ConstraintType::Coincident;
    c.references = {
        m::PointRef{"S0", m::PointType::Control},  // index missing
        m::PointRef{"P0", m::PointType::Center},
    };
    d.constraints.push_back(c);
    const auto r = m::validate(d);
    CHECK(has_code(r, "CONST_MISSING_INDEX"));
}

TEST_CASE("Constraint: CONTROL index out of range") {
    m::SketchDocument d;
    m::Spline s;
    s.degree = 3;
    s.control_points = {{0,0},{1,1},{2,0.5},{3,1.5}};  // size 4
    s.knots = {0,0,0,0,1,1,1,1};
    d.add_primitive(s);
    d.add_primitive(m::Point{});

    m::SketchConstraint c;
    c.id = "c";
    c.constraint_type = m::ConstraintType::Coincident;
    c.references = {
        m::PointRef{"S0", m::PointType::Control, std::nullopt, 99},
        m::PointRef{"P0", m::PointType::Center},
    };
    d.constraints.push_back(c);
    const auto r = m::validate(d);
    CHECK(has_code(r, "CONST_INDEX_OUT_OF_RANGE"));
}

TEST_CASE("Constraint: invalid connection_point") {
    m::SketchDocument d;
    d.add_primitive(m::Line{});
    d.add_primitive(m::Arc{});
    m::SketchConstraint c;
    c.id = "c";
    c.constraint_type = m::ConstraintType::Tangent;
    c.references = {std::string{"L0"}, std::string{"A0"}};
    c.connection_point = m::PointRef{"X99", m::PointType::End};  // missing element
    d.constraints.push_back(c);
    const auto r = m::validate(d);
    CHECK(has_code(r, "CONST_INVALID_CONNECTION_POINT"));
}

TEST_CASE("Document: duplicate constraint IDs are a warning") {
    m::SketchDocument d;
    m::Line l;
    l.start = {0, 0}; l.end = {1, 0};  // non-degenerate
    d.add_primitive(l);
    m::SketchConstraint c1, c2;
    c1.id = "dup"; c1.constraint_type = m::ConstraintType::Horizontal;
    c1.references = {std::string{"L0"}};
    c2 = c1;
    d.constraints.push_back(c1);
    d.constraints.push_back(c2);
    const auto r = m::validate(d);
    CHECK(has_code(r, "CONST_DUPLICATE_ID"));
    CHECK(r.is_valid());  // duplicate IDs are warning-level
}

TEST_CASE("End-to-end: a fully valid sketch produces no issues") {
    m::SketchDocument d;
    d.name = "ValidSketch";

    m::Line l;
    l.start = {0,0}; l.end = {10,0};
    d.add_primitive(l);

    m::Arc a;
    a.center = {10, 5}; a.start_point = {10, 0}; a.end_point = {15, 5};
    d.add_primitive(a);

    m::Circle c;
    c.center = {0, 5}; c.radius = 1.0;
    d.add_primitive(c);

    m::SketchConstraint horiz;
    horiz.id = "h1";
    horiz.constraint_type = m::ConstraintType::Horizontal;
    horiz.references = {std::string{"L0"}};
    d.constraints.push_back(horiz);

    m::SketchConstraint len;
    len.id = "len1";
    len.constraint_type = m::ConstraintType::Length;
    len.references = {std::string{"L0"}};
    len.value = 10.0;
    d.constraints.push_back(len);

    m::SketchConstraint coin;
    coin.id = "co1";
    coin.constraint_type = m::ConstraintType::Coincident;
    coin.references = {
        m::PointRef{"L0", m::PointType::End},
        m::PointRef{"A0", m::PointType::Start},
    };
    d.constraints.push_back(coin);

    const auto r = m::validate(d);
    CHECK(r.is_valid());
    CHECK_FALSE(r.has_warnings());
    CHECK(r.issues.empty());
}
