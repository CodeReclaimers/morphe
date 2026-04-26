#include "doctest/doctest.h"

#include "morphe/primitives.hpp"

#include <algorithm>

namespace m = morphe;

TEST_CASE("Line: length, defaults, equality") {
    m::Line l;
    l.meta.id = "L0";
    l.start = {0.0, 0.0};
    l.end   = {3.0, 4.0};
    CHECK(l.length() == doctest::Approx(5.0));
    CHECK_FALSE(l.meta.construction);
    CHECK(l.meta.confidence == doctest::Approx(1.0));
    CHECK_FALSE(l.meta.source.has_value());

    m::Line m2 = l;
    CHECK(l == m2);
    m2.end = {3.0, 5.0};
    CHECK(l != m2);
}

TEST_CASE("Arc: implicit radius from center to start_point") {
    m::Arc a;
    a.meta.id = "A0";
    a.center = {0.0, 0.0};
    a.start_point = {5.0, 0.0};
    a.end_point   = {0.0, 5.0};
    a.ccw = true;
    CHECK(a.radius() == doctest::Approx(5.0));
}

TEST_CASE("Circle: radius default and equality") {
    m::Circle c;
    c.meta.id = "C0";
    c.center = {1.0, 2.0};
    CHECK(c.radius == doctest::Approx(1.0));  // default
    c.radius = 3.5;
    m::Circle d = c;
    CHECK(c == d);
}

TEST_CASE("Spline: defaults and rationality") {
    m::Spline s;
    s.meta.id = "S0";
    CHECK(s.degree == 3);
    CHECK(s.order() == 4);
    CHECK_FALSE(s.is_rational());
    CHECK_FALSE(s.periodic);
    CHECK_FALSE(s.is_fit_spline);

    s.weights = std::vector<double>{1.0, 2.0, 1.0};
    CHECK(s.is_rational());
}

TEST_CASE("EllipticalArc: end_param defaults to pi/2") {
    m::EllipticalArc ea;
    CHECK(ea.start_param == doctest::Approx(0.0));
    CHECK(ea.end_param == doctest::Approx(1.5707963267948966));
    CHECK(ea.major_radius == doctest::Approx(1.0));
    CHECK(ea.minor_radius == doctest::Approx(0.5));
}

TEST_CASE("Primitive variant: type_tag matches Python wire format") {
    m::Primitive p_line{m::Line{}};
    m::Primitive p_arc{m::Arc{}};
    m::Primitive p_circle{m::Circle{}};
    m::Primitive p_point{m::Point{}};
    m::Primitive p_spline{m::Spline{}};
    m::Primitive p_ellipse{m::Ellipse{}};
    m::Primitive p_earc{m::EllipticalArc{}};

    CHECK(m::type_tag(p_line)    == "line");
    CHECK(m::type_tag(p_arc)     == "arc");
    CHECK(m::type_tag(p_circle)  == "circle");
    CHECK(m::type_tag(p_point)   == "point");
    CHECK(m::type_tag(p_spline)  == "spline");
    CHECK(m::type_tag(p_ellipse) == "ellipse");
    CHECK(m::type_tag(p_earc)    == "ellipticalarc");  // one word, no separator
}

TEST_CASE("Primitive variant: id_prefix") {
    CHECK(m::id_prefix(m::Primitive{m::Line{}})          == 'L');
    CHECK(m::id_prefix(m::Primitive{m::Arc{}})           == 'A');
    CHECK(m::id_prefix(m::Primitive{m::Circle{}})        == 'C');
    CHECK(m::id_prefix(m::Primitive{m::Point{}})         == 'P');
    CHECK(m::id_prefix(m::Primitive{m::Spline{}})        == 'S');
    CHECK(m::id_prefix(m::Primitive{m::Ellipse{}})       == 'E');
    CHECK(m::id_prefix(m::Primitive{m::EllipticalArc{}}) == 'e');
}

TEST_CASE("valid_point_types: matches Python schema for every primitive") {
    using PT = m::PointType;
    auto contains = [](const std::vector<PT>& v, PT t) {
        return std::find(v.begin(), v.end(), t) != v.end();
    };

    auto vp_line = m::valid_point_types(m::Primitive{m::Line{}});
    CHECK(vp_line.size() == 3);
    CHECK(contains(vp_line, PT::Start));
    CHECK(contains(vp_line, PT::End));
    CHECK(contains(vp_line, PT::Midpoint));

    auto vp_arc = m::valid_point_types(m::Primitive{m::Arc{}});
    CHECK(vp_arc.size() == 4);
    CHECK(contains(vp_arc, PT::Start));
    CHECK(contains(vp_arc, PT::End));
    CHECK(contains(vp_arc, PT::Center));
    CHECK(contains(vp_arc, PT::Midpoint));

    auto vp_circle = m::valid_point_types(m::Primitive{m::Circle{}});
    CHECK(vp_circle == std::vector<PT>{PT::Center});

    auto vp_point = m::valid_point_types(m::Primitive{m::Point{}});
    CHECK(vp_point == std::vector<PT>{PT::Center});  // CENTER means "the point itself"

    auto vp_spline = m::valid_point_types(m::Primitive{m::Spline{}});
    CHECK(vp_spline.size() == 3);
    CHECK(contains(vp_spline, PT::Start));
    CHECK(contains(vp_spline, PT::End));
    CHECK(contains(vp_spline, PT::Control));

    auto vp_ellipse = m::valid_point_types(m::Primitive{m::Ellipse{}});
    CHECK(vp_ellipse == std::vector<PT>{PT::Center});

    auto vp_earc = m::valid_point_types(m::Primitive{m::EllipticalArc{}});
    CHECK(vp_earc.size() == 4);
    CHECK(contains(vp_earc, PT::Start));
    CHECK(contains(vp_earc, PT::End));
    CHECK(contains(vp_earc, PT::Center));
    CHECK(contains(vp_earc, PT::Midpoint));
}

TEST_CASE("meta_of: round-trip access through the variant") {
    m::Primitive p{m::Line{}};
    m::meta_of(p).id = "L42";
    m::meta_of(p).construction = true;
    CHECK(m::id_of(p) == "L42");
    CHECK(m::meta_of(p).construction);
}
