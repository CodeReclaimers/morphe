#include "doctest/doctest.h"

#include "morphe/types.hpp"

#include <cmath>

TEST_CASE("Point2D distance and equality") {
    morphe::Point2D a{0.0, 0.0};
    morphe::Point2D b{3.0, 4.0};
    CHECK(a.distance_to(b) == doctest::Approx(5.0));
    CHECK(b.distance_to(a) == doctest::Approx(5.0));
    CHECK(a == morphe::Point2D{0.0, 0.0});
    CHECK(a != b);
}

TEST_CASE("Vector2D magnitude") {
    morphe::Vector2D v{3.0, 4.0};
    CHECK(v.magnitude() == doctest::Approx(5.0));
}

TEST_CASE("PointType <-> string round-trip is exhaustive") {
    using morphe::PointType;
    constexpr PointType all[] = {
        PointType::Start, PointType::End, PointType::Center,
        PointType::Midpoint, PointType::Control, PointType::OnCurve,
    };
    for (auto t : all) {
        CHECK(morphe::point_type_from_string(morphe::to_string(t)) == t);
    }
}

TEST_CASE("PointType wire strings match the specification") {
    using morphe::PointType;
    CHECK(morphe::to_string(PointType::Start)    == "start");
    CHECK(morphe::to_string(PointType::End)      == "end");
    CHECK(morphe::to_string(PointType::Center)   == "center");
    CHECK(morphe::to_string(PointType::Midpoint) == "midpoint");
    CHECK(morphe::to_string(PointType::Control)  == "control");
    CHECK(morphe::to_string(PointType::OnCurve)  == "on_curve");
}

TEST_CASE("PointType from unknown string throws") {
    CHECK_THROWS_AS(morphe::point_type_from_string("not_a_point_type"),
                    std::invalid_argument);
}

TEST_CASE("PointRef construction and equality") {
    morphe::PointRef a{"L0", morphe::PointType::Start};
    morphe::PointRef b{"L0", morphe::PointType::Start};
    morphe::PointRef c{"L0", morphe::PointType::End};
    CHECK(a == b);
    CHECK(a != c);

    morphe::PointRef on_curve{"S2", morphe::PointType::OnCurve, 0.5};
    CHECK(on_curve.parameter.has_value());
    CHECK(*on_curve.parameter == doctest::Approx(0.5));
    CHECK_FALSE(on_curve.index.has_value());

    morphe::PointRef control{"S2", morphe::PointType::Control, std::nullopt, 3};
    CHECK_FALSE(control.parameter.has_value());
    CHECK(control.index.has_value());
    CHECK(*control.index == 3);
}

TEST_CASE("Element prefix conventions") {
    CHECK(morphe::element_prefix::line           == 'L');
    CHECK(morphe::element_prefix::arc            == 'A');
    CHECK(morphe::element_prefix::circle         == 'C');
    CHECK(morphe::element_prefix::point          == 'P');
    CHECK(morphe::element_prefix::spline         == 'S');
    CHECK(morphe::element_prefix::ellipse        == 'E');
    CHECK(morphe::element_prefix::elliptical_arc == 'e');  // lowercase, intentional
}
