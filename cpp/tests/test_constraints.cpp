#include "doctest/doctest.h"

#include "morphe/constraints.hpp"

namespace m = morphe;

namespace {

constexpr m::ConstraintType all_types[] = {
    m::ConstraintType::Coincident,
    m::ConstraintType::Tangent,
    m::ConstraintType::Perpendicular,
    m::ConstraintType::Parallel,
    m::ConstraintType::Concentric,
    m::ConstraintType::Equal,
    m::ConstraintType::Collinear,
    m::ConstraintType::Horizontal,
    m::ConstraintType::Vertical,
    m::ConstraintType::Fixed,
    m::ConstraintType::Distance,
    m::ConstraintType::DistanceX,
    m::ConstraintType::DistanceY,
    m::ConstraintType::Length,
    m::ConstraintType::Radius,
    m::ConstraintType::Diameter,
    m::ConstraintType::Angle,
    m::ConstraintType::Symmetric,
    m::ConstraintType::Midpoint,
};

}  // namespace

TEST_CASE("ConstraintType: round-trip every value through the wire string") {
    for (auto t : all_types) {
        const auto s = m::to_string(t);
        CHECK(m::constraint_type_from_string(s) == t);
    }
}

TEST_CASE("ConstraintType::Midpoint serializes as 'midpoint_constraint'") {
    // This is a footgun: the Python enum value is 'midpoint_constraint' but
    // the C++ enumerator is named Midpoint. Keep them tied.
    CHECK(m::to_string(m::ConstraintType::Midpoint) == "midpoint_constraint");
    CHECK(m::constraint_type_from_string("midpoint_constraint") == m::ConstraintType::Midpoint);
    CHECK_THROWS_AS(m::constraint_type_from_string("midpoint"), std::invalid_argument);
}

TEST_CASE("ConstraintStatus: round-trip every value") {
    constexpr m::ConstraintStatus all[] = {
        m::ConstraintStatus::Unknown,
        m::ConstraintStatus::Satisfied,
        m::ConstraintStatus::Violated,
        m::ConstraintStatus::Redundant,
        m::ConstraintStatus::Conflicting,
    };
    for (auto s : all) {
        CHECK(m::constraint_status_from_string(m::to_string(s)) == s);
    }
}

TEST_CASE("rules_for: every ConstraintType has a rules entry") {
    for (auto t : all_types) {
        const auto& r = m::rules_for(t);
        (void)r;  // just verify lookup does not throw
    }
}

TEST_CASE("rules_for: dimensional constraints require a value") {
    CHECK(m::rules_for(m::ConstraintType::Distance).value_required);
    CHECK(m::rules_for(m::ConstraintType::DistanceX).value_required);
    CHECK(m::rules_for(m::ConstraintType::DistanceY).value_required);
    CHECK(m::rules_for(m::ConstraintType::Length).value_required);
    CHECK(m::rules_for(m::ConstraintType::Radius).value_required);
    CHECK(m::rules_for(m::ConstraintType::Diameter).value_required);
    CHECK(m::rules_for(m::ConstraintType::Angle).value_required);

    CHECK_FALSE(m::rules_for(m::ConstraintType::Coincident).value_required);
    CHECK_FALSE(m::rules_for(m::ConstraintType::Horizontal).value_required);
    CHECK_FALSE(m::rules_for(m::ConstraintType::Tangent).value_required);
}

TEST_CASE("rules_for: chainable constraints have no max_refs") {
    CHECK_FALSE(m::rules_for(m::ConstraintType::Equal).max_refs.has_value());
    CHECK_FALSE(m::rules_for(m::ConstraintType::Collinear).max_refs.has_value());
}

TEST_CASE("rules_for: Symmetric needs exactly 3 refs") {
    const auto& r = m::rules_for(m::ConstraintType::Symmetric);
    CHECK(r.min_refs == 3);
    CHECK(r.max_refs == 3);
    CHECK(r.ref_kinds.size() == 3);
    CHECK(r.ref_kinds[2] == m::RefKind::Line);  // symmetry axis
}

TEST_CASE("rules_for: DistanceX/Y allow either 1 or 2 references") {
    const auto& rx = m::rules_for(m::ConstraintType::DistanceX);
    CHECK(rx.min_refs == 1);
    CHECK(rx.max_refs == 2);
}

TEST_CASE("SketchConstraint: get_element_ids deduplicates and includes connection_point") {
    m::SketchConstraint c;
    c.id = "c1";
    c.constraint_type = m::ConstraintType::Tangent;
    c.references = {std::string{"L0"}, std::string{"A1"}};
    c.connection_point = m::PointRef{"L0", m::PointType::End};

    const auto ids = c.get_element_ids();
    REQUIRE(ids.size() == 2);
    // L0 appears in references[0] and connection_point — should be deduped.
    CHECK((ids[0] == "L0" || ids[1] == "L0"));
    CHECK((ids[0] == "A1" || ids[1] == "A1"));
}

TEST_CASE("ConstraintRef: pulls element ID from either variant") {
    m::ConstraintRef r1{std::string{"L0"}};
    m::ConstraintRef r2{m::PointRef{"A1", m::PointType::Center}};
    CHECK(m::referenced_element_id(r1) == "L0");
    CHECK(m::referenced_element_id(r2) == "A1");
}
