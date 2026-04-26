#include "doctest/doctest.h"

#include "morphe/document.hpp"

namespace m = morphe;

TEST_CASE("SolverStatus: round-trip every value") {
    constexpr m::SolverStatus all[] = {
        m::SolverStatus::Dirty,
        m::SolverStatus::UnderConstrained,
        m::SolverStatus::FullyConstrained,
        m::SolverStatus::OverConstrained,
        m::SolverStatus::Inconsistent,
    };
    for (auto s : all) {
        CHECK(m::solver_status_from_string(m::to_string(s)) == s);
    }
}

TEST_CASE("SketchDocument: defaults") {
    m::SketchDocument d;
    CHECK(d.name == "Untitled");
    CHECK(d.primitives.empty());
    CHECK(d.constraints.empty());
    CHECK(d.solver_status == m::SolverStatus::Dirty);
    CHECK(d.degrees_of_freedom == -1);
}

TEST_CASE("SketchDocument: add_primitive auto-assigns prefixed IDs") {
    m::SketchDocument d;
    auto id1 = d.add_primitive(m::Line{});
    auto id2 = d.add_primitive(m::Line{});
    auto id3 = d.add_primitive(m::Arc{});
    auto id4 = d.add_primitive(m::Circle{});
    auto id5 = d.add_primitive(m::EllipticalArc{});
    CHECK(id1 == "L0");
    CHECK(id2 == "L1");
    CHECK(id3 == "A0");
    CHECK(id4 == "C0");
    CHECK(id5 == "e0");  // lowercase prefix for elliptical arc
}

TEST_CASE("SketchDocument: insertion order is preserved") {
    m::SketchDocument d;
    d.add_primitive(m::Line{});
    d.add_primitive(m::Arc{});
    d.add_primitive(m::Line{});
    d.add_primitive(m::Circle{});

    REQUIRE(d.primitives.size() == 4);
    CHECK(m::id_of(d.primitives[0]) == "L0");
    CHECK(m::id_of(d.primitives[1]) == "A0");
    CHECK(m::id_of(d.primitives[2]) == "L1");
    CHECK(m::id_of(d.primitives[3]) == "C0");
}

TEST_CASE("SketchDocument: add_primitive_with_id updates counter") {
    m::SketchDocument d;
    d.add_primitive_with_id(m::Line{}, "L5");
    auto next_id = d.add_primitive(m::Line{});
    CHECK(next_id == "L6");  // counter advanced past 5
}

TEST_CASE("SketchDocument: duplicate IDs are rejected") {
    m::SketchDocument d;
    d.add_primitive_with_id(m::Line{}, "L0");
    CHECK_THROWS_AS(d.add_primitive_with_id(m::Line{}, "L0"), std::invalid_argument);
}

TEST_CASE("SketchDocument: find_primitive returns nullptr for missing IDs") {
    m::SketchDocument d;
    d.add_primitive(m::Line{});
    CHECK(d.find_primitive("L0") != nullptr);
    CHECK(d.find_primitive("L9") == nullptr);
    CHECK(d.find_primitive("nope") == nullptr);
}

TEST_CASE("SketchDocument: add_constraint validates referenced elements exist") {
    m::SketchDocument d;
    d.add_primitive(m::Line{});  // L0

    m::SketchConstraint c;
    c.id = "h0";
    c.constraint_type = m::ConstraintType::Horizontal;
    c.references = {std::string{"L0"}};
    CHECK_NOTHROW(d.add_constraint(c));

    m::SketchConstraint bad = c;
    bad.id = "h1";
    bad.references = {std::string{"L99"}};
    CHECK_THROWS_AS(d.add_constraint(bad), std::out_of_range);
}

TEST_CASE("SketchDocument: equality compares fields, not internal counters") {
    m::SketchDocument a;
    m::SketchDocument b;
    CHECK(a == b);

    a.add_primitive(m::Line{});
    b.add_primitive(m::Line{});
    CHECK(a == b);

    b.name = "different";
    CHECK(a != b);
}
