#include "doctest/doctest.h"

#include "morphe/morphe.hpp"
#include "morphe/serialization.hpp"

#include "nlohmann/json.hpp"

#include <filesystem>
#include <fstream>

namespace m = morphe;
using json = nlohmann::json;

namespace {

// Build a SketchDocument that exercises every primitive type. IDs are
// assigned via add_primitive so insertion order in the wire format is
// L0, A0, C0, P0, S0, E0, e0.
m::SketchDocument make_full_document() {
    m::SketchDocument d;
    d.name = "AllPrimitives";

    m::Line line;
    line.start = {0.0, 0.0};
    line.end   = {10.0, 0.0};
    d.add_primitive(line);

    m::Arc arc;
    arc.center      = {10.0, 0.0};
    arc.start_point = {10.0, 0.0};
    arc.end_point   = {15.0, 5.0};
    arc.ccw         = true;
    d.add_primitive(arc);

    m::Circle circle;
    circle.center = {0.0, 0.0};
    circle.radius = 5.0;
    d.add_primitive(circle);

    m::Point pt;
    pt.position = {1.5, 2.5};
    d.add_primitive(pt);

    m::Spline sp;
    sp.degree = 3;
    sp.control_points = {{0.0, 0.0}, {1.0, 1.0}, {2.0, 0.5}, {3.0, 1.5}};
    sp.knots = {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
    d.add_primitive(sp);

    m::Ellipse el;
    el.center = {5.0, 5.0};
    el.major_radius = 3.0;
    el.minor_radius = 1.5;
    el.rotation = 0.5;
    d.add_primitive(el);

    m::EllipticalArc ea;
    ea.center = {0.0, 0.0};
    ea.major_radius = 2.0;
    ea.minor_radius = 1.0;
    ea.rotation = 0.0;
    ea.start_param = 0.0;
    ea.end_param = 1.5707963267948966;
    ea.ccw = true;
    d.add_primitive(ea);
    return d;
}

}  // namespace

TEST_CASE("Top-level key order matches Python") {
    m::SketchDocument d;
    const auto s = m::to_json(d, -1);  // compact, no indentation
    // Keys must appear in this exact order: name, primitives, constraints,
    // solver_status, degrees_of_freedom.
    const auto p_name   = s.find("\"name\"");
    const auto p_prims  = s.find("\"primitives\"");
    const auto p_cons   = s.find("\"constraints\"");
    const auto p_status = s.find("\"solver_status\"");
    const auto p_dof    = s.find("\"degrees_of_freedom\"");
    REQUIRE(p_name   != std::string::npos);
    REQUIRE(p_prims  != std::string::npos);
    REQUIRE(p_cons   != std::string::npos);
    REQUIRE(p_status != std::string::npos);
    REQUIRE(p_dof    != std::string::npos);
    CHECK(p_name < p_prims);
    CHECK(p_prims < p_cons);
    CHECK(p_cons < p_status);
    CHECK(p_status < p_dof);
}

TEST_CASE("Round-trip: full document with all 7 primitive types") {
    const auto doc = make_full_document();
    const auto s   = m::to_json(doc);
    const auto rt  = m::from_json(s);
    CHECK(rt == doc);
}

TEST_CASE("Round-trip preserves insertion order across deserialize") {
    const auto doc = make_full_document();
    const auto rt = m::from_json(m::to_json(doc));
    REQUIRE(rt.primitives.size() == doc.primitives.size());
    for (size_t i = 0; i < doc.primitives.size(); ++i) {
        CHECK(m::id_of(rt.primitives[i]) == m::id_of(doc.primitives[i]));
        CHECK(m::type_tag(rt.primitives[i]) == m::type_tag(doc.primitives[i]));
    }
}

TEST_CASE("Type tag for EllipticalArc serializes as one word") {
    m::SketchDocument d;
    m::EllipticalArc ea;
    d.add_primitive(ea);
    const auto j = json::parse(m::to_json(d));
    CHECK(j["primitives"][0]["type"].get<std::string>() == "ellipticalarc");
}

TEST_CASE("Point2D encoded as [x, y] array, not object") {
    m::SketchDocument d;
    m::Line l;
    l.start = {1.0, 2.0};
    l.end   = {3.0, 4.0};
    d.add_primitive(l);
    const auto j = json::parse(m::to_json(d));
    REQUIRE(j["primitives"][0]["start"].is_array());
    CHECK(j["primitives"][0]["start"][0].get<double>() == 1.0);
    CHECK(j["primitives"][0]["start"][1].get<double>() == 2.0);
}

TEST_CASE("Optional primitive metadata: source omitted when null") {
    m::SketchDocument d;
    d.add_primitive(m::Line{});
    const auto j = json::parse(m::to_json(d));
    CHECK(!j["primitives"][0].contains("source"));
}

TEST_CASE("Optional primitive metadata: confidence omitted at default 1.0") {
    m::SketchDocument d;
    d.add_primitive(m::Line{});
    const auto j = json::parse(m::to_json(d));
    CHECK(!j["primitives"][0].contains("confidence"));
}

TEST_CASE("Optional primitive metadata: source emitted when set") {
    m::SketchDocument d;
    m::Line l;
    l.meta.source = "user";
    d.add_primitive(l);
    const auto j = json::parse(m::to_json(d));
    CHECK(j["primitives"][0]["source"].get<std::string>() == "user");
}

TEST_CASE("Optional primitive metadata: confidence emitted when not 1.0") {
    m::SketchDocument d;
    m::Arc a;
    a.meta.confidence = 0.85;
    d.add_primitive(a);
    const auto j = json::parse(m::to_json(d));
    CHECK(j["primitives"][0]["confidence"].get<double>() == 0.85);
}

TEST_CASE("Spline weights: omitted when null, emitted when present") {
    m::SketchDocument d;
    m::Spline non_rational;
    non_rational.control_points = {{0,0},{1,1}};
    non_rational.knots = {0,0,1,1};
    d.add_primitive(non_rational);

    m::Spline rational;
    rational.control_points = {{0,0},{1,1}};
    rational.knots = {0,0,1,1};
    rational.weights = std::vector<double>{1.0, 2.0};
    d.add_primitive(rational);

    const auto j = json::parse(m::to_json(d));
    CHECK(!j["primitives"][0].contains("weights"));
    REQUIRE(j["primitives"][1].contains("weights"));
    CHECK(j["primitives"][1]["weights"][0].get<double>() == 1.0);
    CHECK(j["primitives"][1]["weights"][1].get<double>() == 2.0);
}

TEST_CASE("Constraint: optional fields omitted at defaults") {
    m::SketchDocument d;
    d.add_primitive(m::Line{});
    m::SketchConstraint c;
    c.id = "h";
    c.constraint_type = m::ConstraintType::Horizontal;
    c.references = {std::string{"L0"}};
    d.add_constraint(c);
    const auto j = json::parse(m::to_json(d));
    const auto& jc = j["constraints"][0];
    CHECK(!jc.contains("value"));
    CHECK(!jc.contains("connection_point"));
    CHECK(!jc.contains("inferred"));
    CHECK(!jc.contains("confidence"));
    CHECK(!jc.contains("source"));
    CHECK(!jc.contains("status"));
}

TEST_CASE("Constraint: optional fields emitted when set") {
    m::SketchDocument d;
    d.add_primitive(m::Line{});
    d.add_primitive(m::Arc{});
    m::SketchConstraint c;
    c.id = "t";
    c.constraint_type = m::ConstraintType::Tangent;
    c.references = {std::string{"L0"}, std::string{"A0"}};
    c.connection_point = m::PointRef{"L0", m::PointType::End};
    c.inferred = true;
    c.confidence = 0.9;
    c.source = "ai";
    c.status = m::ConstraintStatus::Satisfied;
    d.add_constraint(c);
    const auto j = json::parse(m::to_json(d));
    const auto& jc = j["constraints"][0];
    REQUIRE(jc.contains("connection_point"));
    CHECK(jc["connection_point"]["element"].get<std::string>() == "L0");
    CHECK(jc["connection_point"]["point"].get<std::string>() == "end");
    CHECK(jc["inferred"].get<bool>() == true);
    CHECK(jc["confidence"].get<double>() == 0.9);
    CHECK(jc["source"].get<std::string>() == "ai");
    CHECK(jc["status"].get<std::string>() == "satisfied");
}

TEST_CASE("Constraint: dimensional value emitted") {
    m::SketchDocument d;
    d.add_primitive(m::Line{});
    m::SketchConstraint c;
    c.id = "len";
    c.constraint_type = m::ConstraintType::Length;
    c.references = {std::string{"L0"}};
    c.value = 12.5;
    d.add_constraint(c);
    const auto j = json::parse(m::to_json(d));
    CHECK(j["constraints"][0]["value"].get<double>() == 12.5);
}

TEST_CASE("Constraint MIDPOINT serializes as 'midpoint_constraint'") {
    m::SketchDocument d;
    d.add_primitive(m::Line{});
    d.add_primitive(m::Point{});
    m::SketchConstraint c;
    c.id = "mp";
    c.constraint_type = m::ConstraintType::Midpoint;
    c.references = {m::PointRef{"P0", m::PointType::Center}, std::string{"L0"}};
    d.add_constraint(c);
    const auto j = json::parse(m::to_json(d));
    CHECK(j["constraints"][0]["type"].get<std::string>() == "midpoint_constraint");

    const auto rt = m::from_json(m::to_json(d));
    REQUIRE(rt.constraints.size() == 1);
    CHECK(rt.constraints[0].constraint_type == m::ConstraintType::Midpoint);
}

TEST_CASE("PointRef: parameter and index emitted only when set") {
    m::SketchDocument d;
    d.add_primitive(m::Spline{});
    d.add_primitive(m::Point{});

    m::SketchConstraint plain;
    plain.id = "c1";
    plain.constraint_type = m::ConstraintType::Coincident;
    plain.references = {
        m::PointRef{"S0", m::PointType::Start},
        m::PointRef{"P0", m::PointType::Center},
    };
    d.add_constraint(plain);

    m::SketchConstraint with_index;
    with_index.id = "c2";
    with_index.constraint_type = m::ConstraintType::Coincident;
    with_index.references = {
        m::PointRef{"S0", m::PointType::Control, std::nullopt, 2},
        m::PointRef{"P0", m::PointType::Center},
    };
    d.add_constraint(with_index);

    m::SketchConstraint with_param;
    with_param.id = "c3";
    with_param.constraint_type = m::ConstraintType::Coincident;
    with_param.references = {
        m::PointRef{"S0", m::PointType::OnCurve, 0.5},
        m::PointRef{"P0", m::PointType::Center},
    };
    d.add_constraint(with_param);

    const auto j = json::parse(m::to_json(d));
    const auto& r1 = j["constraints"][0]["references"][0];
    CHECK(!r1.contains("parameter"));
    CHECK(!r1.contains("index"));

    const auto& r2 = j["constraints"][1]["references"][0];
    CHECK(!r2.contains("parameter"));
    CHECK(r2["index"].get<int>() == 2);

    const auto& r3 = j["constraints"][2]["references"][0];
    CHECK(r3["parameter"].get<double>() == 0.5);
    CHECK(!r3.contains("index"));
}

TEST_CASE("Deserializer: missing point components default to 0.0") {
    const std::string js = R"({
        "name": "partial",
        "primitives": [{"id":"L0","type":"line","construction":false,
                        "start":[1.0],"end":[]}],
        "constraints": [],
        "solver_status": "dirty",
        "degrees_of_freedom": -1
    })";
    const auto d = m::from_json(js);
    REQUIRE(d.primitives.size() == 1);
    const auto& l = std::get<m::Line>(d.primitives[0]);
    CHECK(l.start.x == 1.0);
    CHECK(l.start.y == 0.0);
    CHECK(l.end.x == 0.0);
    CHECK(l.end.y == 0.0);
}

TEST_CASE("Deserializer: unknown primitive type throws") {
    const std::string js = R"({
        "name": "x",
        "primitives": [{"id":"X0","type":"helix","construction":false}],
        "constraints": [],
        "solver_status": "dirty",
        "degrees_of_freedom": -1
    })";
    CHECK_THROWS_AS(m::from_json(js), std::invalid_argument);
}

TEST_CASE("Deserializer: unknown solver_status throws") {
    const std::string js = R"({
        "name": "x", "primitives": [], "constraints": [],
        "solver_status": "perfect",
        "degrees_of_freedom": 0
    })";
    CHECK_THROWS_AS(m::from_json(js), std::invalid_argument);
}

TEST_CASE("Deserializer: unknown constraint type throws") {
    const std::string js = R"({
        "name": "x",
        "primitives": [{"id":"L0","type":"line","construction":false,
                        "start":[0,0],"end":[1,0]}],
        "constraints": [{"id":"c","type":"snap","references":["L0"]}],
        "solver_status": "dirty",
        "degrees_of_freedom": -1
    })";
    CHECK_THROWS_AS(m::from_json(js), std::invalid_argument);
}

TEST_CASE("save and load roundtrip a temp file") {
    auto tmp = std::filesystem::temp_directory_path() / "morphe_cpp_serialization_test.json";
    const auto doc = make_full_document();
    m::save(doc, tmp);
    const auto loaded = m::load(tmp);
    CHECK(loaded == doc);
    std::filesystem::remove(tmp);
}

TEST_CASE("Empty document round-trips") {
    m::SketchDocument empty;
    const auto rt = m::from_json(m::to_json(empty));
    CHECK(rt == empty);
    CHECK(rt.name == "Untitled");
    CHECK(rt.primitives.empty());
    CHECK(rt.constraints.empty());
    CHECK(rt.solver_status == m::SolverStatus::Dirty);
    CHECK(rt.degrees_of_freedom == -1);
}

TEST_CASE("All 20 ConstraintTypes round-trip via JSON") {
    constexpr m::ConstraintType all[] = {
        m::ConstraintType::Coincident,    m::ConstraintType::Tangent,
        m::ConstraintType::Perpendicular, m::ConstraintType::Parallel,
        m::ConstraintType::Concentric,    m::ConstraintType::Equal,
        m::ConstraintType::Collinear,     m::ConstraintType::Horizontal,
        m::ConstraintType::Vertical,      m::ConstraintType::Fixed,
        m::ConstraintType::Distance,      m::ConstraintType::DistanceX,
        m::ConstraintType::DistanceY,     m::ConstraintType::Length,
        m::ConstraintType::Radius,        m::ConstraintType::Diameter,
        m::ConstraintType::Angle,         m::ConstraintType::Symmetric,
        m::ConstraintType::Midpoint,
    };

    for (auto t : all) {
        m::SketchDocument d;
        // Add a generic primitive so references resolve. We do not exercise
        // reference-kind validation here (Phase 4); just need the IDs to exist.
        d.add_primitive(m::Line{});
        m::SketchConstraint c;
        c.id = "c";
        c.constraint_type = t;
        c.references = {std::string{"L0"}};
        if (m::rules_for(t).value_required) c.value = 1.0;
        d.constraints.push_back(c);  // bypass document.add_constraint validation
        const auto rt = m::from_json(m::to_json(d));
        REQUIRE(rt.constraints.size() == 1);
        CHECK(rt.constraints[0].constraint_type == t);
    }
}
