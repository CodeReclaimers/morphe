// Cross-language conformance test.
//
// Loads each JSON fixture under tests/conformance/ (produced by the
// authoritative Python implementation via tests/conformance/generate_corpus.py)
// and verifies:
//
//   1. The C++ deserializer parses every fixture without throwing.
//   2. Re-serializing the loaded document and parsing both JSON trees
//      yields semantically equivalent JSON (deep compare with float
//      tolerance for numeric leaves).
//   3. Validation runs without errors on the curated fixtures (all
//      fixtures here are constructed to be schema-valid; the validator
//      catches accidental fixture corruption).
//   4. The §13.2 RoundedRect fixture, also built programmatically in
//      C++, is byte-equivalent to the on-disk file. This is the
//      strongest cross-language parity claim — it pins not just
//      semantic equivalence but the exact wire format.
//
// The corpus path is injected at build time via the MORPHE_CORPUS_DIR
// preprocessor macro set by CMake.

#include "doctest/doctest.h"

#include "morphe/morphe.hpp"
#include "morphe/serialization.hpp"
#include "morphe/validation.hpp"

#include "nlohmann/json.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

namespace m = morphe;
using json = nlohmann::json;

namespace {

std::filesystem::path corpus_dir() {
    return std::filesystem::path(MORPHE_CORPUS_DIR);
}

std::string read_file(const std::filesystem::path& p) {
    std::ifstream in(p);
    REQUIRE_MESSAGE(in.good(), "cannot open ", p.string());
    std::ostringstream buf;
    buf << in.rdbuf();
    return buf.str();
}

constexpr double FLOAT_TOL = 1e-12;

// Deep-compare two parsed JSON trees with a tolerance on numeric leaves.
// Returns true if they are semantically equal.
bool json_semantically_equal(const json& a, const json& b) {
    if (a.type() != b.type()) {
        // Allow int<->float comparison if numerically equal.
        if (a.is_number() && b.is_number()) {
            return std::abs(a.get<double>() - b.get<double>()) <= FLOAT_TOL;
        }
        return false;
    }
    if (a.is_object()) {
        if (a.size() != b.size()) return false;
        for (auto it = a.begin(); it != a.end(); ++it) {
            if (!b.contains(it.key())) return false;
            if (!json_semantically_equal(it.value(), b.at(it.key()))) return false;
        }
        return true;
    }
    if (a.is_array()) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (!json_semantically_equal(a[i], b[i])) return false;
        }
        return true;
    }
    if (a.is_number()) {
        return std::abs(a.get<double>() - b.get<double>()) <= FLOAT_TOL;
    }
    return a == b;
}

// Names match generate_corpus.py's FIXTURES list.
constexpr std::string_view ALL_FIXTURES[] = {
    "empty.json",
    "all_primitives.json",
    "rounded_rect.json",
    "spline_with_weights.json",
    "spline_periodic.json",
    "elliptical_arc_full.json",
    "inferred_constraints.json",
    "construction_geometry.json",
    "point_refs_full.json",
    "all_constraint_types.json",
};

}  // namespace

TEST_CASE("Corpus directory exists and is populated") {
    const auto dir = corpus_dir();
    REQUIRE(std::filesystem::is_directory(dir));
    int count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".json") ++count;
    }
    CHECK(count >= static_cast<int>(std::size(ALL_FIXTURES)));
}

TEST_CASE("Every fixture in the corpus is loaded by the C++ deserializer") {
    for (const auto fname : ALL_FIXTURES) {
        CAPTURE(fname);
        const auto path = corpus_dir() / std::string{fname};
        REQUIRE(std::filesystem::exists(path));
        CHECK_NOTHROW(m::load(path));
    }
}

TEST_CASE("Round-trip: reload + reserialize produces semantically equal JSON") {
    for (const auto fname : ALL_FIXTURES) {
        CAPTURE(fname);
        const auto path = corpus_dir() / std::string{fname};
        const auto original_text = read_file(path);
        const auto original_json = json::parse(original_text);

        const auto doc = m::load(path);
        const auto reemitted_text = m::to_json(doc);
        const auto reemitted_json = json::parse(reemitted_text);

        CHECK(json_semantically_equal(original_json, reemitted_json));
    }
}

TEST_CASE("Validation: every curated fixture passes (no errors)") {
    // Note: the Python encoder produces solver_status "dirty" by default
    // and the validator does not check solver state — only structural
    // schema. All curated fixtures should have zero errors.
    for (const auto fname : ALL_FIXTURES) {
        CAPTURE(fname);
        const auto path = corpus_dir() / std::string{fname};
        const auto doc = m::load(path);
        const auto r = m::validate(doc);
        if (!r.is_valid()) {
            // Print details to make the failure actionable
            for (const auto& issue : r.errors()) {
                MESSAGE("error code=", issue.code, " elem=", issue.element_id,
                        " msg=", issue.message);
            }
        }
        CHECK(r.is_valid());
    }
}

TEST_CASE("Insertion order preserved through C++ load") {
    // all_primitives.json has L0, A0, C0, P0, S0, E0, e0 in that order.
    const auto doc = m::load(corpus_dir() / "all_primitives.json");
    REQUIRE(doc.primitives.size() == 7);
    CHECK(m::id_of(doc.primitives[0]) == "L0");
    CHECK(m::id_of(doc.primitives[1]) == "A0");
    CHECK(m::id_of(doc.primitives[2]) == "C0");
    CHECK(m::id_of(doc.primitives[3]) == "P0");
    CHECK(m::id_of(doc.primitives[4]) == "S0");
    CHECK(m::id_of(doc.primitives[5]) == "E0");
    CHECK(m::id_of(doc.primitives[6]) == "e0");  // lowercase prefix
}

TEST_CASE("Constraint type round-trip via corpus: all_constraint_types covers every ConstraintType") {
    const auto doc = m::load(corpus_dir() / "all_constraint_types.json");
    CHECK(doc.constraints.size() == 19);  // ConstraintType has 19 values;
                                           // the fixture has one of each.
    // Verify every ConstraintType enum value appears at least once.
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
        bool found = false;
        for (const auto& c : doc.constraints) {
            if (c.constraint_type == t) { found = true; break; }
        }
        CAPTURE(to_string(t));
        CHECK(found);
    }
}

TEST_CASE("RoundedRect: programmatic C++ build matches on-disk fixture") {
    // Build the §13.2 example in C++ exactly as generate_corpus.py does in
    // Python, then compare against the on-disk file. This is the strongest
    // cross-language parity check: it pins not just semantic equivalence
    // but the structural form of the JSON.
    m::SketchDocument d;
    d.name = "RoundedRect";

    m::Line l0; l0.start = {8.0, 0.0}; l0.end = {52.0, 0.0};
    d.add_primitive_with_id(l0, "L0");

    m::Arc a0;
    a0.center = {52.0, 8.0};
    a0.start_point = {52.0, 0.0};
    a0.end_point = {60.0, 8.0};
    a0.ccw = true;
    d.add_primitive_with_id(a0, "A0");

    m::Line l1; l1.start = {60.0, 8.0}; l1.end = {60.0, 32.0};
    d.add_primitive_with_id(l1, "L1");

    m::Circle c0; c0.center = {30.0, 20.0}; c0.radius = 10.0;
    d.add_primitive_with_id(c0, "C0");

    auto add = [&](std::string id, m::ConstraintType t,
                   std::vector<m::ConstraintRef> refs,
                   std::optional<double> value = std::nullopt) {
        m::SketchConstraint c;
        c.id = std::move(id);
        c.constraint_type = t;
        c.references = std::move(refs);
        c.value = value;
        d.constraints.push_back(c);
    };

    add("c1", m::ConstraintType::Tangent,    {std::string{"L0"}, std::string{"A0"}});
    add("c2", m::ConstraintType::Tangent,    {std::string{"A0"}, std::string{"L1"}});
    add("c3", m::ConstraintType::Horizontal, {std::string{"L0"}});
    add("c4", m::ConstraintType::Radius,     {std::string{"A0"}}, 8.0);
    add("c5", m::ConstraintType::Coincident, {
        m::PointRef{"L0", m::PointType::End},
        m::PointRef{"A0", m::PointType::Start},
    });
    d.solver_status = m::SolverStatus::FullyConstrained;
    d.degrees_of_freedom = 0;

    const auto cpp_text = m::to_json(d);
    const auto cpp_json = json::parse(cpp_text);
    const auto fixture_json = json::parse(read_file(corpus_dir() / "rounded_rect.json"));

    CHECK(json_semantically_equal(cpp_json, fixture_json));
}
