#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "morphe/types.hpp"

namespace morphe {

enum class ConstraintType {
    // Point-to-point
    Coincident,
    // Curve-to-curve
    Tangent,
    Perpendicular,
    Parallel,
    Concentric,
    Equal,
    Collinear,
    // Single-element orientation
    Horizontal,
    Vertical,
    Fixed,
    // Dimensional
    Distance,
    DistanceX,
    DistanceY,
    Length,
    Radius,
    Diameter,
    Angle,
    // Symmetry
    Symmetric,
    Midpoint,  // wire string is "midpoint_constraint"
};

inline std::string_view to_string(ConstraintType t) {
    switch (t) {
        case ConstraintType::Coincident:    return "coincident";
        case ConstraintType::Tangent:       return "tangent";
        case ConstraintType::Perpendicular: return "perpendicular";
        case ConstraintType::Parallel:      return "parallel";
        case ConstraintType::Concentric:    return "concentric";
        case ConstraintType::Equal:         return "equal";
        case ConstraintType::Collinear:     return "collinear";
        case ConstraintType::Horizontal:    return "horizontal";
        case ConstraintType::Vertical:      return "vertical";
        case ConstraintType::Fixed:         return "fixed";
        case ConstraintType::Distance:      return "distance";
        case ConstraintType::DistanceX:     return "distance_x";
        case ConstraintType::DistanceY:     return "distance_y";
        case ConstraintType::Length:        return "length";
        case ConstraintType::Radius:        return "radius";
        case ConstraintType::Diameter:      return "diameter";
        case ConstraintType::Angle:         return "angle";
        case ConstraintType::Symmetric:     return "symmetric";
        case ConstraintType::Midpoint:      return "midpoint_constraint";
    }
    throw std::logic_error("unhandled ConstraintType in to_string");
}

inline ConstraintType constraint_type_from_string(std::string_view s) {
    if (s == "coincident")          return ConstraintType::Coincident;
    if (s == "tangent")             return ConstraintType::Tangent;
    if (s == "perpendicular")       return ConstraintType::Perpendicular;
    if (s == "parallel")            return ConstraintType::Parallel;
    if (s == "concentric")          return ConstraintType::Concentric;
    if (s == "equal")               return ConstraintType::Equal;
    if (s == "collinear")           return ConstraintType::Collinear;
    if (s == "horizontal")          return ConstraintType::Horizontal;
    if (s == "vertical")            return ConstraintType::Vertical;
    if (s == "fixed")               return ConstraintType::Fixed;
    if (s == "distance")            return ConstraintType::Distance;
    if (s == "distance_x")          return ConstraintType::DistanceX;
    if (s == "distance_y")          return ConstraintType::DistanceY;
    if (s == "length")              return ConstraintType::Length;
    if (s == "radius")              return ConstraintType::Radius;
    if (s == "diameter")            return ConstraintType::Diameter;
    if (s == "angle")               return ConstraintType::Angle;
    if (s == "symmetric")           return ConstraintType::Symmetric;
    if (s == "midpoint_constraint") return ConstraintType::Midpoint;
    throw std::invalid_argument(std::string{"unknown ConstraintType value: "} + std::string{s});
}

enum class ConstraintStatus {
    Unknown,
    Satisfied,
    Violated,
    Redundant,
    Conflicting,
};

inline std::string_view to_string(ConstraintStatus s) {
    switch (s) {
        case ConstraintStatus::Unknown:     return "unknown";
        case ConstraintStatus::Satisfied:   return "satisfied";
        case ConstraintStatus::Violated:    return "violated";
        case ConstraintStatus::Redundant:   return "redundant";
        case ConstraintStatus::Conflicting: return "conflicting";
    }
    throw std::logic_error("unhandled ConstraintStatus in to_string");
}

inline ConstraintStatus constraint_status_from_string(std::string_view s) {
    if (s == "unknown")     return ConstraintStatus::Unknown;
    if (s == "satisfied")   return ConstraintStatus::Satisfied;
    if (s == "violated")    return ConstraintStatus::Violated;
    if (s == "redundant")   return ConstraintStatus::Redundant;
    if (s == "conflicting") return ConstraintStatus::Conflicting;
    throw std::invalid_argument(std::string{"unknown ConstraintStatus value: "} + std::string{s});
}

// Reference variant: a constraint may name an entire primitive by element ID
// (e.g., Horizontal on a line) or a specific point on a primitive (e.g.,
// Coincident between two endpoints).
using ConstraintRef = std::variant<std::string, PointRef>;

inline const std::string& referenced_element_id(const ConstraintRef& r) {
    struct V {
        const std::string& operator()(const std::string& s) const { return s; }
        const std::string& operator()(const PointRef& p)    const { return p.element_id; }
    };
    return std::visit(V{}, r);
}

// Allowed primitive categories for a constraint reference.
//   Any    - any primitive
//   Point  - the reference must be a PointRef (constraint operates on a specific point)
//   Curve  - Line, Arc, Circle, or Spline (any 1D curve)
//   Line   - Line only
//   Arc    - Arc only
//   Circle - Circle only
//   ArcOrCircle - Arc or Circle (e.g., for Concentric, Radius)
//   LineOrArcOrCircle - Line, Arc, or Circle (e.g., for Equal)
enum class RefKind {
    Any,
    Point,
    Curve,
    Line,
    Arc,
    Circle,
    ArcOrCircle,
    LineOrArcOrCircle,
};

// Applicability rules per ConstraintType. Mirrors CONSTRAINT_RULES in
// morphe/constraints.py. `max_refs` is std::nullopt for chainable constraints
// (Equal, Collinear). `ref_kinds` is either size 1 (homogeneous - all refs must
// match), the exact list of kinds (one per ref position, if max_refs is set),
// or a position-by-position spec used only when sizes match.
struct ConstraintRules {
    int min_refs;
    std::optional<int> max_refs;
    std::vector<RefKind> ref_kinds;  // see notes above
    bool value_required;
};

// Defined in src/constraints.cpp.
const ConstraintRules& rules_for(ConstraintType t);

struct SketchConstraint {
    std::string id;
    ConstraintType constraint_type = ConstraintType::Coincident;
    std::vector<ConstraintRef> references;
    std::optional<double> value;
    std::optional<PointRef> connection_point;

    // Metadata
    bool inferred = false;
    double confidence = 1.0;
    std::optional<std::string> source;
    ConstraintStatus status = ConstraintStatus::Unknown;

    // All distinct element IDs touched by this constraint.
    std::vector<std::string> get_element_ids() const {
        std::vector<std::string> ids;
        auto push = [&](const std::string& elem_id) {
            for (const auto& existing : ids) {
                if (existing == elem_id) return;
            }
            ids.push_back(elem_id);
        };
        for (const auto& r : references) push(referenced_element_id(r));
        if (connection_point) push(connection_point->element_id);
        return ids;
    }

    friend bool operator==(const SketchConstraint& a, const SketchConstraint& b) {
        return a.id == b.id
            && a.constraint_type == b.constraint_type
            && a.references == b.references
            && a.value == b.value
            && a.connection_point == b.connection_point
            && a.inferred == b.inferred
            && a.confidence == b.confidence
            && a.source == b.source
            && a.status == b.status;
    }
    friend bool operator!=(const SketchConstraint& a, const SketchConstraint& b) { return !(a == b); }
};

}  // namespace morphe
