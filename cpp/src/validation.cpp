#include "morphe/validation.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_set>
#include <variant>

namespace morphe {

namespace {

bool is_finite(double v) { return std::isfinite(v); }

template <typename T, typename... Args>
std::string concat(const T& first, const Args&... args) {
    std::ostringstream s;
    s << first;
    ((s << args), ...);
    return s.str();
}

bool point_type_valid_for(const Primitive& p, PointType t) {
    const auto valid = valid_point_types(p);
    return std::find(valid.begin(), valid.end(), t) != valid.end();
}

// ---------- per-primitive ----------

void validate_meta(const PrimitiveMeta& meta, ValidationResult& r) {
    if (meta.id.empty()) {
        r.add_error("Primitive has empty ID", "PRIM_EMPTY_ID", meta.id);
    }
    if (!(meta.confidence >= 0.0 && meta.confidence <= 1.0)) {
        r.add_warning(concat("Confidence ", meta.confidence, " outside [0,1] range"),
                      "PRIM_CONFIDENCE_RANGE", meta.id);
    }
}

void validate_line(const Line& l, ValidationResult& r, double tol) {
    if (l.length() < tol) {
        r.add_error(concat("Line has zero length (length=", l.length(), ")"),
                    "LINE_ZERO_LENGTH", l.meta.id);
    }
    if (!is_finite(l.start.x) || !is_finite(l.start.y) ||
        !is_finite(l.end.x)   || !is_finite(l.end.y)) {
        r.add_error("Line has non-finite coordinates", "LINE_INVALID_COORDS", l.meta.id);
    }
}

void validate_arc(const Arc& a, ValidationResult& r, double tol) {
    if (!is_finite(a.center.x) || !is_finite(a.center.y) ||
        !is_finite(a.start_point.x) || !is_finite(a.start_point.y) ||
        !is_finite(a.end_point.x)   || !is_finite(a.end_point.y)) {
        r.add_error("Arc has non-finite coordinates", "ARC_INVALID_COORDS", a.meta.id);
        return;
    }
    const double r_start = a.center.distance_to(a.start_point);
    const double r_end   = a.center.distance_to(a.end_point);
    if (std::abs(r_start - r_end) > tol) {
        r.add_error(concat("Arc radius inconsistent (start=", r_start, ", end=", r_end, ")"),
                    "ARC_RADIUS_INCONSISTENT", a.meta.id);
    }
    if (r_start < tol) {
        r.add_error("Arc has zero radius", "ARC_ZERO_RADIUS", a.meta.id);
    }
    if (a.start_point.distance_to(a.end_point) < tol) {
        r.add_warning(
            "Arc start and end points are coincident (consider using Circle instead)",
            "ARC_DEGENERATE", a.meta.id);
    }
}

void validate_circle(const Circle& c, ValidationResult& r, double tol) {
    if (!is_finite(c.center.x) || !is_finite(c.center.y) || !is_finite(c.radius)) {
        r.add_error("Circle has non-finite values", "CIRCLE_INVALID_VALUES", c.meta.id);
        return;
    }
    if (c.radius <= 0.0) {
        r.add_error(concat("Circle has non-positive radius (", c.radius, ")"),
                    "CIRCLE_INVALID_RADIUS", c.meta.id);
    } else if (c.radius < tol) {
        r.add_warning(concat("Circle has very small radius (", c.radius, ")"),
                      "CIRCLE_TINY_RADIUS", c.meta.id);
    }
}

void validate_point(const Point& p, ValidationResult& r) {
    if (!is_finite(p.position.x) || !is_finite(p.position.y)) {
        r.add_error("Point has non-finite coordinates", "POINT_INVALID_COORDS", p.meta.id);
    }
}

bool spline_knot_count_ok(const Spline& s) {
    const int n = static_cast<int>(s.control_points.size());
    const int k = s.order();
    const int expected = s.periodic ? (n + 1) : (n + k);
    return static_cast<int>(s.knots.size()) == expected;
}

void validate_spline(const Spline& s, ValidationResult& r, double /*tol*/) {
    if (s.degree < 1) {
        r.add_error(concat("Spline degree must be at least 1 (got ", s.degree, ")"),
                    "SPLINE_INVALID_DEGREE", s.meta.id);
    }
    const int min_points = s.degree + 1;
    if (static_cast<int>(s.control_points.size()) < min_points) {
        r.add_error(concat("Spline needs at least ", min_points,
                           " control points for degree ", s.degree,
                           " (got ", s.control_points.size(), ")"),
                    "SPLINE_INSUFFICIENT_POINTS", s.meta.id);
    }
    for (size_t i = 0; i < s.control_points.size(); ++i) {
        const auto& cp = s.control_points[i];
        if (!is_finite(cp.x) || !is_finite(cp.y)) {
            r.add_error(concat("Spline control point ", i, " has non-finite coordinates"),
                        "SPLINE_INVALID_CONTROL_POINT", s.meta.id);
        }
    }
    if (!s.knots.empty()) {
        for (size_t i = 1; i < s.knots.size(); ++i) {
            if (s.knots[i] < s.knots[i - 1]) {
                r.add_error(concat("Spline knot vector is not non-decreasing at index ", i),
                            "SPLINE_INVALID_KNOTS", s.meta.id);
                break;
            }
        }
        if (!spline_knot_count_ok(s)) {
            const int n = static_cast<int>(s.control_points.size());
            const int expected = s.periodic ? (n + 1) : (n + s.order());
            r.add_warning(concat("Spline knot vector length (", s.knots.size(),
                                 ") doesn't match expected (", expected, ")"),
                          "SPLINE_KNOT_LENGTH_MISMATCH", s.meta.id);
        }
    }
    if (s.weights.has_value()) {
        const auto& w = *s.weights;
        if (w.size() != s.control_points.size()) {
            r.add_error(concat("Spline weights count (", w.size(),
                               ") doesn't match control points (",
                               s.control_points.size(), ")"),
                        "SPLINE_WEIGHT_COUNT_MISMATCH", s.meta.id);
        }
        for (size_t i = 0; i < w.size(); ++i) {
            if (!is_finite(w[i])) {
                r.add_error(concat("Spline weight ", i, " is non-finite"),
                            "SPLINE_INVALID_WEIGHT", s.meta.id);
            } else if (w[i] <= 0.0) {
                r.add_error(concat("Spline weight ", i, " is non-positive (", w[i], ")"),
                            "SPLINE_INVALID_WEIGHT", s.meta.id);
            }
        }
    }
}

void validate_ellipse(const Ellipse& e, ValidationResult& r) {
    if (!is_finite(e.center.x) || !is_finite(e.center.y)) {
        r.add_error("Ellipse has non-finite center coordinates",
                    "ELLIPSE_INVALID_COORDS", e.meta.id);
        return;
    }
    if (e.major_radius <= 0.0) {
        r.add_error(concat("Ellipse major_radius must be positive (got ",
                           e.major_radius, ")"),
                    "ELLIPSE_INVALID_MAJOR_RADIUS", e.meta.id);
    }
    if (e.minor_radius <= 0.0) {
        r.add_error(concat("Ellipse minor_radius must be positive (got ",
                           e.minor_radius, ")"),
                    "ELLIPSE_INVALID_MINOR_RADIUS", e.meta.id);
    }
    if (e.major_radius < e.minor_radius) {
        r.add_error(concat("Ellipse major_radius (", e.major_radius,
                           ") must be >= minor_radius (", e.minor_radius, ")"),
                    "ELLIPSE_MAJOR_LESS_THAN_MINOR", e.meta.id);
    }
}

void validate_elliptical_arc(const EllipticalArc& a, ValidationResult& r) {
    if (!is_finite(a.center.x) || !is_finite(a.center.y)) {
        r.add_error("EllipticalArc has non-finite center coordinates",
                    "ELLIPTICAL_ARC_INVALID_COORDS", a.meta.id);
        return;
    }
    if (a.major_radius <= 0.0) {
        r.add_error(concat("EllipticalArc major_radius must be positive (got ",
                           a.major_radius, ")"),
                    "ELLIPTICAL_ARC_INVALID_MAJOR_RADIUS", a.meta.id);
    }
    if (a.minor_radius <= 0.0) {
        r.add_error(concat("EllipticalArc minor_radius must be positive (got ",
                           a.minor_radius, ")"),
                    "ELLIPTICAL_ARC_INVALID_MINOR_RADIUS", a.meta.id);
    }
    if (a.major_radius < a.minor_radius) {
        r.add_error(concat("EllipticalArc major_radius (", a.major_radius,
                           ") must be >= minor_radius (", a.minor_radius, ")"),
                    "ELLIPTICAL_ARC_MAJOR_LESS_THAN_MINOR", a.meta.id);
    }
    if (!is_finite(a.start_param)) {
        r.add_error("EllipticalArc has non-finite start_param",
                    "ELLIPTICAL_ARC_INVALID_START_PARAM", a.meta.id);
    }
    if (!is_finite(a.end_param)) {
        r.add_error("EllipticalArc has non-finite end_param",
                    "ELLIPTICAL_ARC_INVALID_END_PARAM", a.meta.id);
    }
    if (is_finite(a.start_param) && is_finite(a.end_param) &&
        a.start_param == a.end_param) {
        r.add_error("EllipticalArc start_param and end_param must differ",
                    "ELLIPTICAL_ARC_ZERO_SWEEP", a.meta.id);
    }
}

void validate_primitive(const Primitive& p, ValidationResult& r, double tol) {
    validate_meta(meta_of(p), r);
    struct V {
        ValidationResult& r;
        double tol;
        void operator()(const Line& x)          const { validate_line(x, r, tol); }
        void operator()(const Arc& x)           const { validate_arc(x, r, tol); }
        void operator()(const Circle& x)        const { validate_circle(x, r, tol); }
        void operator()(const Point& x)         const { validate_point(x, r); }
        void operator()(const Spline& x)        const { validate_spline(x, r, tol); }
        void operator()(const Ellipse& x)       const { validate_ellipse(x, r); }
        void operator()(const EllipticalArc& x) const { validate_elliptical_arc(x, r); }
    };
    std::visit(V{r, tol}, p);
}

// ---------- per-constraint ----------

void validate_constraint(const SketchConstraint& c,
                         const SketchDocument& doc,
                         ValidationResult& r) {
    if (c.id.empty()) {
        r.add_warning("Constraint has empty ID", "CONST_EMPTY_ID");
    }

    const ConstraintRules* rules = nullptr;
    try {
        rules = &rules_for(c.constraint_type);
    } catch (...) {
        r.add_error("Unknown constraint type", "CONST_UNKNOWN_TYPE");
        return;
    }

    const int ref_count = static_cast<int>(c.references.size());
    if (ref_count < rules->min_refs) {
        r.add_error(concat(to_string(c.constraint_type),
                           ": Too few references (need ", rules->min_refs,
                           ", got ", ref_count, ")"),
                    "CONST_TOO_FEW_REFS");
    }
    if (rules->max_refs.has_value() && ref_count > *rules->max_refs) {
        r.add_error(concat(to_string(c.constraint_type),
                           ": Too many references (max ", *rules->max_refs,
                           ", got ", ref_count, ")"),
                    "CONST_TOO_MANY_REFS");
    }

    if (rules->value_required && !c.value.has_value()) {
        r.add_error(concat(to_string(c.constraint_type), ": Missing required value"),
                    "CONST_MISSING_VALUE");
    }

    if (c.value.has_value() && !is_finite(*c.value)) {
        r.add_error(concat(to_string(c.constraint_type), ": Value is non-finite"),
                    "CONST_INVALID_VALUE");
    }

    // Sign check on Length / Radius / Diameter / Distance.
    if (c.value.has_value() && is_finite(*c.value)) {
        const auto t = c.constraint_type;
        if (t == ConstraintType::Length   ||
            t == ConstraintType::Radius   ||
            t == ConstraintType::Diameter ||
            t == ConstraintType::Distance) {
            if (*c.value < 0.0) {
                r.add_error(concat(to_string(t),
                                   ": Value must be non-negative (got ",
                                   *c.value, ")"),
                            "CONST_NEGATIVE_VALUE");
            }
        }
    }

    // Reference resolution.
    for (const auto& ref : c.references) {
        std::visit([&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::string>) {
                if (doc.find_primitive(v) == nullptr) {
                    r.add_error(concat("Constraint references non-existent element '", v, "'"),
                                "CONST_INVALID_REF");
                }
            } else {
                const PointRef& pr = v;
                const Primitive* prim = doc.find_primitive(pr.element_id);
                if (prim == nullptr) {
                    r.add_error(concat("Constraint references non-existent element '",
                                       pr.element_id, "'"),
                                "CONST_INVALID_REF");
                    return;
                }
                if (!point_type_valid_for(*prim, pr.point_type)) {
                    r.add_error(concat("Invalid point type ", to_string(pr.point_type),
                                       " for ", pr.element_id),
                                "CONST_INVALID_POINT_TYPE");
                }
                if (pr.point_type == PointType::Control) {
                    if (!pr.index.has_value()) {
                        r.add_error(concat("CONTROL point type requires index for ",
                                           pr.element_id),
                                    "CONST_MISSING_INDEX");
                    } else if (std::holds_alternative<Spline>(*prim)) {
                        const auto& sp = std::get<Spline>(*prim);
                        const int idx = *pr.index;
                        if (idx < 0 || idx >= static_cast<int>(sp.control_points.size())) {
                            r.add_error(concat("Control point index ", idx,
                                               " out of range for ", pr.element_id),
                                        "CONST_INDEX_OUT_OF_RANGE");
                        }
                    }
                }
            }
        }, ref);
    }

    // connection_point.
    if (c.connection_point.has_value()) {
        const auto& cp = *c.connection_point;
        const Primitive* prim = doc.find_primitive(cp.element_id);
        if (prim == nullptr) {
            r.add_error(concat("Connection point references non-existent element '",
                               cp.element_id, "'"),
                        "CONST_INVALID_CONNECTION_POINT");
        } else if (!point_type_valid_for(*prim, cp.point_type)) {
            r.add_error(concat("Invalid connection point type ",
                               to_string(cp.point_type), " for ", cp.element_id),
                        "CONST_INVALID_CONNECTION_POINT_TYPE");
        }
    }

    if (!(c.confidence >= 0.0 && c.confidence <= 1.0)) {
        r.add_warning(concat("Constraint confidence ", c.confidence,
                             " outside [0,1] range"),
                      "CONST_CONFIDENCE_RANGE");
    }
}

void check_duplicate_constraint_ids(const SketchDocument& doc, ValidationResult& r) {
    std::unordered_set<std::string> seen;
    for (const auto& c : doc.constraints) {
        if (!seen.insert(c.id).second) {
            r.add_warning(concat("Duplicate constraint ID: ", c.id),
                          "CONST_DUPLICATE_ID");
        }
    }
}

}  // namespace

ValidationResult validate(const SketchDocument& doc, double tolerance) {
    ValidationResult r;
    for (const auto& p : doc.primitives) validate_primitive(p, r, tolerance);
    for (const auto& c : doc.constraints) validate_constraint(c, doc, r);
    check_duplicate_constraint_ids(doc, r);
    return r;
}

}  // namespace morphe
