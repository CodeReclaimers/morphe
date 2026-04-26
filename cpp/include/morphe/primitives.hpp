#pragma once

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "morphe/types.hpp"

namespace morphe {

// Common metadata carried by every primitive.
//
// Field semantics match morphe.primitives.SketchPrimitive:
//   id           - stable identifier, format "<prefix><index>" (e.g., "L0", "A1")
//   construction - true means reference geometry, not part of the profile
//   source       - origin tag: "fitted", "user", "inferred", etc.; serialized only when set
//   confidence   - reconstruction confidence in [0, 1]; serialized only when != 1.0
struct PrimitiveMeta {
    std::string id;
    bool construction = false;
    std::optional<std::string> source;
    double confidence = 1.0;

    friend bool operator==(const PrimitiveMeta& a, const PrimitiveMeta& b) {
        return a.id == b.id
            && a.construction == b.construction
            && a.source == b.source
            && a.confidence == b.confidence;
    }
    friend bool operator!=(const PrimitiveMeta& a, const PrimitiveMeta& b) { return !(a == b); }
};

struct Line {
    PrimitiveMeta meta;
    Point2D start;
    Point2D end;

    double length() const { return start.distance_to(end); }

    friend bool operator==(const Line& a, const Line& b) {
        return a.meta == b.meta && a.start == b.start && a.end == b.end;
    }
    friend bool operator!=(const Line& a, const Line& b) { return !(a == b); }
};

struct Arc {
    PrimitiveMeta meta;
    Point2D center;
    Point2D start_point;
    Point2D end_point;
    bool ccw = true;

    // Radius is implicit: distance from center to start_point. Validation should
    // ensure |center - start| ~ |center - end| within a tolerance (see validation.hpp).
    double radius() const { return center.distance_to(start_point); }

    friend bool operator==(const Arc& a, const Arc& b) {
        return a.meta == b.meta
            && a.center == b.center
            && a.start_point == b.start_point
            && a.end_point == b.end_point
            && a.ccw == b.ccw;
    }
    friend bool operator!=(const Arc& a, const Arc& b) { return !(a == b); }
};

struct Circle {
    PrimitiveMeta meta;
    Point2D center;
    double radius = 1.0;

    friend bool operator==(const Circle& a, const Circle& b) {
        return a.meta == b.meta && a.center == b.center && a.radius == b.radius;
    }
    friend bool operator!=(const Circle& a, const Circle& b) { return !(a == b); }
};

// Standalone sketch point. PointType::Center references the point itself.
struct Point {
    PrimitiveMeta meta;
    Point2D position;

    friend bool operator==(const Point& a, const Point& b) {
        return a.meta == b.meta && a.position == b.position;
    }
    friend bool operator!=(const Point& a, const Point& b) { return !(a == b); }
};

struct Spline {
    PrimitiveMeta meta;
    int degree = 3;
    std::vector<Point2D> control_points;
    std::vector<double>  knots;
    std::optional<std::vector<double>> weights;  // null = non-rational (uniform)
    bool periodic = false;
    bool is_fit_spline = false;  // true = control_points are interpolation targets

    int order() const { return degree + 1; }
    bool is_rational() const { return weights.has_value(); }

    friend bool operator==(const Spline& a, const Spline& b) {
        return a.meta == b.meta
            && a.degree == b.degree
            && a.control_points == b.control_points
            && a.knots == b.knots
            && a.weights == b.weights
            && a.periodic == b.periodic
            && a.is_fit_spline == b.is_fit_spline;
    }
    friend bool operator!=(const Spline& a, const Spline& b) { return !(a == b); }
};

struct Ellipse {
    PrimitiveMeta meta;
    Point2D center;
    double major_radius = 1.0;
    double minor_radius = 0.5;
    double rotation = 0.0;  // radians, angle of major axis from positive X

    friend bool operator==(const Ellipse& a, const Ellipse& b) {
        return a.meta == b.meta
            && a.center == b.center
            && a.major_radius == b.major_radius
            && a.minor_radius == b.minor_radius
            && a.rotation == b.rotation;
    }
    friend bool operator!=(const Ellipse& a, const Ellipse& b) { return !(a == b); }
};

struct EllipticalArc {
    PrimitiveMeta meta;
    Point2D center;
    double major_radius = 1.0;
    double minor_radius = 0.5;
    double rotation = 0.0;
    double start_param = 0.0;
    double end_param = 1.5707963267948966;  // pi/2, matches Python default
    bool ccw = true;

    friend bool operator==(const EllipticalArc& a, const EllipticalArc& b) {
        return a.meta == b.meta
            && a.center == b.center
            && a.major_radius == b.major_radius
            && a.minor_radius == b.minor_radius
            && a.rotation == b.rotation
            && a.start_param == b.start_param
            && a.end_param == b.end_param
            && a.ccw == b.ccw;
    }
    friend bool operator!=(const EllipticalArc& a, const EllipticalArc& b) { return !(a == b); }
};

using Primitive = std::variant<Line, Arc, Circle, Point, Spline, Ellipse, EllipticalArc>;

// Free-function accessors over the variant. Avoid hand-rolling visit + switch at every call site.

inline const PrimitiveMeta& meta_of(const Primitive& p) {
    return std::visit([](const auto& v) -> const PrimitiveMeta& { return v.meta; }, p);
}
inline PrimitiveMeta& meta_of(Primitive& p) {
    return std::visit([](auto& v) -> PrimitiveMeta& { return v.meta; }, p);
}

inline const std::string& id_of(const Primitive& p) { return meta_of(p).id; }

// Wire-format type tag. Lowercase, no separators. Note "ellipticalarc" is one word
// to match morphe.serialization's `type(p).__name__.lower()` output.
inline std::string_view type_tag(const Primitive& p) {
    struct V {
        std::string_view operator()(const Line&)          const { return "line"; }
        std::string_view operator()(const Arc&)           const { return "arc"; }
        std::string_view operator()(const Circle&)        const { return "circle"; }
        std::string_view operator()(const Point&)         const { return "point"; }
        std::string_view operator()(const Spline&)        const { return "spline"; }
        std::string_view operator()(const Ellipse&)       const { return "ellipse"; }
        std::string_view operator()(const EllipticalArc&) const { return "ellipticalarc"; }
    };
    return std::visit(V{}, p);
}

// ID prefix (single character) for a primitive's stable ID generation.
inline char id_prefix(const Primitive& p) {
    struct V {
        char operator()(const Line&)          const { return element_prefix::line; }
        char operator()(const Arc&)           const { return element_prefix::arc; }
        char operator()(const Circle&)        const { return element_prefix::circle; }
        char operator()(const Point&)         const { return element_prefix::point; }
        char operator()(const Spline&)        const { return element_prefix::spline; }
        char operator()(const Ellipse&)       const { return element_prefix::ellipse; }
        char operator()(const EllipticalArc&) const { return element_prefix::elliptical_arc; }
    };
    return std::visit(V{}, p);
}

// Which PointTypes are referenceable on this primitive? Mirrors
// SketchPrimitive.get_valid_point_types() in morphe/primitives.py.
inline std::vector<PointType> valid_point_types(const Primitive& p) {
    struct V {
        std::vector<PointType> operator()(const Line&) const {
            return {PointType::Start, PointType::End, PointType::Midpoint};
        }
        std::vector<PointType> operator()(const Arc&) const {
            return {PointType::Start, PointType::End, PointType::Center, PointType::Midpoint};
        }
        std::vector<PointType> operator()(const Circle&) const {
            return {PointType::Center};
        }
        std::vector<PointType> operator()(const Point&) const {
            return {PointType::Center};
        }
        std::vector<PointType> operator()(const Spline&) const {
            return {PointType::Start, PointType::End, PointType::Control};
        }
        std::vector<PointType> operator()(const Ellipse&) const {
            return {PointType::Center};
        }
        std::vector<PointType> operator()(const EllipticalArc&) const {
            return {PointType::Start, PointType::End, PointType::Center, PointType::Midpoint};
        }
    };
    return std::visit(V{}, p);
}

}  // namespace morphe
