#pragma once

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace morphe {

struct Point2D {
    double x = 0.0;
    double y = 0.0;

    constexpr Point2D() = default;
    constexpr Point2D(double x_, double y_) : x(x_), y(y_) {}

    double distance_to(const Point2D& other) const {
        const double dx = x - other.x;
        const double dy = y - other.y;
        return std::sqrt(dx * dx + dy * dy);
    }

    friend constexpr bool operator==(const Point2D& a, const Point2D& b) {
        return a.x == b.x && a.y == b.y;
    }
    friend constexpr bool operator!=(const Point2D& a, const Point2D& b) {
        return !(a == b);
    }
};

struct Vector2D {
    double dx = 0.0;
    double dy = 0.0;

    constexpr Vector2D() = default;
    constexpr Vector2D(double dx_, double dy_) : dx(dx_), dy(dy_) {}

    double magnitude() const { return std::sqrt(dx * dx + dy * dy); }

    friend constexpr bool operator==(const Vector2D& a, const Vector2D& b) {
        return a.dx == b.dx && a.dy == b.dy;
    }
    friend constexpr bool operator!=(const Vector2D& a, const Vector2D& b) {
        return !(a == b);
    }
};

// Element ID prefix conventions, matching morphe.types.ElementPrefix.
// Note that ELLIPTICAL_ARC uses lowercase 'e' to disambiguate from Ellipse.
namespace element_prefix {
inline constexpr char line           = 'L';
inline constexpr char arc            = 'A';
inline constexpr char circle         = 'C';
inline constexpr char point          = 'P';
inline constexpr char spline         = 'S';
inline constexpr char ellipse        = 'E';
inline constexpr char elliptical_arc = 'e';
}  // namespace element_prefix

enum class PointType {
    Start,     // Line start, Arc start
    End,       // Line end, Arc end
    Center,    // Arc center, Circle center, Ellipse center, standalone Point
    Midpoint,  // Computed midpoint (lines and arcs)
    Control,   // Spline control point (requires index)
    OnCurve,   // Arbitrary point on curve (requires parameter)
};

inline std::string_view to_string(PointType t) {
    switch (t) {
        case PointType::Start:    return "start";
        case PointType::End:      return "end";
        case PointType::Center:   return "center";
        case PointType::Midpoint: return "midpoint";
        case PointType::Control:  return "control";
        case PointType::OnCurve:  return "on_curve";
    }
    throw std::logic_error("unhandled PointType in to_string");
}

inline PointType point_type_from_string(std::string_view s) {
    if (s == "start")    return PointType::Start;
    if (s == "end")      return PointType::End;
    if (s == "center")   return PointType::Center;
    if (s == "midpoint") return PointType::Midpoint;
    if (s == "control")  return PointType::Control;
    if (s == "on_curve") return PointType::OnCurve;
    throw std::invalid_argument(std::string{"unknown PointType value: "} + std::string{s});
}

// Reference to a specific point on a primitive.
//
//   PointRef{"L0", PointType::Start}                     - start of line L0
//   PointRef{"A1", PointType::Center}                    - center of arc A1
//   PointRef{"S2", PointType::Control, std::nullopt, 3}  - 4th control point of spline S2
//   PointRef{"S2", PointType::OnCurve, 0.5}              - midway on spline S2 (in param)
struct PointRef {
    std::string element_id;
    PointType point_type = PointType::Center;
    std::optional<double> parameter;  // ON_CURVE
    std::optional<int>    index;      // CONTROL

    PointRef() = default;
    PointRef(std::string id, PointType t,
             std::optional<double> param = std::nullopt,
             std::optional<int> idx = std::nullopt)
        : element_id(std::move(id)), point_type(t), parameter(param), index(idx) {}

    friend bool operator==(const PointRef& a, const PointRef& b) {
        return a.element_id == b.element_id
            && a.point_type == b.point_type
            && a.parameter  == b.parameter
            && a.index      == b.index;
    }
    friend bool operator!=(const PointRef& a, const PointRef& b) { return !(a == b); }
};

}  // namespace morphe
