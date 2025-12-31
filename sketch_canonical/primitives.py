"""Geometry primitives for the canonical sketch schema."""

from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod
import math

from .types import Point2D, Vector2D, PointType


@dataclass
class SketchPrimitive(ABC):
    """Base class for all sketch geometry."""
    id: str = ""                             # Stable ID (e.g., "L0", "A1")
    construction: bool = False               # True = reference geometry only

    # Metadata for reconstruction workflow
    source: Optional[str] = None             # Origin: "fitted", "user", "inferred"
    confidence: float = 1.0                  # Fitting confidence (0-1)

    @abstractmethod
    def get_point(self, point_type: PointType) -> Point2D:
        """Get the coordinates of a specific point on this primitive."""
        pass

    @abstractmethod
    def get_valid_point_types(self) -> list[PointType]:
        """Return which PointTypes are valid for this primitive."""
        pass


@dataclass
class Line(SketchPrimitive):
    """
    Line segment defined by two endpoints.

    Valid point types: START, END, MIDPOINT
    """
    start: Point2D = field(default_factory=lambda: Point2D(0, 0))
    end: Point2D = field(default_factory=lambda: Point2D(0, 0))

    @property
    def length(self) -> float:
        """Calculate line segment length."""
        return self.start.distance_to(self.end)

    @property
    def direction(self) -> Vector2D:
        """Get direction vector from start to end."""
        return Vector2D(self.end.x - self.start.x, self.end.y - self.start.y)

    @property
    def midpoint(self) -> Point2D:
        """Get the midpoint of the line segment."""
        return self.start.midpoint(self.end)

    def get_point(self, point_type: PointType) -> Point2D:
        match point_type:
            case PointType.START:
                return self.start
            case PointType.END:
                return self.end
            case PointType.MIDPOINT:
                return self.midpoint
            case _:
                raise ValueError(f"Invalid point type {point_type} for Line")

    def get_valid_point_types(self) -> list[PointType]:
        return [PointType.START, PointType.END, PointType.MIDPOINT]


@dataclass
class Arc(SketchPrimitive):
    """
    Circular arc defined by center, start point, end point, and direction.

    The arc travels from start_point to end_point:
    - If ccw=True: counter-clockwise direction
    - If ccw=False: clockwise direction

    Radius is implicit: distance from center to start_point.
    Validation should ensure |center - start| ≈ |center - end|.

    Valid point types: START, END, CENTER, MIDPOINT
    """
    center: Point2D = field(default_factory=lambda: Point2D(0, 0))
    start_point: Point2D = field(default_factory=lambda: Point2D(0, 0))
    end_point: Point2D = field(default_factory=lambda: Point2D(0, 0))
    ccw: bool = True  # Counter-clockwise from start to end

    @property
    def radius(self) -> float:
        """Calculate arc radius from center to start point."""
        return self.center.distance_to(self.start_point)

    @property
    def start_angle(self) -> float:
        """Angle in radians from center to start_point."""
        return math.atan2(
            self.start_point.y - self.center.y,
            self.start_point.x - self.center.x
        )

    @property
    def end_angle(self) -> float:
        """Angle in radians from center to end_point."""
        return math.atan2(
            self.end_point.y - self.center.y,
            self.end_point.x - self.center.x
        )

    @property
    def sweep_angle(self) -> float:
        """Signed sweep angle in radians (positive = CCW)."""
        delta = self.end_angle - self.start_angle
        if self.ccw:
            return delta if delta > 0 else delta + 2 * math.pi
        else:
            return delta if delta < 0 else delta - 2 * math.pi

    @property
    def arc_length(self) -> float:
        """Calculate arc length."""
        return abs(self.sweep_angle) * self.radius

    @property
    def midpoint(self) -> Point2D:
        """Point at the middle of the arc."""
        mid_angle = self.start_angle + self.sweep_angle / 2
        return Point2D(
            self.center.x + self.radius * math.cos(mid_angle),
            self.center.y + self.radius * math.sin(mid_angle)
        )

    def get_point(self, point_type: PointType) -> Point2D:
        match point_type:
            case PointType.START:
                return self.start_point
            case PointType.END:
                return self.end_point
            case PointType.CENTER:
                return self.center
            case PointType.MIDPOINT:
                return self.midpoint
            case _:
                raise ValueError(f"Invalid point type {point_type} for Arc")

    def get_valid_point_types(self) -> list[PointType]:
        return [PointType.START, PointType.END, PointType.CENTER, PointType.MIDPOINT]

    def to_three_point(self) -> tuple[Point2D, Point2D, Point2D]:
        """Return (start, mid, end) for three-point arc construction."""
        return (self.start_point, self.midpoint, self.end_point)

    def point_at_angle(self, angle: float) -> Point2D:
        """Get point on arc at specified angle (radians from positive X)."""
        return Point2D(
            self.center.x + self.radius * math.cos(angle),
            self.center.y + self.radius * math.sin(angle)
        )


@dataclass
class Circle(SketchPrimitive):
    """
    Full circle defined by center and radius.

    Valid point types: CENTER only
    """
    center: Point2D = field(default_factory=lambda: Point2D(0, 0))
    radius: float = 1.0

    @property
    def diameter(self) -> float:
        """Calculate circle diameter."""
        return self.radius * 2

    @property
    def circumference(self) -> float:
        """Calculate circle circumference."""
        return 2 * math.pi * self.radius

    @property
    def area(self) -> float:
        """Calculate circle area."""
        return math.pi * self.radius**2

    def get_point(self, point_type: PointType) -> Point2D:
        match point_type:
            case PointType.CENTER:
                return self.center
            case _:
                raise ValueError(f"Invalid point type {point_type} for Circle")

    def get_valid_point_types(self) -> list[PointType]:
        return [PointType.CENTER]

    def point_at_angle(self, angle: float) -> Point2D:
        """Get point on circle at specified angle (radians from positive X)."""
        return Point2D(
            self.center.x + self.radius * math.cos(angle),
            self.center.y + self.radius * math.sin(angle)
        )


@dataclass
class Point(SketchPrimitive):
    """
    Standalone sketch point (not an endpoint of another primitive).

    Valid point types: CENTER (the point itself)
    """
    position: Point2D = field(default_factory=lambda: Point2D(0, 0))

    def get_point(self, point_type: PointType) -> Point2D:
        match point_type:
            case PointType.CENTER:
                return self.position
            case _:
                raise ValueError(f"Invalid point type {point_type} for Point")

    def get_valid_point_types(self) -> list[PointType]:
        return [PointType.CENTER]


@dataclass
class Spline(SketchPrimitive):
    """
    B-spline or NURBS curve.

    Two construction modes:
    - Fit-point spline: Curve passes through specified points
    - Control-point spline: Classic B-spline with control polygon

    Valid point types: START, END, CONTROL[i]
    """
    degree: int = 3
    control_points: list[Point2D] = field(default_factory=list)
    knots: list[float] = field(default_factory=list)
    weights: Optional[list[float]] = None  # None = non-rational (uniform weights)
    periodic: bool = False
    is_fit_spline: bool = False  # True = control_points are fit points

    @property
    def order(self) -> int:
        """Spline order = degree + 1"""
        return self.degree + 1

    @property
    def is_rational(self) -> bool:
        """Check if spline is rational (has non-uniform weights)."""
        return self.weights is not None

    @property
    def num_control_points(self) -> int:
        """Get number of control points."""
        return len(self.control_points)

    def get_point(self, point_type: PointType) -> Point2D:
        match point_type:
            case PointType.START:
                if not self.control_points:
                    raise ValueError("Spline has no control points")
                return self.control_points[0]
            case PointType.END:
                if not self.control_points:
                    raise ValueError("Spline has no control points")
                return self.control_points[-1]
            case PointType.CONTROL:
                raise ValueError("CONTROL requires index parameter; use get_control_point()")
            case _:
                raise ValueError(f"Invalid point type {point_type} for Spline")

    def get_control_point(self, index: int) -> Point2D:
        """Get a specific control point by index."""
        if index < 0 or index >= len(self.control_points):
            raise IndexError(f"Control point index {index} out of range [0, {len(self.control_points)})")
        return self.control_points[index]

    def get_valid_point_types(self) -> list[PointType]:
        return [PointType.START, PointType.END, PointType.CONTROL]

    def validate_knot_vector(self) -> bool:
        """
        Validate knot vector consistency with control points and degree.

        For a non-periodic spline: len(knots) = num_control_points + order
        For a periodic spline: len(knots) = num_control_points + 1
        """
        n = len(self.control_points)
        k = self.order
        expected_knots = n + k if not self.periodic else n + 1
        return len(self.knots) == expected_knots

    def evaluate(self, t: float) -> Point2D:
        """
        Evaluate spline at parameter t using De Boor's algorithm.

        Note: This is a simplified implementation. For production use,
        consider using scipy.interpolate.BSpline or similar.
        """
        if not self.control_points or not self.knots:
            raise ValueError("Spline not properly initialized")

        # Clamp t to valid range
        t = max(self.knots[self.degree], min(t, self.knots[-self.degree-1]))

        # Find knot span
        n = len(self.control_points)
        k = self.degree

        # Find the knot span index
        span = k
        for i in range(k, n):
            if self.knots[i] <= t < self.knots[i + 1]:
                span = i
                break
        if t >= self.knots[n]:
            span = n - 1

        # De Boor's algorithm
        d = [Point2D(p.x, p.y) for p in self.control_points[span - k:span + 1]]

        for r in range(1, k + 1):
            for j in range(k, r - 1, -1):
                idx = span - k + j
                denom = self.knots[idx + k + 1 - r] - self.knots[idx]
                if abs(denom) < 1e-10:
                    alpha = 0.0
                else:
                    alpha = (t - self.knots[idx]) / denom
                d[j] = Point2D(
                    (1 - alpha) * d[j - 1].x + alpha * d[j].x,
                    (1 - alpha) * d[j - 1].y + alpha * d[j].y
                )

        return d[k]

    @classmethod
    def create_uniform_bspline(cls, control_points: list[Point2D], degree: int = 3,
                                construction: bool = False) -> 'Spline':
        """
        Create a uniform B-spline with automatically generated knot vector.
        """
        n = len(control_points)
        if n < degree + 1:
            raise ValueError(f"Need at least {degree + 1} control points for degree {degree} spline")

        # Create clamped uniform knot vector
        # First (degree+1) knots are 0, last (degree+1) knots are 1
        # Interior knots are uniformly spaced
        num_knots = n + degree + 1
        knots = []

        # Leading knots (clamped)
        knots.extend([0.0] * (degree + 1))

        # Interior knots
        num_interior = num_knots - 2 * (degree + 1)
        for i in range(num_interior):
            knots.append((i + 1) / (num_interior + 1))

        # Trailing knots (clamped)
        knots.extend([1.0] * (degree + 1))

        return cls(
            degree=degree,
            control_points=control_points,
            knots=knots,
            construction=construction
        )
