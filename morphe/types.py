"""Core data types for the canonical sketch schema."""

import math
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Point2D:
    """Immutable 2D point in millimeters."""
    x: float
    y: float

    def distance_to(self, other: 'Point2D') -> float:
        """Calculate Euclidean distance to another point."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def midpoint(self, other: 'Point2D') -> 'Point2D':
        """Calculate midpoint between this point and another."""
        return Point2D((self.x + other.x) / 2, (self.y + other.y) / 2)

    def __add__(self, other: 'Vector2D') -> 'Point2D':
        """Add a vector to this point."""
        return Point2D(self.x + other.dx, self.y + other.dy)

    def __sub__(self, other: 'Point2D') -> 'Vector2D':
        """Subtract another point to get a vector."""
        return Vector2D(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class Vector2D:
    """2D direction vector (not necessarily normalized)."""
    dx: float
    dy: float

    @property
    def magnitude(self) -> float:
        """Calculate vector magnitude."""
        return math.sqrt(self.dx**2 + self.dy**2)

    def normalized(self) -> 'Vector2D':
        """Return a unit vector in the same direction."""
        mag = self.magnitude
        if mag > 0:
            return Vector2D(self.dx / mag, self.dy / mag)
        return self

    def dot(self, other: 'Vector2D') -> float:
        """Calculate dot product with another vector."""
        return self.dx * other.dx + self.dy * other.dy

    def cross(self, other: 'Vector2D') -> float:
        """Calculate 2D cross product (z-component of 3D cross)."""
        return self.dx * other.dy - self.dy * other.dx

    def __mul__(self, scalar: float) -> 'Vector2D':
        """Multiply vector by a scalar."""
        return Vector2D(self.dx * scalar, self.dy * scalar)

    def __rmul__(self, scalar: float) -> 'Vector2D':
        """Multiply vector by a scalar (reverse)."""
        return self.__mul__(scalar)

    def __add__(self, other: 'Vector2D') -> 'Vector2D':
        """Add two vectors."""
        return Vector2D(self.dx + other.dx, self.dy + other.dy)

    def __neg__(self) -> 'Vector2D':
        """Negate the vector."""
        return Vector2D(-self.dx, -self.dy)


class ElementPrefix:
    """Standard prefixes for element IDs."""
    LINE = "L"
    ARC = "A"
    CIRCLE = "C"
    POINT = "P"
    SPLINE = "S"
    ELLIPSE = "E"
    ELLIPTICAL_ARC = "e"


@dataclass(frozen=True)
class ElementId:
    """
    Stable identifier for sketch elements.
    Format: <type_prefix><index> (e.g., "L0", "A1", "C2")
    """
    prefix: str
    index: int

    def __str__(self) -> str:
        return f"{self.prefix}{self.index}"

    @classmethod
    def parse(cls, s: str) -> 'ElementId':
        """Parse an element ID string like 'L0' or 'A12'."""
        if not s or len(s) < 2:
            raise ValueError(f"Invalid element ID: {s}")
        prefix = s[0]
        try:
            index = int(s[1:])
        except ValueError:
            raise ValueError(f"Invalid element ID index: {s}") from None
        return cls(prefix=prefix, index=index)


class PointType(Enum):
    """Types of referenceable points on primitives."""
    START = "start"        # Line start, Arc start
    END = "end"            # Line end, Arc end
    CENTER = "center"      # Arc center, Circle center
    MIDPOINT = "midpoint"  # Computed midpoint (lines and arcs)

    # For splines
    CONTROL = "control"    # Control point (requires index)
    ON_CURVE = "on_curve"  # Arbitrary point (requires parameter)


@dataclass(frozen=True)
class PointRef:
    """
    Reference to a specific point on a primitive.

    Examples:
        PointRef("L0", PointType.START)  - Start of line L0
        PointRef("A1", PointType.CENTER) - Center of arc A1
        PointRef("C2", PointType.CENTER) - Center of circle C2
    """
    element_id: str
    point_type: PointType
    parameter: float | None = None  # For ON_CURVE type
    index: int | None = None        # For CONTROL type

    def __str__(self) -> str:
        if self.point_type == PointType.CONTROL:
            return f"{self.element_id}.{self.point_type.value}[{self.index}]"
        elif self.point_type == PointType.ON_CURVE:
            return f"{self.element_id}.{self.point_type.value}({self.parameter})"
        return f"{self.element_id}.{self.point_type.value}"
