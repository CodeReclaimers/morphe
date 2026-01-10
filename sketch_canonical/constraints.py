"""Constraint types and data structures for the canonical sketch schema."""

import uuid
from dataclasses import dataclass
from enum import Enum

from .types import PointRef


class ConstraintType(Enum):
    """All supported constraint types."""

    # === Point-to-Point Constraints ===
    COINCIDENT = "coincident"        # Two points at same location

    # === Curve-to-Curve Constraints ===
    TANGENT = "tangent"              # Smooth connection (G1 continuity)
    PERPENDICULAR = "perpendicular"  # 90° angle between lines
    PARALLEL = "parallel"            # Lines have same direction
    CONCENTRIC = "concentric"        # Arcs/circles share center
    EQUAL = "equal"                  # Same size (length or radius)
    COLLINEAR = "collinear"          # Lines on same infinite line

    # === Single-Element Orientation ===
    HORIZONTAL = "horizontal"        # Line parallel to X axis
    VERTICAL = "vertical"            # Line parallel to Y axis
    FIXED = "fixed"                  # Lock all degrees of freedom

    # === Dimensional Constraints ===
    DISTANCE = "distance"            # Distance between two points
    DISTANCE_X = "distance_x"        # Horizontal distance (signed)
    DISTANCE_Y = "distance_y"        # Vertical distance (signed)
    LENGTH = "length"                # Line segment length
    RADIUS = "radius"                # Arc or circle radius
    DIAMETER = "diameter"            # Arc or circle diameter
    ANGLE = "angle"                  # Angle between two lines

    # === Symmetry ===
    SYMMETRIC = "symmetric"          # Two elements symmetric about a line
    MIDPOINT = "midpoint_constraint" # Point at midpoint of line


class ConstraintStatus(Enum):
    """Solver status for a constraint."""
    UNKNOWN = "unknown"          # Not yet evaluated
    SATISFIED = "satisfied"      # Constraint is met
    VIOLATED = "violated"        # Constraint cannot be satisfied
    REDUNDANT = "redundant"      # Constraint is redundant with others
    CONFLICTING = "conflicting"  # Conflicts with other constraints


# Constraint applicability rules
CONSTRAINT_RULES: dict[ConstraintType, dict] = {
    ConstraintType.COINCIDENT: {
        "min_refs": 2,
        "max_refs": 2,
        "ref_types": ["point"],
        "value_required": False,
    },
    ConstraintType.TANGENT: {
        "min_refs": 2,
        "max_refs": 2,
        "ref_types": ["curve"],  # Line, Arc, Circle, Spline
        "value_required": False,
        "notes": "At least one must be Arc, Circle, or Spline"
    },
    ConstraintType.PERPENDICULAR: {
        "min_refs": 2,
        "max_refs": 2,
        "ref_types": ["line"],
        "value_required": False,
    },
    ConstraintType.PARALLEL: {
        "min_refs": 2,
        "max_refs": 2,
        "ref_types": ["line"],
        "value_required": False,
    },
    ConstraintType.CONCENTRIC: {
        "min_refs": 2,
        "max_refs": 2,
        "ref_types": ["arc", "circle"],
        "value_required": False,
    },
    ConstraintType.EQUAL: {
        "min_refs": 2,
        "max_refs": None,  # Can chain multiple
        "ref_types": ["line", "arc", "circle"],  # All same type
        "value_required": False,
    },
    ConstraintType.COLLINEAR: {
        "min_refs": 2,
        "max_refs": None,  # Can chain multiple
        "ref_types": ["line"],
        "value_required": False,
    },
    ConstraintType.HORIZONTAL: {
        "min_refs": 1,
        "max_refs": 1,
        "ref_types": ["line"],
        "value_required": False,
    },
    ConstraintType.VERTICAL: {
        "min_refs": 1,
        "max_refs": 1,
        "ref_types": ["line"],
        "value_required": False,
    },
    ConstraintType.FIXED: {
        "min_refs": 1,
        "max_refs": 1,
        "ref_types": ["any"],
        "value_required": False,
    },
    ConstraintType.DISTANCE: {
        "min_refs": 2,
        "max_refs": 2,
        "ref_types": ["point"],
        "value_required": True,
    },
    ConstraintType.DISTANCE_X: {
        "min_refs": 1,
        "max_refs": 2,
        "ref_types": ["point"],
        "value_required": True,
        "notes": "One point = distance from origin; two points = horizontal distance between them"
    },
    ConstraintType.DISTANCE_Y: {
        "min_refs": 1,
        "max_refs": 2,
        "ref_types": ["point"],
        "value_required": True,
        "notes": "One point = distance from origin; two points = vertical distance between them"
    },
    ConstraintType.LENGTH: {
        "min_refs": 1,
        "max_refs": 1,
        "ref_types": ["line"],
        "value_required": True,
    },
    ConstraintType.RADIUS: {
        "min_refs": 1,
        "max_refs": 1,
        "ref_types": ["arc", "circle"],
        "value_required": True,
    },
    ConstraintType.DIAMETER: {
        "min_refs": 1,
        "max_refs": 1,
        "ref_types": ["arc", "circle"],
        "value_required": True,
    },
    ConstraintType.ANGLE: {
        "min_refs": 2,
        "max_refs": 2,
        "ref_types": ["line"],
        "value_required": True,
        "notes": "Value in degrees"
    },
    ConstraintType.SYMMETRIC: {
        "min_refs": 3,
        "max_refs": 3,
        "ref_types": ["any", "any", "line"],  # Two elements + symmetry axis
        "value_required": False,
        "notes": "Third reference is the symmetry axis (line)"
    },
    ConstraintType.MIDPOINT: {
        "min_refs": 2,
        "max_refs": 2,
        "ref_types": ["point", "line"],
        "value_required": False,
        "notes": "Point constrained to midpoint of line"
    },
}


@dataclass
class SketchConstraint:
    """
    A geometric or dimensional constraint.

    References can be:
    - Element IDs (str): For constraints on whole primitives (e.g., Horizontal("L0"))
    - PointRefs: For constraints on specific points (e.g., Coincident(L0.END, A1.START))

    The interpretation depends on constraint type:
    - COINCIDENT: requires two PointRefs
    - HORIZONTAL: requires one element ID (line)
    - TANGENT: requires two element IDs (curves), optionally with connection point hints
    """
    id: str                                          # Unique constraint ID
    constraint_type: ConstraintType
    references: list[str | PointRef]           # Element IDs or PointRefs
    value: float | None = None                    # For dimensional constraints (mm or degrees)

    # Connection hints for curve-to-curve constraints
    connection_point: PointRef | None = None      # Where tangent/perpendicular occurs

    # Metadata
    inferred: bool = False                           # True if AI/algorithm suggested
    confidence: float = 1.0                          # Confidence for inferred constraints
    source: str | None = None                     # Origin: "user", "ai", "detected"

    # Status (populated after solving)
    status: ConstraintStatus = ConstraintStatus.UNKNOWN

    def __str__(self) -> str:
        refs_str = ", ".join(str(r) for r in self.references)
        if self.value is not None:
            return f"{self.constraint_type.value}({refs_str}, {self.value})"
        return f"{self.constraint_type.value}({refs_str})"

    def get_element_ids(self) -> set[str]:
        """Extract all element IDs referenced by this constraint."""
        ids = set()
        for ref in self.references:
            if isinstance(ref, str):
                ids.add(ref)
            elif isinstance(ref, PointRef):
                ids.add(ref.element_id)
        if self.connection_point:
            ids.add(self.connection_point.element_id)
        return ids


def _generate_id() -> str:
    """Generate a short unique ID for constraints."""
    return str(uuid.uuid4())[:8]


# Convenience constructors
def Coincident(pt1: PointRef, pt2: PointRef, **kwargs) -> SketchConstraint:
    """Create a coincident constraint between two points."""
    return SketchConstraint(
        id=kwargs.pop('id', _generate_id()),
        constraint_type=ConstraintType.COINCIDENT,
        references=[pt1, pt2],
        **kwargs
    )


def Tangent(elem1: str, elem2: str, at: PointRef | None = None, **kwargs) -> SketchConstraint:
    """Create a tangent constraint between two curves."""
    return SketchConstraint(
        id=kwargs.pop('id', _generate_id()),
        constraint_type=ConstraintType.TANGENT,
        references=[elem1, elem2],
        connection_point=at,
        **kwargs
    )


def Perpendicular(elem1: str, elem2: str, **kwargs) -> SketchConstraint:
    """Create a perpendicular constraint between two lines."""
    return SketchConstraint(
        id=kwargs.pop('id', _generate_id()),
        constraint_type=ConstraintType.PERPENDICULAR,
        references=[elem1, elem2],
        **kwargs
    )


def Parallel(elem1: str, elem2: str, **kwargs) -> SketchConstraint:
    """Create a parallel constraint between two lines."""
    return SketchConstraint(
        id=kwargs.pop('id', _generate_id()),
        constraint_type=ConstraintType.PARALLEL,
        references=[elem1, elem2],
        **kwargs
    )


def Concentric(elem1: str, elem2: str, **kwargs) -> SketchConstraint:
    """Create a concentric constraint between two arcs/circles."""
    return SketchConstraint(
        id=kwargs.pop('id', _generate_id()),
        constraint_type=ConstraintType.CONCENTRIC,
        references=[elem1, elem2],
        **kwargs
    )


def Equal(*elements: str, **kwargs) -> SketchConstraint:
    """Create an equal constraint between multiple elements."""
    if len(elements) < 2:
        raise ValueError("Equal constraint requires at least 2 elements")
    return SketchConstraint(
        id=kwargs.pop('id', _generate_id()),
        constraint_type=ConstraintType.EQUAL,
        references=list(elements),
        **kwargs
    )


def Collinear(*lines: str, **kwargs) -> SketchConstraint:
    """Create a collinear constraint between multiple lines."""
    if len(lines) < 2:
        raise ValueError("Collinear constraint requires at least 2 lines")
    return SketchConstraint(
        id=kwargs.pop('id', _generate_id()),
        constraint_type=ConstraintType.COLLINEAR,
        references=list(lines),
        **kwargs
    )


def Horizontal(elem: str, **kwargs) -> SketchConstraint:
    """Create a horizontal constraint on a line."""
    return SketchConstraint(
        id=kwargs.pop('id', _generate_id()),
        constraint_type=ConstraintType.HORIZONTAL,
        references=[elem],
        **kwargs
    )


def Vertical(elem: str, **kwargs) -> SketchConstraint:
    """Create a vertical constraint on a line."""
    return SketchConstraint(
        id=kwargs.pop('id', _generate_id()),
        constraint_type=ConstraintType.VERTICAL,
        references=[elem],
        **kwargs
    )


def Fixed(elem: str, **kwargs) -> SketchConstraint:
    """Create a fixed constraint on an element."""
    return SketchConstraint(
        id=kwargs.pop('id', _generate_id()),
        constraint_type=ConstraintType.FIXED,
        references=[elem],
        **kwargs
    )


def Distance(pt1: PointRef, pt2: PointRef, value: float, **kwargs) -> SketchConstraint:
    """Create a distance constraint between two points."""
    return SketchConstraint(
        id=kwargs.pop('id', _generate_id()),
        constraint_type=ConstraintType.DISTANCE,
        references=[pt1, pt2],
        value=value,
        **kwargs
    )


def DistanceX(pt: PointRef, value: float, pt2: PointRef | None = None, **kwargs) -> SketchConstraint:
    """Create a horizontal distance constraint."""
    refs: list[str | PointRef] = [pt] if pt2 is None else [pt, pt2]
    return SketchConstraint(
        id=kwargs.pop('id', _generate_id()),
        constraint_type=ConstraintType.DISTANCE_X,
        references=refs,
        value=value,
        **kwargs
    )


def DistanceY(pt: PointRef, value: float, pt2: PointRef | None = None, **kwargs) -> SketchConstraint:
    """Create a vertical distance constraint."""
    refs: list[str | PointRef] = [pt] if pt2 is None else [pt, pt2]
    return SketchConstraint(
        id=kwargs.pop('id', _generate_id()),
        constraint_type=ConstraintType.DISTANCE_Y,
        references=refs,
        value=value,
        **kwargs
    )


def Length(elem: str, value: float, **kwargs) -> SketchConstraint:
    """Create a length constraint on a line."""
    return SketchConstraint(
        id=kwargs.pop('id', _generate_id()),
        constraint_type=ConstraintType.LENGTH,
        references=[elem],
        value=value,
        **kwargs
    )


def Radius(elem: str, value: float, **kwargs) -> SketchConstraint:
    """Create a radius constraint on an arc or circle."""
    return SketchConstraint(
        id=kwargs.pop('id', _generate_id()),
        constraint_type=ConstraintType.RADIUS,
        references=[elem],
        value=value,
        **kwargs
    )


def Diameter(elem: str, value: float, **kwargs) -> SketchConstraint:
    """Create a diameter constraint on an arc or circle."""
    return SketchConstraint(
        id=kwargs.pop('id', _generate_id()),
        constraint_type=ConstraintType.DIAMETER,
        references=[elem],
        value=value,
        **kwargs
    )


def Angle(elem1: str, elem2: str, value: float, **kwargs) -> SketchConstraint:
    """Create an angle constraint between two lines (value in degrees)."""
    return SketchConstraint(
        id=kwargs.pop('id', _generate_id()),
        constraint_type=ConstraintType.ANGLE,
        references=[elem1, elem2],
        value=value,
        **kwargs
    )


def Symmetric(elem1: str | PointRef, elem2: str | PointRef,
              axis: str, **kwargs) -> SketchConstraint:
    """Create a symmetric constraint about a line axis."""
    return SketchConstraint(
        id=kwargs.pop('id', _generate_id()),
        constraint_type=ConstraintType.SYMMETRIC,
        references=[elem1, elem2, axis],
        **kwargs
    )


def MidpointConstraint(pt: PointRef, line: str, **kwargs) -> SketchConstraint:
    """Create a midpoint constraint (point at midpoint of line)."""
    return SketchConstraint(
        id=kwargs.pop('id', _generate_id()),
        constraint_type=ConstraintType.MIDPOINT,
        references=[pt, line],
        **kwargs
    )
