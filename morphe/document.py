"""Sketch document structure for the canonical sketch schema."""

from dataclasses import dataclass, field
from enum import Enum

from .constraints import SketchConstraint
from .primitives import (
    Arc,
    Circle,
    Ellipse,
    EllipticalArc,
    Line,
    Point,
    SketchPrimitive,
    Spline,
)
from .types import ElementPrefix, Point2D, PointRef, PointType


class SolverStatus(Enum):
    """Overall sketch constraint status."""
    DIRTY = "dirty"                      # Constraints changed, needs re-solve
    UNDER_CONSTRAINED = "under_constrained"
    FULLY_CONSTRAINED = "fully_constrained"
    OVER_CONSTRAINED = "over_constrained"
    INCONSISTENT = "inconsistent"        # Conflicting constraints


@dataclass
class SketchDocument:
    """
    Complete representation of a 2D sketch.

    This is the main container for sketch geometry and constraints.
    It provides methods for adding/removing elements, querying relationships,
    and generating human-readable descriptions.
    """
    name: str = "Untitled"

    # Geometry
    primitives: dict[str, SketchPrimitive] = field(default_factory=dict)

    # Constraints
    constraints: list[SketchConstraint] = field(default_factory=list)

    # Solver state
    solver_status: SolverStatus = SolverStatus.DIRTY
    degrees_of_freedom: int = -1  # -1 = not computed

    # ID counters for stable ID generation
    _next_index: dict[str, int] = field(default_factory=lambda: {
        ElementPrefix.LINE: 0,
        ElementPrefix.ARC: 0,
        ElementPrefix.CIRCLE: 0,
        ElementPrefix.POINT: 0,
        ElementPrefix.SPLINE: 0,
        ElementPrefix.ELLIPSE: 0,
        ElementPrefix.ELLIPTICAL_ARC: 0,
    })

    def _get_prefix_for_type(self, primitive_type: type[SketchPrimitive]) -> str:
        """Get the ID prefix for a primitive type."""
        prefix_map: dict[type[SketchPrimitive], str] = {
            Line: ElementPrefix.LINE,
            Arc: ElementPrefix.ARC,
            Circle: ElementPrefix.CIRCLE,
            Point: ElementPrefix.POINT,
            Spline: ElementPrefix.SPLINE,
            Ellipse: ElementPrefix.ELLIPSE,
            EllipticalArc: ElementPrefix.ELLIPTICAL_ARC,
        }
        prefix = prefix_map.get(primitive_type)
        if prefix is None:
            raise ValueError(f"Unknown primitive type: {primitive_type}")
        return prefix

    def add_primitive(self, primitive: SketchPrimitive) -> str:
        """
        Add primitive and assign stable ID.

        Returns:
            The assigned ID string.
        """
        prefix = self._get_prefix_for_type(type(primitive))

        idx = self._next_index[prefix]
        self._next_index[prefix] += 1

        primitive.id = f"{prefix}{idx}"
        self.primitives[primitive.id] = primitive
        self.solver_status = SolverStatus.DIRTY

        return primitive.id

    def add_primitive_with_id(self, primitive: SketchPrimitive, element_id: str) -> str:
        """
        Add primitive with a specific ID.

        This is useful when reconstructing a sketch from serialized data.
        Updates the next_index counter if necessary.

        Returns:
            The assigned ID string.
        """
        if element_id in self.primitives:
            raise ValueError(f"Element ID '{element_id}' already exists")

        primitive.id = element_id
        self.primitives[element_id] = primitive

        # Update next_index if necessary
        if len(element_id) >= 2:
            prefix = element_id[0]
            try:
                idx = int(element_id[1:])
                if prefix in self._next_index:
                    self._next_index[prefix] = max(self._next_index[prefix], idx + 1)
            except ValueError:
                pass

        self.solver_status = SolverStatus.DIRTY
        return element_id

    def remove_primitive(self, element_id: str) -> bool:
        """
        Remove a primitive by ID.

        Also removes any constraints referencing this primitive.

        Returns:
            True if removed, False if not found.
        """
        if element_id not in self.primitives:
            return False

        del self.primitives[element_id]

        # Remove constraints referencing this element
        self.constraints = [
            c for c in self.constraints
            if element_id not in c.get_element_ids()
        ]

        self.solver_status = SolverStatus.DIRTY
        return True

    def get_primitive(self, element_id: str) -> SketchPrimitive | None:
        """Get a primitive by its ID."""
        return self.primitives.get(element_id)

    def get_point(self, ref: PointRef) -> Point2D:
        """Resolve a PointRef to actual coordinates."""
        prim = self.primitives.get(ref.element_id)
        if prim is None:
            raise KeyError(f"Element '{ref.element_id}' not found")

        if ref.point_type == PointType.CONTROL and isinstance(prim, Spline):
            if ref.index is None:
                raise ValueError("CONTROL point type requires index")
            return prim.get_control_point(ref.index)

        return prim.get_point(ref.point_type)

    def add_constraint(self, constraint: SketchConstraint) -> None:
        """Add a constraint to the sketch."""
        # Validate that all referenced elements exist
        for elem_id in constraint.get_element_ids():
            if elem_id not in self.primitives:
                raise KeyError(f"Constraint references non-existent element '{elem_id}'")

        self.constraints.append(constraint)
        self.solver_status = SolverStatus.DIRTY

    def remove_constraint(self, constraint_id: str) -> bool:
        """
        Remove a constraint by ID.

        Returns:
            True if removed, False if not found.
        """
        for i, c in enumerate(self.constraints):
            if c.id == constraint_id:
                del self.constraints[i]
                self.solver_status = SolverStatus.DIRTY
                return True
        return False

    def get_constraint(self, constraint_id: str) -> SketchConstraint | None:
        """Get a constraint by its ID."""
        for c in self.constraints:
            if c.id == constraint_id:
                return c
        return None

    def get_constraints_for(self, element_id: str) -> list[SketchConstraint]:
        """Get all constraints involving a specific element."""
        return [c for c in self.constraints if element_id in c.get_element_ids()]

    def get_primitives_by_type(self, prim_type: type) -> list[SketchPrimitive]:
        """Get all primitives of a specific type."""
        return [p for p in self.primitives.values() if isinstance(p, prim_type)]

    def get_lines(self) -> list[Line]:
        """Get all lines in the sketch."""
        return [p for p in self.primitives.values() if isinstance(p, Line)]

    def get_arcs(self) -> list[Arc]:
        """Get all arcs in the sketch."""
        return [p for p in self.primitives.values() if isinstance(p, Arc)]

    def get_circles(self) -> list[Circle]:
        """Get all circles in the sketch."""
        return [p for p in self.primitives.values() if isinstance(p, Circle)]

    def get_splines(self) -> list[Spline]:
        """Get all splines in the sketch."""
        return [p for p in self.primitives.values() if isinstance(p, Spline)]

    def get_ellipses(self) -> list[Ellipse]:
        """Get all ellipses in the sketch."""
        return [p for p in self.primitives.values() if isinstance(p, Ellipse)]

    def get_elliptical_arcs(self) -> list[EllipticalArc]:
        """Get all elliptical arcs in the sketch."""
        return [p for p in self.primitives.values() if isinstance(p, EllipticalArc)]

    def get_construction_geometry(self) -> list[SketchPrimitive]:
        """Get all construction (reference) geometry."""
        return [p for p in self.primitives.values() if p.construction]

    def get_profile_geometry(self) -> list[SketchPrimitive]:
        """Get all non-construction geometry."""
        return [p for p in self.primitives.values() if not p.construction]

    def clear(self) -> None:
        """Remove all geometry and constraints."""
        self.primitives.clear()
        self.constraints.clear()
        self._next_index = {
            ElementPrefix.LINE: 0,
            ElementPrefix.ARC: 0,
            ElementPrefix.CIRCLE: 0,
            ElementPrefix.POINT: 0,
            ElementPrefix.SPLINE: 0,
            ElementPrefix.ELLIPSE: 0,
            ElementPrefix.ELLIPTICAL_ARC: 0,
        }
        self.solver_status = SolverStatus.DIRTY
        self.degrees_of_freedom = -1

    def to_text_description(self, include_point_coords: bool = False) -> str:
        """
        Generate human/AI-readable description of the sketch.

        Args:
            include_point_coords: If True, list all referenceable points with coordinates
        """
        lines = [f"Sketch: {self.name}", "", "Elements:"]

        for id, prim in sorted(self.primitives.items()):
            lines.append(f"  {self._describe_primitive(prim)}")
            if include_point_coords:
                for pt_type in prim.get_valid_point_types():
                    if pt_type == PointType.CONTROL:
                        # Handle spline control points specially
                        if isinstance(prim, Spline):
                            for i, cp in enumerate(prim.control_points):
                                lines.append(f"    {id}.{pt_type.value}[{i}]: ({cp.x:.2f}, {cp.y:.2f})")
                    else:
                        try:
                            pt = prim.get_point(pt_type)
                            lines.append(f"    {id}.{pt_type.value}: ({pt.x:.2f}, {pt.y:.2f})")
                        except ValueError:
                            pass

        lines.append("\nConstraints:")
        if self.constraints:
            for c in self.constraints:
                lines.append(f"  {c}")
        else:
            lines.append("  (none)")

        lines.append(f"\nStatus: {self.solver_status.value}")
        if self.degrees_of_freedom >= 0:
            lines.append(f"Degrees of Freedom: {self.degrees_of_freedom}")

        return "\n".join(lines)

    def _describe_primitive(self, p: SketchPrimitive) -> str:
        """Generate a text description of a primitive."""
        const_marker = " [C]" if p.construction else ""

        if isinstance(p, Line):
            return f"{p.id}: Line ({p.start.x:.2f},{p.start.y:.2f}) → ({p.end.x:.2f},{p.end.y:.2f}){const_marker}"
        elif isinstance(p, Arc):
            direction = "CCW" if p.ccw else "CW"
            return f"{p.id}: Arc center=({p.center.x:.2f},{p.center.y:.2f}) r={p.radius:.2f} {direction}{const_marker}"
        elif isinstance(p, Circle):
            return f"{p.id}: Circle center=({p.center.x:.2f},{p.center.y:.2f}) r={p.radius:.2f}{const_marker}"
        elif isinstance(p, Point):
            return f"{p.id}: Point ({p.position.x:.2f},{p.position.y:.2f}){const_marker}"
        elif isinstance(p, Spline):
            periodic = "periodic" if p.periodic else "open"
            return f"{p.id}: Spline degree={p.degree} points={len(p.control_points)} {periodic}{const_marker}"
        elif isinstance(p, Ellipse):
            import math
            rot_deg = math.degrees(p.rotation)
            return f"{p.id}: Ellipse center=({p.center.x:.2f},{p.center.y:.2f}) a={p.major_radius:.2f} b={p.minor_radius:.2f} rot={rot_deg:.1f}°{const_marker}"
        elif isinstance(p, EllipticalArc):
            import math
            direction = "CCW" if p.ccw else "CW"
            rot_deg = math.degrees(p.rotation)
            return f"{p.id}: EllipticalArc center=({p.center.x:.2f},{p.center.y:.2f}) a={p.major_radius:.2f} b={p.minor_radius:.2f} rot={rot_deg:.1f}° {direction}{const_marker}"
        else:
            return f"{p.id}: {type(p).__name__}{const_marker}"

    def __repr__(self) -> str:
        return f"SketchDocument(name={self.name!r}, primitives={len(self.primitives)}, constraints={len(self.constraints)})"
