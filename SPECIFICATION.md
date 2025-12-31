# Canonical Sketch Geometry and Constraint Schema

This document defines a CAD-agnostic representation for 2D sketch geometry and constraints, along with platform-specific adaptation requirements for FreeCAD, SolidWorks, Inventor, and Fusion360.

## 1. Design Principles

1. **Explicit over implicit**: All relationships are stated explicitly (e.g., tangent does not imply coincident)
2. **Point references are first-class**: Constraints reference specific points on primitives, not just primitives
3. **Unitless Internal**: Internal coordinates are unitless; conversion happens at adapter boundary using a user-provided 2D affine transform
4. **IDs are stable**: Element IDs persist across modifications for reliable AI/human reference
5. **Validation is layered**: Schema validity vs. geometric validity vs. solver satisfiability

### 1.1 Project structure
```
sketch-canonical/           # Core schema, validation, serialization (Python)
sketch-adapter-freecad/     # FreeCAD adapter (Python, open source)
sketch-adapter-solidworks/  # SolidWorks adapter (C#, commercial)
sketch-adapter-inventor/    # Inventor adapter (C#, commercial)  
sketch-adapter-fusion360/   # Fusion 360 adapter (Python, commercial)
```

Each adapter project has:
- The adapter implementation
- Platform-specific test harness
- Any licensing/registration handling

Each adapter may be of two different types:
- Full backend (FreeCAD, possibly Fusion 360): Can create, export, AND visualize with point cloud overlay
- Creation-only backend (SolidWorks, Inventor): Can create and export sketches, but visualization happens elsewhere

The core project has:
- Canonical schema definitions
- Validation logic
- JSON serialization
- MCP protocol definitions
- Integration tests using mock adapters



---

## 2. Core Data Types

### 2.1 Basic Types

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union
import uuid

@dataclass(frozen=True)
class Point2D:
    """Immutable 2D point in millimeters."""
    x: float
    y: float
    
    def distance_to(self, other: 'Point2D') -> float:
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5
    
    def midpoint(self, other: 'Point2D') -> 'Point2D':
        return Point2D((self.x + other.x) / 2, (self.y + other.y) / 2)

@dataclass(frozen=True)
class Vector2D:
    """2D direction vector (not necessarily normalized)."""
    dx: float
    dy: float
    
    def normalized(self) -> 'Vector2D':
        mag = (self.dx**2 + self.dy**2)**0.5
        return Vector2D(self.dx / mag, self.dy / mag) if mag > 0 else self
```

### 2.2 Element ID System

```python
@dataclass(frozen=True)
class ElementId:
    """
    Stable identifier for sketch elements.
    Format: <type_prefix><index> (e.g., "L0", "A1", "C2")
    """
    prefix: str  # "L" for line, "A" for arc, "C" for circle, "P" for point
    index: int
    
    def __str__(self) -> str:
        return f"{self.prefix}{self.index}"
    
    @classmethod
    def parse(cls, s: str) -> 'ElementId':
        return cls(prefix=s[0], index=int(s[1:]))

class ElementPrefix:
    LINE = "L"
    ARC = "A"
    CIRCLE = "C"
    POINT = "P"
    SPLINE = "S"
```

---

## 3. Geometry Primitives

### 3.1 Point Reference System

Many constraints apply to specific points on primitives rather than the primitive as a whole.

```python
class PointType(Enum):
    """Types of referenceable points on primitives."""
    START = "start"        # Line start, Arc start
    END = "end"            # Line end, Arc end  
    CENTER = "center"      # Arc center, Circle center
    MIDPOINT = "midpoint"  # Computed midpoint (lines and arcs)
    
    # For splines (future)
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
    parameter: Optional[float] = None  # For ON_CURVE type
    index: Optional[int] = None        # For CONTROL type
    
    def __str__(self) -> str:
        if self.point_type == PointType.CONTROL:
            return f"{self.element_id}.{self.point_type.value}[{self.index}]"
        return f"{self.element_id}.{self.point_type.value}"
```

### 3.2 Primitive Base Class

```python
@dataclass
class SketchPrimitive:
    """Base class for all sketch geometry."""
    id: str                              # Stable ID (e.g., "L0", "A1")
    construction: bool = False           # True = reference geometry only
    
    # Metadata for reconstruction workflow
    source: Optional[str] = None         # Origin: "fitted", "user", "inferred"
    confidence: float = 1.0              # Fitting confidence (0-1)
    
    def get_point(self, point_type: PointType) -> Point2D:
        """Get the coordinates of a specific point on this primitive."""
        raise NotImplementedError
    
    def get_valid_point_types(self) -> list[PointType]:
        """Return which PointTypes are valid for this primitive."""
        raise NotImplementedError
```

### 3.3 Line

```python
@dataclass
class Line(SketchPrimitive):
    """
    Line segment defined by two endpoints.
    
    Valid point types: START, END, MIDPOINT
    """
    start: Point2D
    end: Point2D
    
    @property
    def length(self) -> float:
        return self.start.distance_to(self.end)
    
    @property
    def direction(self) -> Vector2D:
        return Vector2D(self.end.x - self.start.x, self.end.y - self.start.y)
    
    @property
    def midpoint(self) -> Point2D:
        return self.start.midpoint(self.end)
    
    def get_point(self, point_type: PointType) -> Point2D:
        match point_type:
            case PointType.START: return self.start
            case PointType.END: return self.end
            case PointType.MIDPOINT: return self.midpoint
            case _: raise ValueError(f"Invalid point type {point_type} for Line")
    
    def get_valid_point_types(self) -> list[PointType]:
        return [PointType.START, PointType.END, PointType.MIDPOINT]
```

### 3.4 Arc

```python
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
    center: Point2D
    start_point: Point2D
    end_point: Point2D
    ccw: bool  # Counter-clockwise from start to end
    
    @property
    def radius(self) -> float:
        return self.center.distance_to(self.start_point)
    
    @property
    def start_angle(self) -> float:
        """Angle in radians from center to start_point."""
        import math
        return math.atan2(
            self.start_point.y - self.center.y,
            self.start_point.x - self.center.x
        )
    
    @property
    def end_angle(self) -> float:
        """Angle in radians from center to end_point."""
        import math
        return math.atan2(
            self.end_point.y - self.center.y,
            self.end_point.x - self.center.x
        )
    
    @property
    def sweep_angle(self) -> float:
        """Signed sweep angle in radians (positive = CCW)."""
        import math
        delta = self.end_angle - self.start_angle
        if self.ccw:
            return delta if delta > 0 else delta + 2 * math.pi
        else:
            return delta if delta < 0 else delta - 2 * math.pi
    
    @property
    def midpoint(self) -> Point2D:
        """Point at the middle of the arc."""
        import math
        mid_angle = self.start_angle + self.sweep_angle / 2
        return Point2D(
            self.center.x + self.radius * math.cos(mid_angle),
            self.center.y + self.radius * math.sin(mid_angle)
        )
    
    def get_point(self, point_type: PointType) -> Point2D:
        match point_type:
            case PointType.START: return self.start_point
            case PointType.END: return self.end_point
            case PointType.CENTER: return self.center
            case PointType.MIDPOINT: return self.midpoint
            case _: raise ValueError(f"Invalid point type {point_type} for Arc")
    
    def get_valid_point_types(self) -> list[PointType]:
        return [PointType.START, PointType.END, PointType.CENTER, PointType.MIDPOINT]
    
    def to_three_point(self) -> tuple[Point2D, Point2D, Point2D]:
        """Return (start, mid, end) for three-point arc construction."""
        return (self.start_point, self.midpoint, self.end_point)
```

### 3.5 Circle

```python
@dataclass
class Circle(SketchPrimitive):
    """
    Full circle defined by center and radius.
    
    Valid point types: CENTER only
    (Quadrant points could be added if needed)
    """
    center: Point2D
    radius: float
    
    def get_point(self, point_type: PointType) -> Point2D:
        match point_type:
            case PointType.CENTER: return self.center
            case _: raise ValueError(f"Invalid point type {point_type} for Circle")
    
    def get_valid_point_types(self) -> list[PointType]:
        return [PointType.CENTER]

    @property
    def diameter(self) -> float:
        return self.radius * 2
```

### 3.6 Standalone Point

```python
@dataclass
class Point(SketchPrimitive):
    """
    Standalone sketch point (not an endpoint of another primitive).
    
    Valid point types: CENTER (the point itself)
    """
    position: Point2D
    
    def get_point(self, point_type: PointType) -> Point2D:
        match point_type:
            case PointType.CENTER: return self.position
            case _: raise ValueError(f"Invalid point type {point_type} for Point")
    
    def get_valid_point_types(self) -> list[PointType]:
        return [PointType.CENTER]
```

### 3.7 Spline

```python
@dataclass
class Spline(SketchPrimitive):
    """
    B-spline or NURBS curve.
    
    Two construction modes:
    - Fit-point spline: Curve passes through specified points
    - Control-point spline: Classic B-spline with control polygon
    
    Valid point types: START, END, CONTROL[i]
    """
    degree: int
    control_points: list[Point2D]
    knots: list[float]
    weights: Optional[list[float]] = None  # None = non-rational (uniform weights)
    periodic: bool = False
    is_fit_spline: bool = False  # True = control_points are fit points
    
    @property
    def order(self) -> int:
        """Spline order = degree + 1"""
        return self.degree + 1
    
    @property
    def is_rational(self) -> bool:
        return self.weights is not None
    
    def get_point(self, point_type: PointType) -> Point2D:
        match point_type:
            case PointType.START: return self.control_points[0]
            case PointType.END: return self.control_points[-1]
            case PointType.CONTROL: 
                raise ValueError("CONTROL requires index parameter")
            case _: raise ValueError(f"Invalid point type {point_type} for Spline")
    
    def get_control_point(self, index: int) -> Point2D:
        return self.control_points[index]
    
    def get_valid_point_types(self) -> list[PointType]:
        return [PointType.START, PointType.END, PointType.CONTROL]
```

---

## 4. Constraints

### 4.1 Constraint Type Definitions

```python
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


# Constraint applicability rules
CONSTRAINT_RULES = {
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
}
```

### 4.2 Constraint Data Structure

```python
class ConstraintStatus(Enum):
    """Solver status for a constraint."""
    UNKNOWN = "unknown"          # Not yet evaluated
    SATISFIED = "satisfied"      # Constraint is met
    VIOLATED = "violated"        # Constraint cannot be satisfied
    REDUNDANT = "redundant"      # Constraint is redundant with others
    CONFLICTING = "conflicting"  # Conflicts with other constraints

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
    references: list[Union[str, PointRef]]           # Element IDs or PointRefs
    value: Optional[float] = None                    # For dimensional constraints (mm or degrees)
    
    # Connection hints for curve-to-curve constraints
    connection_point: Optional[PointRef] = None      # Where tangent/perpendicular occurs
    
    # Metadata
    inferred: bool = False                           # True if AI/algorithm suggested
    confidence: float = 1.0                          # Confidence for inferred constraints
    source: Optional[str] = None                     # Origin: "user", "ai", "detected"
    
    # Status (populated after solving)
    status: ConstraintStatus = ConstraintStatus.UNKNOWN
    
    def __str__(self) -> str:
        refs_str = ", ".join(str(r) for r in self.references)
        if self.value is not None:
            return f"{self.constraint_type.value}({refs_str}, {self.value})"
        return f"{self.constraint_type.value}({refs_str})"


# Convenience constructors
def Coincident(pt1: PointRef, pt2: PointRef, **kwargs) -> SketchConstraint:
    return SketchConstraint(
        id=kwargs.get('id', str(uuid.uuid4())[:8]),
        constraint_type=ConstraintType.COINCIDENT,
        references=[pt1, pt2],
        **{k: v for k, v in kwargs.items() if k != 'id'}
    )

def Tangent(elem1: str, elem2: str, at: Optional[PointRef] = None, **kwargs) -> SketchConstraint:
    return SketchConstraint(
        id=kwargs.get('id', str(uuid.uuid4())[:8]),
        constraint_type=ConstraintType.TANGENT,
        references=[elem1, elem2],
        connection_point=at,
        **{k: v for k, v in kwargs.items() if k != 'id'}
    )

def Horizontal(elem: str, **kwargs) -> SketchConstraint:
    return SketchConstraint(
        id=kwargs.get('id', str(uuid.uuid4())[:8]),
        constraint_type=ConstraintType.HORIZONTAL,
        references=[elem],
        **{k: v for k, v in kwargs.items() if k != 'id'}
    )

def Radius(elem: str, value: float, **kwargs) -> SketchConstraint:
    return SketchConstraint(
        id=kwargs.get('id', str(uuid.uuid4())[:8]),
        constraint_type=ConstraintType.RADIUS,
        references=[elem],
        value=value,
        **{k: v for k, v in kwargs.items() if k != 'id'}
    )

# ... similar constructors for other constraint types
```

---

## 5. Sketch Document

### 5.1 Complete Sketch Structure

```python
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
    """
    name: str
    
    # Geometry
    primitives: dict[str, SketchPrimitive]  # ID -> Primitive
    
    # Constraints
    constraints: list[SketchConstraint]
    
    # Solver state
    solver_status: SolverStatus = SolverStatus.DIRTY
    degrees_of_freedom: int = -1            # -1 = not computed
    
    # ID counters for stable ID generation
    _next_index: dict[str, int] = field(default_factory=lambda: {
        ElementPrefix.LINE: 0, 
        ElementPrefix.ARC: 0, 
        ElementPrefix.CIRCLE: 0, 
        ElementPrefix.POINT: 0, 
        ElementPrefix.SPLINE: 0
    })
    
    def add_primitive(self, primitive: SketchPrimitive) -> str:
        """Add primitive and assign stable ID. Returns the assigned ID."""
        prefix = {
            Line: ElementPrefix.LINE, 
            Arc: ElementPrefix.ARC, 
            Circle: ElementPrefix.CIRCLE,
            Point: ElementPrefix.POINT, 
            Spline: ElementPrefix.SPLINE
        }[type(primitive)]
        
        idx = self._next_index[prefix]
        self._next_index[prefix] += 1
        primitive.id = f"{prefix}{idx}"
        self.primitives[primitive.id] = primitive
        return primitive.id
    
    def get_primitive(self, id: str) -> Optional[SketchPrimitive]:
        return self.primitives.get(id)
    
    def get_point(self, ref: PointRef) -> Point2D:
        """Resolve a PointRef to actual coordinates."""
        prim = self.primitives[ref.element_id]
        if ref.point_type == PointType.CONTROL and isinstance(prim, Spline):
            return prim.get_control_point(ref.index)
        return prim.get_point(ref.point_type)
    
    def add_constraint(self, constraint: SketchConstraint) -> None:
        """Add a constraint to the sketch."""
        self.constraints.append(constraint)
        self.solver_status = SolverStatus.DIRTY  # Mark as needing re-solve
    
    def get_constraints_for(self, element_id: str) -> list[SketchConstraint]:
        """Get all constraints involving a specific element."""
        result = []
        for c in self.constraints:
            for ref in c.references:
                if isinstance(ref, str) and ref == element_id:
                    result.append(c)
                    break
                elif isinstance(ref, PointRef) and ref.element_id == element_id:
                    result.append(c)
                    break
        return result
    
    def to_text_description(self, include_point_coords: bool = False) -> str:
        """
        Generate human/AI-readable description of the sketch.
        
        Args:
            include_point_coords: If True, list all referenceable points with coordinates
        """
        lines = ["Elements:"]
        for id, prim in sorted(self.primitives.items()):
            lines.append(f"  {self._describe_primitive(prim)}")
            if include_point_coords:
                for pt_type in prim.get_valid_point_types():
                    pt = prim.get_point(pt_type)
                    lines.append(f"    {id}.{pt_type.value}: ({pt.x:.2f}, {pt.y:.2f})")
        
        lines.append("\nConstraints:")
        for c in self.constraints:
            lines.append(f"  {c}")
        
        lines.append(f"\nStatus: {self.solver_status.value}")
        if self.degrees_of_freedom >= 0:
            lines.append(f"Degrees of Freedom: {self.degrees_of_freedom}")
        
        return "\n".join(lines)
    
    def _describe_primitive(self, p: SketchPrimitive) -> str:
        if isinstance(p, Line):
            return f"{p.id}: Line ({p.start.x:.2f},{p.start.y:.2f}) → ({p.end.x:.2f},{p.end.y:.2f})"
        elif isinstance(p, Arc):
            return f"{p.id}: Arc center=({p.center.x:.2f},{p.center.y:.2f}) r={p.radius:.2f} {'CCW' if p.ccw else 'CW'}"
        elif isinstance(p, Circle):
            return f"{p.id}: Circle center=({p.center.x:.2f},{p.center.y:.2f}) r={p.radius:.2f}"
        elif isinstance(p, Point):
            return f"{p.id}: Point ({p.position.x:.2f},{p.position.y:.2f})"
        elif isinstance(p, Spline):
            return f"{p.id}: Spline degree={p.degree} points={len(p.control_points)} {'periodic' if p.periodic else 'open'}"
        else:
            return f"{p.id}: {type(p).__name__}"
```

---

## 6. Platform Adapters

### 6.1 Adapter Interface

```python
from abc import ABC, abstractmethod

class SketchBackendAdapter(ABC):
    """Abstract interface for CAD platform adapters."""
    
    @abstractmethod
    def create_sketch(self, name: str, plane: Optional[Any] = None) -> None:
        """
        Create a new empty sketch.
        
        Args:
            name: Sketch name
            plane: Platform-specific plane/face reference (required for some platforms)
        """
        pass
    
    @abstractmethod
    def load_sketch(self, sketch: SketchDocument) -> None:
        """Load a canonical sketch into the CAD system."""
        pass
    
    @abstractmethod
    def export_sketch(self) -> SketchDocument:
        """Export the current CAD sketch to canonical form."""
        pass
    
    @abstractmethod
    def add_primitive(self, primitive: SketchPrimitive) -> Any:
        """
        Add a single primitive, return CAD-internal reference.
        
        Returns platform-specific entity that can be used for chaining.
        """
        pass
    
    @abstractmethod
    def add_constraint(self, constraint: SketchConstraint) -> bool:
        """Add a constraint, return success status."""
        pass
    
    @abstractmethod
    def get_solver_status(self) -> tuple[SolverStatus, int]:
        """Get (status, degrees_of_freedom)."""
        pass
    
    @abstractmethod
    def capture_image(self, width: int, height: int) -> bytes:
        """Capture sketch visualization as PNG bytes."""
        pass
```

### 6.2 Geometry Chaining Pattern

When creating connected paths, CAD platforms work best when you use SketchPoint objects from previous segments rather than raw coordinates. This ensures proper topological connectivity.

```python
# CORRECT: Use SketchPoint objects for chaining
seg1 = create_line(p0, p1)
seg2 = create_line(seg1.end_sketch_point, p2)  # Use SketchPoint, not coordinates
seg3 = create_arc(seg2.end_sketch_point, mid, p3)

# INCORRECT: Creates disconnected segments that need explicit coincident constraints
seg1 = create_line(p0, p1)
seg2 = create_line(p1, p2)  # Same coordinates, but not topologically connected
```

Each platform section below documents the specific chaining pattern.

---

## 7. FreeCAD Adapter

### 7.1 Overview

| Aspect | FreeCAD Behavior |
|--------|------------------|
| Native unit | Millimeters |
| Arc representation | ArcOfCircle (angles) or Arc (3-point) |
| Tangent implies coincident? | **No** — must add separately |
| Constraint indexing | Geometry index + vertex index |
| Point cloud support | Points workbench |

### 7.2 Geometry Conversion

```python
class FreeCADAdapter(SketchBackendAdapter):
    """FreeCAD Sketcher adapter."""
    
    # ID to FreeCAD geometry index mapping
    _id_to_index: dict[str, int] = {}
    _index_to_id: dict[int, str] = {}
    
    def add_primitive(self, primitive: SketchPrimitive) -> int:
        sketch = self._get_active_sketch()
        
        if isinstance(primitive, Line):
            idx = sketch.addGeometry(
                Part.LineSegment(
                    App.Vector(primitive.start.x, primitive.start.y, 0),
                    App.Vector(primitive.end.x, primitive.end.y, 0)
                ),
                primitive.construction
            )
        
        elif isinstance(primitive, Arc):
            # USE THREE-POINT CONSTRUCTION for reliable arc direction
            start, mid, end = primitive.to_three_point()
            idx = sketch.addGeometry(
                Part.Arc(
                    App.Vector(start.x, start.y, 0),
                    App.Vector(mid.x, mid.y, 0),
                    App.Vector(end.x, end.y, 0)
                ),
                primitive.construction
            )
        
        elif isinstance(primitive, Circle):
            idx = sketch.addGeometry(
                Part.Circle(
                    App.Vector(primitive.center.x, primitive.center.y, 0),
                    App.Vector(0, 0, 1),
                    primitive.radius
                ),
                primitive.construction
            )
        
        self._id_to_index[primitive.id] = idx
        self._index_to_id[idx] = primitive.id
        return idx
```

### 7.3 Constraint Conversion

**FreeCAD vertex indexing:**
- Lines: vertex 1 = start, vertex 2 = end
- Arcs: vertex 1 = start, vertex 2 = end, vertex 3 = center
- Circles: vertex 3 = center (no start/end)

```python
    # PointType to FreeCAD vertex index
    VERTEX_MAP = {
        Line: {PointType.START: 1, PointType.END: 2},
        Arc: {PointType.START: 1, PointType.END: 2, PointType.CENTER: 3},
        Circle: {PointType.CENTER: 3},
    }
    
    def _point_ref_to_freecad(self, ref: PointRef) -> tuple[int, int]:
        """Convert PointRef to (geometry_index, vertex_index)."""
        prim = self._sketch_doc.get_primitive(ref.element_id)
        geo_idx = self._id_to_index[ref.element_id]
        vertex_idx = self.VERTEX_MAP[type(prim)][ref.point_type]
        return (geo_idx, vertex_idx)
    
    def add_constraint(self, constraint: SketchConstraint) -> bool:
        sketch = self._get_active_sketch()
        
        match constraint.constraint_type:
            case ConstraintType.COINCIDENT:
                pt1 = self._point_ref_to_freecad(constraint.references[0])
                pt2 = self._point_ref_to_freecad(constraint.references[1])
                sketch.addConstraint(Sketcher.Constraint(
                    'Coincident', pt1[0], pt1[1], pt2[0], pt2[1]
                ))
            
            case ConstraintType.TANGENT:
                # IMPORTANT: Use tangent-only, NOT coincident+tangent
                idx1 = self._id_to_index[constraint.references[0]]
                idx2 = self._id_to_index[constraint.references[1]]
                
                if constraint.connection_point:
                    # Tangent at specific point
                    pt = self._point_ref_to_freecad(constraint.connection_point)
                    sketch.addConstraint(Sketcher.Constraint(
                        'Tangent', idx1, pt[1], idx2, pt[1]
                    ))
                else:
                    # General tangent (FreeCAD picks connection point)
                    sketch.addConstraint(Sketcher.Constraint('Tangent', idx1, idx2))
            
            case ConstraintType.HORIZONTAL:
                idx = self._id_to_index[constraint.references[0]]
                sketch.addConstraint(Sketcher.Constraint('Horizontal', idx))
            
            case ConstraintType.VERTICAL:
                idx = self._id_to_index[constraint.references[0]]
                sketch.addConstraint(Sketcher.Constraint('Vertical', idx))
            
            case ConstraintType.PERPENDICULAR:
                idx1 = self._id_to_index[constraint.references[0]]
                idx2 = self._id_to_index[constraint.references[1]]
                sketch.addConstraint(Sketcher.Constraint('Perpendicular', idx1, idx2))
            
            case ConstraintType.PARALLEL:
                idx1 = self._id_to_index[constraint.references[0]]
                idx2 = self._id_to_index[constraint.references[1]]
                sketch.addConstraint(Sketcher.Constraint('Parallel', idx1, idx2))
            
            case ConstraintType.EQUAL:
                indices = [self._id_to_index[r] for r in constraint.references]
                for i in range(len(indices) - 1):
                    sketch.addConstraint(Sketcher.Constraint('Equal', indices[i], indices[i+1]))
            
            case ConstraintType.RADIUS:
                idx = self._id_to_index[constraint.references[0]]
                sketch.addConstraint(Sketcher.Constraint('Radius', idx, constraint.value))
            
            case ConstraintType.DISTANCE:
                pt1 = self._point_ref_to_freecad(constraint.references[0])
                pt2 = self._point_ref_to_freecad(constraint.references[1])
                sketch.addConstraint(Sketcher.Constraint(
                    'Distance', pt1[0], pt1[1], pt2[0], pt2[1], constraint.value
                ))
            
            case ConstraintType.DISTANCE_X:
                pt = self._point_ref_to_freecad(constraint.references[0])
                # -1, 1 = origin point
                sketch.addConstraint(Sketcher.Constraint(
                    'DistanceX', -1, 1, pt[0], pt[1], constraint.value
                ))
            
            case ConstraintType.DISTANCE_Y:
                pt = self._point_ref_to_freecad(constraint.references[0])
                sketch.addConstraint(Sketcher.Constraint(
                    'DistanceY', -1, 1, pt[0], pt[1], constraint.value
                ))
        
        return True
```

### 7.4 FreeCAD-Specific Considerations

| Issue | Handling |
|-------|----------|
| Arc angle ambiguity | Always use three-point arc construction (`Part.Arc(start, mid, end)`) |
| Coincident + Tangent redundancy | Apply only Tangent with point specifiers; do NOT add separate Coincident |
| Constraint solve failures | Check `sketch.solve()` return value; inspect conflicting constraints |
| Label visualization | Create Draft.Text objects at element midpoints |
| Origin reference | Use geometry index -1, vertex 1 for origin point |

### 7.5 Exporting from FreeCAD

```python
    def export_sketch(self) -> SketchDocument:
        """Read FreeCAD sketch into canonical form."""
        sketch = self._get_active_sketch()
        doc = SketchDocument(name=sketch.Label)
        
        # Export geometry
        for i, geo in enumerate(sketch.Geometry):
            if isinstance(geo, Part.LineSegment):
                prim = Line(
                    start=Point2D(geo.StartPoint.x, geo.StartPoint.y),
                    end=Point2D(geo.EndPoint.x, geo.EndPoint.y),
                )
            elif isinstance(geo, Part.ArcOfCircle):
                # Convert angle-based to point-based
                center = Point2D(geo.Center.x, geo.Center.y)
                prim = Arc(
                    center=center,
                    start_point=Point2D(geo.StartPoint.x, geo.StartPoint.y),
                    end_point=Point2D(geo.EndPoint.x, geo.EndPoint.y),
                    ccw=geo.EndAngle > geo.StartAngle,  # Simplified; may need adjustment
                )
            elif isinstance(geo, Part.Circle):
                prim = Circle(
                    center=Point2D(geo.Center.x, geo.Center.y),
                    radius=geo.Radius,
                )
            else:
                continue
            
            doc.add_primitive(prim)
            self._index_to_id[i] = prim.id
        
        # Export constraints
        for c in sketch.Constraints:
            canonical = self._convert_freecad_constraint(c)
            if canonical:
                doc.add_constraint(canonical)
        
        return doc
```

---

## 8. SolidWorks Adapter

### 8.1 Overview

| Aspect | SolidWorks Behavior |
|--------|---------------------|
| Native unit | **Meters** |
| Arc representation | Center, start, end, direction (±1) |
| Tangent implies coincident? | Often auto-inferred at endpoints |
| API style | COM Interop (C#/C++) |
| Coordinate transform | Model-to-sketch transform required for 2D |
| Angle units | Radians |

### 8.2 Unit Conversion

```csharp
// C# adapter
public class SolidWorksAdapter : ISketchBackendAdapter
{
    private const double MM_TO_METERS = 0.001;
    private const double METERS_TO_MM = 1000.0;
    
    private (double x, double y, double z) ToSwCoords(Point2D pt)
    {
        return (pt.X * MM_TO_METERS, pt.Y * MM_TO_METERS, 0.0);
    }
}
```

### 8.3 Geometry Conversion

```csharp
public object AddPrimitive(SketchPrimitive primitive)
{
    var sm = _sketchManager;
    SketchSegment segment = null;
    
    if (primitive is Line line)
    {
        var (x0, y0, z0) = ToSwCoords(line.Start);
        var (x1, y1, z1) = ToSwCoords(line.End);
        segment = sm.CreateLine(x0, y0, z0, x1, y1, z1);
    }
    else if (primitive is Arc arc)
    {
        var (cx, cy, cz) = ToSwCoords(arc.Center);
        var (sx, sy, sz) = ToSwCoords(arc.StartPoint);
        var (ex, ey, ez) = ToSwCoords(arc.EndPoint);
        short direction = (short)(arc.Ccw ? 1 : -1);
        segment = sm.CreateArc(cx, cy, cz, sx, sy, sz, ex, ey, ez, direction);
    }
    else if (primitive is Circle circle)
    {
        var (cx, cy, cz) = ToSwCoords(circle.Center);
        double radiusM = circle.Radius * MM_TO_METERS;
        segment = sm.CreateCircleByRadius(cx, cy, cz, radiusM);
    }
    
    if (primitive.Construction)
        segment.ConstructionGeometry = true;
    
    return RegisterSegment(segment, primitive.Id);
}
```

#### Arc Direction Handling

SolidWorks supports two equivalent approaches for arc direction:

**Approach 1: Variable direction parameter**
```csharp
// Pass direction based on ccw flag
short direction = (short)(arc.Ccw ? 1 : -1);
segment = sm.CreateArc(cx, cy, cz, sx, sy, sz, ex, ey, ez, direction);
```

**Approach 2: Swap endpoints, always use direction=1**
```csharp
// Swap start/end points based on ccw, always pass 1
var p0 = arc.Ccw ? arc.StartPoint : arc.EndPoint;
var p1 = arc.Ccw ? arc.EndPoint : arc.StartPoint;
segment = sm.CreateArc(cx, cy, cz, p0.X, p0.Y, 0, p1.X, p1.Y, 0, 1);
```

Both approaches produce equivalent results. Use Approach 1 for clarity.

### 8.4 Constraint Conversion

#### Implementation Status

| Constraint | Status | Method |
|------------|--------|--------|
| FIXED | ✓ Implemented | `swConstraintType_e.swConstraintType_FIXED` |
| HORIZONTAL | ✓ Implemented | `swConstraintType_e.swConstraintType_HORIZONTAL` |
| VERTICAL | ✓ Implemented | `swConstraintType_e.swConstraintType_VERTICAL` |
| COINCIDENT | Use Alternative | `sgMERGEPOINTS` for path connections |
| TANGENT | ⚠ Not Implemented | Would need implementation |
| PERPENDICULAR | ⚠ Not Implemented | Would need implementation |
| PARALLEL | ⚠ Not Implemented | Would need implementation |
| EQUAL | ⚠ Not Implemented | Would need implementation |
| CONCENTRIC | ⚠ Not Implemented | Would need implementation |

```csharp
public bool AddConstraint(SketchConstraint constraint)
{
    var md = (ModelDoc2)_solidWorks.ActiveDoc;
    
    switch (constraint.ConstraintType)
    {
        case ConstraintType.FIXED:
            // Select entity first, then add constraint
            SelectEntities(constraint.References);
            md.SketchAddConstraints("sgFIXED");
            break;
            
        case ConstraintType.HORIZONTAL:
            SelectEntities(constraint.References);
            md.SketchAddConstraints("sgHORIZONTAL2D");
            break;
            
        case ConstraintType.VERTICAL:
            SelectEntities(constraint.References);
            md.SketchAddConstraints("sgVERTICAL2D");
            break;
    }
    
    return true;
}
```

### 8.5 Geometry Chaining Pattern

For connected paths, use `sgMERGEPOINTS` instead of explicit COINCIDENT constraints:

```csharp
// After creating a sequence of connected segments:
public void JoinSegments(List<SketchSegment> segments)
{
    // Select all segment endpoints that should connect
    foreach (var seg in segments)
    {
        seg.Select4(true, null);  // Append to selection
    }
    
    // Merge coincident points
    ((ModelDoc2)_solidWorks.ActiveDoc).SketchAddConstraints("sgMERGEPOINTS");
}
```

### 8.6 Spline Creation

```csharp
// Point-based spline with optional simplification
public SketchSegment CreateSpline(List<Point2D> points, bool closed)
{
    double[] pointsMeters = new double[points.Count * 3];
    for (int i = 0; i < points.Count; i++)
    {
        pointsMeters[i * 3] = points[i].X * MM_TO_METERS;
        pointsMeters[i * 3 + 1] = points[i].Y * MM_TO_METERS;
        pointsMeters[i * 3 + 2] = 0;
    }
    
    var spline = _sketchManager.CreateSpline2(pointsMeters, closed);
    
    // Optional: Simplify to reduce control points
    // spline.Simplify(tolerance * MM_TO_METERS);
    
    return spline;
}

// Parametric B-spline with explicit control
public SketchSegment CreateParametricSpline(Spline canonical)
{
    var splineParams = _sketchManager.CreateSplineParamData();
    splineParams.Dimension = 3;
    splineParams.Order = canonical.Order;  // degree + 1
    splineParams.Periodic = canonical.Periodic ? 1 : 0;
    splineParams.ControlPointsCount = canonical.ControlPoints.Count;
    
    // Convert knots
    double[] knots = canonical.Knots.ToArray();
    splineParams.SetKnotPoints(knots);
    
    // Convert control points to meters
    double[] controls = new double[canonical.ControlPoints.Count * 3];
    for (int i = 0; i < canonical.ControlPoints.Count; i++)
    {
        controls[i * 3] = canonical.ControlPoints[i].X * MM_TO_METERS;
        controls[i * 3 + 1] = canonical.ControlPoints[i].Y * MM_TO_METERS;
        controls[i * 3 + 2] = 0;
    }
    splineParams.SetControlPoints(controls);
    
    return _sketchManager.CreateSplinesByEqnParams2(splineParams);
}
```

### 8.7 Transform Matrix Format

The model-to-sketch transform is a 12-element array in column-major order:

```csharp
public GeoMatrix2 GetModelToSketchTransform()
{
    var sketch = _sketchManager.ActiveSketch;
    var transform = sketch.ModelToSketchTransform;
    double[] data = (double[])transform.ArrayData;
    
    // Array layout:
    // [0-2]:  First column of 3x3 rotation
    // [3-5]:  Second column of 3x3 rotation  
    // [6-8]:  Third column of 3x3 rotation
    // [9-11]: Translation vector
    
    var matrix = new GeoMatrix2();
    matrix.ImportSWMatrix(data);
    return matrix;
}
```

### 8.8 Mirrored Transform Handling

When applying transforms that include reflection, arc direction must be flipped:

```csharp
public Arc TransformArc(Arc arc, GeoMatrix2 transform)
{
    // Check if transform is mirrored (negative determinant)
    double det2 = transform[0, 0] * transform[1, 1] - transform[0, 1] * transform[1, 0];
    
    var result = new Arc
    {
        Center = transform.Apply(arc.Center),
        StartPoint = transform.Apply(arc.StartPoint),
        EndPoint = transform.Apply(arc.EndPoint),
        Ccw = det2 < 0 ? !arc.Ccw : arc.Ccw  // Flip direction for mirrored transforms
    };
    
    return result;
}
```

### 8.9 SolidWorks-Specific Considerations

| Issue | Handling |
|-------|----------|
| Unit conversion | All coords × 0.001 (mm → meters) |
| Model-to-sketch transform | Apply via `Sketch.ModelToSketchTransform` before creating 2D elements |
| Retry logic | SW API sometimes fails; retry up to 10× |
| Batch mode | Use `SWAddToDBContext` wrapper for performance |
| MERGEPOINTS vs COINCIDENT | Use `sgMERGEPOINTS` for connecting sequential path segments |
| Arc direction | +1 = CCW, -1 = CW |
| Mirrored transforms | Check determinant; flip arc direction if negative |

### 8.10 Exporting from SolidWorks

```csharp
public SketchDocument ExportSketch()
{
    var doc = new SketchDocument { Name = _sketch.Name };
    
    foreach (var seg in _sketch.GetSketchSegments())
    {
        SketchPrimitive prim = null;
        
        switch (seg.GetType())
        {
            case (int)swSketchSegments_e.swSketchLINE:
                var line = (SketchLine)seg;
                var sp = line.GetStartPoint2();
                var ep = line.GetEndPoint2();
                prim = new Line
                {
                    Start = new Point2D(sp.X * METERS_TO_MM, sp.Y * METERS_TO_MM),
                    End = new Point2D(ep.X * METERS_TO_MM, ep.Y * METERS_TO_MM),
                    Construction = seg.ConstructionGeometry
                };
                break;
                
            case (int)swSketchSegments_e.swSketchARC:
                var arc = (SketchArc)seg;
                var center = arc.GetCenterPoint2();
                var start = arc.GetStartPoint2();
                var end = arc.GetEndPoint2();
                short dir = arc.GetDirection();
                prim = new Arc
                {
                    Center = new Point2D(center.X * METERS_TO_MM, center.Y * METERS_TO_MM),
                    StartPoint = new Point2D(start.X * METERS_TO_MM, start.Y * METERS_TO_MM),
                    EndPoint = new Point2D(end.X * METERS_TO_MM, end.Y * METERS_TO_MM),
                    Ccw = dir > 0
                };
                break;
        }
        
        if (prim != null)
            doc.AddPrimitive(prim);
    }
    
    return doc;
}
```

---

## 9. Inventor Adapter

### 9.1 Overview

| Aspect | Inventor Behavior |
|--------|-------------------|
| Native unit | **Centimeters** |
| Arc representation | Center, start, end, ccw flag |
| Tangent implies coincident? | Context-dependent |
| API style | COM Interop (C#/C++) |
| Sketch types | PlanarSketch (2D) and Sketch3D |

### 9.2 Unit Conversion

```csharp
public class InventorAdapter : ISketchBackendAdapter
{
    private const double MM_TO_CM = 0.1;
    private const double CM_TO_MM = 10.0;
    
    private TransientGeometry _tg;  // Required for point creation
    
    public InventorAdapter(Application app)
    {
        _tg = app.TransientGeometry;  // Must cache this
    }
    
    private Point2d ToInvCoords(Point2D pt)
    {
        return _tg.CreatePoint2d(pt.X * MM_TO_CM, pt.Y * MM_TO_CM);
    }
    
    private Point ToInvCoords3D(Point2D pt)
    {
        return _tg.CreatePoint(pt.X * MM_TO_CM, pt.Y * MM_TO_CM, 0);
    }
}
```

### 9.3 Geometry Conversion

```csharp
public object AddPrimitive(SketchPrimitive primitive)
{
    if (primitive is Line line)
    {
        var p0 = ToInvCoords(line.Start);
        var p1 = ToInvCoords(line.End);
        var segment = _sketch.SketchLines.AddByTwoPoints(p0, p1);
        if (primitive.Construction)
            segment.Construction = true;
        return RegisterSegment(segment, primitive.Id);
    }
    else if (primitive is Arc arc)
    {
        var center = ToInvCoords(arc.Center);
        var start = ToInvCoords(arc.StartPoint);
        var end = ToInvCoords(arc.EndPoint);
        var segment = _sketch.SketchArcs.AddByCenterStartEndPoint(center, start, end, arc.Ccw);
        if (primitive.Construction)
            segment.Construction = true;
        return RegisterSegment(segment, primitive.Id);
    }
    else if (primitive is Circle circle)
    {
        var center = ToInvCoords(circle.Center);
        double radiusCm = circle.Radius * MM_TO_CM;
        var segment = _sketch.SketchCircles.AddByCenterRadius(center, radiusCm);
        if (primitive.Construction)
            segment.Construction = true;
        return RegisterSegment(segment, primitive.Id);
    }
    
    return null;
}
```

### 9.4 Geometry Chaining Pattern

For connected paths, use SketchPoint objects from previous segments:

```csharp
public void CreateConnectedPath(List<SketchPrimitive> primitives)
{
    SketchPoint currentEnd = null;
    SketchPoint firstStart = null;
    
    foreach (var prim in primitives)
    {
        if (prim is Line line)
        {
            SketchLine segment;
            if (currentEnd != null)
            {
                // Use SketchPoint from previous segment
                var p1 = ToInvCoords(line.End);
                segment = _sketch.SketchLines.AddByTwoPoints(currentEnd, p1);
            }
            else
            {
                var p0 = ToInvCoords(line.Start);
                var p1 = ToInvCoords(line.End);
                segment = _sketch.SketchLines.AddByTwoPoints(p0, p1);
                firstStart = segment.StartSketchPoint;
            }
            currentEnd = segment.EndSketchPoint;
        }
        else if (prim is Arc arc)
        {
            // Similar pattern for arcs
            // Note: StartSketchPoint/EndSketchPoint mapping depends on ccw flag
            SketchArc segment = /* create arc */;
            currentEnd = arc.Ccw ? segment.EndSketchPoint : segment.StartSketchPoint;
        }
    }
    
    // Close path if needed
    if (firstStart != null && currentEnd != null)
    {
        try
        {
            _sketch.GeometricConstraints.AddCoincident(
                (SketchEntity)firstStart, 
                (SketchEntity)currentEnd
            );
        }
        catch (Exception)
        {
            // Constraint may fail if points already coincident
        }
    }
}
```

### 9.5 Arc Endpoint Mapping

The `StartSketchPoint` and `EndSketchPoint` properties depend on the `ccw` flag:

```csharp
// When ccw=true:  StartSketchPoint = arc start, EndSketchPoint = arc end
// When ccw=false: StartSketchPoint = arc end,   EndSketchPoint = arc start

public SketchPoint GetArcStartPoint(SketchArc arc, bool ccw)
{
    return ccw ? arc.StartSketchPoint : arc.EndSketchPoint;
}

public SketchPoint GetArcEndPoint(SketchArc arc, bool ccw)
{
    return ccw ? arc.EndSketchPoint : arc.StartSketchPoint;
}
```

### 9.6 Constraint Conversion

#### Implementation Status

| Constraint | Status | Method |
|------------|--------|--------|
| FIXED (Ground) | ✓ Verified | `AddGround(entity)` |
| HORIZONTAL | ✓ Verified | `AddHorizontal(entity)` |
| VERTICAL | ✓ Verified | `AddVertical(entity)` |
| COINCIDENT | ✓ Verified | `AddCoincident(e1, e2)` — wrap in try-catch |
| TANGENT | ⚠ Unverified | Likely `AddTangent(e1, e2)` |
| PERPENDICULAR | ⚠ Unverified | Likely `AddPerpendicular(e1, e2)` |
| PARALLEL | ⚠ Unverified | Likely `AddParallel(e1, e2)` |
| EQUAL | ⚠ Unverified | Likely `AddEqual(e1, e2)` |

```csharp
public bool AddConstraint(SketchConstraint constraint)
{
    var gc = _sketch.GeometricConstraints;
    
    switch (constraint.ConstraintType)
    {
        case ConstraintType.COINCIDENT:
            var e1 = GetSketchPoint(constraint.References[0]) as SketchEntity;
            var e2 = GetSketchPoint(constraint.References[1]) as SketchEntity;
            try
            {
                gc.AddCoincident(e1, e2);
            }
            catch (Exception)
            {
                // Constraint may fail if solver determines points already coincident
                // This is normal - verify geometry rather than constraint success
            }
            break;
            
        case ConstraintType.HORIZONTAL:
            var entity = GetEntity(constraint.References[0]) as SketchEntity;
            gc.AddHorizontal(entity);
            break;
            
        case ConstraintType.VERTICAL:
            gc.AddVertical(GetEntity(constraint.References[0]) as SketchEntity);
            break;
            
        case ConstraintType.FIXED:
            gc.AddGround(GetEntity(constraint.References[0]) as SketchEntity);
            break;
            
        case ConstraintType.TANGENT:
            gc.AddTangent(
                GetEntity(constraint.References[0]),
                GetEntity(constraint.References[1])
            );
            break;
            
        case ConstraintType.PERPENDICULAR:
            gc.AddPerpendicular(
                GetEntity(constraint.References[0]),
                GetEntity(constraint.References[1])
            );
            break;
            
        case ConstraintType.PARALLEL:
            gc.AddParallel(
                GetEntity(constraint.References[0]),
                GetEntity(constraint.References[1])
            );
            break;
            
        case ConstraintType.RADIUS:
            var dc = _sketch.DimensionConstraints;
            var dim = dc.AddRadial(
                GetEntity(constraint.References[0]),
                ToInvCoords(GetDimensionPlacement(constraint))  // Text placement
            );
            dim.Parameter.Value = constraint.Value * MM_TO_CM;
            break;
    }
    
    return true;
}
```

### 9.7 3D Sketch Constraints

3D sketches use a separate constraint system:

```csharp
// 2D sketch
sketch2d.GeometricConstraints.AddGround((SketchEntity)element);

// 3D sketch - different types
sketch3d.GeometricConstraints3D.AddGround((SketchEntity3D)element);
```

### 9.8 Spline Creation

```csharp
// Fit-point spline
public SketchSpline CreateFitSpline(List<Point2D> fitPoints)
{
    var points = _app.TransientObjects.CreateObjectCollection();
    foreach (var pt in fitPoints)
    {
        points.Add(ToInvCoords(pt));
    }
    return _sketch.SketchSplines.Add(points, SplineFitMethodEnum.kSweetSplineFit);
}

// B-spline with control points (3D sketch)
public SketchFixedSpline3D CreateBSpline3D(Spline canonical)
{
    // Build pole array
    object[] poles = new object[canonical.ControlPoints.Count];
    for (int i = 0; i < canonical.ControlPoints.Count; i++)
    {
        poles[i] = ToInvCoords3D(canonical.ControlPoints[i]);
    }
    
    double[] knots = canonical.Knots.ToArray();
    double[] weights = canonical.Weights?.ToArray() ?? new double[0];
    
    var curve = _tg.CreateBSplineCurve(
        canonical.Order,        // order = degree + 1
        ref poles,
        ref knots, 
        ref weights,
        canonical.Periodic
    );
    
    return _sketch3d.SketchFixedSplines3D.Add(curve);
}
```

### 9.9 Transform Matrix Import

```csharp
public GeoMatrix2 GetModelToSketchTransform()
{
    var sketch = _sketch as PlanarSketch;
    var transform = sketch.ModelToSketchTransform;
    
    double[] matrixData = new double[16];
    transform.GetMatrixData(ref matrixData);
    
    var matrix = new GeoMatrix2();
    matrix.ImportINVMatrix(matrixData);
    return matrix;
}
```

### 9.10 Inventor-Specific Considerations

| Issue | Handling |
|-------|----------|
| Unit conversion | All coords × 0.1 (mm → cm) |
| TransientGeometry | Must cache `app.TransientGeometry` for point creation |
| Model-to-sketch transform | `PlanarSketch.ModelToSketchTransform` — import as 16-element matrix |
| DeferUpdates | Set `True` during batch operations, `False` after |
| Transaction wrapping | Wrap operations in `TransactionManager.StartTransaction()` |
| Full circle from arc | When `start_point == end_point`, use `AddByCenterRadius` instead |
| Endpoint access | Use `StartSketchPoint` / `EndSketchPoint` for chaining |
| Arc CCW mapping | `StartSketchPoint`/`EndSketchPoint` swap based on `ccw` flag |
| Coincident failures | Wrap in try-catch; solver may reject if already satisfied |
| 3D constraints | Use `GeometricConstraints3D` with `SketchEntity3D` types |
| ObjectCollection | Use `TransientObjects.CreateObjectCollection()` for point arrays |

---

## 10. Fusion 360 Adapter

### 10.1 Overview

| Aspect | Fusion 360 Behavior |
|--------|---------------------|
| Native unit | **Centimeters** |
| Arc representation | Three-point OR center-start-sweep |
| Tangent implies coincident? | No explicit rule; depends on context |
| API style | Python or C++ API |
| Profile detection | Automatic from closed loops |
| Sketch creation | Must be on ConstructionPlane or BRepFace |

### 10.2 Unit Conversion

```python
class Fusion360Adapter(SketchBackendAdapter):
    """Autodesk Fusion 360 adapter."""
    
    MM_TO_CM = 0.1
    CM_TO_MM = 10.0
    
    def _to_f360_point(self, pt: Point2D) -> 'Point3D':
        """Convert to Point3D (for standalone geometry or arc midpoint)."""
        return adsk.core.Point3D.create(pt.x * self.MM_TO_CM, pt.y * self.MM_TO_CM, 0)
```

### 10.3 Sketch Creation

Fusion 360 requires sketches to be created on a plane or face:

```python
def create_sketch(self, name: str, plane=None) -> None:
    """
    Create sketch on plane.
    
    Args:
        name: Sketch name
        plane: ConstructionPlane or BRepFace. Defaults to XY plane.
    """
    root = self._app.activeProduct.rootComponent
    sketches = root.sketches
    
    if plane is None:
        plane = root.xYConstructionPlane
    
    self._sketch = sketches.add(plane)
    self._sketch.name = name
```

### 10.4 SketchPoint vs Point3D

Fusion 360 distinguishes between `Point3D` (raw coordinates) and `SketchPoint` (topology element):

| Use Case | Type |
|----------|------|
| Standalone geometry creation | Point3D |
| Chained/connected geometry | SketchPoint |
| Arc midpoint (three-point) | Point3D |
| Constraint point references | SketchPoint |

### 10.5 Geometry Conversion

```python
def add_primitive(self, primitive: SketchPrimitive) -> Any:
    curves = self._sketch.sketchCurves
    points = self._sketch.sketchPoints
    
    if isinstance(primitive, Line):
        # Use SketchPoints for proper connectivity
        p0 = points.add(self._to_f360_point(primitive.start))
        p1 = points.add(self._to_f360_point(primitive.end))
        entity = curves.sketchLines.addByTwoPoints(p0, p1)
        
    elif isinstance(primitive, Arc):
        # Three-point arc: start and end are SketchPoints, mid is Point3D
        start, mid, end = primitive.to_three_point()
        p_start = points.add(self._to_f360_point(start))
        p_mid = self._to_f360_point(mid)  # Point3D, NOT SketchPoint
        p_end = points.add(self._to_f360_point(end))
        entity = curves.sketchArcs.addByThreePoints(p_start, p_mid, p_end)
        
    elif isinstance(primitive, Circle):
        center = self._to_f360_point(primitive.center)
        radius_cm = primitive.radius * self.MM_TO_CM
        entity = curves.sketchCircles.addByCenterRadius(center, radius_cm)
    
    entity.isConstruction = primitive.construction
    
    # Health check after creation
    if self._sketch.healthState != adsk.fusion.FeatureHealthStates.HealthyFeatureHealthState:
        raise SketchCreationError(f"Sketch became unhealthy after adding {primitive.id}")
    
    return self._register(entity, primitive.id)
```

### 10.6 Geometry Chaining Pattern

```python
def create_connected_path(self, primitives: list[SketchPrimitive]) -> None:
    """Create connected geometry using SketchPoints for topology."""
    curves = self._sketch.sketchCurves
    points = self._sketch.sketchPoints
    
    current_end: Optional[SketchPoint] = None
    first_start: Optional[SketchPoint] = None
    
    for prim in primitives:
        if isinstance(prim, Line):
            if current_end is not None:
                # Chain from previous segment
                p1 = points.add(self._to_f360_point(prim.end))
                segment = curves.sketchLines.addByTwoPoints(current_end, p1)
            else:
                # First segment
                p0 = points.add(self._to_f360_point(prim.start))
                p1 = points.add(self._to_f360_point(prim.end))
                segment = curves.sketchLines.addByTwoPoints(p0, p1)
                first_start = segment.startSketchPoint
            current_end = segment.endSketchPoint
            
        elif isinstance(prim, Arc):
            start, mid, end = prim.to_three_point()
            if current_end is not None:
                p_mid = self._to_f360_point(mid)
                p_end = points.add(self._to_f360_point(end))
                segment = curves.sketchArcs.addByThreePoints(current_end, p_mid, p_end)
            else:
                p_start = points.add(self._to_f360_point(start))
                p_mid = self._to_f360_point(mid)
                p_end = points.add(self._to_f360_point(end))
                segment = curves.sketchArcs.addByThreePoints(p_start, p_mid, p_end)
                first_start = segment.startSketchPoint
            current_end = segment.endSketchPoint
    
    # Close path if needed
    if first_start and current_end:
        self._sketch.geometricConstraints.addCoincident(first_start, current_end)
```

### 10.7 Constraint Conversion

#### Implementation Status

| Constraint | Status | Method |
|------------|--------|--------|
| COINCIDENT | ✓ Verified | `addCoincident(p1, p2)` |
| TANGENT | ✓ Verified | `addTangent(e1, e2)` |
| HORIZONTAL | ✓ Verified | `addHorizontal(entity)` |
| VERTICAL | ✓ Verified | `addVertical(entity)` |
| PERPENDICULAR | ✓ Verified | `addPerpendicular(e1, e2)` |
| PARALLEL | ✓ Verified | `addParallel(e1, e2)` |
| CONCENTRIC | ✓ Verified | `addConcentric(e1, e2)` |
| EQUAL | ✓ Verified | `addEqual(e1, e2)` |
| FIXED | ⚠ Verify | `addFix(entity)` or `addGround(entity)` |
| COLLINEAR | ⚠ Verify | May not exist |
| SYMMETRIC | ⚠ Verify | May not exist |
| MIDPOINT | ⚠ Verify | May not exist |

```python
def add_constraint(self, constraint: SketchConstraint) -> bool:
    gc = self._sketch.geometricConstraints
    dc = self._sketch.sketchDimensions
    
    match constraint.constraint_type:
        case ConstraintType.COINCIDENT:
            p1 = self._get_sketch_point(constraint.references[0])
            p2 = self._get_sketch_point(constraint.references[1])
            gc.addCoincident(p1, p2)
        
        case ConstraintType.TANGENT:
            e1 = self._get_entity(constraint.references[0])
            e2 = self._get_entity(constraint.references[1])
            gc.addTangent(e1, e2)
        
        case ConstraintType.HORIZONTAL:
            e = self._get_entity(constraint.references[0])
            gc.addHorizontal(e)
        
        case ConstraintType.VERTICAL:
            e = self._get_entity(constraint.references[0])
            gc.addVertical(e)
        
        case ConstraintType.PERPENDICULAR:
            e1 = self._get_entity(constraint.references[0])
            e2 = self._get_entity(constraint.references[1])
            gc.addPerpendicular(e1, e2)
        
        case ConstraintType.PARALLEL:
            e1 = self._get_entity(constraint.references[0])
            e2 = self._get_entity(constraint.references[1])
            gc.addParallel(e1, e2)
        
        case ConstraintType.CONCENTRIC:
            e1 = self._get_entity(constraint.references[0])
            e2 = self._get_entity(constraint.references[1])
            gc.addConcentric(e1, e2)
        
        case ConstraintType.EQUAL:
            e1 = self._get_entity(constraint.references[0])
            e2 = self._get_entity(constraint.references[1])
            gc.addEqual(e1, e2)
        
        case ConstraintType.FIXED:
            e = self._get_entity(constraint.references[0])
            gc.addFix(e)  # Verify: might be addGround
        
        # Dimensional constraints
        case ConstraintType.RADIUS:
            e = self._get_entity(constraint.references[0])
            placement = self._get_dimension_placement(e)
            dim = dc.addRadialDimension(e, placement)
            dim.value = constraint.value * self.MM_TO_CM
        
        case ConstraintType.DISTANCE:
            p1 = self._get_sketch_point(constraint.references[0])
            p2 = self._get_sketch_point(constraint.references[1])
            placement = self._get_dimension_placement_between(p1, p2)
            dim = dc.addDistanceDimension(
                p1, p2,
                adsk.fusion.DimensionOrientations.AlignedDimensionOrientation,
                placement
            )
            dim.value = constraint.value * self.MM_TO_CM
    
    # Verify health after constraint
    if self._sketch.healthState != adsk.fusion.FeatureHealthStates.HealthyFeatureHealthState:
        return False
    
    return True

def _get_dimension_placement(self, entity) -> 'Point3D':
    """Calculate sensible dimension text placement (not origin)."""
    # Example: offset from entity center
    if hasattr(entity, 'centerSketchPoint'):
        center = entity.centerSketchPoint.geometry
        return adsk.core.Point3D.create(center.x + 1.0, center.y + 1.0, 0)
    elif hasattr(entity, 'startSketchPoint'):
        start = entity.startSketchPoint.geometry
        return adsk.core.Point3D.create(start.x + 1.0, start.y + 1.0, 0)
    return adsk.core.Point3D.create(1.0, 1.0, 0)
```

### 10.8 Exporting from Fusion 360

```python
def export_sketch(self) -> SketchDocument:
    doc = SketchDocument(name=self._sketch.name)
    
    for curve in self._sketch.sketchCurves:
        prim = None
        
        if isinstance(curve, adsk.fusion.SketchLine):
            prim = Line(
                start=Point2D(
                    curve.startSketchPoint.geometry.x * self.CM_TO_MM,
                    curve.startSketchPoint.geometry.y * self.CM_TO_MM
                ),
                end=Point2D(
                    curve.endSketchPoint.geometry.x * self.CM_TO_MM,
                    curve.endSketchPoint.geometry.y * self.CM_TO_MM
                ),
                construction=curve.isConstruction
            )
        elif isinstance(curve, adsk.fusion.SketchArc):
            geo = curve.geometry  # Arc3D
            # Derive CCW from sweep angle sign
            sweep = geo.sweepAngle  # Positive = CCW
            prim = Arc(
                center=Point2D(
                    curve.centerSketchPoint.geometry.x * self.CM_TO_MM,
                    curve.centerSketchPoint.geometry.y * self.CM_TO_MM
                ),
                start_point=Point2D(
                    curve.startSketchPoint.geometry.x * self.CM_TO_MM,
                    curve.startSketchPoint.geometry.y * self.CM_TO_MM
                ),
                end_point=Point2D(
                    curve.endSketchPoint.geometry.x * self.CM_TO_MM,
                    curve.endSketchPoint.geometry.y * self.CM_TO_MM
                ),
                ccw=(sweep > 0),  # Use sweep angle sign for CCW
                construction=curve.isConstruction
            )
        elif isinstance(curve, adsk.fusion.SketchCircle):
            prim = Circle(
                center=Point2D(
                    curve.centerSketchPoint.geometry.x * self.CM_TO_MM,
                    curve.centerSketchPoint.geometry.y * self.CM_TO_MM
                ),
                radius=curve.radius * self.CM_TO_MM,
                construction=curve.isConstruction
            )
        
        if prim:
            doc.add_primitive(prim)
    
    return doc
```

### 10.9 Fusion 360-Specific Considerations

| Issue | Handling |
|-------|----------|
| Unit conversion | All coords and dimensions × 0.1 (mm → cm) |
| Angular units | Radians internally |
| Sketch creation | Must be on ConstructionPlane or BRepFace |
| SketchPoint vs Point3D | Use SketchPoint for chaining, Point3D for arc midpoints |
| Arc three-point signature | `addByThreePoints(SketchPoint, Point3D, SketchPoint)` |
| Arc construction | `addByThreePoints` preferred over center-sweep |
| CCW on export | Use `geometry.sweepAngle > 0`, not geometry computation |
| Health state | Check `sketch.healthState == HealthyFeatureHealthState` after operations |
| Profile detection | Automatic; access via `sketch.profiles` |
| NURBS/Splines | Full support via `NurbsCurve3D` |
| Sketch point access | Lines have `startSketchPoint`, `endSketchPoint` properties |
| Tolerance | Point coincidence ≈ 0.001 cm |
| Dimension placement | Calculate proper positions; don't use (0,0) |

---

## 11. Conversion Summary Table

### 11.1 Units

| Platform | Internal Unit | × from mm | × to mm |
|----------|---------------|-----------|---------|
| Canonical | mm | 1.0 | 1.0 |
| FreeCAD | mm | 1.0 | 1.0 |
| SolidWorks | m | 0.001 | 1000.0 |
| Inventor | cm | 0.1 | 10.0 |
| Fusion 360 | cm | 0.1 | 10.0 |

### 11.2 Arc Representation

| Platform | Canonical → Platform | Platform → Canonical |
|----------|----------------------|----------------------|
| FreeCAD | Use `Part.Arc(start, mid, end)` (three-point) | Read `StartPoint`, `EndPoint`, compute `ccw` from angles |
| SolidWorks | `CreateArc(cx, cy, cz, sx, sy, sz, ex, ey, ez, dir)` where `dir` = 1 (CCW) or -1 (CW) | Read arc params, `ccw = (dir > 0)` |
| Inventor | `AddByCenterStartEndPoint(center, start, end, ccw)` | Direct mapping; note endpoint swap based on `ccw` |
| Fusion 360 | `addByThreePoints(SketchPoint, Point3D, SketchPoint)` | Use `geometry.sweepAngle > 0` for `ccw` |

### 11.3 Constraint Naming and Status

| Canonical | FreeCAD | SolidWorks | Inventor | Fusion 360 |
|-----------|---------|------------|----------|------------|
| COINCIDENT | Coincident ✓ | sgMERGEPOINTS ✓ | AddCoincident ✓ | addCoincident ✓ |
| TANGENT | Tangent ✓ | ⚠ Not impl | AddTangent ⚠ | addTangent ✓ |
| PERPENDICULAR | Perpendicular ✓ | ⚠ Not impl | AddPerpendicular ⚠ | addPerpendicular ✓ |
| PARALLEL | Parallel ✓ | ⚠ Not impl | AddParallel ⚠ | addParallel ✓ |
| HORIZONTAL | Horizontal ✓ | sgHORIZONTAL2D ✓ | AddHorizontal ✓ | addHorizontal ✓ |
| VERTICAL | Vertical ✓ | sgVERTICAL2D ✓ | AddVertical ✓ | addVertical ✓ |
| EQUAL | Equal ✓ | ⚠ Not impl | AddEqual ⚠ | addEqual ✓ |
| CONCENTRIC | Concentric ✓ | ⚠ Not impl | AddConcentric ⚠ | addConcentric ✓ |
| FIXED | Fixed ✓ | sgFIXED ✓ | AddGround ✓ | addFix ⚠ |
| RADIUS | Radius ✓ | Radial dim | AddRadial ⚠ | addRadialDimension ✓ |
| DISTANCE | Distance ✓ | Linear dim | AddTwoPointDistance ⚠ | addDistanceDimension ✓ |

Legend: ✓ = Verified, ⚠ = Unverified/Not Implemented

### 11.4 Special Handling Requirements

| Platform | Special Cases |
|----------|---------------|
| FreeCAD | Tangent does NOT imply coincident; arc angles can be ambiguous; use vertex indices for point refs |
| SolidWorks | Use `sgMERGEPOINTS` for path connections; retry API calls; batch with `AddToDB=true`; check transform determinant for arc direction |
| Inventor | Use `DeferUpdates` for batching; wrap in transactions; coincident may throw if already satisfied; 3D uses separate constraint API |
| Fusion 360 | Check `healthState` after operations; sketches require plane; use SketchPoint for chaining; three-point arc has mixed signature |

---

## 12. Validation Layer

### 12.1 Schema Validation

```python
def validate_sketch(sketch: SketchDocument) -> list[str]:
    """Validate sketch for schema correctness. Returns list of errors."""
    errors = []
    
    for prim in sketch.primitives.values():
        errors.extend(validate_primitive(prim))
    
    for constraint in sketch.constraints:
        errors.extend(validate_constraint(constraint, sketch))
    
    return errors

def validate_primitive(prim: SketchPrimitive) -> list[str]:
    errors = []
    
    if isinstance(prim, Arc):
        # Radius consistency check
        r_start = prim.center.distance_to(prim.start_point)
        r_end = prim.center.distance_to(prim.end_point)
        if abs(r_start - r_end) > 0.001:  # 1 micron tolerance
            errors.append(f"{prim.id}: Arc radius inconsistent (start={r_start:.4f}, end={r_end:.4f})")
        
        # Degenerate arc check
        if prim.start_point.distance_to(prim.end_point) < 0.001:
            errors.append(f"{prim.id}: Arc start and end points are coincident (use Circle instead)")
    
    elif isinstance(prim, Line):
        if prim.length < 0.001:
            errors.append(f"{prim.id}: Line has zero length")
    
    elif isinstance(prim, Circle):
        if prim.radius <= 0:
            errors.append(f"{prim.id}: Circle has non-positive radius")
    
    return errors

def validate_constraint(c: SketchConstraint, sketch: SketchDocument) -> list[str]:
    errors = []
    rules = CONSTRAINT_RULES.get(c.constraint_type)
    
    if not rules:
        errors.append(f"Unknown constraint type: {c.constraint_type}")
        return errors
    
    # Check reference count
    if len(c.references) < rules["min_refs"]:
        errors.append(f"{c}: Too few references (min {rules['min_refs']})")
    if rules["max_refs"] and len(c.references) > rules["max_refs"]:
        errors.append(f"{c}: Too many references (max {rules['max_refs']})")
    
    # Check value requirement
    if rules["value_required"] and c.value is None:
        errors.append(f"{c}: Missing required value")
    
    # Check reference types
    for ref in c.references:
        if isinstance(ref, PointRef):
            prim = sketch.get_primitive(ref.element_id)
            if prim and ref.point_type not in prim.get_valid_point_types():
                errors.append(f"{c}: Invalid point type {ref.point_type} for {ref.element_id}")
    
    return errors
```

---

## 13. Serialization

### 13.1 JSON Schema

```python
import json
from dataclasses import asdict

def sketch_to_json(sketch: SketchDocument) -> str:
    """Serialize sketch to JSON."""
    data = {
        "name": sketch.name,
        "primitives": [primitive_to_dict(p) for p in sketch.primitives.values()],
        "constraints": [constraint_to_dict(c) for c in sketch.constraints],
        "solver_status": sketch.solver_status.value,
        "degrees_of_freedom": sketch.degrees_of_freedom,
    }
    return json.dumps(data, indent=2)

def primitive_to_dict(p: SketchPrimitive) -> dict:
    base = {
        "id": p.id,
        "type": type(p).__name__.lower(),
        "construction": p.construction,
    }
    
    if isinstance(p, Line):
        base.update({
            "start": [p.start.x, p.start.y],
            "end": [p.end.x, p.end.y],
        })
    elif isinstance(p, Arc):
        base.update({
            "center": [p.center.x, p.center.y],
            "start_point": [p.start_point.x, p.start_point.y],
            "end_point": [p.end_point.x, p.end_point.y],
            "ccw": p.ccw,
        })
    elif isinstance(p, Circle):
        base.update({
            "center": [p.center.x, p.center.y],
            "radius": p.radius,
        })
    elif isinstance(p, Point):
        base.update({
            "position": [p.position.x, p.position.y],
        })
    elif isinstance(p, Spline):
        base.update({
            "degree": p.degree,
            "control_points": [[pt.x, pt.y] for pt in p.control_points],
            "knots": p.knots,
            "weights": p.weights,
            "periodic": p.periodic,
            "is_fit_spline": p.is_fit_spline,
        })
    
    return base

def constraint_to_dict(c: SketchConstraint) -> dict:
    refs = []
    for r in c.references:
        if isinstance(r, PointRef):
            refs.append({"element": r.element_id, "point": r.point_type.value})
        else:
            refs.append(r)
    
    return {
        "id": c.id,
        "type": c.constraint_type.value,
        "references": refs,
        "value": c.value,
        "inferred": c.inferred,
        "confidence": c.confidence,
    }
```

### 13.2 Example JSON Output

```json
{
  "name": "RoundedRect",
  "primitives": [
    {"id": "L0", "type": "line", "construction": false,
     "start": [8.0, 0.0], "end": [52.0, 0.0]},
    {"id": "A1", "type": "arc", "construction": false,
     "center": [52.0, 8.0], "start_point": [52.0, 0.0], "end_point": [60.0, 8.0], "ccw": true},
    {"id": "L2", "type": "line", "construction": false,
     "start": [60.0, 8.0], "end": [60.0, 32.0]},
    {"id": "C8", "type": "circle", "construction": false,
     "center": [30.0, 20.0], "radius": 10.0}
  ],
  "constraints": [
    {"id": "c1", "type": "tangent", "references": ["L0", "A1"], "value": null, "inferred": false, "confidence": 1.0},
    {"id": "c2", "type": "tangent", "references": ["A1", "L2"], "value": null, "inferred": false, "confidence": 1.0},
    {"id": "c3", "type": "horizontal", "references": ["L0"], "value": null, "inferred": false, "confidence": 1.0},
    {"id": "c4", "type": "equal", "references": ["A1", "A3", "A5", "A7"], "value": null, "inferred": true, "confidence": 0.95},
    {"id": "c5", "type": "radius", "references": ["A1"], "value": 8.0, "inferred": false, "confidence": 1.0},
    {"id": "c6", "type": "coincident", "references": [
      {"element": "L0", "point": "end"},
      {"element": "A1", "point": "start"}
    ], "value": null, "inferred": false, "confidence": 1.0}
  ],
  "solver_status": "fully_constrained",
  "degrees_of_freedom": 0
}
```

---

## 14. Future Extensions

### 14.1 Additional Primitives
- **Ellipse**: center, major_axis, minor_axis, rotation
- **Conic sections**: Generic conic representation
- **Text**: For annotations (typically not constrained)

### 14.2 Additional Constraints
- **SYMMETRIC**: Two elements symmetric about a line
- **MIDPOINT_CONSTRAINT**: Point constrained to midpoint of line
- **COLINEAR**: Multiple lines on same infinite line
- **PATTERN**: Linear or circular patterns of elements

### 14.3 3D Sketch Support
- Extend `Point2D` → `Point3D`
- Add work plane references
- Handle 3D-specific constraints
- Inventor: Use `GeometricConstraints3D` and `SketchEntity3D`

### 14.4 Assembly Constraints
- Mate constraints between sketches/features
- External references

### 14.5 Cloud-Based Systems

**Onshape:**
- REST API for sketch operations
- Similar primitive types and constraints
- Units: meters (like SolidWorks)
- Adapter would be HTTP-based

**Autodesk Platform Services (formerly Forge):**
- Cloud-backed Fusion 360 documents
- REST APIs available but sketch-level manipulation limited
- Rate limiting considerations for interactive use
