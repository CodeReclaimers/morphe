"""
Canonical Sketch Geometry and Constraint Schema

A CAD-agnostic representation for 2D sketch geometry and constraints,
with support for platform-specific adaptation (FreeCAD, SolidWorks,
Inventor, Fusion 360).

Example usage:
    from sketch_canonical import (
        SketchDocument, Point2D, Line, Arc, Circle, Spline,
        Coincident, Tangent, Horizontal, Radius,
        PointRef, PointType,
        validate_sketch, sketch_to_json, sketch_from_json
    )

    # Create a sketch
    sketch = SketchDocument(name="MySketch")

    # Add geometry
    line_id = sketch.add_primitive(Line(
        start=Point2D(0, 0),
        end=Point2D(100, 0)
    ))

    arc_id = sketch.add_primitive(Arc(
        center=Point2D(100, 10),
        start_point=Point2D(100, 0),
        end_point=Point2D(110, 10),
        ccw=True
    ))

    # Add constraints
    sketch.add_constraint(Coincident(
        PointRef(line_id, PointType.END),
        PointRef(arc_id, PointType.START)
    ))
    sketch.add_constraint(Horizontal(line_id))

    # Validate
    result = validate_sketch(sketch)
    if result.is_valid:
        print("Sketch is valid!")

    # Serialize
    json_str = sketch_to_json(sketch)
"""

__version__ = "0.1.0"

# Core types
from .types import (
    Point2D,
    Vector2D,
    ElementId,
    ElementPrefix,
    PointType,
    PointRef,
)

# Geometry primitives
from .primitives import (
    SketchPrimitive,
    Line,
    Arc,
    Circle,
    Point,
    Spline,
)

# Constraints
from .constraints import (
    ConstraintType,
    ConstraintStatus,
    SketchConstraint,
    CONSTRAINT_RULES,
    # Convenience constructors
    Coincident,
    Tangent,
    Perpendicular,
    Parallel,
    Concentric,
    Equal,
    Collinear,
    Horizontal,
    Vertical,
    Fixed,
    Distance,
    DistanceX,
    DistanceY,
    Length,
    Radius,
    Diameter,
    Angle,
    Symmetric,
    MidpointConstraint,
)

# Document
from .document import (
    SolverStatus,
    SketchDocument,
)

# Validation
from .validation import (
    ValidationSeverity,
    ValidationIssue,
    ValidationResult,
    validate_sketch,
    validate_primitive,
    validate_constraint,
    DEFAULT_TOLERANCE,
)

# Serialization
from .serialization import (
    SketchEncoder,
    sketch_to_json,
    sketch_from_json,
    sketch_to_dict,
    dict_to_sketch,
    primitive_to_dict,
    dict_to_primitive,
    constraint_to_dict,
    dict_to_constraint,
    point_ref_to_dict,
    dict_to_point_ref,
    save_sketch,
    load_sketch,
)

# Adapter interface
from .adapter import (
    SketchBackendAdapter,
    AdapterError,
    SketchCreationError,
    GeometryError,
    ConstraintError,
    ExportError,
)

__all__ = [
    # Version
    "__version__",

    # Core types
    "Point2D",
    "Vector2D",
    "ElementId",
    "ElementPrefix",
    "PointType",
    "PointRef",

    # Primitives
    "SketchPrimitive",
    "Line",
    "Arc",
    "Circle",
    "Point",
    "Spline",

    # Constraints
    "ConstraintType",
    "ConstraintStatus",
    "SketchConstraint",
    "CONSTRAINT_RULES",
    "Coincident",
    "Tangent",
    "Perpendicular",
    "Parallel",
    "Concentric",
    "Equal",
    "Collinear",
    "Horizontal",
    "Vertical",
    "Fixed",
    "Distance",
    "DistanceX",
    "DistanceY",
    "Length",
    "Radius",
    "Diameter",
    "Angle",
    "Symmetric",
    "MidpointConstraint",

    # Document
    "SolverStatus",
    "SketchDocument",

    # Validation
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationResult",
    "validate_sketch",
    "validate_primitive",
    "validate_constraint",
    "DEFAULT_TOLERANCE",

    # Serialization
    "SketchEncoder",
    "sketch_to_json",
    "sketch_from_json",
    "sketch_to_dict",
    "dict_to_sketch",
    "primitive_to_dict",
    "dict_to_primitive",
    "constraint_to_dict",
    "dict_to_constraint",
    "point_ref_to_dict",
    "dict_to_point_ref",
    "save_sketch",
    "load_sketch",

    # Adapter interface
    "SketchBackendAdapter",
    "AdapterError",
    "SketchCreationError",
    "GeometryError",
    "ConstraintError",
    "ExportError",
]
