"""JSON serialization and deserialization for the canonical sketch schema."""

import json
from typing import Any

from .constraints import ConstraintStatus, ConstraintType, SketchConstraint
from .document import SketchDocument, SolverStatus
from .primitives import Arc, Circle, Ellipse, EllipticalArc, Line, Point, SketchPrimitive, Spline
from .types import Point2D, PointRef, PointType


class SketchEncoder(json.JSONEncoder):
    """Custom JSON encoder for sketch types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, SketchDocument):
            return sketch_to_dict(obj)
        elif isinstance(obj, SketchPrimitive):
            return primitive_to_dict(obj)
        elif isinstance(obj, SketchConstraint):
            return constraint_to_dict(obj)
        elif isinstance(obj, Point2D):
            return [obj.x, obj.y]
        elif isinstance(obj, PointRef):
            return point_ref_to_dict(obj)
        elif isinstance(obj, ConstraintType | ConstraintStatus | SolverStatus | PointType):
            return obj.value
        return super().default(obj)


def sketch_to_json(sketch: SketchDocument, indent: int | None = 2) -> str:
    """
    Serialize a sketch document to JSON string.

    Args:
        sketch: The sketch document to serialize
        indent: JSON indentation level (None for compact)

    Returns:
        JSON string representation
    """
    return json.dumps(sketch_to_dict(sketch), indent=indent)


def sketch_from_json(json_str: str) -> SketchDocument:
    """
    Deserialize a sketch document from JSON string.

    Args:
        json_str: JSON string to parse

    Returns:
        Reconstructed SketchDocument
    """
    data = json.loads(json_str)
    return dict_to_sketch(data)


def sketch_to_dict(sketch: SketchDocument) -> dict:
    """Convert a sketch document to a dictionary."""
    return {
        "name": sketch.name,
        "primitives": [primitive_to_dict(p) for p in sketch.primitives.values()],
        "constraints": [constraint_to_dict(c) for c in sketch.constraints],
        "solver_status": sketch.solver_status.value,
        "degrees_of_freedom": sketch.degrees_of_freedom,
    }


def dict_to_sketch(data: dict) -> SketchDocument:
    """Convert a dictionary to a sketch document."""
    sketch = SketchDocument(name=data.get("name", "Untitled"))

    # Load primitives
    for prim_data in data.get("primitives", []):
        prim = dict_to_primitive(prim_data)
        sketch.add_primitive_with_id(prim, prim.id)

    # Load constraints
    for const_data in data.get("constraints", []):
        const = dict_to_constraint(const_data)
        sketch.constraints.append(const)

    # Load solver state
    status_str = data.get("solver_status", "dirty")
    sketch.solver_status = SolverStatus(status_str)
    sketch.degrees_of_freedom = data.get("degrees_of_freedom", -1)

    return sketch


def primitive_to_dict(p: SketchPrimitive) -> dict:
    """Convert a primitive to a dictionary."""
    base = {
        "id": p.id,
        "type": type(p).__name__.lower(),
        "construction": p.construction,
    }

    # Add optional metadata
    if p.source is not None:
        base["source"] = p.source
    if p.confidence != 1.0:
        base["confidence"] = p.confidence

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
            "periodic": p.periodic,
            "is_fit_spline": p.is_fit_spline,
        })
        if p.weights is not None:
            base["weights"] = p.weights
    elif isinstance(p, Ellipse):
        base.update({
            "center": [p.center.x, p.center.y],
            "major_radius": p.major_radius,
            "minor_radius": p.minor_radius,
            "rotation": p.rotation,
        })
    elif isinstance(p, EllipticalArc):
        base.update({
            "center": [p.center.x, p.center.y],
            "major_radius": p.major_radius,
            "minor_radius": p.minor_radius,
            "rotation": p.rotation,
            "start_param": p.start_param,
            "end_param": p.end_param,
            "ccw": p.ccw,
        })

    return base


def dict_to_primitive(data: dict) -> SketchPrimitive:
    """Convert a dictionary to a primitive."""
    prim_type = data.get("type", "").lower()

    # Common fields
    id_val = data.get("id", "")
    construction = data.get("construction", False)
    source = data.get("source")
    confidence = data.get("confidence", 1.0)

    prim: SketchPrimitive
    if prim_type == "line":
        start = _parse_point(data.get("start", [0, 0]))
        end = _parse_point(data.get("end", [0, 0]))
        prim = Line(
            id=id_val,
            construction=construction,
            source=source,
            confidence=confidence,
            start=start,
            end=end
        )
    elif prim_type == "arc":
        center = _parse_point(data.get("center", [0, 0]))
        start_point = _parse_point(data.get("start_point", [0, 0]))
        end_point = _parse_point(data.get("end_point", [0, 0]))
        ccw = data.get("ccw", True)
        prim = Arc(
            id=id_val,
            construction=construction,
            source=source,
            confidence=confidence,
            center=center,
            start_point=start_point,
            end_point=end_point,
            ccw=ccw
        )
    elif prim_type == "circle":
        center = _parse_point(data.get("center", [0, 0]))
        radius = data.get("radius", 1.0)
        prim = Circle(
            id=id_val,
            construction=construction,
            source=source,
            confidence=confidence,
            center=center,
            radius=radius
        )
    elif prim_type == "point":
        position = _parse_point(data.get("position", [0, 0]))
        prim = Point(
            id=id_val,
            construction=construction,
            source=source,
            confidence=confidence,
            position=position
        )
    elif prim_type == "spline":
        control_points = [_parse_point(pt) for pt in data.get("control_points", [])]
        prim = Spline(
            id=id_val,
            construction=construction,
            source=source,
            confidence=confidence,
            degree=data.get("degree", 3),
            control_points=control_points,
            knots=data.get("knots", []),
            weights=data.get("weights"),
            periodic=data.get("periodic", False),
            is_fit_spline=data.get("is_fit_spline", False)
        )
    elif prim_type == "ellipse":
        center = _parse_point(data.get("center", [0, 0]))
        prim = Ellipse(
            id=id_val,
            construction=construction,
            source=source,
            confidence=confidence,
            center=center,
            major_radius=data.get("major_radius", 1.0),
            minor_radius=data.get("minor_radius", 0.5),
            rotation=data.get("rotation", 0.0),
        )
    elif prim_type == "ellipticalarc":
        center = _parse_point(data.get("center", [0, 0]))
        prim = EllipticalArc(
            id=id_val,
            construction=construction,
            source=source,
            confidence=confidence,
            center=center,
            major_radius=data.get("major_radius", 1.0),
            minor_radius=data.get("minor_radius", 0.5),
            rotation=data.get("rotation", 0.0),
            start_param=data.get("start_param", 0.0),
            end_param=data.get("end_param", 1.5707963267948966),  # pi/2
            ccw=data.get("ccw", True),
        )
    else:
        raise ValueError(f"Unknown primitive type: {prim_type}")

    return prim


def constraint_to_dict(c: SketchConstraint) -> dict[str, Any]:
    """Convert a constraint to a dictionary."""
    refs: list[dict[str, Any] | str] = []
    for r in c.references:
        if isinstance(r, PointRef):
            refs.append(point_ref_to_dict(r))
        else:
            refs.append(r)

    result: dict[str, Any] = {
        "id": c.id,
        "type": c.constraint_type.value,
        "references": refs,
    }

    # Add optional fields only if they have non-default values
    if c.value is not None:
        result["value"] = c.value

    if c.connection_point is not None:
        result["connection_point"] = point_ref_to_dict(c.connection_point)

    if c.inferred:
        result["inferred"] = c.inferred

    if c.confidence != 1.0:
        result["confidence"] = c.confidence

    if c.source is not None:
        result["source"] = c.source

    if c.status != ConstraintStatus.UNKNOWN:
        result["status"] = c.status.value

    return result


def dict_to_constraint(data: dict[str, Any]) -> SketchConstraint:
    """Convert a dictionary to a constraint."""
    refs: list[str | PointRef] = []
    for r in data.get("references", []):
        if isinstance(r, dict):
            refs.append(dict_to_point_ref(r))
        else:
            refs.append(r)

    connection_point = None
    if "connection_point" in data:
        connection_point = dict_to_point_ref(data["connection_point"])

    status = ConstraintStatus.UNKNOWN
    if "status" in data:
        status = ConstraintStatus(data["status"])

    return SketchConstraint(
        id=data.get("id", ""),
        constraint_type=ConstraintType(data.get("type", "coincident")),
        references=refs,
        value=data.get("value"),
        connection_point=connection_point,
        inferred=data.get("inferred", False),
        confidence=data.get("confidence", 1.0),
        source=data.get("source"),
        status=status
    )


def point_ref_to_dict(ref: PointRef) -> dict[str, Any]:
    """Convert a PointRef to a dictionary."""
    result: dict[str, Any] = {
        "element": ref.element_id,
        "point": ref.point_type.value,
    }
    if ref.parameter is not None:
        result["parameter"] = ref.parameter
    if ref.index is not None:
        result["index"] = ref.index
    return result


def dict_to_point_ref(data: dict) -> PointRef:
    """Convert a dictionary to a PointRef."""
    return PointRef(
        element_id=data.get("element", ""),
        point_type=PointType(data.get("point", "center")),
        parameter=data.get("parameter"),
        index=data.get("index")
    )


def _parse_point(data: list) -> Point2D:
    """Parse a point from [x, y] array format."""
    if len(data) < 2:
        return Point2D(0, 0)
    return Point2D(float(data[0]), float(data[1]))


# File I/O utilities

def save_sketch(sketch: SketchDocument, filepath: str, indent: int | None = 2) -> None:
    """
    Save a sketch document to a JSON file.

    Args:
        sketch: The sketch document to save
        filepath: Path to the output file
        indent: JSON indentation level (None for compact)
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(sketch_to_json(sketch, indent))


def load_sketch(filepath: str) -> SketchDocument:
    """
    Load a sketch document from a JSON file.

    Args:
        filepath: Path to the input file

    Returns:
        Loaded SketchDocument
    """
    with open(filepath, encoding='utf-8') as f:
        return sketch_from_json(f.read())
