"""
SolidWorks sketch point mapping utilities.

SolidWorks uses ISketchPoint objects for vertices. This module provides
utilities for mapping between canonical PointType and SolidWorks sketch
point properties.

SolidWorks sketch entities have these point access patterns:
- SketchLine: GetStartPoint2(), GetEndPoint2()
- SketchArc: GetStartPoint2(), GetEndPoint2(), GetCenterPoint2()
- SketchCircle: GetCenterPoint2()
- SketchPoint: (the point itself)
- SketchSpline: GetPoints2() returns array of fit points
"""

from typing import Any

from morphe import PointType


def get_sketch_point_from_entity(entity: Any, point_type: PointType) -> Any:
    """
    Get the SolidWorks SketchPoint from an entity based on point type.

    Args:
        entity: SolidWorks sketch entity (SketchLine, SketchArc, etc.)
        point_type: Canonical point type

    Returns:
        SolidWorks SketchPoint object

    Raises:
        ValueError: If point type is not valid for the entity type
    """
    entity_type = _get_entity_type(entity)

    # Try direct COM methods first
    try:
        if entity_type == "SketchLine":
            if point_type == PointType.START:
                return entity.GetStartPoint2()
            elif point_type == PointType.END:
                return entity.GetEndPoint2()
            else:
                raise ValueError(f"Invalid point type {point_type} for SketchLine")

        elif entity_type == "SketchArc":
            if point_type == PointType.START:
                return entity.GetStartPoint2()
            elif point_type == PointType.END:
                return entity.GetEndPoint2()
            elif point_type == PointType.CENTER:
                return entity.GetCenterPoint2()
            else:
                raise ValueError(f"Invalid point type {point_type} for SketchArc")

        elif entity_type == "SketchCircle":
            if point_type == PointType.CENTER:
                return entity.GetCenterPoint2()
            else:
                raise ValueError(f"Invalid point type {point_type} for SketchCircle")

        elif entity_type == "SketchPoint":
            if point_type == PointType.CENTER:
                return entity
            else:
                raise ValueError(f"Invalid point type {point_type} for SketchPoint")

        elif entity_type == "SketchSpline":
            if point_type == PointType.START:
                points = entity.GetPoints2()
                if points and len(points) >= 3:
                    return _create_point_from_coords(entity, points[0], points[1], points[2])
                return None
            elif point_type == PointType.END:
                points = entity.GetPoints2()
                if points and len(points) >= 3:
                    return _create_point_from_coords(entity, points[-3], points[-2], points[-1])
                return None
            else:
                raise ValueError(f"Invalid point type {point_type} for SketchSpline")

        elif entity_type == "SketchEllipse":
            if point_type == PointType.CENTER:
                return entity.GetCenterPoint2()
            else:
                raise ValueError(f"Invalid point type {point_type} for SketchEllipse")

        elif entity_type == "SketchEllipticalArc":
            if point_type == PointType.START:
                return entity.GetStartPoint2()
            elif point_type == PointType.END:
                return entity.GetEndPoint2()
            elif point_type == PointType.CENTER:
                return entity.GetCenterPoint2()
            else:
                raise ValueError(f"Invalid point type {point_type} for SketchEllipticalArc")

        else:
            raise ValueError(f"Unknown entity type: {entity_type}")

    except Exception:
        # COM methods may not be available in late binding
        # or may fail with com_error
        # Return None and let caller handle it
        return None


def get_point_type_for_sketch_point(entity: Any, sketch_point: Any) -> PointType | None:
    """
    Determine the canonical PointType for a SketchPoint on an entity.

    Args:
        entity: SolidWorks sketch entity that may contain the point
        sketch_point: SolidWorks SketchPoint to find

    Returns:
        PointType if the point belongs to this entity, None otherwise
    """
    entity_type = _get_entity_type(entity)

    try:
        if entity_type == "SketchLine":
            if _same_point(entity.GetStartPoint2(), sketch_point):
                return PointType.START
            elif _same_point(entity.GetEndPoint2(), sketch_point):
                return PointType.END

        elif entity_type == "SketchArc":
            if _same_point(entity.GetStartPoint2(), sketch_point):
                return PointType.START
            elif _same_point(entity.GetEndPoint2(), sketch_point):
                return PointType.END
            elif _same_point(entity.GetCenterPoint2(), sketch_point):
                return PointType.CENTER

        elif entity_type == "SketchCircle":
            if _same_point(entity.GetCenterPoint2(), sketch_point):
                return PointType.CENTER

        elif entity_type == "SketchPoint":
            if _same_point(entity, sketch_point):
                return PointType.CENTER

        elif entity_type == "SketchEllipse":
            if _same_point(entity.GetCenterPoint2(), sketch_point):
                return PointType.CENTER

        elif entity_type == "SketchEllipticalArc":
            if _same_point(entity.GetStartPoint2(), sketch_point):
                return PointType.START
            elif _same_point(entity.GetEndPoint2(), sketch_point):
                return PointType.END
            elif _same_point(entity.GetCenterPoint2(), sketch_point):
                return PointType.CENTER

    except Exception:
        pass

    return None


def _get_entity_type(entity: Any) -> str:
    """Get the type name of a SolidWorks sketch entity."""
    try:
        # Try to get type from COM object
        type_name = type(entity).__name__
        if "SketchLine" in type_name:
            return "SketchLine"
        elif "SketchArc" in type_name:
            return "SketchArc"
        elif "SketchCircle" in type_name:
            return "SketchCircle"
        elif "SketchPoint" in type_name:
            return "SketchPoint"
        elif "SketchSpline" in type_name:
            return "SketchSpline"
        elif "SketchEllipticalArc" in type_name:
            return "SketchEllipticalArc"
        elif "SketchEllipse" in type_name:
            return "SketchEllipse"

        # Try checking via interface
        if hasattr(entity, "GetStartPoint2") and hasattr(entity, "GetEndPoint2"):
            if hasattr(entity, "GetCenterPoint2"):
                # Could be arc or ellipse
                if hasattr(entity, "GetRadius"):
                    return "SketchArc"
                return "SketchEllipse"
            return "SketchLine"
        elif hasattr(entity, "GetCenterPoint2"):
            if hasattr(entity, "GetRadius"):
                return "SketchCircle"
        elif hasattr(entity, "GetPoints2"):
            return "SketchSpline"

        return type_name
    except Exception:
        return "Unknown"


def _same_point(pt1: Any, pt2: Any) -> bool:
    """Check if two sketch points are the same (by geometry comparison)."""
    try:
        if pt1 is pt2:
            return True

        # Get coordinates - SolidWorks points have X, Y, Z properties
        tolerance = 1e-9  # SolidWorks uses meters, so tighter tolerance

        x1 = pt1.X if hasattr(pt1, "X") else pt1[0]
        y1 = pt1.Y if hasattr(pt1, "Y") else pt1[1]
        x2 = pt2.X if hasattr(pt2, "X") else pt2[0]
        y2 = pt2.Y if hasattr(pt2, "Y") else pt2[1]

        return bool(abs(x1 - x2) < tolerance and abs(y1 - y2) < tolerance)
    except Exception:
        return False


def _create_point_from_coords(entity: Any, x: float, y: float, z: float) -> Any:
    """Create a point-like object from coordinates (for spline endpoints)."""
    # Return a simple object with X, Y, Z properties
    class PointCoords:
        def __init__(self, x: float, y: float, z: float):
            self.X = x
            self.Y = y
            self.Z = z

    return PointCoords(x, y, z)


def get_valid_point_types(entity: Any) -> list[PointType]:
    """
    Get the valid point types for a SolidWorks sketch entity.

    Args:
        entity: SolidWorks sketch entity

    Returns:
        List of valid PointType values for this entity type
    """
    entity_type = _get_entity_type(entity)

    if entity_type == "SketchLine":
        return [PointType.START, PointType.END]
    elif entity_type == "SketchEllipticalArc":
        return [PointType.START, PointType.END, PointType.CENTER]
    elif entity_type == "SketchArc":
        return [PointType.START, PointType.END, PointType.CENTER]
    elif entity_type == "SketchEllipse":
        return [PointType.CENTER]
    elif entity_type == "SketchCircle":
        return [PointType.CENTER]
    elif entity_type == "SketchPoint":
        return [PointType.CENTER]
    elif entity_type == "SketchSpline":
        return [PointType.START, PointType.END]
    else:
        return []
