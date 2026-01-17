"""
Autodesk Inventor sketch point mapping utilities.

Inventor uses SketchPoint objects for vertices rather than numeric indices.
This module provides utilities for mapping between canonical PointType
and Inventor's sketch point properties.

Inventor sketch entities have these point properties:
- SketchLine: StartSketchPoint, EndSketchPoint
- SketchArc: StartSketchPoint, EndSketchPoint, CenterSketchPoint
- SketchCircle: CenterSketchPoint
- SketchPoint: (the point itself)
- SketchSpline: StartPoint, EndPoint, FitPoints collection
"""

from typing import Any

from morphe import PointType


def get_sketch_point_from_entity(entity: Any, point_type: PointType) -> Any:
    """
    Get the Inventor SketchPoint from an entity based on point type.

    Args:
        entity: Inventor sketch entity (SketchLine, SketchArc, etc.)
        point_type: Canonical point type

    Returns:
        Inventor SketchPoint object

    Raises:
        ValueError: If point type is not valid for the entity type
    """
    entity_type = type(entity).__name__

    if 'SketchLine' in entity_type:
        if point_type == PointType.START:
            return entity.StartSketchPoint
        elif point_type == PointType.END:
            return entity.EndSketchPoint
        else:
            raise ValueError(f"Invalid point type {point_type} for SketchLine")

    elif 'SketchArc' in entity_type:
        if point_type == PointType.START:
            return entity.StartSketchPoint
        elif point_type == PointType.END:
            return entity.EndSketchPoint
        elif point_type == PointType.CENTER:
            return entity.CenterSketchPoint
        else:
            raise ValueError(f"Invalid point type {point_type} for SketchArc")

    elif 'SketchCircle' in entity_type:
        if point_type == PointType.CENTER:
            return entity.CenterSketchPoint
        else:
            raise ValueError(f"Invalid point type {point_type} for SketchCircle")

    elif 'SketchPoint' in entity_type:
        if point_type == PointType.CENTER:
            return entity
        else:
            raise ValueError(f"Invalid point type {point_type} for SketchPoint")

    elif 'SketchSpline' in entity_type:
        if point_type == PointType.START:
            return entity.StartPoint
        elif point_type == PointType.END:
            return entity.EndPoint
        else:
            raise ValueError(f"Invalid point type {point_type} for SketchSpline")

    elif 'SketchEllipse' in entity_type:
        if point_type == PointType.CENTER:
            return entity.CenterSketchPoint
        else:
            raise ValueError(f"Invalid point type {point_type} for SketchEllipse")

    elif 'SketchEllipticalArc' in entity_type:
        if point_type == PointType.START:
            return entity.StartSketchPoint
        elif point_type == PointType.END:
            return entity.EndSketchPoint
        elif point_type == PointType.CENTER:
            return entity.CenterSketchPoint
        else:
            raise ValueError(f"Invalid point type {point_type} for SketchEllipticalArc")

    else:
        raise ValueError(f"Unknown entity type: {entity_type}")


def get_point_type_for_sketch_point(entity: Any, sketch_point: Any) -> PointType | None:
    """
    Determine the canonical PointType for a SketchPoint on an entity.

    Args:
        entity: Inventor sketch entity that may contain the point
        sketch_point: Inventor SketchPoint to find

    Returns:
        PointType if the point belongs to this entity, None otherwise
    """
    entity_type = type(entity).__name__

    try:
        if 'SketchLine' in entity_type:
            if _same_point(entity.StartSketchPoint, sketch_point):
                return PointType.START
            elif _same_point(entity.EndSketchPoint, sketch_point):
                return PointType.END

        elif 'SketchArc' in entity_type:
            if _same_point(entity.StartSketchPoint, sketch_point):
                return PointType.START
            elif _same_point(entity.EndSketchPoint, sketch_point):
                return PointType.END
            elif _same_point(entity.CenterSketchPoint, sketch_point):
                return PointType.CENTER

        elif 'SketchCircle' in entity_type:
            if _same_point(entity.CenterSketchPoint, sketch_point):
                return PointType.CENTER

        elif 'SketchPoint' in entity_type:
            if _same_point(entity, sketch_point):
                return PointType.CENTER

        elif 'SketchSpline' in entity_type:
            if _same_point(entity.StartPoint, sketch_point):
                return PointType.START
            elif _same_point(entity.EndPoint, sketch_point):
                return PointType.END

        elif 'SketchEllipse' in entity_type:
            if _same_point(entity.CenterSketchPoint, sketch_point):
                return PointType.CENTER

        elif 'SketchEllipticalArc' in entity_type:
            if _same_point(entity.StartSketchPoint, sketch_point):
                return PointType.START
            elif _same_point(entity.EndSketchPoint, sketch_point):
                return PointType.END
            elif _same_point(entity.CenterSketchPoint, sketch_point):
                return PointType.CENTER

    except Exception:
        pass

    return None


def _same_point(pt1: Any, pt2: Any) -> bool:
    """Check if two sketch points are the same (by geometry comparison)."""
    try:
        # Try direct comparison first
        if pt1 is pt2:
            return True

        # Compare by geometry
        g1 = pt1.Geometry
        g2 = pt2.Geometry
        tolerance = 1e-8
        return bool(
            abs(g1.X - g2.X) < tolerance and
            abs(g1.Y - g2.Y) < tolerance
        )
    except Exception:
        return False


def get_valid_point_types(entity: Any) -> list[PointType]:
    """
    Get the valid point types for an Inventor sketch entity.

    Args:
        entity: Inventor sketch entity

    Returns:
        List of valid PointType values for this entity type
    """
    entity_type = type(entity).__name__

    if 'SketchLine' in entity_type:
        return [PointType.START, PointType.END]
    elif 'SketchEllipticalArc' in entity_type:
        return [PointType.START, PointType.END, PointType.CENTER]
    elif 'SketchArc' in entity_type:
        return [PointType.START, PointType.END, PointType.CENTER]
    elif 'SketchEllipse' in entity_type:
        return [PointType.CENTER]
    elif 'SketchCircle' in entity_type:
        return [PointType.CENTER]
    elif 'SketchPoint' in entity_type:
        return [PointType.CENTER]
    elif 'SketchSpline' in entity_type:
        return [PointType.START, PointType.END]
    else:
        return []
