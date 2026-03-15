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

Note: COM late-binding returns 'CDispatch' for all types, so we use
duck-typing (checking available properties) to identify entity kinds.
"""

from typing import Any

from morphe import PointType


def _entity_kind(entity: Any) -> str:
    """Determine the sketch entity kind via duck-typing.

    Returns one of: 'line', 'arc', 'circle', 'point', 'spline',
    'ellipse', 'elliptical_arc', or 'unknown'.
    """
    # Check for SketchLine (has Start/End but no Center, no Radius)
    has_start = hasattr(entity, 'StartSketchPoint')
    has_end = hasattr(entity, 'EndSketchPoint')
    has_center = hasattr(entity, 'CenterSketchPoint')
    has_radius = hasattr(entity, 'Radius')
    has_major = hasattr(entity, 'MajorRadius')
    has_fit_points = hasattr(entity, 'FitPoints')
    has_geometry_pt = hasattr(entity, 'Geometry')

    if has_fit_points:
        return 'spline'
    if has_major and has_start and has_end:
        return 'elliptical_arc'
    if has_major and has_center:
        return 'ellipse'
    if has_center and has_start and has_end:
        return 'arc'
    if has_center and has_radius and not has_start:
        return 'circle'
    if has_start and has_end and not has_center:
        return 'line'
    if has_geometry_pt and not has_start and not has_end and not has_center:
        return 'point'

    return 'unknown'


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
    kind = _entity_kind(entity)

    if kind == 'line':
        if point_type == PointType.START:
            return entity.StartSketchPoint
        elif point_type == PointType.END:
            return entity.EndSketchPoint
        else:
            raise ValueError(f"Invalid point type {point_type} for SketchLine")

    elif kind == 'arc':
        if point_type == PointType.START:
            return entity.StartSketchPoint
        elif point_type == PointType.END:
            return entity.EndSketchPoint
        elif point_type == PointType.CENTER:
            return entity.CenterSketchPoint
        else:
            raise ValueError(f"Invalid point type {point_type} for SketchArc")

    elif kind == 'circle':
        if point_type == PointType.CENTER:
            return entity.CenterSketchPoint
        else:
            raise ValueError(f"Invalid point type {point_type} for SketchCircle")

    elif kind == 'point':
        if point_type == PointType.CENTER:
            return entity
        else:
            raise ValueError(f"Invalid point type {point_type} for SketchPoint")

    elif kind == 'spline':
        if point_type == PointType.START:
            return entity.StartSketchPoint
        elif point_type == PointType.END:
            return entity.EndSketchPoint
        else:
            raise ValueError(f"Invalid point type {point_type} for SketchSpline")

    elif kind == 'ellipse':
        if point_type == PointType.CENTER:
            return entity.CenterSketchPoint
        else:
            raise ValueError(f"Invalid point type {point_type} for SketchEllipse")

    elif kind == 'elliptical_arc':
        if point_type == PointType.START:
            return entity.StartSketchPoint
        elif point_type == PointType.END:
            return entity.EndSketchPoint
        elif point_type == PointType.CENTER:
            return entity.CenterSketchPoint
        else:
            raise ValueError(f"Invalid point type {point_type} for SketchEllipticalArc")

    else:
        raise ValueError(f"Unknown entity kind for: {type(entity).__name__}")


def get_point_type_for_sketch_point(entity: Any, sketch_point: Any) -> PointType | None:
    """
    Determine the canonical PointType for a SketchPoint on an entity.

    Args:
        entity: Inventor sketch entity that may contain the point
        sketch_point: Inventor SketchPoint to find

    Returns:
        PointType if the point belongs to this entity, None otherwise
    """
    kind = _entity_kind(entity)

    try:
        if kind == 'line':
            if _same_point(entity.StartSketchPoint, sketch_point):
                return PointType.START
            elif _same_point(entity.EndSketchPoint, sketch_point):
                return PointType.END

        elif kind == 'arc':
            if _same_point(entity.StartSketchPoint, sketch_point):
                return PointType.START
            elif _same_point(entity.EndSketchPoint, sketch_point):
                return PointType.END
            elif _same_point(entity.CenterSketchPoint, sketch_point):
                return PointType.CENTER

        elif kind == 'circle':
            if _same_point(entity.CenterSketchPoint, sketch_point):
                return PointType.CENTER

        elif kind == 'point':
            if _same_point(entity, sketch_point):
                return PointType.CENTER

        elif kind == 'spline':
            if _same_point(entity.StartSketchPoint, sketch_point):
                return PointType.START
            elif _same_point(entity.EndSketchPoint, sketch_point):
                return PointType.END

        elif kind == 'ellipse':
            if _same_point(entity.CenterSketchPoint, sketch_point):
                return PointType.CENTER

        elif kind == 'elliptical_arc':
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
    """Check if two sketch points are the same (by COM identity or geometry)."""
    try:
        if pt1._oleobj_ == pt2._oleobj_:
            return True
    except Exception:
        pass

    try:
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
    kind = _entity_kind(entity)

    if kind == 'line':
        return [PointType.START, PointType.END]
    elif kind == 'elliptical_arc':
        return [PointType.START, PointType.END, PointType.CENTER]
    elif kind == 'arc':
        return [PointType.START, PointType.END, PointType.CENTER]
    elif kind == 'ellipse':
        return [PointType.CENTER]
    elif kind == 'circle':
        return [PointType.CENTER]
    elif kind == 'point':
        return [PointType.CENTER]
    elif kind == 'spline':
        return [PointType.START, PointType.END]
    else:
        return []
