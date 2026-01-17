"""
FreeCAD vertex index mapping.

FreeCAD uses specific vertex indices for different geometry types:
- Lines: vertex 1 = start, vertex 2 = end
- Arcs: vertex 1 = start, vertex 2 = end, vertex 3 = center
- Circles: vertex 3 = center (no start/end)
- Points: vertex 1 = the point itself

The origin point uses geometry index -1, vertex 1.
"""

from dataclasses import dataclass

from morphe import Arc, Circle, Ellipse, EllipticalArc, Line, Point, PointType, Spline


@dataclass
class VertexMap:
    """
    Mapping between canonical PointType and FreeCAD vertex indices
    for each primitive type.
    """
    # Line vertex indices
    LINE_START = 1
    LINE_END = 2
    LINE_MIDPOINT = 3  # FreeCAD uses 3 for midpoint of lines

    # Arc vertex indices
    ARC_START = 1
    ARC_END = 2
    ARC_CENTER = 3

    # Circle vertex indices
    CIRCLE_CENTER = 3

    # Point vertex indices
    POINT_CENTER = 1

    # Spline vertex indices
    SPLINE_START = 1
    SPLINE_END = 2

    # Ellipse vertex indices (similar to Circle)
    ELLIPSE_CENTER = 3

    # EllipticalArc vertex indices (similar to Arc)
    ELLIPTICAL_ARC_START = 1
    ELLIPTICAL_ARC_END = 2
    ELLIPTICAL_ARC_CENTER = 3

    # Origin reference
    ORIGIN_GEO_INDEX = -1
    ORIGIN_VERTEX = 1

    # External geometry base index
    EXTERNAL_GEO_BASE = -2


def get_vertex_index(primitive_type: type, point_type: PointType) -> int | None:
    """
    Get the FreeCAD vertex index for a point type on a primitive.

    Args:
        primitive_type: The type of primitive (Line, Arc, Circle, etc.)
        point_type: The canonical point type

    Returns:
        FreeCAD vertex index, or None if invalid combination.
    """
    if primitive_type == Line:
        mapping = {
            PointType.START: VertexMap.LINE_START,
            PointType.END: VertexMap.LINE_END,
        }
    elif primitive_type == Arc:
        mapping = {
            PointType.START: VertexMap.ARC_START,
            PointType.END: VertexMap.ARC_END,
            PointType.CENTER: VertexMap.ARC_CENTER,
        }
    elif primitive_type == Circle:
        mapping = {
            PointType.CENTER: VertexMap.CIRCLE_CENTER,
        }
    elif primitive_type == Point:
        mapping = {
            PointType.CENTER: VertexMap.POINT_CENTER,
        }
    elif primitive_type == Spline:
        mapping = {
            PointType.START: VertexMap.SPLINE_START,
            PointType.END: VertexMap.SPLINE_END,
        }
    elif primitive_type == Ellipse:
        mapping = {
            PointType.CENTER: VertexMap.ELLIPSE_CENTER,
        }
    elif primitive_type == EllipticalArc:
        mapping = {
            PointType.START: VertexMap.ELLIPTICAL_ARC_START,
            PointType.END: VertexMap.ELLIPTICAL_ARC_END,
            PointType.CENTER: VertexMap.ELLIPTICAL_ARC_CENTER,
        }
    else:
        return None

    return mapping.get(point_type)


def get_point_type_from_vertex(primitive_type: type, vertex_index: int) -> PointType | None:
    """
    Get the canonical point type from a FreeCAD vertex index.

    Args:
        primitive_type: The type of primitive
        vertex_index: FreeCAD vertex index

    Returns:
        Canonical PointType, or None if invalid.
    """
    if primitive_type == Line:
        mapping = {
            VertexMap.LINE_START: PointType.START,
            VertexMap.LINE_END: PointType.END,
        }
    elif primitive_type == Arc:
        mapping = {
            VertexMap.ARC_START: PointType.START,
            VertexMap.ARC_END: PointType.END,
            VertexMap.ARC_CENTER: PointType.CENTER,
        }
    elif primitive_type == Circle:
        mapping = {
            VertexMap.CIRCLE_CENTER: PointType.CENTER,
        }
    elif primitive_type == Point:
        mapping = {
            VertexMap.POINT_CENTER: PointType.CENTER,
        }
    elif primitive_type == Spline:
        mapping = {
            VertexMap.SPLINE_START: PointType.START,
            VertexMap.SPLINE_END: PointType.END,
        }
    elif primitive_type == Ellipse:
        mapping = {
            VertexMap.ELLIPSE_CENTER: PointType.CENTER,
        }
    elif primitive_type == EllipticalArc:
        mapping = {
            VertexMap.ELLIPTICAL_ARC_START: PointType.START,
            VertexMap.ELLIPTICAL_ARC_END: PointType.END,
            VertexMap.ELLIPTICAL_ARC_CENTER: PointType.CENTER,
        }
    else:
        return None

    return mapping.get(vertex_index)
