"""Vertex mapping utilities for Fusion 360 sketch adapter.

Unlike FreeCAD which uses numeric vertex indices, Fusion 360 provides direct
property access to sketch points (startSketchPoint, endSketchPoint, centerSketchPoint).
This module provides a consistent interface for accessing these points based on
canonical PointType values.
"""

from morphe.types import PointType


class VertexMap:
    """Maps between canonical PointType and Fusion 360 sketch point properties.

    Fusion 360 sketch entities expose points as properties:
    - SketchLine: startSketchPoint, endSketchPoint
    - SketchArc: startSketchPoint, endSketchPoint, centerSketchPoint
    - SketchCircle: centerSketchPoint
    - SketchPoint: geometry (Point3D)
    - SketchFittedSpline: startSketchPoint, endSketchPoint, fitPoints (collection)
    """

    # Mapping from (primitive_type, PointType) to Fusion 360 property name
    POINT_PROPERTY_MAP = {
        # Line points
        ("line", PointType.START): "startSketchPoint",
        ("line", PointType.END): "endSketchPoint",

        # Arc points
        ("arc", PointType.START): "startSketchPoint",
        ("arc", PointType.END): "endSketchPoint",
        ("arc", PointType.CENTER): "centerSketchPoint",

        # Circle points
        ("circle", PointType.CENTER): "centerSketchPoint",

        # Ellipse points
        ("ellipse", PointType.CENTER): "centerSketchPoint",

        # EllipticalArc points
        ("ellipticalarc", PointType.START): "startSketchPoint",
        ("ellipticalarc", PointType.END): "endSketchPoint",
        ("ellipticalarc", PointType.CENTER): "centerSketchPoint",

        # Point
        ("point", PointType.CENTER): "geometry",

        # Spline points
        ("spline", PointType.START): "startSketchPoint",
        ("spline", PointType.END): "endSketchPoint",
    }

    @classmethod
    def get_point_property(cls, primitive_type: str, point_type: PointType) -> str:
        """Get the Fusion 360 property name for accessing a specific point.

        Args:
            primitive_type: Type of primitive ("line", "arc", "circle", "point", "spline")
            point_type: The canonical PointType

        Returns:
            Property name to access on the Fusion 360 sketch entity

        Raises:
            ValueError: If the combination is not valid
        """
        key = (primitive_type.lower(), point_type)
        if key not in cls.POINT_PROPERTY_MAP:
            raise ValueError(
                f"Invalid point type {point_type} for primitive type {primitive_type}"
            )
        return cls.POINT_PROPERTY_MAP[key]

    @classmethod
    def get_sketch_point(cls, entity, primitive_type: str, point_type: PointType):
        """Get a SketchPoint from a Fusion 360 sketch entity.

        Args:
            entity: Fusion 360 sketch entity (SketchLine, SketchArc, etc.)
            primitive_type: Type of primitive
            point_type: The canonical PointType

        Returns:
            The SketchPoint object for use in constraints
        """
        # Special case: SketchPoint entities should return themselves
        # for constraint purposes (not their geometry which is Point3D)
        if primitive_type.lower() == "point" and point_type == PointType.CENTER:
            return entity  # Return the SketchPoint itself

        prop_name = cls.get_point_property(primitive_type, point_type)
        return getattr(entity, prop_name)

    @classmethod
    def get_point_types_for_primitive(cls, primitive_type: str) -> list:
        """Get all valid PointTypes for a given primitive type.

        Args:
            primitive_type: Type of primitive

        Returns:
            List of valid PointType values
        """
        ptype = primitive_type.lower()
        return [
            pt for (pt_type, pt), _ in cls.POINT_PROPERTY_MAP.items()
            if pt_type == ptype
        ]

    @classmethod
    def get_point_type_from_property(cls, primitive_type: str, property_name: str) -> PointType:
        """Get the canonical PointType from a Fusion 360 property name.

        Args:
            primitive_type: Type of primitive
            property_name: Fusion 360 property name

        Returns:
            The corresponding PointType

        Raises:
            ValueError: If the property is not recognized
        """
        ptype = primitive_type.lower()
        for (pt_type, point_type), prop in cls.POINT_PROPERTY_MAP.items():
            if pt_type == ptype and prop == property_name:
                return point_type
        raise ValueError(
            f"Unknown property {property_name} for primitive type {primitive_type}"
        )


def get_point_from_sketch_entity(entity, point_type: PointType):
    """Extract a Point3D from a Fusion 360 sketch entity at the specified point type.

    This is a convenience function that handles the different entity types
    and returns the actual Point3D geometry.

    Args:
        entity: Fusion 360 sketch entity
        point_type: The canonical PointType

    Returns:
        adsk.core.Point3D object
    """
    # Determine primitive type from entity
    entity_type = entity.objectType

    if "SketchLine" in entity_type:
        primitive_type = "line"
    elif "SketchEllipticalArc" in entity_type:
        primitive_type = "ellipticalarc"
    elif "SketchArc" in entity_type:
        primitive_type = "arc"
    elif "SketchEllipse" in entity_type:
        primitive_type = "ellipse"
    elif "SketchCircle" in entity_type:
        primitive_type = "circle"
    elif "SketchPoint" in entity_type:
        primitive_type = "point"
    elif "SketchFittedSpline" in entity_type or "SketchFixedSpline" in entity_type:
        primitive_type = "spline"
    else:
        raise ValueError(f"Unknown entity type: {entity_type}")

    sketch_point = VertexMap.get_sketch_point(entity, primitive_type, point_type)

    # For most entities, sketch_point is a SketchPoint with a geometry property
    # For SketchPoint entities with CENTER, it's already a Point3D
    if hasattr(sketch_point, "geometry"):
        return sketch_point.geometry
    return sketch_point
