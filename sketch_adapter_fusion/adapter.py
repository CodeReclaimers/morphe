"""Fusion 360 adapter for canonical sketch representation.

This module provides the FusionAdapter class that implements the
SketchBackendAdapter interface for Autodesk Fusion 360.

Note: Fusion 360 internally uses centimeters, while the canonical format
uses millimeters. This adapter handles the conversion automatically.
"""

import math
from typing import Any, Dict, Optional, Tuple

from sketch_canonical.adapter import (
    AdapterError,
    ConstraintError,
    ExportError,
    GeometryError,
    SketchBackendAdapter,
    SketchCreationError,
)
from sketch_canonical.constraints import ConstraintType, SketchConstraint
from sketch_canonical.document import SketchDocument, SolverStatus
from sketch_canonical.primitives import Arc, Circle, Line, Point, SketchPrimitive, Spline
from sketch_canonical.types import ElementPrefix, PointRef, PointType

from .vertex_map import VertexMap, get_point_from_sketch_entity

# Fusion 360 uses centimeters internally, canonical format uses millimeters
MM_TO_CM = 0.1
CM_TO_MM = 10.0


class FusionAdapter(SketchBackendAdapter):
    """Fusion 360 implementation of SketchBackendAdapter.

    This adapter translates between the canonical sketch representation
    and Fusion 360's native sketch API. It requires Fusion 360 to be
    running and accessible via the adsk module.

    Attributes:
        _app: Fusion 360 Application object
        _design: Active Fusion design
        _sketch: Current active sketch
        _id_to_entity: Mapping from canonical IDs to Fusion sketch entities
        _entity_to_id: Mapping from Fusion entities to canonical IDs
    """

    def __init__(self, document=None):
        """Initialize the Fusion 360 adapter.

        Args:
            document: Optional Fusion 360 document. If None, uses active document.

        Raises:
            ImportError: If Fusion 360 API is not available
            AdapterError: If no active design is found
        """
        try:
            import adsk.core
            import adsk.fusion
            self._adsk_core = adsk.core
            self._adsk_fusion = adsk.fusion
        except ImportError as e:
            raise ImportError(
                "Fusion 360 API not available. This adapter must be run within Fusion 360."
            ) from e

        self._app = adsk.core.Application.get()
        if not self._app:
            raise AdapterError("Could not get Fusion 360 application instance")

        if document is not None:
            self._document = document
        else:
            self._document = self._app.activeDocument

        if self._document is None:
            raise AdapterError("No active Fusion 360 document")

        self._design = adsk.fusion.Design.cast(self._app.activeProduct)
        if not self._design:
            raise AdapterError("No active Fusion 360 design")

        self._sketch = None
        self._id_to_entity: Dict[str, Any] = {}
        self._entity_to_id: Dict[Any, str] = {}

    def create_sketch(self, name: str, plane=None) -> None:
        """Create a new sketch in Fusion 360.

        Args:
            name: Name for the new sketch
            plane: Optional plane specification. Can be:
                - None: Uses XY construction plane
                - "XY", "XZ", "YZ": Standard construction planes
                - A Fusion 360 ConstructionPlane or BRepFace object

        Raises:
            SketchCreationError: If sketch creation fails
        """
        try:
            root_comp = self._design.rootComponent
            sketches = root_comp.sketches

            # Determine the plane to use
            if plane is None or plane == "XY":
                sketch_plane = root_comp.xYConstructionPlane
            elif plane == "XZ":
                sketch_plane = root_comp.xZConstructionPlane
            elif plane == "YZ":
                sketch_plane = root_comp.yZConstructionPlane
            else:
                sketch_plane = plane

            self._sketch = sketches.add(sketch_plane)
            self._sketch.name = name

            # Clear mappings for new sketch
            self._id_to_entity.clear()
            self._entity_to_id.clear()

        except Exception as e:
            raise SketchCreationError(f"Failed to create sketch: {e}") from e

    def load_sketch(self, sketch: SketchDocument) -> None:
        """Load a SketchDocument into a new Fusion 360 sketch.

        Creates a new sketch and populates it with the primitives and
        constraints from the provided SketchDocument.

        Args:
            sketch: The SketchDocument to load

        Raises:
            SketchCreationError: If sketch creation fails
            GeometryError: If geometry creation fails
            ConstraintError: If constraint creation fails
        """
        # Create the sketch
        self.create_sketch(sketch.name)

        # Add all primitives
        for prim_id, primitive in sketch.primitives.items():
            self.add_primitive(primitive)

        # Add all constraints
        for constraint in sketch.constraints:
            try:
                self.add_constraint(constraint)
            except ConstraintError:
                # Log but continue - some constraints may not be supported
                pass

    def export_sketch(self) -> SketchDocument:
        """Export the current Fusion 360 sketch to a SketchDocument.

        Returns:
            A SketchDocument representing the current sketch

        Raises:
            ExportError: If export fails or no sketch is active
        """
        if not self._sketch:
            raise ExportError("No active sketch to export")

        try:
            doc = SketchDocument(name=self._sketch.name)

            # Export all geometry
            self._export_lines(doc)
            self._export_arcs(doc)
            self._export_circles(doc)
            self._export_points(doc)
            self._export_splines(doc)

            # Export constraints
            self._export_geometric_constraints(doc)
            self._export_dimensional_constraints(doc)

            # Update solver status
            status, dof = self.get_solver_status()
            doc.solver_status = status
            doc.degrees_of_freedom = dof

            return doc

        except Exception as e:
            raise ExportError(f"Failed to export sketch: {e}") from e

    def add_primitive(self, primitive: SketchPrimitive) -> Any:
        """Add a primitive to the current Fusion 360 sketch.

        Args:
            primitive: The primitive to add

        Returns:
            The created Fusion 360 sketch entity

        Raises:
            GeometryError: If the primitive cannot be added
        """
        if not self._sketch:
            raise GeometryError("No active sketch")

        try:
            if isinstance(primitive, Line):
                entity = self._add_line(primitive)
            elif isinstance(primitive, Arc):
                entity = self._add_arc(primitive)
            elif isinstance(primitive, Circle):
                entity = self._add_circle(primitive)
            elif isinstance(primitive, Point):
                entity = self._add_point(primitive)
            elif isinstance(primitive, Spline):
                entity = self._add_spline(primitive)
            else:
                raise GeometryError(f"Unsupported primitive type: {type(primitive)}")

            # Set construction geometry flag if needed
            if primitive.construction and hasattr(entity, "isConstruction"):
                entity.isConstruction = True

            # Store mapping
            self._id_to_entity[primitive.id] = entity
            self._entity_to_id[entity.entityToken] = primitive.id

            return entity

        except Exception as e:
            if isinstance(e, GeometryError):
                raise
            raise GeometryError(f"Failed to add primitive {primitive.id}: {e}") from e

    def _add_line(self, line: Line) -> Any:
        """Add a line to the sketch."""
        lines = self._sketch.sketchCurves.sketchLines

        start_pt = self._point2d_to_point3d(line.start)
        end_pt = self._point2d_to_point3d(line.end)

        return lines.addByTwoPoints(start_pt, end_pt)

    def _add_arc(self, arc: Arc) -> Any:
        """Add an arc to the sketch.

        Uses three-point construction for reliable direction representation.
        """
        arcs = self._sketch.sketchCurves.sketchArcs

        # Get three points for arc construction
        three_pts = arc.to_three_point()
        start_pt = self._point2d_to_point3d(three_pts["start"])
        mid_pt = self._point2d_to_point3d(three_pts["mid"])
        end_pt = self._point2d_to_point3d(three_pts["end"])

        return arcs.addByThreePoints(start_pt, mid_pt, end_pt)

    def _add_circle(self, circle: Circle) -> Any:
        """Add a circle to the sketch."""
        circles = self._sketch.sketchCurves.sketchCircles

        center_pt = self._point2d_to_point3d(circle.center)
        radius_cm = circle.radius * MM_TO_CM

        return circles.addByCenterRadius(center_pt, radius_cm)

    def _add_point(self, point: Point) -> Any:
        """Add a sketch point."""
        points = self._sketch.sketchPoints

        pt = self._point2d_to_point3d(point.position)

        return points.add(pt)

    def _add_spline(self, spline: Spline) -> Any:
        """Add a spline to the sketch.

        Fusion 360 supports both fitted splines (through points) and
        control-point-based splines via NurbsCurve3D. We use the NURBS
        approach for precise control point specification.
        """
        # Create a NurbsCurve3D from the spline data
        poles = []
        for pole in spline.poles:
            poles.append(self._point2d_to_point3d(pole))

        # Create ObjectCollection for control points
        control_points = self._adsk_core.ObjectCollection.create()
        for pole in poles:
            control_points.add(pole)

        # Extract knot vector and weights
        knots = list(spline.knots)
        degree = spline.degree
        weights = list(spline.weights) if spline.weights else [1.0] * len(spline.poles)

        # Create the NURBS curve (transient geometry)
        # For rational (weighted) B-splines:
        if spline.weights:
            nurbs_curve = self._adsk_core.NurbsCurve3D.createRational(
                control_points,
                degree,
                knots,
                weights,
                spline.periodic
            )
        else:
            # For non-rational B-splines:
            nurbs_curve = self._adsk_core.NurbsCurve3D.createNonRational(
                control_points,
                degree,
                knots,
                spline.periodic
            )

        # Add as a fitted spline using the NURBS curve
        # Note: addByNurbsCurve is on sketchFittedSplines collection
        splines = self._sketch.sketchCurves.sketchFittedSplines
        return splines.addByNurbsCurve(nurbs_curve)

    def add_constraint(self, constraint: SketchConstraint) -> bool:
        """Add a constraint to the current sketch.

        Args:
            constraint: The constraint to add

        Returns:
            True if the constraint was added successfully

        Raises:
            ConstraintError: If the constraint cannot be added
        """
        if not self._sketch:
            raise ConstraintError("No active sketch")

        try:
            ctype = constraint.constraint_type
            refs = constraint.references
            value = constraint.value

            # Geometric constraints
            if ctype == ConstraintType.COINCIDENT:
                return self._add_coincident(refs)
            elif ctype == ConstraintType.HORIZONTAL:
                return self._add_horizontal(refs)
            elif ctype == ConstraintType.VERTICAL:
                return self._add_vertical(refs)
            elif ctype == ConstraintType.PARALLEL:
                return self._add_parallel(refs)
            elif ctype == ConstraintType.PERPENDICULAR:
                return self._add_perpendicular(refs)
            elif ctype == ConstraintType.TANGENT:
                return self._add_tangent(refs)
            elif ctype == ConstraintType.EQUAL:
                return self._add_equal(refs)
            elif ctype == ConstraintType.CONCENTRIC:
                return self._add_concentric(refs)
            elif ctype == ConstraintType.COLLINEAR:
                return self._add_collinear(refs)
            elif ctype == ConstraintType.FIXED:
                return self._add_fixed(refs)
            elif ctype == ConstraintType.SYMMETRIC:
                return self._add_symmetric(refs)
            elif ctype == ConstraintType.MIDPOINT:
                return self._add_midpoint(refs)

            # Dimensional constraints
            elif ctype == ConstraintType.DISTANCE:
                return self._add_distance(refs, value)
            elif ctype == ConstraintType.DISTANCE_X:
                return self._add_distance_x(refs, value)
            elif ctype == ConstraintType.DISTANCE_Y:
                return self._add_distance_y(refs, value)
            elif ctype == ConstraintType.LENGTH:
                return self._add_length(refs, value)
            elif ctype == ConstraintType.RADIUS:
                return self._add_radius(refs, value)
            elif ctype == ConstraintType.DIAMETER:
                return self._add_diameter(refs, value)
            elif ctype == ConstraintType.ANGLE:
                return self._add_angle(refs, value)
            else:
                raise ConstraintError(f"Unsupported constraint type: {ctype}")

        except Exception as e:
            if isinstance(e, ConstraintError):
                raise
            raise ConstraintError(f"Failed to add constraint: {e}") from e

    def _get_entity_for_ref(self, ref) -> Any:
        """Get the Fusion entity for a reference (string ID or PointRef)."""
        if isinstance(ref, PointRef):
            element_id = ref.element_id
        else:
            element_id = str(ref)

        if element_id not in self._id_to_entity:
            raise ConstraintError(f"Unknown element ID: {element_id}")

        return self._id_to_entity[element_id]

    def _get_sketch_point_for_ref(self, ref: PointRef) -> Any:
        """Get a SketchPoint for a PointRef."""
        entity = self._get_entity_for_ref(ref)
        primitive_type = self._get_primitive_type_for_entity(entity)
        return VertexMap.get_sketch_point(entity, primitive_type, ref.point_type)

    def _get_primitive_type_for_entity(self, entity) -> str:
        """Determine the primitive type from a Fusion entity."""
        obj_type = entity.objectType
        if "SketchLine" in obj_type:
            return "line"
        elif "SketchArc" in obj_type:
            return "arc"
        elif "SketchCircle" in obj_type:
            return "circle"
        elif "SketchPoint" in obj_type:
            return "point"
        elif "Spline" in obj_type:
            return "spline"
        raise ConstraintError(f"Unknown entity type: {obj_type}")

    # Geometric constraint implementations

    def _add_coincident(self, refs) -> bool:
        """Add a coincident constraint between two points."""
        if len(refs) != 2:
            raise ConstraintError("COINCIDENT requires exactly 2 references")

        constraints = self._sketch.geometricConstraints

        pt1 = self._get_sketch_point_for_ref(refs[0])
        pt2 = self._get_sketch_point_for_ref(refs[1])

        constraints.addCoincident(pt1, pt2)
        return True

    def _add_horizontal(self, refs) -> bool:
        """Add a horizontal constraint."""
        constraints = self._sketch.geometricConstraints

        if len(refs) == 1:
            # Single line
            entity = self._get_entity_for_ref(refs[0])
            constraints.addHorizontal(entity)
        elif len(refs) == 2:
            # Two points - add horizontal constraint between them
            pt1 = self._get_sketch_point_for_ref(refs[0])
            pt2 = self._get_sketch_point_for_ref(refs[1])
            constraints.addHorizontalPoints(pt1, pt2)
        else:
            raise ConstraintError("HORIZONTAL requires 1 or 2 references")

        return True

    def _add_vertical(self, refs) -> bool:
        """Add a vertical constraint."""
        constraints = self._sketch.geometricConstraints

        if len(refs) == 1:
            entity = self._get_entity_for_ref(refs[0])
            constraints.addVertical(entity)
        elif len(refs) == 2:
            pt1 = self._get_sketch_point_for_ref(refs[0])
            pt2 = self._get_sketch_point_for_ref(refs[1])
            constraints.addVerticalPoints(pt1, pt2)
        else:
            raise ConstraintError("VERTICAL requires 1 or 2 references")

        return True

    def _add_parallel(self, refs) -> bool:
        """Add a parallel constraint between two lines."""
        if len(refs) < 2:
            raise ConstraintError("PARALLEL requires at least 2 references")

        constraints = self._sketch.geometricConstraints

        # Add pairwise constraints
        first = self._get_entity_for_ref(refs[0])
        for i in range(1, len(refs)):
            other = self._get_entity_for_ref(refs[i])
            constraints.addParallel(first, other)

        return True

    def _add_perpendicular(self, refs) -> bool:
        """Add a perpendicular constraint between two lines."""
        if len(refs) != 2:
            raise ConstraintError("PERPENDICULAR requires exactly 2 references")

        constraints = self._sketch.geometricConstraints
        line1 = self._get_entity_for_ref(refs[0])
        line2 = self._get_entity_for_ref(refs[1])

        constraints.addPerpendicular(line1, line2)
        return True

    def _add_tangent(self, refs) -> bool:
        """Add a tangent constraint between curves."""
        if len(refs) < 2:
            raise ConstraintError("TANGENT requires at least 2 references")

        constraints = self._sketch.geometricConstraints

        # Add pairwise tangent constraints
        first = self._get_entity_for_ref(refs[0])
        for i in range(1, len(refs)):
            other = self._get_entity_for_ref(refs[i])
            constraints.addTangent(first, other)

        return True

    def _add_equal(self, refs) -> bool:
        """Add an equal constraint between curves."""
        if len(refs) < 2:
            raise ConstraintError("EQUAL requires at least 2 references")

        constraints = self._sketch.geometricConstraints

        first = self._get_entity_for_ref(refs[0])
        for i in range(1, len(refs)):
            other = self._get_entity_for_ref(refs[i])
            constraints.addEqual(first, other)

        return True

    def _add_concentric(self, refs) -> bool:
        """Add a concentric constraint between circles/arcs."""
        if len(refs) < 2:
            raise ConstraintError("CONCENTRIC requires at least 2 references")

        constraints = self._sketch.geometricConstraints

        first = self._get_entity_for_ref(refs[0])
        for i in range(1, len(refs)):
            other = self._get_entity_for_ref(refs[i])
            constraints.addConcentric(first, other)

        return True

    def _add_collinear(self, refs) -> bool:
        """Add a collinear constraint between lines."""
        if len(refs) < 2:
            raise ConstraintError("COLLINEAR requires at least 2 references")

        constraints = self._sketch.geometricConstraints

        first = self._get_entity_for_ref(refs[0])
        for i in range(1, len(refs)):
            other = self._get_entity_for_ref(refs[i])
            constraints.addCollinear(first, other)

        return True

    def _add_fixed(self, refs) -> bool:
        """Add a fixed/lock constraint.

        Note: The exact API method name may vary between Fusion 360 versions.
        This implementation tries 'addFix' which is the expected method name.
        """
        constraints = self._sketch.geometricConstraints

        for ref in refs:
            entity = self._get_entity_for_ref(ref)
            # The method name is 'addFix' in Fusion 360's API
            if hasattr(constraints, 'addFix'):
                constraints.addFix(entity)
            else:
                raise ConstraintError(
                    "FIXED constraint not supported in this Fusion 360 version"
                )

        return True

    def _add_symmetric(self, refs) -> bool:
        """Add a symmetry constraint.

        Expects 3 references: two entities and a symmetry line.
        """
        if len(refs) != 3:
            raise ConstraintError("SYMMETRIC requires exactly 3 references (2 entities + line)")

        constraints = self._sketch.geometricConstraints

        # First two refs are the symmetric entities, third is the symmetry line
        if isinstance(refs[0], PointRef) and isinstance(refs[1], PointRef):
            # Point symmetry
            pt1 = self._get_sketch_point_for_ref(refs[0])
            pt2 = self._get_sketch_point_for_ref(refs[1])
            line = self._get_entity_for_ref(refs[2])
            constraints.addSymmetry(pt1, pt2, line)
        else:
            # Entity symmetry
            entity1 = self._get_entity_for_ref(refs[0])
            entity2 = self._get_entity_for_ref(refs[1])
            line = self._get_entity_for_ref(refs[2])
            constraints.addSymmetry(entity1, entity2, line)

        return True

    def _add_midpoint(self, refs) -> bool:
        """Add a midpoint constraint.

        Expects 2 references: point and line.
        """
        if len(refs) != 2:
            raise ConstraintError("MIDPOINT requires exactly 2 references")

        constraints = self._sketch.geometricConstraints

        # Determine which is the point and which is the line
        ref0_is_point = isinstance(refs[0], PointRef)
        ref1_is_point = isinstance(refs[1], PointRef)

        if ref0_is_point and not ref1_is_point:
            point = self._get_sketch_point_for_ref(refs[0])
            line = self._get_entity_for_ref(refs[1])
        elif ref1_is_point and not ref0_is_point:
            point = self._get_sketch_point_for_ref(refs[1])
            line = self._get_entity_for_ref(refs[0])
        else:
            raise ConstraintError("MIDPOINT requires one point reference and one line reference")

        constraints.addMidPoint(point, line)
        return True

    # Dimensional constraint implementations

    def _add_distance(self, refs, value: float) -> bool:
        """Add a distance constraint."""
        if value is None:
            raise ConstraintError("DISTANCE requires a value")

        dims = self._sketch.sketchDimensions
        distance_cm = value * MM_TO_CM

        if len(refs) == 2:
            # Distance between two points
            pt1 = self._get_sketch_point_for_ref(refs[0])
            pt2 = self._get_sketch_point_for_ref(refs[1])

            # Need a text position for the dimension
            text_pt = self._adsk_core.Point3D.create(
                (pt1.geometry.x + pt2.geometry.x) / 2,
                (pt1.geometry.y + pt2.geometry.y) / 2 + 0.5,
                0
            )

            dim = dims.addDistanceDimension(pt1, pt2,
                self._adsk_fusion.DimensionOrientations.AlignedDimensionOrientation,
                text_pt)
            dim.parameter.value = distance_cm
        elif len(refs) == 1:
            # Distance from origin - use offset dimension
            pt = self._get_sketch_point_for_ref(refs[0])
            origin = self._sketch.originPoint

            text_pt = self._adsk_core.Point3D.create(
                pt.geometry.x / 2,
                pt.geometry.y / 2 + 0.5,
                0
            )

            dim = dims.addDistanceDimension(origin, pt,
                self._adsk_fusion.DimensionOrientations.AlignedDimensionOrientation,
                text_pt)
            dim.parameter.value = distance_cm
        else:
            raise ConstraintError("DISTANCE requires 1 or 2 references")

        return True

    def _add_distance_x(self, refs, value: float) -> bool:
        """Add a horizontal distance constraint."""
        if value is None:
            raise ConstraintError("DISTANCE_X requires a value")

        dims = self._sketch.sketchDimensions
        distance_cm = value * MM_TO_CM

        if len(refs) == 2:
            pt1 = self._get_sketch_point_for_ref(refs[0])
            pt2 = self._get_sketch_point_for_ref(refs[1])

            text_pt = self._adsk_core.Point3D.create(
                (pt1.geometry.x + pt2.geometry.x) / 2,
                max(pt1.geometry.y, pt2.geometry.y) + 0.5,
                0
            )

            dim = dims.addDistanceDimension(pt1, pt2,
                self._adsk_fusion.DimensionOrientations.HorizontalDimensionOrientation,
                text_pt)
            dim.parameter.value = distance_cm
        elif len(refs) == 1:
            pt = self._get_sketch_point_for_ref(refs[0])
            origin = self._sketch.originPoint

            text_pt = self._adsk_core.Point3D.create(
                pt.geometry.x / 2,
                pt.geometry.y + 0.5,
                0
            )

            dim = dims.addDistanceDimension(origin, pt,
                self._adsk_fusion.DimensionOrientations.HorizontalDimensionOrientation,
                text_pt)
            dim.parameter.value = distance_cm
        else:
            raise ConstraintError("DISTANCE_X requires 1 or 2 references")

        return True

    def _add_distance_y(self, refs, value: float) -> bool:
        """Add a vertical distance constraint."""
        if value is None:
            raise ConstraintError("DISTANCE_Y requires a value")

        dims = self._sketch.sketchDimensions
        distance_cm = value * MM_TO_CM

        if len(refs) == 2:
            pt1 = self._get_sketch_point_for_ref(refs[0])
            pt2 = self._get_sketch_point_for_ref(refs[1])

            text_pt = self._adsk_core.Point3D.create(
                max(pt1.geometry.x, pt2.geometry.x) + 0.5,
                (pt1.geometry.y + pt2.geometry.y) / 2,
                0
            )

            dim = dims.addDistanceDimension(pt1, pt2,
                self._adsk_fusion.DimensionOrientations.VerticalDimensionOrientation,
                text_pt)
            dim.parameter.value = distance_cm
        elif len(refs) == 1:
            pt = self._get_sketch_point_for_ref(refs[0])
            origin = self._sketch.originPoint

            text_pt = self._adsk_core.Point3D.create(
                pt.geometry.x + 0.5,
                pt.geometry.y / 2,
                0
            )

            dim = dims.addDistanceDimension(origin, pt,
                self._adsk_fusion.DimensionOrientations.VerticalDimensionOrientation,
                text_pt)
            dim.parameter.value = distance_cm
        else:
            raise ConstraintError("DISTANCE_Y requires 1 or 2 references")

        return True

    def _add_length(self, refs, value: float) -> bool:
        """Add a length constraint to a line."""
        if value is None or len(refs) != 1:
            raise ConstraintError("LENGTH requires exactly 1 reference and a value")

        dims = self._sketch.sketchDimensions
        length_cm = value * MM_TO_CM

        entity = self._get_entity_for_ref(refs[0])

        # Get midpoint for dimension text placement
        if hasattr(entity, "startSketchPoint") and hasattr(entity, "endSketchPoint"):
            start = entity.startSketchPoint.geometry
            end = entity.endSketchPoint.geometry
            text_pt = self._adsk_core.Point3D.create(
                (start.x + end.x) / 2,
                (start.y + end.y) / 2 + 0.5,
                0
            )
        else:
            text_pt = self._adsk_core.Point3D.create(0, 0.5, 0)

        dim = dims.addDistanceDimension(
            entity.startSketchPoint,
            entity.endSketchPoint,
            self._adsk_fusion.DimensionOrientations.AlignedDimensionOrientation,
            text_pt
        )
        dim.parameter.value = length_cm

        return True

    def _add_radius(self, refs, value: float) -> bool:
        """Add a radius constraint to a circle or arc."""
        if value is None or len(refs) != 1:
            raise ConstraintError("RADIUS requires exactly 1 reference and a value")

        dims = self._sketch.sketchDimensions
        radius_cm = value * MM_TO_CM

        entity = self._get_entity_for_ref(refs[0])

        # Text position near the entity
        if hasattr(entity, "centerSketchPoint"):
            center = entity.centerSketchPoint.geometry
            text_pt = self._adsk_core.Point3D.create(center.x + radius_cm, center.y, 0)
        else:
            text_pt = self._adsk_core.Point3D.create(0, 0, 0)

        dim = dims.addRadialDimension(entity, text_pt)
        dim.parameter.value = radius_cm

        return True

    def _add_diameter(self, refs, value: float) -> bool:
        """Add a diameter constraint to a circle or arc."""
        if value is None or len(refs) != 1:
            raise ConstraintError("DIAMETER requires exactly 1 reference and a value")

        dims = self._sketch.sketchDimensions
        diameter_cm = value * MM_TO_CM

        entity = self._get_entity_for_ref(refs[0])

        if hasattr(entity, "centerSketchPoint"):
            center = entity.centerSketchPoint.geometry
            text_pt = self._adsk_core.Point3D.create(center.x + diameter_cm / 2, center.y, 0)
        else:
            text_pt = self._adsk_core.Point3D.create(0, 0, 0)

        dim = dims.addDiameterDimension(entity, text_pt)
        dim.parameter.value = diameter_cm

        return True

    def _add_angle(self, refs, value: float) -> bool:
        """Add an angle constraint between two lines.

        Args:
            refs: Two line references
            value: Angle in degrees
        """
        if value is None or len(refs) != 2:
            raise ConstraintError("ANGLE requires exactly 2 references and a value")

        dims = self._sketch.sketchDimensions
        angle_rad = math.radians(value)

        line1 = self._get_entity_for_ref(refs[0])
        line2 = self._get_entity_for_ref(refs[1])

        # Find intersection point for text placement
        text_pt = self._adsk_core.Point3D.create(0, 0, 0)

        dim = dims.addAngularDimension(line1, line2, text_pt)
        dim.parameter.value = angle_rad

        return True

    def get_solver_status(self) -> Tuple[SolverStatus, int]:
        """Get the current solver status and degrees of freedom.

        Returns:
            Tuple of (SolverStatus, degrees_of_freedom)
        """
        if not self._sketch:
            return SolverStatus.DIRTY, -1

        # Fusion 360 doesn't expose solver status directly in the same way as FreeCAD
        # We infer it from the sketch's constraint state
        try:
            # Check if fully constrained
            # Fusion uses different mechanisms - we check profile validity
            # and constraint count vs geometry

            # For now, return a reasonable default
            # In practice, Fusion 360 doesn't expose DOF directly
            return SolverStatus.UNDER_CONSTRAINED, -1

        except Exception:
            return SolverStatus.DIRTY, -1

    def capture_image(self, width: int, height: int) -> bytes:
        """Capture an image of the current sketch.

        Note: This requires Fusion 360's UI to be active.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            PNG image data as bytes

        Raises:
            AdapterError: If image capture fails
        """
        if not self._sketch:
            raise AdapterError("No active sketch")

        try:
            import tempfile
            import os

            # Activate the sketch for viewing
            self._sketch.isVisible = True

            # Get the viewport
            viewport = self._app.activeViewport

            # Create a temp file for the image
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                temp_path = f.name

            try:
                # Save image to temp file
                viewport.saveAsImageFile(temp_path, width, height)

                # Read the image data
                with open(temp_path, "rb") as f:
                    image_data = f.read()

                return image_data

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            raise AdapterError(f"Failed to capture image: {e}") from e

    def close_sketch(self) -> None:
        """Close the current sketch editing session."""
        if self._sketch:
            # Fusion doesn't require explicit close, but we can finish edit mode
            try:
                self._design.timeline.moveToEnd()
            except Exception:
                pass

    def get_element_by_id(self, element_id: str) -> Optional[Any]:
        """Get a Fusion 360 entity by its canonical ID.

        Args:
            element_id: The canonical element ID

        Returns:
            The Fusion 360 entity, or None if not found
        """
        return self._id_to_entity.get(element_id)

    def supports_feature(self, feature: str) -> bool:
        """Check if a feature is supported by this adapter.

        Args:
            feature: Feature name to check

        Returns:
            True if the feature is supported
        """
        supported = {
            "spline": True,
            "three_point_arc": True,
            "image_capture": True,
            "solver_status": False,  # Limited support
            "construction_geometry": True,
            "fixed_spline": True,
            "fitted_spline": True,
        }
        return supported.get(feature, False)

    # Helper methods

    def _point2d_to_point3d(self, point) -> Any:
        """Convert a canonical Point2D to a Fusion Point3D.

        Handles unit conversion from mm to cm.
        """
        from sketch_canonical.types import Point2D

        if isinstance(point, Point2D):
            return self._adsk_core.Point3D.create(
                point.x * MM_TO_CM,
                point.y * MM_TO_CM,
                0
            )
        elif isinstance(point, (list, tuple)):
            return self._adsk_core.Point3D.create(
                point[0] * MM_TO_CM,
                point[1] * MM_TO_CM,
                0
            )
        else:
            raise ValueError(f"Cannot convert {type(point)} to Point3D")

    def _point3d_to_point2d(self, point3d) -> 'Point2D':
        """Convert a Fusion Point3D to a canonical Point2D.

        Handles unit conversion from cm to mm.
        """
        from sketch_canonical.types import Point2D

        return Point2D(
            point3d.x * CM_TO_MM,
            point3d.y * CM_TO_MM
        )

    # Export helper methods

    def _export_lines(self, doc: SketchDocument) -> None:
        """Export all lines from the sketch."""
        lines = self._sketch.sketchCurves.sketchLines
        for i in range(lines.count):
            line = lines.item(i)

            start = self._point3d_to_point2d(line.startSketchPoint.geometry)
            end = self._point3d_to_point2d(line.endSketchPoint.geometry)

            canonical_line = Line(
                start=start,
                end=end,
                construction=line.isConstruction
            )

            prim_id = doc.add_primitive(canonical_line)
            self._id_to_entity[prim_id] = line
            self._entity_to_id[line.entityToken] = prim_id

    def _export_arcs(self, doc: SketchDocument) -> None:
        """Export all arcs from the sketch."""
        arcs = self._sketch.sketchCurves.sketchArcs
        for i in range(arcs.count):
            arc = arcs.item(i)

            center = self._point3d_to_point2d(arc.centerSketchPoint.geometry)
            start = self._point3d_to_point2d(arc.startSketchPoint.geometry)
            end = self._point3d_to_point2d(arc.endSketchPoint.geometry)

            # Determine CCW from the arc geometry
            # Fusion arcs have geometry.startAngle and geometry.endAngle
            geom = arc.geometry
            start_angle = geom.startAngle
            end_angle = geom.endAngle

            # If end > start in default (CCW), then ccw=True
            # Otherwise ccw=False
            ccw = (end_angle > start_angle)

            canonical_arc = Arc(
                center=center,
                start=start,
                end=end,
                ccw=ccw,
                construction=arc.isConstruction
            )

            prim_id = doc.add_primitive(canonical_arc)
            self._id_to_entity[prim_id] = arc
            self._entity_to_id[arc.entityToken] = prim_id

    def _export_circles(self, doc: SketchDocument) -> None:
        """Export all circles from the sketch."""
        circles = self._sketch.sketchCurves.sketchCircles
        for i in range(circles.count):
            circle = circles.item(i)

            center = self._point3d_to_point2d(circle.centerSketchPoint.geometry)
            radius = circle.radius * CM_TO_MM

            canonical_circle = Circle(
                center=center,
                radius=radius,
                construction=circle.isConstruction
            )

            prim_id = doc.add_primitive(canonical_circle)
            self._id_to_entity[prim_id] = circle
            self._entity_to_id[circle.entityToken] = prim_id

    def _export_points(self, doc: SketchDocument) -> None:
        """Export all sketch points from the sketch."""
        points = self._sketch.sketchPoints
        for i in range(points.count):
            point = points.item(i)

            # Skip origin point
            if point == self._sketch.originPoint:
                continue

            position = self._point3d_to_point2d(point.geometry)

            canonical_point = Point(
                position=position,
                construction=False  # Points don't have construction flag in Fusion
            )

            prim_id = doc.add_primitive(canonical_point)
            self._id_to_entity[prim_id] = point
            self._entity_to_id[point.entityToken] = prim_id

    def _export_splines(self, doc: SketchDocument) -> None:
        """Export all splines from the sketch."""
        # Export fitted splines
        fitted_splines = self._sketch.sketchCurves.sketchFittedSplines
        for i in range(fitted_splines.count):
            spline = fitted_splines.item(i)
            self._export_single_spline(doc, spline)

        # Export fixed splines (NURBS)
        fixed_splines = self._sketch.sketchCurves.sketchFixedSplines
        for i in range(fixed_splines.count):
            spline = fixed_splines.item(i)
            self._export_single_spline(doc, spline)

    def _export_single_spline(self, doc: SketchDocument, spline) -> None:
        """Export a single spline entity."""
        from sketch_canonical.types import Point2D

        # Get the NURBS data from the spline
        geom = spline.geometry
        nurbs = geom.asNurbsCurve

        # Extract control points
        _, control_points = nurbs.controlPoints
        poles = []
        for pt in control_points:
            poles.append(Point2D(pt.x * CM_TO_MM, pt.y * CM_TO_MM))

        # Extract knots
        _, knots = nurbs.knots
        knots = list(knots)

        # Extract degree
        degree = nurbs.degree

        # Extract weights if rational
        weights = None
        if nurbs.isRational:
            _, weights = nurbs.weights
            weights = list(weights)

        # Check if periodic
        periodic = nurbs.isPeriodic

        canonical_spline = Spline(
            poles=poles,
            degree=degree,
            knots=knots,
            weights=weights,
            periodic=periodic,
            construction=spline.isConstruction
        )

        prim_id = doc.add_primitive(canonical_spline)
        self._id_to_entity[prim_id] = spline
        self._entity_to_id[spline.entityToken] = prim_id

    def _export_geometric_constraints(self, doc: SketchDocument) -> None:
        """Export geometric constraints from the sketch."""
        constraints = self._sketch.geometricConstraints

        for i in range(constraints.count):
            constraint = constraints.item(i)
            canonical = self._convert_geometric_constraint(constraint)
            if canonical:
                doc.add_constraint(canonical)

    def _convert_geometric_constraint(self, constraint) -> Optional[SketchConstraint]:
        """Convert a Fusion geometric constraint to canonical form."""
        obj_type = constraint.objectType

        try:
            if "CoincidentConstraint" in obj_type:
                return self._convert_coincident(constraint)
            elif "HorizontalConstraint" in obj_type:
                return self._convert_horizontal(constraint)
            elif "VerticalConstraint" in obj_type:
                return self._convert_vertical(constraint)
            elif "ParallelConstraint" in obj_type:
                return self._convert_parallel(constraint)
            elif "PerpendicularConstraint" in obj_type:
                return self._convert_perpendicular(constraint)
            elif "TangentConstraint" in obj_type:
                return self._convert_tangent(constraint)
            elif "EqualConstraint" in obj_type:
                return self._convert_equal(constraint)
            elif "ConcentricConstraint" in obj_type:
                return self._convert_concentric(constraint)
            elif "CollinearConstraint" in obj_type:
                return self._convert_collinear(constraint)
            elif "FixConstraint" in obj_type:
                return self._convert_fixed(constraint)
            elif "SymmetryConstraint" in obj_type:
                return self._convert_symmetric(constraint)
            elif "MidPointConstraint" in obj_type:
                return self._convert_midpoint(constraint)
            else:
                # Unknown constraint type
                return None
        except Exception:
            return None

    def _get_id_for_entity(self, entity) -> Optional[str]:
        """Get the canonical ID for a Fusion entity."""
        token = entity.entityToken
        return self._entity_to_id.get(token)

    def _convert_coincident(self, constraint) -> Optional[SketchConstraint]:
        """Convert a coincident constraint."""
        # Get the two points involved
        pt1 = constraint.point
        pt2 = constraint.entity  # Could be point or curve

        # For point-to-point coincident
        if hasattr(pt2, "geometry"):
            # Both are points
            id1 = self._get_id_for_entity_or_parent(pt1)
            id2 = self._get_id_for_entity_or_parent(pt2)
            if id1 and id2:
                ref1 = self._point_to_ref(pt1, id1)
                ref2 = self._point_to_ref(pt2, id2)
                return SketchConstraint(
                    constraint_type=ConstraintType.COINCIDENT,
                    references=[ref1, ref2]
                )
        return None

    def _get_id_for_entity_or_parent(self, entity) -> Optional[str]:
        """Get ID for an entity, checking parent curve if it's a sketch point."""
        # First check if this entity has a direct mapping
        if hasattr(entity, "entityToken"):
            entity_id = self._entity_to_id.get(entity.entityToken)
            if entity_id:
                return entity_id

        # For sketch points that are part of curves, find the parent
        if hasattr(entity, "geometry") and hasattr(entity, "connectedEntities"):
            # It's a SketchPoint - find its parent curve
            for connected in entity.connectedEntities:
                if hasattr(connected, "entityToken"):
                    return self._entity_to_id.get(connected.entityToken)

        return None

    def _point_to_ref(self, point, element_id: str) -> PointRef:
        """Convert a Fusion SketchPoint to a PointRef."""
        # Determine which point type this is on its parent
        parent = None
        for connected in point.connectedEntities:
            if hasattr(connected, "entityToken"):
                if self._entity_to_id.get(connected.entityToken) == element_id:
                    parent = connected
                    break

        if not parent:
            return PointRef(element_id, PointType.CENTER)

        # Determine point type based on which property matches
        obj_type = parent.objectType
        if "SketchLine" in obj_type:
            if hasattr(parent, "startSketchPoint") and parent.startSketchPoint == point:
                return PointRef(element_id, PointType.START)
            elif hasattr(parent, "endSketchPoint") and parent.endSketchPoint == point:
                return PointRef(element_id, PointType.END)
        elif "SketchArc" in obj_type:
            if hasattr(parent, "startSketchPoint") and parent.startSketchPoint == point:
                return PointRef(element_id, PointType.START)
            elif hasattr(parent, "endSketchPoint") and parent.endSketchPoint == point:
                return PointRef(element_id, PointType.END)
            elif hasattr(parent, "centerSketchPoint") and parent.centerSketchPoint == point:
                return PointRef(element_id, PointType.CENTER)
        elif "SketchCircle" in obj_type:
            return PointRef(element_id, PointType.CENTER)

        return PointRef(element_id, PointType.CENTER)

    def _convert_horizontal(self, constraint) -> Optional[SketchConstraint]:
        """Convert a horizontal constraint."""
        entity = constraint.line
        entity_id = self._get_id_for_entity(entity)
        if entity_id:
            return SketchConstraint(
                constraint_type=ConstraintType.HORIZONTAL,
                references=[entity_id]
            )
        return None

    def _convert_vertical(self, constraint) -> Optional[SketchConstraint]:
        """Convert a vertical constraint."""
        entity = constraint.line
        entity_id = self._get_id_for_entity(entity)
        if entity_id:
            return SketchConstraint(
                constraint_type=ConstraintType.VERTICAL,
                references=[entity_id]
            )
        return None

    def _convert_parallel(self, constraint) -> Optional[SketchConstraint]:
        """Convert a parallel constraint."""
        line1 = constraint.lineOne
        line2 = constraint.lineTwo
        id1 = self._get_id_for_entity(line1)
        id2 = self._get_id_for_entity(line2)
        if id1 and id2:
            return SketchConstraint(
                constraint_type=ConstraintType.PARALLEL,
                references=[id1, id2]
            )
        return None

    def _convert_perpendicular(self, constraint) -> Optional[SketchConstraint]:
        """Convert a perpendicular constraint."""
        line1 = constraint.lineOne
        line2 = constraint.lineTwo
        id1 = self._get_id_for_entity(line1)
        id2 = self._get_id_for_entity(line2)
        if id1 and id2:
            return SketchConstraint(
                constraint_type=ConstraintType.PERPENDICULAR,
                references=[id1, id2]
            )
        return None

    def _convert_tangent(self, constraint) -> Optional[SketchConstraint]:
        """Convert a tangent constraint."""
        curve1 = constraint.curveOne
        curve2 = constraint.curveTwo
        id1 = self._get_id_for_entity(curve1)
        id2 = self._get_id_for_entity(curve2)
        if id1 and id2:
            return SketchConstraint(
                constraint_type=ConstraintType.TANGENT,
                references=[id1, id2]
            )
        return None

    def _convert_equal(self, constraint) -> Optional[SketchConstraint]:
        """Convert an equal constraint."""
        curve1 = constraint.curveOne
        curve2 = constraint.curveTwo
        id1 = self._get_id_for_entity(curve1)
        id2 = self._get_id_for_entity(curve2)
        if id1 and id2:
            return SketchConstraint(
                constraint_type=ConstraintType.EQUAL,
                references=[id1, id2]
            )
        return None

    def _convert_concentric(self, constraint) -> Optional[SketchConstraint]:
        """Convert a concentric constraint."""
        entity1 = constraint.entityOne
        entity2 = constraint.entityTwo
        id1 = self._get_id_for_entity(entity1)
        id2 = self._get_id_for_entity(entity2)
        if id1 and id2:
            return SketchConstraint(
                constraint_type=ConstraintType.CONCENTRIC,
                references=[id1, id2]
            )
        return None

    def _convert_collinear(self, constraint) -> Optional[SketchConstraint]:
        """Convert a collinear constraint."""
        line1 = constraint.lineOne
        line2 = constraint.lineTwo
        id1 = self._get_id_for_entity(line1)
        id2 = self._get_id_for_entity(line2)
        if id1 and id2:
            return SketchConstraint(
                constraint_type=ConstraintType.COLLINEAR,
                references=[id1, id2]
            )
        return None

    def _convert_fixed(self, constraint) -> Optional[SketchConstraint]:
        """Convert a fixed constraint."""
        entity = constraint.entity
        entity_id = self._get_id_for_entity(entity)
        if entity_id:
            return SketchConstraint(
                constraint_type=ConstraintType.FIXED,
                references=[entity_id]
            )
        return None

    def _convert_symmetric(self, constraint) -> Optional[SketchConstraint]:
        """Convert a symmetry constraint."""
        entity1 = constraint.entityOne
        entity2 = constraint.entityTwo
        line = constraint.symmetryLine
        id1 = self._get_id_for_entity(entity1)
        id2 = self._get_id_for_entity(entity2)
        line_id = self._get_id_for_entity(line)
        if id1 and id2 and line_id:
            return SketchConstraint(
                constraint_type=ConstraintType.SYMMETRIC,
                references=[id1, id2, line_id]
            )
        return None

    def _convert_midpoint(self, constraint) -> Optional[SketchConstraint]:
        """Convert a midpoint constraint."""
        point = constraint.point
        line = constraint.midPointCurve
        point_id = self._get_id_for_entity_or_parent(point)
        line_id = self._get_id_for_entity(line)
        if point_id and line_id:
            ref = self._point_to_ref(point, point_id)
            return SketchConstraint(
                constraint_type=ConstraintType.MIDPOINT,
                references=[ref, line_id]
            )
        return None

    def _export_dimensional_constraints(self, doc: SketchDocument) -> None:
        """Export dimensional constraints from the sketch."""
        dims = self._sketch.sketchDimensions

        for i in range(dims.count):
            dim = dims.item(i)
            canonical = self._convert_dimensional_constraint(dim)
            if canonical:
                doc.add_constraint(canonical)

    def _convert_dimensional_constraint(self, dim) -> Optional[SketchConstraint]:
        """Convert a Fusion dimensional constraint to canonical form."""
        obj_type = dim.objectType

        try:
            # Get the dimension value in mm
            value_cm = dim.parameter.value
            value_mm = value_cm * CM_TO_MM

            if "SketchLinearDimension" in obj_type:
                return self._convert_linear_dimension(dim, value_mm)
            elif "SketchRadialDimension" in obj_type:
                return self._convert_radial_dimension(dim, value_mm)
            elif "SketchDiameterDimension" in obj_type:
                return self._convert_diameter_dimension(dim, value_mm)
            elif "SketchAngularDimension" in obj_type:
                return self._convert_angular_dimension(dim)
            else:
                return None
        except Exception:
            return None

    def _convert_linear_dimension(self, dim, value: float) -> Optional[SketchConstraint]:
        """Convert a linear dimension constraint."""
        # Determine if it's distance, length, or offset dimension
        orientation = dim.orientation

        entity1 = dim.entityOne
        entity2 = dim.entityTwo

        # If both entities are points, it's a distance constraint
        if entity2 is not None:
            id1 = self._get_id_for_entity_or_parent(entity1)
            id2 = self._get_id_for_entity_or_parent(entity2)
            if id1 and id2:
                ref1 = self._point_to_ref(entity1, id1) if hasattr(entity1, "geometry") else id1
                ref2 = self._point_to_ref(entity2, id2) if hasattr(entity2, "geometry") else id2

                # Check orientation for X/Y constraints
                if orientation == self._adsk_fusion.DimensionOrientations.HorizontalDimensionOrientation:
                    return SketchConstraint(
                        constraint_type=ConstraintType.DISTANCE_X,
                        references=[ref1, ref2],
                        value=value
                    )
                elif orientation == self._adsk_fusion.DimensionOrientations.VerticalDimensionOrientation:
                    return SketchConstraint(
                        constraint_type=ConstraintType.DISTANCE_Y,
                        references=[ref1, ref2],
                        value=value
                    )
                else:
                    return SketchConstraint(
                        constraint_type=ConstraintType.DISTANCE,
                        references=[ref1, ref2],
                        value=value
                    )
        else:
            # Single entity - could be length
            entity_id = self._get_id_for_entity(entity1)
            if entity_id:
                return SketchConstraint(
                    constraint_type=ConstraintType.LENGTH,
                    references=[entity_id],
                    value=value
                )

        return None

    def _convert_radial_dimension(self, dim, value: float) -> Optional[SketchConstraint]:
        """Convert a radial dimension constraint."""
        entity = dim.entity
        entity_id = self._get_id_for_entity(entity)
        if entity_id:
            return SketchConstraint(
                constraint_type=ConstraintType.RADIUS,
                references=[entity_id],
                value=value
            )
        return None

    def _convert_diameter_dimension(self, dim, value: float) -> Optional[SketchConstraint]:
        """Convert a diameter dimension constraint."""
        entity = dim.entity
        entity_id = self._get_id_for_entity(entity)
        if entity_id:
            return SketchConstraint(
                constraint_type=ConstraintType.DIAMETER,
                references=[entity_id],
                value=value
            )
        return None

    def _convert_angular_dimension(self, dim) -> Optional[SketchConstraint]:
        """Convert an angular dimension constraint."""
        # Value is in radians, convert to degrees
        value_rad = dim.parameter.value
        value_deg = math.degrees(value_rad)

        line1 = dim.lineOne
        line2 = dim.lineTwo
        id1 = self._get_id_for_entity(line1)
        id2 = self._get_id_for_entity(line2)

        if id1 and id2:
            return SketchConstraint(
                constraint_type=ConstraintType.ANGLE,
                references=[id1, id2],
                value=value_deg
            )
        return None
