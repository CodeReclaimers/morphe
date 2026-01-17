"""Autodesk Inventor adapter for canonical sketch representation.

This module provides the InventorAdapter class that implements the
SketchBackendAdapter interface for Autodesk Inventor.

Note: Inventor internally uses centimeters, while the canonical format
uses millimeters. This adapter handles the conversion automatically.

This adapter uses the COM API via win32com, which requires:
- Windows operating system
- Autodesk Inventor installed
- pywin32 package installed (pip install pywin32)
"""

import math
from typing import Any

from morphe import (
    Arc,
    Circle,
    ConstraintError,
    ConstraintType,
    Ellipse,
    EllipticalArc,
    ExportError,
    GeometryError,
    Line,
    Point,
    Point2D,
    PointRef,
    SketchBackendAdapter,
    SketchConstraint,
    SketchCreationError,
    SketchDocument,
    SketchPrimitive,
    SolverStatus,
    Spline,
)

from .vertex_map import get_point_type_for_sketch_point, get_sketch_point_from_entity

# Inventor uses centimeters internally, canonical format uses millimeters
MM_TO_CM = 0.1
CM_TO_MM = 10.0

# Try to import win32com for COM automation
INVENTOR_AVAILABLE = False
_inventor_app = None

try:
    import win32com.client
    INVENTOR_AVAILABLE = True
except ImportError:
    win32com = None


def get_inventor_application():
    """Get or create connection to Inventor application.

    Returns:
        Inventor Application COM object

    Raises:
        ImportError: If win32com is not available
        ConnectionError: If Inventor is not running or cannot be connected
    """
    global _inventor_app

    if not INVENTOR_AVAILABLE:
        raise ImportError(
            "win32com is not available. Install with: pip install pywin32"
        )

    if _inventor_app is not None:
        try:
            # Test if connection is still valid
            _ = _inventor_app.Documents
            return _inventor_app
        except Exception:
            _inventor_app = None

    try:
        # Try to connect to running Inventor instance
        _inventor_app = win32com.client.GetActiveObject("Inventor.Application")
        return _inventor_app
    except Exception:
        pass

    try:
        # Try to start new Inventor instance
        _inventor_app = win32com.client.Dispatch("Inventor.Application")
        _inventor_app.Visible = True
        return _inventor_app
    except Exception as e:
        raise ConnectionError(
            f"Could not connect to Autodesk Inventor. "
            f"Ensure Inventor is installed and running. Error: {e}"
        ) from e


class InventorAdapter(SketchBackendAdapter):
    """Autodesk Inventor implementation of SketchBackendAdapter.

    This adapter translates between the canonical sketch representation
    and Inventor's native sketch API via COM automation.

    Attributes:
        _app: Inventor Application COM object
        _document: Active Inventor part document
        _sketch: Current active sketch
        _id_to_entity: Mapping from canonical IDs to Inventor sketch entities
        _entity_to_id: Mapping from Inventor entities to canonical IDs
    """

    def __init__(self, document: Any | None = None):
        """Initialize the Inventor adapter.

        Args:
            document: Optional Inventor document. If None, creates a new part.

        Raises:
            ImportError: If win32com is not available
            ConnectionError: If Inventor cannot be connected
        """
        self._app = get_inventor_application()

        if document is not None:
            self._document = document
        else:
            self._document = None

        self._sketch = None
        self._sketch_def = None  # PlanarSketch object
        self._id_to_entity: dict[str, Any] = {}
        self._entity_to_id: dict[int, str] = {}  # Use hash for COM objects
        self._ground_constraints: set[str] = set()  # Track grounded entities

    def _ensure_document(self) -> None:
        """Ensure we have an active part document."""
        if self._document is None:
            # Create a new part document
            self._document = self._app.Documents.Add(
                12163,  # kPartDocumentObject
                "",     # Default template
                True    # Create visible
            )

    def create_sketch(self, name: str, plane: Any | None = None) -> None:
        """Create a new sketch in Inventor.

        Args:
            name: Name for the new sketch
            plane: Optional plane specification. Can be:
                - None: Uses XY work plane
                - "XY", "XZ", "YZ": Standard work planes
                - An Inventor WorkPlane or Face object

        Raises:
            SketchCreationError: If sketch creation fails
        """
        try:
            self._ensure_document()

            part_def = self._document.ComponentDefinition
            sketches = part_def.Sketches

            # Determine the plane to use
            if plane is None or plane == "XY":
                work_planes = part_def.WorkPlanes
                sketch_plane = work_planes.Item(3)  # XY plane (index 3)
            elif plane == "XZ":
                work_planes = part_def.WorkPlanes
                sketch_plane = work_planes.Item(2)  # XZ plane (index 2)
            elif plane == "YZ":
                work_planes = part_def.WorkPlanes
                sketch_plane = work_planes.Item(1)  # YZ plane (index 1)
            else:
                sketch_plane = plane

            new_sketch = sketches.Add(sketch_plane)
            new_sketch.Name = name
            self._sketch_def = new_sketch
            self._sketch = new_sketch

            # Clear mappings for new sketch
            self._id_to_entity.clear()
            self._entity_to_id.clear()
            self._ground_constraints.clear()

        except Exception as e:
            raise SketchCreationError(f"Failed to create sketch: {e}") from e

    def load_sketch(self, sketch: SketchDocument) -> None:
        """Load a SketchDocument into a new Inventor sketch.

        Args:
            sketch: The SketchDocument to load

        Raises:
            SketchCreationError: If sketch creation fails
            GeometryError: If geometry creation fails
            ConstraintError: If constraint creation fails
        """
        # Create the sketch if not already created
        if self._sketch is None:
            self.create_sketch(sketch.name)

        # Add all primitives
        for _prim_id, primitive in sketch.primitives.items():
            self.add_primitive(primitive)

        # Add all constraints
        for constraint in sketch.constraints:
            try:
                self.add_constraint(constraint)
            except ConstraintError:
                # Log but continue - some constraints may fail
                pass

    def export_sketch(self) -> SketchDocument:
        """Export the current Inventor sketch to canonical form.

        Returns:
            A new SketchDocument containing the canonical representation.

        Raises:
            ExportError: If export fails
        """
        if self._sketch is None:
            raise ExportError("No active sketch to export")

        try:
            sketch = self._sketch  # Local reference for mypy
            doc = SketchDocument(name=sketch.Name)

            # Clear and rebuild mappings
            self._id_to_entity.clear()
            self._entity_to_id.clear()

            # Export lines
            for line in sketch.SketchLines:
                if self._is_reference_geometry(line):
                    continue
                prim = self._export_line(line)
                doc.add_primitive(prim)
                self._entity_to_id[id(line)] = prim.id
                self._id_to_entity[prim.id] = line

            # Export circles
            for circle in sketch.SketchCircles:
                if self._is_reference_geometry(circle):
                    continue
                prim = self._export_circle(circle)
                doc.add_primitive(prim)
                self._entity_to_id[id(circle)] = prim.id
                self._id_to_entity[prim.id] = circle

            # Export arcs
            for arc in sketch.SketchArcs:
                if self._is_reference_geometry(arc):
                    continue
                prim = self._export_arc(arc)
                doc.add_primitive(prim)
                self._entity_to_id[id(arc)] = prim.id
                self._id_to_entity[prim.id] = arc

            # Export ellipses
            for ellipse in sketch.SketchEllipses:
                if self._is_reference_geometry(ellipse):
                    continue
                prim = self._export_ellipse(ellipse)
                doc.add_primitive(prim)
                self._entity_to_id[id(ellipse)] = prim.id
                self._id_to_entity[prim.id] = ellipse

            # Export elliptical arcs
            for elliptical_arc in sketch.SketchEllipticalArcs:
                if self._is_reference_geometry(elliptical_arc):
                    continue
                prim = self._export_elliptical_arc(elliptical_arc)
                doc.add_primitive(prim)
                self._entity_to_id[id(elliptical_arc)] = prim.id
                self._id_to_entity[prim.id] = elliptical_arc

            # Export points
            for point in sketch.SketchPoints:
                if self._is_reference_geometry(point):
                    continue
                # Skip points that are part of other geometry
                if self._is_dependent_point(point):
                    continue
                prim = self._export_point(point)
                doc.add_primitive(prim)
                self._entity_to_id[id(point)] = prim.id
                self._id_to_entity[prim.id] = point

            # Export constraints
            self._export_geometric_constraints(doc)
            self._export_dimension_constraints(doc)

            # Get solver status
            status, dof = self.get_solver_status()
            doc.solver_status = status
            doc.degrees_of_freedom = dof

            return doc

        except Exception as e:
            raise ExportError(f"Failed to export sketch: {e}") from e

    def _is_reference_geometry(self, entity: Any) -> bool:
        """Check if entity is reference/projected geometry."""
        try:
            return bool(entity.Reference)
        except Exception:
            return False

    def _is_dependent_point(self, point: Any) -> bool:
        """Check if a point is dependent on other geometry (e.g., line endpoint)."""
        try:
            # Points with non-empty DependentObjects are part of other geometry
            return bool(point.DependentObjects.Count > 0)
        except Exception:
            return False

    def add_primitive(self, primitive: SketchPrimitive) -> Any:
        """Add a single primitive to the sketch.

        Args:
            primitive: The canonical primitive to add

        Returns:
            Inventor sketch entity

        Raises:
            GeometryError: If geometry creation fails
        """
        if self._sketch is None:
            raise GeometryError("No active sketch")

        try:
            if isinstance(primitive, Line):
                entity = self._add_line(primitive)
            elif isinstance(primitive, Circle):
                entity = self._add_circle(primitive)
            elif isinstance(primitive, Arc):
                entity = self._add_arc(primitive)
            elif isinstance(primitive, Point):
                entity = self._add_point(primitive)
            elif isinstance(primitive, Spline):
                entity = self._add_spline(primitive)
            elif isinstance(primitive, Ellipse):
                entity = self._add_ellipse(primitive)
            elif isinstance(primitive, EllipticalArc):
                entity = self._add_elliptical_arc(primitive)
            else:
                raise GeometryError(f"Unsupported primitive type: {type(primitive)}")

            # Store mapping
            self._id_to_entity[primitive.id] = entity
            self._entity_to_id[id(entity)] = primitive.id

            # Set construction mode if needed
            if primitive.construction:
                try:
                    entity.Construction = True
                except Exception:
                    pass  # Some entities may not support construction mode

            return entity

        except Exception as e:
            raise GeometryError(f"Failed to add {type(primitive).__name__}: {e}") from e

    def _add_line(self, line: Line) -> Any:
        """Add a line to the sketch."""
        assert self._sketch is not None
        lines = self._sketch.SketchLines
        start = self._app.TransientGeometry.CreatePoint2d(
            line.start.x * MM_TO_CM,
            line.start.y * MM_TO_CM
        )
        end = self._app.TransientGeometry.CreatePoint2d(
            line.end.x * MM_TO_CM,
            line.end.y * MM_TO_CM
        )
        return lines.AddByTwoPoints(start, end)

    def _add_circle(self, circle: Circle) -> Any:
        """Add a circle to the sketch."""
        assert self._sketch is not None
        circles = self._sketch.SketchCircles
        center = self._app.TransientGeometry.CreatePoint2d(
            circle.center.x * MM_TO_CM,
            circle.center.y * MM_TO_CM
        )
        radius_cm = circle.radius * MM_TO_CM
        return circles.AddByCenterRadius(center, radius_cm)

    def _add_arc(self, arc: Arc) -> Any:
        """Add an arc to the sketch."""
        assert self._sketch is not None
        arcs = self._sketch.SketchArcs
        center = self._app.TransientGeometry.CreatePoint2d(
            arc.center.x * MM_TO_CM,
            arc.center.y * MM_TO_CM
        )
        start_pt = self._app.TransientGeometry.CreatePoint2d(
            arc.start_point.x * MM_TO_CM,
            arc.start_point.y * MM_TO_CM
        )
        end_pt = self._app.TransientGeometry.CreatePoint2d(
            arc.end_point.x * MM_TO_CM,
            arc.end_point.y * MM_TO_CM
        )

        # Inventor's AddByCenterStartEndPoint expects counterclockwise direction
        # If arc is clockwise, we swap start and end
        if arc.ccw:
            return arcs.AddByCenterStartEndPoint(center, start_pt, end_pt)
        else:
            return arcs.AddByCenterStartEndPoint(center, end_pt, start_pt)

    def _add_point(self, point: Point) -> Any:
        """Add a point to the sketch."""
        assert self._sketch is not None
        points = self._sketch.SketchPoints
        pt = self._app.TransientGeometry.CreatePoint2d(
            point.position.x * MM_TO_CM,
            point.position.y * MM_TO_CM
        )
        return points.Add(pt)

    def _add_spline(self, spline: Spline) -> Any:
        """Add a spline to the sketch."""
        assert self._sketch is not None
        splines = self._sketch.SketchSplines

        # Create fit points array
        fit_points = self._app.TransientObjects.CreateObjectCollection()
        for pt in spline.control_points:
            point = self._app.TransientGeometry.CreatePoint2d(
                pt.x * MM_TO_CM,
                pt.y * MM_TO_CM
            )
            fit_points.Add(point)

        # Use fit points method (simpler than control point method)
        # For more accurate B-spline, would need SplineControlPointDefinitions
        return splines.Add(fit_points)

    def _add_ellipse(self, ellipse: Ellipse) -> Any:
        """Add an ellipse to the sketch."""
        assert self._sketch is not None
        ellipses = self._sketch.SketchEllipses

        center = self._app.TransientGeometry.CreatePoint2d(
            ellipse.center.x * MM_TO_CM,
            ellipse.center.y * MM_TO_CM
        )

        # Calculate major axis endpoint
        major_axis_x = ellipse.major_radius * math.cos(ellipse.rotation)
        major_axis_y = ellipse.major_radius * math.sin(ellipse.rotation)
        major_pt = self._app.TransientGeometry.CreatePoint2d(
            (ellipse.center.x + major_axis_x) * MM_TO_CM,
            (ellipse.center.y + major_axis_y) * MM_TO_CM
        )

        # Calculate minor axis endpoint (perpendicular to major)
        minor_axis_x = ellipse.minor_radius * math.cos(ellipse.rotation + math.pi / 2)
        minor_axis_y = ellipse.minor_radius * math.sin(ellipse.rotation + math.pi / 2)
        minor_pt = self._app.TransientGeometry.CreatePoint2d(
            (ellipse.center.x + minor_axis_x) * MM_TO_CM,
            (ellipse.center.y + minor_axis_y) * MM_TO_CM
        )

        # Inventor's AddByThreePoints(center, majorAxisPoint, minorAxisPoint)
        return ellipses.Add(center, major_pt, minor_pt)

    def _add_elliptical_arc(self, arc: EllipticalArc) -> Any:
        """Add an elliptical arc to the sketch."""
        assert self._sketch is not None
        elliptical_arcs = self._sketch.SketchEllipticalArcs

        center = self._app.TransientGeometry.CreatePoint2d(
            arc.center.x * MM_TO_CM,
            arc.center.y * MM_TO_CM
        )

        # Calculate major axis endpoint
        major_axis_x = arc.major_radius * math.cos(arc.rotation)
        major_axis_y = arc.major_radius * math.sin(arc.rotation)
        major_pt = self._app.TransientGeometry.CreatePoint2d(
            (arc.center.x + major_axis_x) * MM_TO_CM,
            (arc.center.y + major_axis_y) * MM_TO_CM
        )

        # Minor axis ratio
        minor_ratio = arc.minor_radius / arc.major_radius

        # Calculate start and end points
        start_pt_canonical = arc.start_point
        end_pt_canonical = arc.end_point

        start_pt = self._app.TransientGeometry.CreatePoint2d(
            start_pt_canonical.x * MM_TO_CM,
            start_pt_canonical.y * MM_TO_CM
        )
        end_pt = self._app.TransientGeometry.CreatePoint2d(
            end_pt_canonical.x * MM_TO_CM,
            end_pt_canonical.y * MM_TO_CM
        )

        # Inventor's AddByMajorMinorAxisStartEnd(center, majorAxisPoint, minorRatio, startPoint, endPoint)
        return elliptical_arcs.Add(center, major_pt, minor_ratio, start_pt, end_pt)

    # =========================================================================
    # Constraint Methods
    # =========================================================================

    def add_constraint(self, constraint: SketchConstraint) -> bool:
        """Add a constraint to the sketch.

        Args:
            constraint: The canonical constraint to add

        Returns:
            True if successful

        Raises:
            ConstraintError: If constraint creation fails
        """
        if self._sketch is None:
            raise ConstraintError("No active sketch")

        try:
            ctype = constraint.constraint_type
            refs = constraint.references
            value = constraint.value

            geom_constraints = self._sketch.GeometricConstraints
            dim_constraints = self._sketch.DimensionConstraints

            # Geometric constraints
            if ctype == ConstraintType.COINCIDENT:
                return self._add_coincident(geom_constraints, refs)
            elif ctype == ConstraintType.TANGENT:
                return self._add_tangent(geom_constraints, refs)
            elif ctype == ConstraintType.PERPENDICULAR:
                return self._add_perpendicular(geom_constraints, refs)
            elif ctype == ConstraintType.PARALLEL:
                return self._add_parallel(geom_constraints, refs)
            elif ctype == ConstraintType.HORIZONTAL:
                return self._add_horizontal(geom_constraints, refs)
            elif ctype == ConstraintType.VERTICAL:
                return self._add_vertical(geom_constraints, refs)
            elif ctype == ConstraintType.EQUAL:
                return self._add_equal(geom_constraints, refs)
            elif ctype == ConstraintType.CONCENTRIC:
                return self._add_concentric(geom_constraints, refs)
            elif ctype == ConstraintType.COLLINEAR:
                return self._add_collinear(geom_constraints, refs)
            elif ctype == ConstraintType.SYMMETRIC:
                return self._add_symmetric(geom_constraints, refs)
            elif ctype == ConstraintType.MIDPOINT:
                return self._add_midpoint(geom_constraints, refs)
            elif ctype == ConstraintType.FIXED:
                return self._add_ground(geom_constraints, refs)

            # Dimensional constraints
            elif ctype == ConstraintType.DISTANCE:
                return self._add_distance(dim_constraints, refs, value)
            elif ctype == ConstraintType.DISTANCE_X:
                return self._add_distance_x(dim_constraints, refs, value)
            elif ctype == ConstraintType.DISTANCE_Y:
                return self._add_distance_y(dim_constraints, refs, value)
            elif ctype == ConstraintType.LENGTH:
                return self._add_length(dim_constraints, refs, value)
            elif ctype == ConstraintType.RADIUS:
                return self._add_radius(dim_constraints, refs, value)
            elif ctype == ConstraintType.DIAMETER:
                return self._add_diameter(dim_constraints, refs, value)
            elif ctype == ConstraintType.ANGLE:
                return self._add_angle(dim_constraints, refs, value)

            else:
                raise ConstraintError(f"Unsupported constraint type: {ctype}")

        except Exception as e:
            raise ConstraintError(f"Failed to add constraint: {e}") from e

    def _get_entity(self, ref: str | PointRef) -> Any:
        """Get Inventor entity from reference."""
        if isinstance(ref, PointRef):
            entity_id = ref.element_id
        else:
            entity_id = ref

        entity = self._id_to_entity.get(entity_id)
        if entity is None:
            raise ConstraintError(f"Unknown entity: {entity_id}")
        return entity

    def _get_sketch_point(self, ref: PointRef) -> Any:
        """Get Inventor SketchPoint from PointRef."""
        entity = self._get_entity(ref)
        return get_sketch_point_from_entity(entity, ref.point_type)

    # Geometric constraint implementations

    def _add_coincident(self, constraints: Any, refs: list) -> bool:
        """Add a coincident constraint."""
        if len(refs) < 2:
            raise ConstraintError("Coincident requires 2 references")

        pt1 = self._get_sketch_point(refs[0])
        pt2 = self._get_sketch_point(refs[1])
        constraints.AddCoincident(pt1, pt2)
        return True

    def _add_tangent(self, constraints: Any, refs: list) -> bool:
        """Add a tangent constraint."""
        if len(refs) < 2:
            raise ConstraintError("Tangent requires 2 references")

        entity1 = self._get_entity(refs[0])
        entity2 = self._get_entity(refs[1])
        constraints.AddTangent(entity1, entity2)
        return True

    def _add_perpendicular(self, constraints: Any, refs: list) -> bool:
        """Add a perpendicular constraint."""
        if len(refs) < 2:
            raise ConstraintError("Perpendicular requires 2 references")

        entity1 = self._get_entity(refs[0])
        entity2 = self._get_entity(refs[1])
        constraints.AddPerpendicular(entity1, entity2)
        return True

    def _add_parallel(self, constraints: Any, refs: list) -> bool:
        """Add a parallel constraint."""
        if len(refs) < 2:
            raise ConstraintError("Parallel requires 2 references")

        entity1 = self._get_entity(refs[0])
        entity2 = self._get_entity(refs[1])
        constraints.AddParallel(entity1, entity2)
        return True

    def _add_horizontal(self, constraints: Any, refs: list) -> bool:
        """Add a horizontal constraint."""
        if len(refs) < 1:
            raise ConstraintError("Horizontal requires 1 reference")

        entity = self._get_entity(refs[0])
        constraints.AddHorizontal(entity)
        return True

    def _add_vertical(self, constraints: Any, refs: list) -> bool:
        """Add a vertical constraint."""
        if len(refs) < 1:
            raise ConstraintError("Vertical requires 1 reference")

        entity = self._get_entity(refs[0])
        constraints.AddVertical(entity)
        return True

    def _add_equal(self, constraints: Any, refs: list) -> bool:
        """Add an equal constraint (length or radius)."""
        if len(refs) < 2:
            raise ConstraintError("Equal requires at least 2 references")

        first = self._get_entity(refs[0])

        # Chain equal constraints for multiple elements
        for i in range(1, len(refs)):
            other = self._get_entity(refs[i])
            # Inventor uses different methods for lines vs circles
            try:
                # Try EqualLength first (for lines)
                constraints.AddEqualLength(first, other)
            except Exception:
                try:
                    # Try EqualRadius (for circles/arcs)
                    constraints.AddEqualRadius(first, other)
                except Exception as err:
                    raise ConstraintError(
                        "Could not create equal constraint between entities"
                    ) from err
        return True

    def _add_concentric(self, constraints: Any, refs: list) -> bool:
        """Add a concentric constraint."""
        if len(refs) < 2:
            raise ConstraintError("Concentric requires 2 references")

        entity1 = self._get_entity(refs[0])
        entity2 = self._get_entity(refs[1])
        constraints.AddConcentric(entity1, entity2)
        return True

    def _add_collinear(self, constraints: Any, refs: list) -> bool:
        """Add a collinear constraint."""
        if len(refs) < 2:
            raise ConstraintError("Collinear requires at least 2 references")

        first = self._get_entity(refs[0])
        for i in range(1, len(refs)):
            other = self._get_entity(refs[i])
            constraints.AddCollinear(first, other)
        return True

    def _add_symmetric(self, constraints: Any, refs: list) -> bool:
        """Add a symmetric constraint."""
        if len(refs) < 3:
            raise ConstraintError("Symmetric requires 3 references")

        # refs[0], refs[1] are the elements to be symmetric
        # refs[2] is the symmetry axis
        if isinstance(refs[0], PointRef):
            entity1 = self._get_sketch_point(refs[0])
        else:
            entity1 = self._get_entity(refs[0])

        if isinstance(refs[1], PointRef):
            entity2 = self._get_sketch_point(refs[1])
        else:
            entity2 = self._get_entity(refs[1])

        axis = self._get_entity(refs[2])
        constraints.AddSymmetry(entity1, entity2, axis)
        return True

    def _add_midpoint(self, constraints: Any, refs: list) -> bool:
        """Add a midpoint constraint."""
        if len(refs) != 2:
            raise ConstraintError("Midpoint requires exactly 2 references")

        # Determine which is point and which is line
        ref0 = refs[0]
        ref1 = refs[1]

        if isinstance(ref0, PointRef):
            point = self._get_sketch_point(ref0)
            line = self._get_entity(ref1)
        elif isinstance(ref1, PointRef):
            point = self._get_sketch_point(ref1)
            line = self._get_entity(ref0)
        else:
            raise ConstraintError("Midpoint requires one point reference")

        constraints.AddMidpoint(point, line)
        return True

    def _add_ground(self, constraints: Any, refs: list) -> bool:
        """Add a ground (fixed) constraint."""
        if len(refs) < 1:
            raise ConstraintError("Ground requires 1 reference")

        entity = self._get_entity(refs[0])
        constraints.AddGround(entity)
        self._ground_constraints.add(refs[0] if isinstance(refs[0], str) else refs[0].element_id)
        return True

    # Dimensional constraint implementations

    def _add_distance(self, constraints: Any, refs: list, value: float | None) -> bool:
        """Add a distance constraint between two points."""
        if value is None or len(refs) < 2:
            raise ConstraintError("Distance requires 2 references and a value")

        pt1 = self._get_sketch_point(refs[0])
        pt2 = self._get_sketch_point(refs[1])

        # Create dimension at midpoint
        mid_x = (pt1.Geometry.X + pt2.Geometry.X) / 2
        mid_y = (pt1.Geometry.Y + pt2.Geometry.Y) / 2
        dim_pos = self._app.TransientGeometry.CreatePoint2d(mid_x, mid_y + 1.0)

        dim = constraints.AddTwoPointDistance(pt1, pt2, dim_pos)
        dim.Parameter.Value = value * MM_TO_CM
        return True

    def _add_distance_x(self, constraints: Any, refs: list, value: float | None) -> bool:
        """Add a horizontal distance constraint."""
        assert self._sketch is not None
        if value is None:
            raise ConstraintError("DistanceX requires a value")

        if len(refs) == 1:
            # Distance from origin
            pt = self._get_sketch_point(refs[0])
            origin = self._sketch.OriginPoint
            dim_pos = self._app.TransientGeometry.CreatePoint2d(
                pt.Geometry.X / 2, pt.Geometry.Y + 1.0
            )
            dim = constraints.AddTwoPointDistance(origin, pt, dim_pos)
        else:
            pt1 = self._get_sketch_point(refs[0])
            pt2 = self._get_sketch_point(refs[1])
            dim_pos = self._app.TransientGeometry.CreatePoint2d(
                (pt1.Geometry.X + pt2.Geometry.X) / 2,
                max(pt1.Geometry.Y, pt2.Geometry.Y) + 1.0
            )
            dim = constraints.AddTwoPointDistance(pt1, pt2, dim_pos)

        dim.Parameter.Value = abs(value) * MM_TO_CM
        return True

    def _add_distance_y(self, constraints: Any, refs: list, value: float | None) -> bool:
        """Add a vertical distance constraint."""
        assert self._sketch is not None
        if value is None:
            raise ConstraintError("DistanceY requires a value")

        if len(refs) == 1:
            # Distance from origin
            pt = self._get_sketch_point(refs[0])
            origin = self._sketch.OriginPoint
            dim_pos = self._app.TransientGeometry.CreatePoint2d(
                pt.Geometry.X + 1.0, pt.Geometry.Y / 2
            )
            dim = constraints.AddTwoPointDistance(origin, pt, dim_pos)
        else:
            pt1 = self._get_sketch_point(refs[0])
            pt2 = self._get_sketch_point(refs[1])
            dim_pos = self._app.TransientGeometry.CreatePoint2d(
                max(pt1.Geometry.X, pt2.Geometry.X) + 1.0,
                (pt1.Geometry.Y + pt2.Geometry.Y) / 2
            )
            dim = constraints.AddTwoPointDistance(pt1, pt2, dim_pos)

        dim.Parameter.Value = abs(value) * MM_TO_CM
        return True

    def _add_length(self, constraints: Any, refs: list, value: float | None) -> bool:
        """Add a length constraint to a line."""
        if value is None or len(refs) < 1:
            raise ConstraintError("Length requires 1 reference and a value")

        line = self._get_entity(refs[0])
        # Position dimension above the line
        mid_x = (line.StartSketchPoint.Geometry.X + line.EndSketchPoint.Geometry.X) / 2
        mid_y = (line.StartSketchPoint.Geometry.Y + line.EndSketchPoint.Geometry.Y) / 2
        dim_pos = self._app.TransientGeometry.CreatePoint2d(mid_x, mid_y + 1.0)

        dim = constraints.AddLinearDimension(line, dim_pos)
        dim.Parameter.Value = value * MM_TO_CM
        return True

    def _add_radius(self, constraints: Any, refs: list, value: float | None) -> bool:
        """Add a radius constraint."""
        if value is None or len(refs) < 1:
            raise ConstraintError("Radius requires 1 reference and a value")

        entity = self._get_entity(refs[0])
        # Position dimension outside the arc/circle
        try:
            center = entity.CenterSketchPoint.Geometry
        except Exception:
            center = entity.Geometry.Center
        dim_pos = self._app.TransientGeometry.CreatePoint2d(
            center.X + value * MM_TO_CM * 1.5,
            center.Y
        )

        dim = constraints.AddRadiusDimension(entity, dim_pos)
        dim.Parameter.Value = value * MM_TO_CM
        return True

    def _add_diameter(self, constraints: Any, refs: list, value: float | None) -> bool:
        """Add a diameter constraint."""
        if value is None or len(refs) < 1:
            raise ConstraintError("Diameter requires 1 reference and a value")

        entity = self._get_entity(refs[0])
        try:
            center = entity.CenterSketchPoint.Geometry
        except Exception:
            center = entity.Geometry.Center
        dim_pos = self._app.TransientGeometry.CreatePoint2d(
            center.X + value * MM_TO_CM,
            center.Y
        )

        dim = constraints.AddDiameterDimension(entity, dim_pos)
        dim.Parameter.Value = value * MM_TO_CM
        return True

    def _add_angle(self, constraints: Any, refs: list, value: float | None) -> bool:
        """Add an angle constraint (value in degrees)."""
        if value is None or len(refs) < 2:
            raise ConstraintError("Angle requires 2 references and a value")

        entity1 = self._get_entity(refs[0])
        entity2 = self._get_entity(refs[1])

        # Position at intersection or midpoint
        dim_pos = self._app.TransientGeometry.CreatePoint2d(0, 0)

        dim = constraints.AddTwoLineAngle(entity1, entity2, dim_pos)
        # Inventor uses radians for angle dimensions
        dim.Parameter.Value = math.radians(value)
        return True

    # =========================================================================
    # Export Methods
    # =========================================================================

    def _export_line(self, line: Any) -> Line:
        """Export an Inventor line to canonical format."""
        start = Point2D(
            line.StartSketchPoint.Geometry.X * CM_TO_MM,
            line.StartSketchPoint.Geometry.Y * CM_TO_MM
        )
        end = Point2D(
            line.EndSketchPoint.Geometry.X * CM_TO_MM,
            line.EndSketchPoint.Geometry.Y * CM_TO_MM
        )
        return Line(
            start=start,
            end=end,
            construction=line.Construction
        )

    def _export_circle(self, circle: Any) -> Circle:
        """Export an Inventor circle to canonical format."""
        center = Point2D(
            circle.CenterSketchPoint.Geometry.X * CM_TO_MM,
            circle.CenterSketchPoint.Geometry.Y * CM_TO_MM
        )
        radius = circle.Radius * CM_TO_MM
        return Circle(
            center=center,
            radius=radius,
            construction=circle.Construction
        )

    def _export_arc(self, arc: Any) -> Arc:
        """Export an Inventor arc to canonical format."""
        center = Point2D(
            arc.CenterSketchPoint.Geometry.X * CM_TO_MM,
            arc.CenterSketchPoint.Geometry.Y * CM_TO_MM
        )
        start_pt = Point2D(
            arc.StartSketchPoint.Geometry.X * CM_TO_MM,
            arc.StartSketchPoint.Geometry.Y * CM_TO_MM
        )
        end_pt = Point2D(
            arc.EndSketchPoint.Geometry.X * CM_TO_MM,
            arc.EndSketchPoint.Geometry.Y * CM_TO_MM
        )

        # Determine direction - Inventor arcs are always CCW
        # Check sweep angle sign
        try:
            sweep = arc.SweepAngle
            ccw = sweep > 0
        except Exception:
            ccw = True

        return Arc(
            center=center,
            start_point=start_pt,
            end_point=end_pt,
            ccw=ccw,
            construction=arc.Construction
        )

    def _export_point(self, point: Any) -> Point:
        """Export an Inventor point to canonical format."""
        pos = Point2D(
            point.Geometry.X * CM_TO_MM,
            point.Geometry.Y * CM_TO_MM
        )
        return Point(position=pos)

    def _export_ellipse(self, ellipse: Any) -> Ellipse:
        """Export an Inventor ellipse to canonical format."""
        center = Point2D(
            ellipse.CenterSketchPoint.Geometry.X * CM_TO_MM,
            ellipse.CenterSketchPoint.Geometry.Y * CM_TO_MM
        )

        # Get major and minor radii
        major_radius = ellipse.MajorRadius * CM_TO_MM
        minor_radius = ellipse.MinorRadius * CM_TO_MM

        # Get rotation from major axis direction
        # Inventor provides MajorAxisVector property
        major_axis = ellipse.MajorAxisVector
        rotation = math.atan2(major_axis.Y, major_axis.X)

        return Ellipse(
            center=center,
            major_radius=major_radius,
            minor_radius=minor_radius,
            rotation=rotation,
            construction=ellipse.Construction
        )

    def _export_elliptical_arc(self, arc: Any) -> EllipticalArc:
        """Export an Inventor elliptical arc to canonical format."""
        center = Point2D(
            arc.CenterSketchPoint.Geometry.X * CM_TO_MM,
            arc.CenterSketchPoint.Geometry.Y * CM_TO_MM
        )

        # Get major and minor radii
        major_radius = arc.MajorRadius * CM_TO_MM
        minor_radius = arc.MinorRadius * CM_TO_MM

        # Get rotation from major axis direction
        major_axis = arc.MajorAxisVector
        rotation = math.atan2(major_axis.Y, major_axis.X)

        # Get start and end points to calculate parametric angles
        start_pt = Point2D(
            arc.StartSketchPoint.Geometry.X * CM_TO_MM,
            arc.StartSketchPoint.Geometry.Y * CM_TO_MM
        )
        end_pt = Point2D(
            arc.EndSketchPoint.Geometry.X * CM_TO_MM,
            arc.EndSketchPoint.Geometry.Y * CM_TO_MM
        )

        # Calculate parametric angles from points
        # Transform to ellipse-local coordinates and find angle
        def point_to_param(pt: Point2D) -> float:
            # Translate to center
            dx = pt.x - center.x
            dy = pt.y - center.y
            # Rotate by -rotation to align with axes
            cos_r = math.cos(-rotation)
            sin_r = math.sin(-rotation)
            local_x = dx * cos_r - dy * sin_r
            local_y = dx * sin_r + dy * cos_r
            # Now local_x = major_radius * cos(t), local_y = minor_radius * sin(t)
            return math.atan2(local_y / minor_radius, local_x / major_radius)

        start_param = point_to_param(start_pt)
        end_param = point_to_param(end_pt)

        # Determine direction from sweep angle
        try:
            sweep = arc.SweepAngle
            ccw = sweep > 0
        except Exception:
            ccw = True

        return EllipticalArc(
            center=center,
            major_radius=major_radius,
            minor_radius=minor_radius,
            rotation=rotation,
            start_param=start_param,
            end_param=end_param,
            ccw=ccw,
            construction=arc.Construction
        )

    def _export_geometric_constraints(self, doc: SketchDocument) -> None:
        """Export geometric constraints from Inventor sketch."""
        assert self._sketch is not None
        for constraint in self._sketch.GeometricConstraints:
            canonical = self._convert_geometric_constraint(constraint)
            if canonical is not None:
                doc.constraints.append(canonical)

    def _export_dimension_constraints(self, doc: SketchDocument) -> None:
        """Export dimensional constraints from Inventor sketch."""
        assert self._sketch is not None
        for dim in self._sketch.DimensionConstraints:
            canonical = self._convert_dimension_constraint(dim)
            if canonical is not None:
                doc.constraints.append(canonical)

    def _convert_geometric_constraint(self, constraint: Any) -> SketchConstraint | None:
        """Convert Inventor geometric constraint to canonical form."""
        try:
            ctype_name = constraint.Type.ToString() if hasattr(constraint.Type, 'ToString') else str(constraint.Type)

            # Map Inventor constraint types to canonical
            type_map = {
                'kCoincidentConstraint': ConstraintType.COINCIDENT,
                'kTangentConstraint': ConstraintType.TANGENT,
                'kPerpendicularConstraint': ConstraintType.PERPENDICULAR,
                'kParallelConstraint': ConstraintType.PARALLEL,
                'kHorizontalConstraint': ConstraintType.HORIZONTAL,
                'kVerticalConstraint': ConstraintType.VERTICAL,
                'kEqualLengthConstraint': ConstraintType.EQUAL,
                'kEqualRadiusConstraint': ConstraintType.EQUAL,
                'kConcentricConstraint': ConstraintType.CONCENTRIC,
                'kCollinearConstraint': ConstraintType.COLLINEAR,
                'kSymmetryConstraint': ConstraintType.SYMMETRIC,
                'kMidpointConstraint': ConstraintType.MIDPOINT,
                'kGroundConstraint': ConstraintType.FIXED,
            }

            # Try to get type from numeric value
            constraint_type = None
            for key, value in type_map.items():
                if key in ctype_name or str(constraint.Type) == key:
                    constraint_type = value
                    break

            if constraint_type is None:
                return None

            # Extract references
            refs = self._extract_constraint_refs(constraint, constraint_type)
            if refs is None:
                return None

            return SketchConstraint(
                id="",
                constraint_type=constraint_type,
                references=refs
            )

        except Exception:
            return None

    def _convert_dimension_constraint(self, dim: Any) -> SketchConstraint | None:
        """Convert Inventor dimension constraint to canonical form."""
        try:
            dim_type = str(dim.Type)
            value = dim.Parameter.Value * CM_TO_MM

            # Determine constraint type
            if 'LinearDimension' in dim_type or 'TwoPointDistance' in dim_type:
                # Could be LENGTH, DISTANCE, DISTANCE_X, or DISTANCE_Y
                # Simplified: treat as DISTANCE for now
                constraint_type = ConstraintType.DISTANCE
            elif 'Radius' in dim_type:
                constraint_type = ConstraintType.RADIUS
            elif 'Diameter' in dim_type:
                constraint_type = ConstraintType.DIAMETER
            elif 'Angle' in dim_type:
                constraint_type = ConstraintType.ANGLE
                value = math.degrees(dim.Parameter.Value)
            else:
                return None

            # Extract entity references
            refs = self._extract_dimension_refs(dim)
            if refs is None:
                return None

            return SketchConstraint(
                id="",
                constraint_type=constraint_type,
                references=refs,
                value=value
            )

        except Exception:
            return None

    def _extract_constraint_refs(
        self, constraint: Any, constraint_type: ConstraintType
    ) -> list[str | PointRef] | None:
        """Extract references from Inventor geometric constraint."""
        try:
            # Point-based constraints
            if constraint_type == ConstraintType.COINCIDENT:
                pt1 = constraint.PointOne
                pt2 = constraint.PointTwo
                ref1 = self._sketch_point_to_ref(pt1)
                ref2 = self._sketch_point_to_ref(pt2)
                if ref1 and ref2:
                    return [ref1, ref2]

            # Element-based constraints
            elif constraint_type in (
                ConstraintType.TANGENT, ConstraintType.PERPENDICULAR,
                ConstraintType.PARALLEL, ConstraintType.EQUAL,
                ConstraintType.COLLINEAR, ConstraintType.CONCENTRIC
            ):
                e1 = constraint.EntityOne
                e2 = constraint.EntityTwo
                id1 = self._entity_to_id.get(id(e1))
                id2 = self._entity_to_id.get(id(e2))
                if id1 and id2:
                    return [id1, id2]

            # Single-element constraints
            elif constraint_type in (
                ConstraintType.HORIZONTAL, ConstraintType.VERTICAL,
                ConstraintType.FIXED
            ):
                entity = constraint.Entity
                eid = self._entity_to_id.get(id(entity))
                if eid:
                    return [eid]

            # Symmetric constraint
            elif constraint_type == ConstraintType.SYMMETRIC:
                e1 = constraint.EntityOne
                e2 = constraint.EntityTwo
                axis = constraint.SymmetryLine
                id1 = self._entity_to_id.get(id(e1))
                id2 = self._entity_to_id.get(id(e2))
                axis_id = self._entity_to_id.get(id(axis))
                if id1 and id2 and axis_id:
                    return [id1, id2, axis_id]

            # Midpoint constraint
            elif constraint_type == ConstraintType.MIDPOINT:
                point = constraint.Point
                line = constraint.Entity
                ref = self._sketch_point_to_ref(point)
                line_id = self._entity_to_id.get(id(line))
                if ref and line_id:
                    return [ref, line_id]

            return None

        except Exception:
            return None

    def _extract_dimension_refs(self, dim: Any) -> list[str | PointRef] | None:
        """Extract references from Inventor dimension constraint."""
        try:
            refs: list[str | PointRef] = []

            # Try to get entities involved
            if hasattr(dim, 'EntityOne'):
                e1 = dim.EntityOne
                eid = self._entity_to_id.get(id(e1))
                if eid:
                    refs.append(eid)

            if hasattr(dim, 'EntityTwo'):
                e2 = dim.EntityTwo
                eid = self._entity_to_id.get(id(e2))
                if eid:
                    refs.append(eid)

            if hasattr(dim, 'PointOne'):
                pt1 = dim.PointOne
                ref = self._sketch_point_to_ref(pt1)
                if ref:
                    refs.append(ref)

            if hasattr(dim, 'PointTwo'):
                pt2 = dim.PointTwo
                ref = self._sketch_point_to_ref(pt2)
                if ref:
                    refs.append(ref)

            return refs if refs else None

        except Exception:
            return None

    def _sketch_point_to_ref(self, sketch_point: Any) -> PointRef | None:
        """Convert Inventor SketchPoint to canonical PointRef."""
        try:
            # Find which entity owns this point
            for entity_id, entity in self._id_to_entity.items():
                point_type = get_point_type_for_sketch_point(entity, sketch_point)
                if point_type is not None:
                    return PointRef(entity_id, point_type)
            return None
        except Exception:
            return None

    # =========================================================================
    # Solver Status
    # =========================================================================

    def get_solver_status(self) -> tuple[SolverStatus, int]:
        """Get the constraint solver status.

        Returns:
            Tuple of (SolverStatus, degrees_of_freedom)
        """
        if self._sketch is None:
            return (SolverStatus.DIRTY, -1)

        try:
            # Inventor doesn't expose DOF directly like FreeCAD
            # We can check if sketch is fully constrained
            if self._sketch.IsFullyConstrained:
                return (SolverStatus.FULLY_CONSTRAINED, 0)
            else:
                # Estimate DOF based on geometry and constraints
                # This is a rough approximation
                dof = self._estimate_dof()
                return (SolverStatus.UNDER_CONSTRAINED, dof)

        except Exception:
            return (SolverStatus.INCONSISTENT, -1)

    def _estimate_dof(self) -> int:
        """Estimate degrees of freedom (rough approximation)."""
        if self._sketch is None:
            return -1
        try:
            sketch = self._sketch  # Local reference for mypy
            dof = 0

            # Each line has 4 DOF (2 points x 2 coordinates)
            dof += sketch.SketchLines.Count * 4

            # Each circle has 3 DOF (center x,y + radius)
            dof += sketch.SketchCircles.Count * 3

            # Each arc has 5 DOF (center x,y + radius + 2 angles)
            dof += sketch.SketchArcs.Count * 5

            # Each point has 2 DOF
            for pt in sketch.SketchPoints:
                if not self._is_dependent_point(pt):
                    dof += 2

            # Subtract constraints (rough estimate)
            dof -= sketch.GeometricConstraints.Count
            dof -= sketch.DimensionConstraints.Count * 1

            return max(0, dof)

        except Exception:
            return -1

    def capture_image(self, width: int, height: int) -> bytes:
        """Capture a visualization of the sketch.

        Note: This requires Inventor GUI to be available.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            PNG image data as bytes
        """
        raise NotImplementedError(
            "Image capture not yet implemented for Inventor adapter"
        )
