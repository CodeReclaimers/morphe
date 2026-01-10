"""SolidWorks adapter for canonical sketch representation.

This module provides the SolidWorksAdapter class that implements the
SketchBackendAdapter interface for SolidWorks.

Note: SolidWorks internally uses meters, while the canonical format
uses millimeters. This adapter handles the conversion automatically.

This adapter uses the COM API via win32com, which requires:
- Windows operating system
- SolidWorks installed
- pywin32 package installed (pip install pywin32)
"""

import math
from typing import Any

from sketch_canonical import (
    Arc,
    Circle,
    ConstraintError,
    ConstraintType,
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

from .vertex_map import get_sketch_point_from_entity

# SolidWorks uses meters internally, canonical format uses millimeters
MM_TO_M = 0.001
M_TO_MM = 1000.0

# Try to import win32com for COM automation
SOLIDWORKS_AVAILABLE = False
_solidworks_app = None

try:
    import win32com.client

    SOLIDWORKS_AVAILABLE = True
except ImportError:
    win32com = None  # type: ignore[assignment]


# SolidWorks constraint type constants (from swConstraintType_e)
class SwConstraintType:
    """SolidWorks constraint type enumeration values."""

    COINCIDENT = 3
    CONCENTRIC = 4
    TANGENT = 5
    HORIZONTAL = 6
    VERTICAL = 7
    PERPENDICULAR = 8
    PARALLEL = 9
    EQUAL = 10
    FIX = 11
    MIDPOINT = 12
    SYMMETRIC = 13
    COLLINEAR = 14
    CORADIAL = 15


# SolidWorks sketch segment type constants
class SwSketchSegments:
    """SolidWorks sketch segment type enumeration."""

    LINE = 0
    ARC = 1
    ELLIPSE = 2
    SPLINE = 3
    TEXT = 4
    PARABOLA = 5


def get_solidworks_application() -> Any:
    """Get or create a SolidWorks application instance.

    Returns:
        SolidWorks Application COM object

    Raises:
        ImportError: If win32com is not available
        ConnectionError: If SolidWorks cannot be connected
    """
    global _solidworks_app

    if not SOLIDWORKS_AVAILABLE:
        raise ImportError(
            "win32com is not available. Install pywin32: pip install pywin32"
        )

    if _solidworks_app is not None:
        try:
            # Test if still connected
            _ = _solidworks_app.Visible
            return _solidworks_app
        except Exception:
            _solidworks_app = None

    try:
        # Try to connect to running SolidWorks instance
        _solidworks_app = win32com.client.GetActiveObject("SldWorks.Application")
        return _solidworks_app
    except Exception:
        pass

    try:
        # Try to start new SolidWorks instance
        _solidworks_app = win32com.client.Dispatch("SldWorks.Application")
        _solidworks_app.Visible = True
        return _solidworks_app
    except Exception as e:
        raise ConnectionError(
            f"Could not connect to SolidWorks. "
            f"Ensure SolidWorks is installed and running. Error: {e}"
        ) from e


class SolidWorksAdapter(SketchBackendAdapter):
    """SolidWorks implementation of SketchBackendAdapter.

    This adapter translates between the canonical sketch representation
    and SolidWorks's native sketch API via COM automation.

    Attributes:
        _app: SolidWorks Application COM object
        _document: Active SolidWorks part document
        _sketch: Current active sketch
        _sketch_manager: Sketch manager for geometry creation
        _id_to_entity: Mapping from canonical IDs to SolidWorks sketch entities
        _entity_to_id: Mapping from SolidWorks entities to canonical IDs
    """

    def __init__(self, document: Any | None = None):
        """Initialize the SolidWorks adapter.

        Args:
            document: Optional existing SolidWorks document to use.
                     If None, a new part document will be created when needed.

        Raises:
            ImportError: If win32com is not available
            ConnectionError: If SolidWorks cannot be connected
        """
        self._app = get_solidworks_application()

        if document is not None:
            self._document = document
        else:
            self._document = None

        self._sketch = None
        self._sketch_manager = None
        self._id_to_entity: dict[str, Any] = {}
        self._entity_to_id: dict[int, str] = {}
        self._ground_constraints: set[str] = set()

    def _ensure_document(self) -> None:
        """Ensure we have an active part document."""
        if self._document is None:
            # Create a new part document
            # NewDocument(TemplateName, PaperSize, Width, Height)
            # Use empty string for default template
            self._document = self._app.NewDocument(
                "",  # Default part template
                0,   # Paper size (not used for parts)
                0,   # Width (not used for parts)
                0    # Height (not used for parts)
            )

            if self._document is None:
                # Try alternative method
                self._document = self._app.NewPart()

    def create_sketch(self, name: str, plane: str | Any = "XY") -> None:
        """Create a new sketch on the specified plane.

        Args:
            name: Name for the new sketch
            plane: Either a plane name ("XY", "XZ", "YZ") or a SolidWorks
                   plane/face object

        Raises:
            SketchCreationError: If sketch creation fails
        """
        try:
            self._ensure_document()
            assert self._document is not None

            model = self._document
            self._sketch_manager = model.SketchManager

            # Select the appropriate plane
            if isinstance(plane, str):
                # Get reference plane by name
                if plane == "XY" or plane == "Front":
                    plane_name = "Front Plane"
                elif plane == "XZ" or plane == "Top":
                    plane_name = "Top Plane"
                elif plane == "YZ" or plane == "Right":
                    plane_name = "Right Plane"
                else:
                    plane_name = plane

                # Select the plane
                model.Extension.SelectByID2(
                    plane_name, "PLANE", 0, 0, 0, False, 0, None, 0
                )
            else:
                # Assume it's a plane object - select it
                plane.Select(False)

            # Insert a new sketch
            assert self._sketch_manager is not None
            self._sketch_manager.InsertSketch(True)
            self._sketch = self._sketch_manager.ActiveSketch

            # Rename the sketch if possible
            if self._sketch is not None:
                try:
                    feature = self._sketch
                    if hasattr(feature, "Name"):
                        feature.Name = name
                except Exception:
                    pass  # Renaming may not always work

            # Clear mappings for new sketch
            self._id_to_entity.clear()
            self._entity_to_id.clear()
            self._ground_constraints.clear()

        except Exception as e:
            raise SketchCreationError(f"Failed to create sketch: {e}") from e

    def load_sketch(self, sketch: SketchDocument) -> None:
        """Load a canonical sketch into SolidWorks.

        Args:
            sketch: The canonical SketchDocument to load

        Raises:
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
        """Export the current SolidWorks sketch to canonical form.

        Returns:
            A new SketchDocument containing the canonical representation.

        Raises:
            ExportError: If export fails
        """
        if self._sketch is None:
            raise ExportError("No active sketch to export")

        try:
            sketch = self._sketch
            doc = SketchDocument(name=getattr(sketch, "Name", "ExportedSketch"))

            # Clear and rebuild mappings
            self._id_to_entity.clear()
            self._entity_to_id.clear()

            # Get all sketch segments
            segments = sketch.GetSketchSegments()
            if segments:
                for segment in segments:
                    if self._is_construction(segment):
                        construction = True
                    else:
                        construction = False

                    prim = self._export_segment(segment, construction)
                    if prim is not None:
                        doc.add_primitive(prim)
                        self._entity_to_id[id(segment)] = prim.id
                        self._id_to_entity[prim.id] = segment

            # Export standalone points
            points = sketch.GetSketchPoints2()
            if points:
                for point in points:
                    # Skip points that are part of other geometry
                    if self._is_dependent_point(point):
                        continue
                    prim = self._export_point(point)
                    doc.add_primitive(prim)
                    self._entity_to_id[id(point)] = prim.id
                    self._id_to_entity[prim.id] = point

            # Export constraints
            self._export_constraints(doc)

            # Get solver status
            status, dof = self.get_solver_status()
            doc.solver_status = status
            doc.degrees_of_freedom = dof

            return doc

        except Exception as e:
            raise ExportError(f"Failed to export sketch: {e}") from e

    def _is_construction(self, segment: Any) -> bool:
        """Check if segment is construction geometry."""
        try:
            return bool(segment.ConstructionGeometry)
        except Exception:
            return False

    def _is_dependent_point(self, point: Any) -> bool:
        """Check if a point is dependent on other geometry."""
        try:
            # Check if point has constraints linking it to other geometry
            return False  # For now, include all standalone points
        except Exception:
            return False

    def add_primitive(self, primitive: SketchPrimitive) -> Any:
        """Add a single primitive to the sketch.

        Args:
            primitive: The canonical primitive to add

        Returns:
            SolidWorks sketch entity

        Raises:
            GeometryError: If geometry creation fails
        """
        if self._sketch_manager is None:
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
            else:
                raise GeometryError(f"Unsupported primitive type: {type(primitive)}")

            # Store mapping
            if entity is not None:
                self._id_to_entity[primitive.id] = entity
                self._entity_to_id[id(entity)] = primitive.id

                # Set construction mode if needed
                if primitive.construction:
                    try:
                        entity.ConstructionGeometry = True
                    except Exception:
                        pass

            return entity

        except Exception as e:
            raise GeometryError(f"Failed to add {type(primitive).__name__}: {e}") from e

    def _add_line(self, line: Line) -> Any:
        """Add a line to the sketch."""
        assert self._sketch_manager is not None
        # CreateLine(X1, Y1, Z1, X2, Y2, Z2)
        # SolidWorks uses meters
        segment = self._sketch_manager.CreateLine(
            line.start.x * MM_TO_M,
            line.start.y * MM_TO_M,
            0,  # Z = 0 for 2D sketch
            line.end.x * MM_TO_M,
            line.end.y * MM_TO_M,
            0
        )
        return segment

    def _add_circle(self, circle: Circle) -> Any:
        """Add a circle to the sketch."""
        assert self._sketch_manager is not None
        # CreateCircle(Xc, Yc, Zc, Xp, Yp, Zp)
        # Center point and a point on the circle
        segment = self._sketch_manager.CreateCircle(
            circle.center.x * MM_TO_M,
            circle.center.y * MM_TO_M,
            0,
            (circle.center.x + circle.radius) * MM_TO_M,
            circle.center.y * MM_TO_M,
            0
        )
        return segment

    def _add_arc(self, arc: Arc) -> Any:
        """Add an arc to the sketch."""
        assert self._sketch_manager is not None
        # CreateArc(Xc, Yc, Zc, Xs, Ys, Zs, Xe, Ye, Ze, Direction)
        # Direction: 1 = counter-clockwise, -1 = clockwise
        direction = 1 if arc.ccw else -1
        segment = self._sketch_manager.CreateArc(
            arc.center.x * MM_TO_M,
            arc.center.y * MM_TO_M,
            0,
            arc.start_point.x * MM_TO_M,
            arc.start_point.y * MM_TO_M,
            0,
            arc.end_point.x * MM_TO_M,
            arc.end_point.y * MM_TO_M,
            0,
            direction
        )
        return segment

    def _add_point(self, point: Point) -> Any:
        """Add a point to the sketch."""
        assert self._sketch_manager is not None
        # CreatePoint(X, Y, Z)
        sketch_point = self._sketch_manager.CreatePoint(
            point.position.x * MM_TO_M,
            point.position.y * MM_TO_M,
            0
        )
        return sketch_point

    def _add_spline(self, spline: Spline) -> Any:
        """Add a spline to the sketch."""
        assert self._sketch_manager is not None

        # Build points array for spline
        # CreateSpline expects an array of doubles: [x1,y1,z1, x2,y2,z2, ...]
        points = []
        for pt in spline.control_points:
            points.extend([
                pt.x * MM_TO_M,
                pt.y * MM_TO_M,
                0
            ])

        # Convert to variant array for COM
        import pythoncom
        from win32com.client import VARIANT

        points_array = VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, points)

        segment = self._sketch_manager.CreateSpline2(
            points_array,
            False  # Not periodic
        )
        return segment

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
        if self._sketch is None or self._document is None:
            raise ConstraintError("No active sketch")

        try:
            ctype = constraint.constraint_type
            refs = constraint.references
            value = constraint.value

            model = self._document

            # Geometric constraints
            if ctype == ConstraintType.COINCIDENT:
                return self._add_coincident(model, refs)
            elif ctype == ConstraintType.TANGENT:
                return self._add_tangent(model, refs)
            elif ctype == ConstraintType.PERPENDICULAR:
                return self._add_perpendicular(model, refs)
            elif ctype == ConstraintType.PARALLEL:
                return self._add_parallel(model, refs)
            elif ctype == ConstraintType.HORIZONTAL:
                return self._add_horizontal(model, refs)
            elif ctype == ConstraintType.VERTICAL:
                return self._add_vertical(model, refs)
            elif ctype == ConstraintType.EQUAL:
                return self._add_equal(model, refs)
            elif ctype == ConstraintType.CONCENTRIC:
                return self._add_concentric(model, refs)
            elif ctype == ConstraintType.COLLINEAR:
                return self._add_collinear(model, refs)
            elif ctype == ConstraintType.MIDPOINT:
                return self._add_midpoint(model, refs)
            elif ctype == ConstraintType.FIXED:
                return self._add_fixed(model, refs)

            # Dimensional constraints
            elif ctype == ConstraintType.DISTANCE:
                return self._add_distance(model, refs, value)
            elif ctype == ConstraintType.RADIUS:
                return self._add_radius(model, refs, value)
            elif ctype == ConstraintType.DIAMETER:
                return self._add_diameter(model, refs, value)
            elif ctype == ConstraintType.ANGLE:
                return self._add_angle(model, refs, value)
            elif ctype == ConstraintType.LENGTH:
                return self._add_length(model, refs, value)
            elif ctype == ConstraintType.DISTANCE_X:
                return self._add_distance_x(model, refs, value)
            elif ctype == ConstraintType.DISTANCE_Y:
                return self._add_distance_y(model, refs, value)

            else:
                raise ConstraintError(f"Unsupported constraint type: {ctype}")

        except ConstraintError:
            raise
        except Exception as e:
            raise ConstraintError(f"Failed to add constraint: {e}") from e

    def _select_entity(self, ref: str | PointRef, append: bool = False) -> bool:
        """Select an entity or point for constraint creation."""
        try:
            if isinstance(ref, PointRef):
                entity = self._id_to_entity.get(ref.element_id)
                if entity is None:
                    return False
                point = get_sketch_point_from_entity(entity, ref.point_type)
                if point is None:
                    return False
                # Select the point
                return bool(point.Select4(append, None))
            else:
                entity = self._id_to_entity.get(ref)
                if entity is None:
                    return False
                return bool(entity.Select4(append, None))
        except Exception:
            return False

    def _add_coincident(self, model: Any, refs: list) -> bool:
        """Add a coincident constraint."""
        if len(refs) < 2:
            raise ConstraintError("Coincident requires 2 references")

        model.ClearSelection2(True)
        if not self._select_entity(refs[0], False):
            raise ConstraintError("Could not select first entity")
        if not self._select_entity(refs[1], True):
            raise ConstraintError("Could not select second entity")

        model.SketchAddConstraints("sgCOINCIDENT")
        return True

    def _add_tangent(self, model: Any, refs: list) -> bool:
        """Add a tangent constraint."""
        if len(refs) < 2:
            raise ConstraintError("Tangent requires 2 references")

        model.ClearSelection2(True)
        if not self._select_entity(refs[0], False):
            raise ConstraintError("Could not select first entity")
        if not self._select_entity(refs[1], True):
            raise ConstraintError("Could not select second entity")

        model.SketchAddConstraints("sgTANGENT")
        return True

    def _add_perpendicular(self, model: Any, refs: list) -> bool:
        """Add a perpendicular constraint."""
        if len(refs) < 2:
            raise ConstraintError("Perpendicular requires 2 references")

        model.ClearSelection2(True)
        if not self._select_entity(refs[0], False):
            raise ConstraintError("Could not select first entity")
        if not self._select_entity(refs[1], True):
            raise ConstraintError("Could not select second entity")

        model.SketchAddConstraints("sgPERPENDICULAR")
        return True

    def _add_parallel(self, model: Any, refs: list) -> bool:
        """Add a parallel constraint."""
        if len(refs) < 2:
            raise ConstraintError("Parallel requires 2 references")

        model.ClearSelection2(True)
        if not self._select_entity(refs[0], False):
            raise ConstraintError("Could not select first entity")
        if not self._select_entity(refs[1], True):
            raise ConstraintError("Could not select second entity")

        model.SketchAddConstraints("sgPARALLEL")
        return True

    def _add_horizontal(self, model: Any, refs: list) -> bool:
        """Add a horizontal constraint."""
        if len(refs) < 1:
            raise ConstraintError("Horizontal requires at least 1 reference")

        model.ClearSelection2(True)
        if not self._select_entity(refs[0], False):
            raise ConstraintError("Could not select entity")

        model.SketchAddConstraints("sgHORIZONTAL2D")
        return True

    def _add_vertical(self, model: Any, refs: list) -> bool:
        """Add a vertical constraint."""
        if len(refs) < 1:
            raise ConstraintError("Vertical requires at least 1 reference")

        model.ClearSelection2(True)
        if not self._select_entity(refs[0], False):
            raise ConstraintError("Could not select entity")

        model.SketchAddConstraints("sgVERTICAL2D")
        return True

    def _add_equal(self, model: Any, refs: list) -> bool:
        """Add an equal constraint."""
        if len(refs) < 2:
            raise ConstraintError("Equal requires 2 references")

        model.ClearSelection2(True)
        if not self._select_entity(refs[0], False):
            raise ConstraintError("Could not select first entity")
        if not self._select_entity(refs[1], True):
            raise ConstraintError("Could not select second entity")

        model.SketchAddConstraints("sgSAMELENGTH")
        return True

    def _add_concentric(self, model: Any, refs: list) -> bool:
        """Add a concentric constraint."""
        if len(refs) < 2:
            raise ConstraintError("Concentric requires 2 references")

        model.ClearSelection2(True)
        if not self._select_entity(refs[0], False):
            raise ConstraintError("Could not select first entity")
        if not self._select_entity(refs[1], True):
            raise ConstraintError("Could not select second entity")

        model.SketchAddConstraints("sgCONCENTRIC")
        return True

    def _add_collinear(self, model: Any, refs: list) -> bool:
        """Add a collinear constraint."""
        if len(refs) < 2:
            raise ConstraintError("Collinear requires 2 references")

        model.ClearSelection2(True)
        if not self._select_entity(refs[0], False):
            raise ConstraintError("Could not select first entity")
        if not self._select_entity(refs[1], True):
            raise ConstraintError("Could not select second entity")

        model.SketchAddConstraints("sgCOLINEAR")
        return True

    def _add_midpoint(self, model: Any, refs: list) -> bool:
        """Add a midpoint constraint."""
        if len(refs) < 2:
            raise ConstraintError("Midpoint requires 2 references")

        model.ClearSelection2(True)
        # First ref should be the point, second the line
        if not self._select_entity(refs[0], False):
            raise ConstraintError("Could not select point")
        if not self._select_entity(refs[1], True):
            raise ConstraintError("Could not select line")

        model.SketchAddConstraints("sgATMIDDLE")
        return True

    def _add_fixed(self, model: Any, refs: list) -> bool:
        """Add a fixed constraint."""
        if len(refs) < 1:
            raise ConstraintError("Fixed requires at least 1 reference")

        model.ClearSelection2(True)
        if not self._select_entity(refs[0], False):
            raise ConstraintError("Could not select entity")

        model.SketchAddConstraints("sgFIXED")
        return True

    def _add_distance(self, model: Any, refs: list, value: float | None) -> bool:
        """Add a distance constraint."""
        if value is None:
            raise ConstraintError("Distance requires a value")
        if len(refs) < 2:
            raise ConstraintError("Distance requires 2 references")

        model.ClearSelection2(True)
        if not self._select_entity(refs[0], False):
            raise ConstraintError("Could not select first entity")
        if not self._select_entity(refs[1], True):
            raise ConstraintError("Could not select second entity")

        # Add dimension
        dim = model.AddDimension2(0, 0, 0)
        if dim is not None:
            # Set the value (convert mm to meters)
            dim.SystemValue = value * MM_TO_M
        return True

    def _add_radius(self, model: Any, refs: list, value: float | None) -> bool:
        """Add a radius constraint."""
        if value is None:
            raise ConstraintError("Radius requires a value")
        if len(refs) < 1:
            raise ConstraintError("Radius requires 1 reference")

        model.ClearSelection2(True)
        entity_ref = refs[0]
        entity_id = entity_ref.element_id if isinstance(entity_ref, PointRef) else entity_ref
        entity = self._id_to_entity.get(entity_id)
        if entity is None:
            raise ConstraintError("Could not find entity")

        entity.Select4(False, None)

        # Add dimension - for circles/arcs, this creates radius dimension
        dim = model.AddDimension2(0, 0, 0)
        if dim is not None:
            dim.SystemValue = value * MM_TO_M
        return True

    def _add_diameter(self, model: Any, refs: list, value: float | None) -> bool:
        """Add a diameter constraint."""
        if value is None:
            raise ConstraintError("Diameter requires a value")

        # SolidWorks uses radius, so convert diameter to radius
        return self._add_radius(model, refs, value / 2)

    def _add_angle(self, model: Any, refs: list, value: float | None) -> bool:
        """Add an angle constraint."""
        if value is None:
            raise ConstraintError("Angle requires a value")
        if len(refs) < 2:
            raise ConstraintError("Angle requires 2 references")

        model.ClearSelection2(True)
        if not self._select_entity(refs[0], False):
            raise ConstraintError("Could not select first entity")
        if not self._select_entity(refs[1], True):
            raise ConstraintError("Could not select second entity")

        # Add angle dimension
        dim = model.AddDimension2(0, 0, 0)
        if dim is not None:
            # Angle in radians
            dim.SystemValue = math.radians(value)
        return True

    def _add_length(self, model: Any, refs: list, value: float | None) -> bool:
        """Add a length constraint to a line."""
        if value is None:
            raise ConstraintError("Length requires a value")
        if len(refs) < 1:
            raise ConstraintError("Length requires 1 reference")

        model.ClearSelection2(True)
        entity_ref = refs[0]
        entity_id = entity_ref.element_id if isinstance(entity_ref, PointRef) else entity_ref
        entity = self._id_to_entity.get(entity_id)
        if entity is None:
            raise ConstraintError("Could not find entity")

        entity.Select4(False, None)

        # Add dimension
        dim = model.AddDimension2(0, 0, 0)
        if dim is not None:
            dim.SystemValue = value * MM_TO_M
        return True

    def _add_distance_x(self, model: Any, refs: list, value: float | None) -> bool:
        """Add a horizontal distance constraint."""
        if value is None:
            raise ConstraintError("DistanceX requires a value")

        model.ClearSelection2(True)
        if len(refs) >= 2:
            if not self._select_entity(refs[0], False):
                raise ConstraintError("Could not select first entity")
            if not self._select_entity(refs[1], True):
                raise ConstraintError("Could not select second entity")
        else:
            if not self._select_entity(refs[0], False):
                raise ConstraintError("Could not select entity")

        # Add horizontal dimension
        dim = model.Extension.AddDimension(0, 0, 0, 0)  # swHorDimension
        if dim is not None:
            dim.SystemValue = abs(value) * MM_TO_M
        return True

    def _add_distance_y(self, model: Any, refs: list, value: float | None) -> bool:
        """Add a vertical distance constraint."""
        if value is None:
            raise ConstraintError("DistanceY requires a value")

        model.ClearSelection2(True)
        if len(refs) >= 2:
            if not self._select_entity(refs[0], False):
                raise ConstraintError("Could not select first entity")
            if not self._select_entity(refs[1], True):
                raise ConstraintError("Could not select second entity")
        else:
            if not self._select_entity(refs[0], False):
                raise ConstraintError("Could not select entity")

        # Add vertical dimension
        dim = model.Extension.AddDimension(0, 0, 0, 1)  # swVerDimension
        if dim is not None:
            dim.SystemValue = abs(value) * MM_TO_M
        return True

    # =========================================================================
    # Export Methods
    # =========================================================================

    def _export_segment(self, segment: Any, construction: bool = False) -> SketchPrimitive | None:
        """Export a SolidWorks sketch segment to canonical format."""
        try:
            seg_type = segment.GetType()

            if seg_type == SwSketchSegments.LINE:
                return self._export_line(segment, construction)
            elif seg_type == SwSketchSegments.ARC:
                return self._export_arc(segment, construction)
            elif seg_type == SwSketchSegments.SPLINE:
                return self._export_spline(segment, construction)
            # Circles are handled differently in SolidWorks
            # They may come as arcs or need special handling
            else:
                return None
        except Exception:
            return None

    def _export_line(self, segment: Any, construction: bool = False) -> Line:
        """Export a SolidWorks line to canonical format."""
        start_pt = segment.GetStartPoint2()
        end_pt = segment.GetEndPoint2()

        return Line(
            start=Point2D(start_pt.X * M_TO_MM, start_pt.Y * M_TO_MM),
            end=Point2D(end_pt.X * M_TO_MM, end_pt.Y * M_TO_MM),
            construction=construction
        )

    def _export_arc(self, segment: Any, construction: bool = False) -> Arc | Circle:
        """Export a SolidWorks arc to canonical format."""
        start_pt = segment.GetStartPoint2()
        end_pt = segment.GetEndPoint2()
        center_pt = segment.GetCenterPoint2()

        # Check if it's a full circle (start == end)
        start_x = start_pt.X * M_TO_MM
        start_y = start_pt.Y * M_TO_MM
        end_x = end_pt.X * M_TO_MM
        end_y = end_pt.Y * M_TO_MM
        center_x = center_pt.X * M_TO_MM
        center_y = center_pt.Y * M_TO_MM

        dist = math.sqrt((start_x - end_x)**2 + (start_y - end_y)**2)
        if dist < 1e-6:
            # Full circle
            radius = math.sqrt((start_x - center_x)**2 + (start_y - center_y)**2)
            return Circle(
                center=Point2D(center_x, center_y),
                radius=radius,
                construction=construction
            )
        else:
            # Arc - determine direction
            # SolidWorks arcs: check if counter-clockwise
            # We can determine this from the cross product of vectors
            v1x = start_x - center_x
            v1y = start_y - center_y
            v2x = end_x - center_x
            v2y = end_y - center_y
            cross = v1x * v2y - v1y * v2x
            ccw = cross > 0

            return Arc(
                center=Point2D(center_x, center_y),
                start_point=Point2D(start_x, start_y),
                end_point=Point2D(end_x, end_y),
                ccw=ccw,
                construction=construction
            )

    def _export_spline(self, segment: Any, construction: bool = False) -> Spline:
        """Export a SolidWorks spline to canonical format."""
        points_data = segment.GetPoints2()
        control_points = []

        if points_data:
            # Points come as flat array [x1,y1,z1, x2,y2,z2, ...]
            for i in range(0, len(points_data), 3):
                control_points.append(Point2D(
                    points_data[i] * M_TO_MM,
                    points_data[i + 1] * M_TO_MM
                ))

        return Spline(
            control_points=control_points,
            degree=3,  # Default degree
            construction=construction
        )

    def _export_point(self, point: Any) -> Point:
        """Export a SolidWorks point to canonical format."""
        return Point(
            position=Point2D(point.X * M_TO_MM, point.Y * M_TO_MM)
        )

    def _export_constraints(self, doc: SketchDocument) -> None:
        """Export constraints from SolidWorks sketch."""
        if self._sketch is None:
            return

        try:
            # Get sketch relations
            relations = self._sketch.GetSketchRelations()
            if relations:
                for relation in relations:
                    canonical = self._convert_relation(relation)
                    if canonical is not None:
                        doc.constraints.append(canonical)
        except Exception:
            pass

    def _convert_relation(self, relation: Any) -> SketchConstraint | None:
        """Convert a SolidWorks sketch relation to canonical constraint."""
        try:
            rel_type = relation.GetRelationType()

            # Map SolidWorks relation types to canonical
            type_map = {
                SwConstraintType.HORIZONTAL: ConstraintType.HORIZONTAL,
                SwConstraintType.VERTICAL: ConstraintType.VERTICAL,
                SwConstraintType.COINCIDENT: ConstraintType.COINCIDENT,
                SwConstraintType.TANGENT: ConstraintType.TANGENT,
                SwConstraintType.PERPENDICULAR: ConstraintType.PERPENDICULAR,
                SwConstraintType.PARALLEL: ConstraintType.PARALLEL,
                SwConstraintType.EQUAL: ConstraintType.EQUAL,
                SwConstraintType.CONCENTRIC: ConstraintType.CONCENTRIC,
                SwConstraintType.COLLINEAR: ConstraintType.COLLINEAR,
                SwConstraintType.FIX: ConstraintType.FIXED,
                SwConstraintType.MIDPOINT: ConstraintType.MIDPOINT,
            }

            if rel_type not in type_map:
                return None

            ctype = type_map[rel_type]

            # Get entities involved
            entities = relation.GetEntities()
            refs: list[str | PointRef] = []
            if entities:
                for entity in entities:
                    entity_id = self._entity_to_id.get(id(entity))
                    if entity_id:
                        refs.append(entity_id)

            if not refs:
                return None

            # Generate a unique constraint ID
            import uuid
            constraint_id = f"C_{uuid.uuid4().hex[:8]}"

            return SketchConstraint(
                id=constraint_id,
                constraint_type=ctype,
                references=refs
            )

        except Exception:
            return None

    def get_solver_status(self) -> tuple[SolverStatus, int]:
        """Get the constraint solver status.

        Returns:
            Tuple of (SolverStatus, degrees_of_freedom)
        """
        if self._sketch is None:
            return (SolverStatus.DIRTY, -1)

        try:
            # SolidWorks sketch states:
            # 1 = Under defined (blue)
            # 2 = Fully defined (black)
            # 3 = Over defined (red)
            status_val = self._sketch.GetConstrainedStatus()

            if status_val == 2:
                return (SolverStatus.FULLY_CONSTRAINED, 0)
            elif status_val == 3:
                return (SolverStatus.OVER_CONSTRAINED, 0)
            else:
                # Under defined - estimate DOF
                dof = self._estimate_dof()
                return (SolverStatus.UNDER_CONSTRAINED, dof)

        except Exception:
            return (SolverStatus.INCONSISTENT, -1)

    def _estimate_dof(self) -> int:
        """Estimate degrees of freedom (rough approximation)."""
        if self._sketch is None:
            return -1

        try:
            sketch = self._sketch
            dof = 0

            # Count geometry
            segments = sketch.GetSketchSegments()
            if segments:
                for segment in segments:
                    seg_type = segment.GetType()
                    if seg_type == SwSketchSegments.LINE:
                        dof += 4  # 2 points x 2 coords
                    elif seg_type == SwSketchSegments.ARC:
                        dof += 5  # center + radius + 2 angles
                    elif seg_type == SwSketchSegments.SPLINE:
                        points = segment.GetPoints2()
                        if points:
                            dof += (len(points) // 3) * 2

            # Subtract for relations
            relations = sketch.GetSketchRelations()
            if relations:
                dof -= len(relations)

            return max(0, dof)

        except Exception:
            return -1

    def capture_image(self, width: int, height: int) -> bytes:
        """Capture a visualization of the sketch.

        Note: Image capture is not directly supported via COM.
        This returns an empty bytes object.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Empty bytes (not implemented)
        """
        return b""
