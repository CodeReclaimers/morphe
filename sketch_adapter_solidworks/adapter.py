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
        # Store original primitive data for export (since COM access is limited)
        # Use a list indexed by creation order since COM object ids are not stable
        self._segment_geometry_list: list[dict] = []

    def _ensure_document(self) -> None:
        """Ensure we have an active part document."""
        if self._document is None:
            # First, check if there's already an active document
            try:
                active_doc = self._app.ActiveDoc
                if active_doc is not None:
                    self._document = active_doc
                    return
            except Exception as e:
                pass

            # Try to find part template using various methods
            template_path = self._find_part_template()

            # Create a new part document
            if template_path:
                self._document = self._app.NewDocument(
                    template_path,
                    0,   # Paper size (not used for parts)
                    0,   # Width (not used for parts)
                    0    # Height (not used for parts)
                )

            if self._document is None:
                raise SketchCreationError(
                    "Could not create a new part document. "
                    "Please ensure SolidWorks has a valid part template configured."
                )

    def _find_part_template(self) -> str:
        """Find a valid part template path."""
        import os

        # Try various user preference string values for part template
        # Different SolidWorks versions use different constants
        preference_indices = [
            7,   # swDefaultTemplatePart in some versions
            17,  # Another possible index
            27,  # Another possible index
        ]

        for idx in preference_indices:
            try:
                path = self._app.GetUserPreferenceStringValue(idx)
                if path and path.lower().endswith('.prtdot'):
                    if os.path.exists(path):
                        return path
            except Exception:
                pass

        # Try to get the templates folder and search for .prtdot files
        template_folders = []

        # Try swFileLocationsDocumentTemplates = 23
        try:
            folder = self._app.GetUserPreferenceStringValue(23)
            if folder:
                template_folders.append(folder)
        except Exception:
            pass

        # Common SolidWorks template locations
        program_data = os.environ.get('ProgramData', 'C:\\ProgramData')
        for year in ['2024', '2023', '2022', '2021', '2020']:
            template_folders.extend([
                f"{program_data}\\SolidWorks\\SOLIDWORKS {year}\\templates",
                f"{program_data}\\SolidWorks\\SOLIDWORKS {year}\\lang\\english\\Tutorial",
                f"C:\\Program Files\\SOLIDWORKS Corp\\SOLIDWORKS\\lang\\english\\Tutorial",
            ])

        for folder in template_folders:
            if os.path.isdir(folder):
                for filename in os.listdir(folder):
                    if filename.lower().endswith('.prtdot'):
                        full_path = os.path.join(folder, filename)
                        return full_path

        return ""

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
            plane_feature = None
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

                # Try to get the plane feature directly
                try:
                    # Get FeatureManager to access features
                    plane_feature = model.FeatureByName(plane_name)
                except Exception as e:
                    pass

                if plane_feature is not None:
                    # Select the plane feature
                    plane_feature.Select2(False, 0)
                else:
                    # Fallback: try selecting via feature manager tree traversal
                    feat = model.FirstFeature()
                    while feat is not None:
                        feat_name = feat.Name
                        if feat_name == plane_name:
                            plane_feature = feat
                            plane_feature.Select2(False, 0)
                            break
                        feat = feat.GetNextFeature()
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
            self._segment_geometry_list.clear()

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

            # Track point coordinates used by segments to avoid duplicating them
            used_point_coords: set[tuple[float, float]] = set()

            # Get all sketch segments
            # Note: In COM late binding, GetSketchSegments may be a property returning
            # a tuple rather than a callable method
            segments = self._get_com_result(sketch, "GetSketchSegments")
            if segments:
                for seg_idx, segment in enumerate(segments):
                    if self._is_construction(segment):
                        construction = True
                    else:
                        construction = False

                    prim = self._export_segment(segment, construction, seg_idx)
                    if prim is not None:
                        doc.add_primitive(prim)
                        self._entity_to_id[id(segment)] = prim.id
                        self._id_to_entity[prim.id] = segment

                        # Track coordinates used by this primitive
                        if isinstance(prim, Line):
                            used_point_coords.add((round(prim.start.x, 6), round(prim.start.y, 6)))
                            used_point_coords.add((round(prim.end.x, 6), round(prim.end.y, 6)))
                        elif isinstance(prim, Arc):
                            used_point_coords.add((round(prim.start_point.x, 6), round(prim.start_point.y, 6)))
                            used_point_coords.add((round(prim.end_point.x, 6), round(prim.end_point.y, 6)))
                            used_point_coords.add((round(prim.center.x, 6), round(prim.center.y, 6)))
                        elif isinstance(prim, Circle):
                            used_point_coords.add((round(prim.center.x, 6), round(prim.center.y, 6)))

            # Export standalone points (skip points that are part of segments)
            points = self._get_com_result(sketch, "GetSketchPoints2")
            if points:
                for point in points:
                    # Skip points that are part of other geometry
                    if self._is_dependent_point(point):
                        continue

                    # Export the point
                    prim = self._export_point(point)

                    # Skip if this point's coordinates match a segment endpoint
                    point_coords = (round(prim.position.x, 6), round(prim.position.y, 6))
                    if point_coords in used_point_coords:
                        continue

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

    def _get_com_result(self, obj: Any, attr_name: str) -> Any:
        """Get a COM result, handling both property and method access.

        In win32com late binding, some methods are exposed as properties
        that return tuples instead of callable methods.
        """
        attr = getattr(obj, attr_name, None)
        if attr is None:
            return None
        # If it's callable (a method), call it
        if callable(attr):
            try:
                return attr()
            except TypeError:
                # If calling fails, it might be a property that looks callable
                return attr
        else:
            # It's a property, return its value directly
            return attr

    def _is_construction(self, segment: Any) -> bool:
        """Check if segment is construction geometry."""
        try:
            return bool(segment.ConstructionGeometry)
        except Exception:
            return False

    def _is_dependent_point(self, point: Any) -> bool:
        """Check if a point is dependent on other geometry (e.g., endpoint of a line).

        Returns True if this point is part of a line, arc, or other segment.
        """
        try:
            # In SolidWorks, we can check if the point has any sketch segments
            # that use it as an endpoint
            # Try GetSketchSegmentCount or similar
            seg_count = self._get_com_result(point, "GetSketchSegmentCount")
            if seg_count is not None and seg_count > 0:
                return True

            # Alternative: check if the point is constrained/connected
            # Points that are endpoints of lines/arcs usually have constraints
            return False
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

        # Store geometry data for export (since COM access to segment points is limited)
        if segment is not None:
            self._segment_geometry_list.append({
                'type': 'line',
                'start': (line.start.x, line.start.y),
                'end': (line.end.x, line.end.y),
                'construction': line.construction
            })

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

        # Store geometry data for export
        if segment is not None:
            self._segment_geometry_list.append({
                'type': 'circle',
                'center': (circle.center.x, circle.center.y),
                'radius': circle.radius,
                'construction': circle.construction
            })

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

        # Store geometry data for export
        if segment is not None:
            self._segment_geometry_list.append({
                'type': 'arc',
                'center': (arc.center.x, arc.center.y),
                'start': (arc.start_point.x, arc.start_point.y),
                'end': (arc.end_point.x, arc.end_point.y),
                'ccw': arc.ccw,
                'construction': arc.construction
            })

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

    def _export_segment(self, segment: Any, construction: bool = False, segment_index: int = -1) -> SketchPrimitive | None:
        """Export a SolidWorks sketch segment to canonical format."""
        try:
            # First, check if we have stored geometry data for this segment by index
            if 0 <= segment_index < len(self._segment_geometry_list):
                geom = self._segment_geometry_list[segment_index]
                if geom['type'] == 'line':
                    return Line(
                        start=Point2D(geom['start'][0], geom['start'][1]),
                        end=Point2D(geom['end'][0], geom['end'][1]),
                        construction=geom.get('construction', construction)
                    )
                elif geom['type'] == 'circle':
                    return Circle(
                        center=Point2D(geom['center'][0], geom['center'][1]),
                        radius=geom['radius'],
                        construction=geom.get('construction', construction)
                    )
                elif geom['type'] == 'arc':
                    return Arc(
                        center=Point2D(geom['center'][0], geom['center'][1]),
                        start_point=Point2D(geom['start'][0], geom['start'][1]),
                        end_point=Point2D(geom['end'][0], geom['end'][1]),
                        ccw=geom['ccw'],
                        construction=geom.get('construction', construction)
                    )

            # Fall back to COM-based export if no stored geometry
            # Debug: List available attributes on the segment

            seg_type = self._get_com_result(segment, "GetType")

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
        except Exception as e:
            return None

    def _export_line(self, segment: Any, construction: bool = False) -> Line:
        """Export a SolidWorks line to canonical format."""
        start_pt = None
        end_pt = None

        # Method 1: Cast to ISketchLine and use its methods
        try:
            sketch_line = win32com.client.CastTo(segment, "ISketchLine")
            start_pt = sketch_line.GetStartPoint2()
            end_pt = sketch_line.GetEndPoint2()
        except Exception as e:
            pass

        # Method 2: Try ISketchSegment interface
        if start_pt is None:
            try:
                sketch_seg = win32com.client.CastTo(segment, "ISketchSegment")
                start_pt = sketch_seg.GetStartPoint2()
                end_pt = sketch_seg.GetEndPoint2()
            except Exception as e:
                pass

        # Method 3: Try to get points from the sketch directly
        if start_pt is None and self._sketch is not None:
            try:
                # Get all sketch points and match by position
                points = self._get_com_result(self._sketch, "GetSketchPoints2")
                if points and len(points) >= 2:
                    # For a line, the first two points should be the endpoints
                    # (This is a rough approximation)
                    start_pt = points[0]
                    end_pt = points[1]
            except Exception as e:
                pass

        # Method 4: Try direct attribute access with different casing
        if start_pt is None:
            for start_attr in ["GetStartPoint2", "getStartPoint2", "StartPoint", "startPoint"]:
                for end_attr in ["GetEndPoint2", "getEndPoint2", "EndPoint", "endPoint"]:
                    try:
                        start_func = getattr(segment, start_attr, None)
                        end_func = getattr(segment, end_attr, None)
                        if start_func and end_func:
                            if callable(start_func):
                                start_pt = start_func()
                                end_pt = end_func()
                            else:
                                start_pt = start_func
                                end_pt = end_func
                            if start_pt and end_pt:
                                break
                    except Exception:
                        continue
                if start_pt:
                    break

        if start_pt is None or end_pt is None:
            raise ValueError("Could not get line endpoints")

        return Line(
            start=Point2D(start_pt.X * M_TO_MM, start_pt.Y * M_TO_MM),
            end=Point2D(end_pt.X * M_TO_MM, end_pt.Y * M_TO_MM),
            construction=construction
        )

    def _export_arc(self, segment: Any, construction: bool = False) -> Arc | Circle:
        """Export a SolidWorks arc to canonical format."""
        start_pt = None
        end_pt = None
        center_pt = None
        radius = None

        # Method 1: Try to get radius directly (works for circles)
        try:
            radius = self._get_com_result(segment, "GetRadius")
            if radius is not None:
                radius = radius * M_TO_MM
        except Exception as e:
            pass

        # Method 2: Get points from the sketch (like we did for lines)
        if self._sketch is not None:
            try:
                points = self._get_com_result(self._sketch, "GetSketchPoints2")

                if points:
                    # For a circle, there should be 1 center point
                    # For an arc, there should be 3 points: center, start, end

                    if len(points) == 1:
                        # Single point = center of circle
                        center_pt = points[0]
                        if radius is not None:
                            center_x = center_pt.X * M_TO_MM
                            center_y = center_pt.Y * M_TO_MM
                            return Circle(
                                center=Point2D(center_x, center_y),
                                radius=radius,
                                construction=construction
                            )

                    elif len(points) >= 3 and radius is not None:
                        # Arc: we have center, start, end points
                        # Figure out which point is the center by checking distance to radius
                        point_coords = []
                        for pt in points:
                            point_coords.append((pt.X * M_TO_MM, pt.Y * M_TO_MM))

                        # Find the center: it's the point equidistant to other points at radius distance
                        center_idx = None
                        for i, (cx, cy) in enumerate(point_coords):
                            distances = []
                            for j, (px, py) in enumerate(point_coords):
                                if i != j:
                                    dist = math.sqrt((cx - px)**2 + (cy - py)**2)
                                    distances.append(dist)
                            # If both other points are at radius distance, this is center
                            if len(distances) == 2 and all(abs(d - radius) < 0.01 for d in distances):
                                center_idx = i
                                break

                        if center_idx is not None:
                            center_x, center_y = point_coords[center_idx]
                            other_points = [p for i, p in enumerate(point_coords) if i != center_idx]
                            start_x, start_y = other_points[0]
                            end_x, end_y = other_points[1]

                            # Determine CCW direction using cross product
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

            except Exception as e:
                pass

        # Method 3: Try to get curve and extract parameters
        try:
            curve = self._get_com_result(segment, "GetCurve")
            if curve:
                # For arcs/circles, the curve should have circle data
                is_circle = self._get_com_result(curve, "IsCircle")

                if is_circle:
                    # Get circle params: returns array [cx, cy, cz, ax, ay, az, radius]
                    # where (cx,cy,cz) is center and (ax,ay,az) is axis
                    circle_params = self._get_com_result(curve, "CircleParams")
                    if circle_params:
                        center_x = circle_params[0] * M_TO_MM
                        center_y = circle_params[1] * M_TO_MM
                        radius = circle_params[6] * M_TO_MM

                        return Circle(
                            center=Point2D(center_x, center_y),
                            radius=radius,
                            construction=construction
                        )
        except Exception as e:
            pass

        # If we get here, we couldn't export the arc/circle
        raise ValueError("Could not get arc/circle geometry")

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
