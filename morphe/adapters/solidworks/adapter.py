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
    PointType,
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


# SolidWorks sketch constraint type constants (from swConstraintType_e)
# Reference: https://help.solidworks.com/2024/english/api/swconst/swConstraintType_e.html
class SwConstraintType:
    """SolidWorks swConstraintType_e enumeration values."""

    # Geometric constraints
    HORIZONTAL = 4          # swConstraintType_HORIZONTAL
    VERTICAL = 5            # swConstraintType_VERTICAL
    TANGENT = 6             # swConstraintType_TANGENT
    PARALLEL = 7            # swConstraintType_PARALLEL
    PERPENDICULAR = 8       # swConstraintType_PERPENDICULAR
    COINCIDENT = 9          # swConstraintType_COINCIDENT
    CONCENTRIC = 10         # swConstraintType_CONCENTRIC
    SYMMETRIC = 11          # swConstraintType_SYMMETRIC
    MIDPOINT = 12           # swConstraintType_ATMIDDLE
    ATINTERSECT = 13        # swConstraintType_ATINTERSECT
    EQUAL = 14              # swConstraintType_SAMELENGTH
    FIX = 17                # swConstraintType_FIXED
    COLLINEAR = 27          # swConstraintType_COLINEAR
    CORADIAL = 28           # swConstraintType_CORADIAL

    # Dimensional constraints (for reference - accessed differently)
    DISTANCE = 1            # swConstraintType_DISTANCE
    ANGLE = 2               # swConstraintType_ANGLE
    RADIUS = 3              # swConstraintType_RADIUS
    DIAMETER = 15           # swConstraintType_DIAMETER


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

        # Disable dimension input dialog to prevent blocking
        self._disable_dimension_dialog()

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
        # Track if constraints have been applied (requires reading actual geometry)
        self._constraints_applied: bool = False
        # Track standalone Point primitive IDs (to preserve during export)
        self._standalone_point_ids: set[str] = set()
        # Property-based mappings for constraint entity matching
        # (COM returns different wrapper objects, so id() matching doesn't work)
        self._length_to_id: dict[float, str] = {}
        self._radius_to_id: dict[float, str] = {}
        self._midpoint_to_id: dict[tuple[float, float], str] = {}  # (x, y) -> id
        self._segment_index_to_id: list[str | None] = []  # index -> id

    def _disable_dimension_dialog(self) -> None:
        """Disable the dimension input dialog that blocks automation.

        SolidWorks shows a 'Modify' dialog when adding dimensions via API.
        This tries multiple approaches to disable it.
        """
        # swInputDimValOnCreate - controls whether dimension dialog appears
        # Try multiple possible indices as they vary by SW version
        preference_indices = [8, 78, 108]

        for idx in preference_indices:
            try:
                self._app.SetUserPreferenceToggle(idx, False)
            except Exception:
                pass

        # Also try setting the string preference for default dimension behavior
        try:
            # swDetailingDimInput = 201 in some versions
            self._app.SetUserPreferenceIntegerValue(201, 0)
        except Exception:
            pass

    def _ensure_document(self) -> None:
        """Ensure we have an active part document."""
        if self._document is None:
            # First, check if there's already an active document
            try:
                active_doc = self._app.ActiveDoc
                if active_doc is not None:
                    self._document = active_doc
                    return
            except Exception:
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
                "C:\\Program Files\\SOLIDWORKS Corp\\SOLIDWORKS\\lang\\english\\Tutorial",
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
                except Exception:
                    pass

                if plane_feature is not None:
                    # Select the plane feature
                    plane_feature.Select2(False, 0)
                else:
                    # Fallback: try selecting via feature manager tree traversal
                    # Use FeatureByPositionReverse since FirstFeature/GetNextFeature
                    # don't work with late-bound COM
                    try:
                        fm = model.FeatureManager
                        feature_count = fm.GetFeatureCount(True)
                        for i in range(feature_count):
                            feat = model.FeatureByPositionReverse(i)
                            if feat is not None and feat.Name == plane_name:
                                plane_feature = feat
                                plane_feature.Select2(False, 0)
                                break
                    except Exception:
                        pass
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
            self._constraints_applied = False

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

            # Save standalone point entities before clearing mappings
            standalone_point_entities = {
                pid: self._id_to_entity.get(pid)
                for pid in self._standalone_point_ids
                if self._id_to_entity.get(pid) is not None
            }

            # Clear and rebuild mappings
            self._id_to_entity.clear()
            self._entity_to_id.clear()

            # Clear matched geometry tracking for this export
            self._matched_geometry_ids = set()

            # Track point coordinates used by segments to avoid duplicating them
            used_point_coords: set[tuple[float, float]] = set()

            # Pre-compute segment-to-points matching for lines with same length
            self._used_point_pairs: set[tuple[tuple[float, float], tuple[float, float]]] = set()

            # Build a property-based mapping for matching entities from relations
            # COM returns different wrapper objects, so we can't use id() for matching
            self._length_to_id: dict[float, str] = {}
            self._radius_to_id: dict[float, str] = {}
            self._midpoint_to_id: dict[tuple[float, float], str] = {}  # (x, y) -> id
            self._segment_index_to_id: list[str | None] = []  # index -> id (for constraint export)

            # Handle stored ellipse geometry specially - SolidWorks may decompose
            # ellipses into multiple arc segments, so we use stored geometry directly
            # if constraints haven't been applied
            stored_ellipse_ids: set[str] = set()
            if not self._constraints_applied:
                for geom in self._segment_geometry_list:
                    if geom['type'] == 'ellipse':
                        prim = Ellipse(
                            id=geom.get('element_id'),
                            center=Point2D(geom['center'][0], geom['center'][1]),
                            major_radius=geom['major_radius'],
                            minor_radius=geom['minor_radius'],
                            rotation=geom.get('rotation', 0.0),
                            construction=geom.get('construction', False)
                        )
                        doc.add_primitive(prim)
                        stored_ellipse_ids.add(geom.get('element_id', ''))
                        # Add all ellipse-related points to used_point_coords
                        # SolidWorks creates points at center and major/minor axis endpoints
                        cx, cy = prim.center.x, prim.center.y
                        rot = prim.rotation
                        cos_r, sin_r = math.cos(rot), math.sin(rot)
                        used_point_coords.add((round(cx, 6), round(cy, 6)))  # center
                        # Major axis endpoints
                        used_point_coords.add((round(cx + prim.major_radius * cos_r, 6),
                                               round(cy + prim.major_radius * sin_r, 6)))
                        used_point_coords.add((round(cx - prim.major_radius * cos_r, 6),
                                               round(cy - prim.major_radius * sin_r, 6)))
                        # Minor axis endpoints (perpendicular to major axis)
                        used_point_coords.add((round(cx - prim.minor_radius * sin_r, 6),
                                               round(cy + prim.minor_radius * cos_r, 6)))
                        used_point_coords.add((round(cx + prim.minor_radius * sin_r, 6),
                                               round(cy - prim.minor_radius * cos_r, 6)))
                    elif geom['type'] == 'elliptical_arc':
                        prim = EllipticalArc(
                            id=geom.get('element_id'),
                            center=Point2D(geom['center'][0], geom['center'][1]),
                            major_radius=geom['major_radius'],
                            minor_radius=geom['minor_radius'],
                            rotation=geom.get('rotation', 0.0),
                            start_param=geom['start_param'],
                            end_param=geom['end_param'],
                            ccw=geom['ccw'],
                            construction=geom.get('construction', False)
                        )
                        doc.add_primitive(prim)
                        stored_ellipse_ids.add(geom.get('element_id', ''))
                        # Add all elliptical arc related points
                        # SolidWorks creates points at center, start, end, and axis endpoints
                        cx, cy = prim.center.x, prim.center.y
                        rot = prim.rotation
                        cos_r, sin_r = math.cos(rot), math.sin(rot)
                        used_point_coords.add((round(cx, 6), round(cy, 6)))  # center
                        used_point_coords.add((round(prim.start_point.x, 6), round(prim.start_point.y, 6)))
                        used_point_coords.add((round(prim.end_point.x, 6), round(prim.end_point.y, 6)))
                        # Major axis endpoints (SolidWorks may create these)
                        used_point_coords.add((round(cx + prim.major_radius * cos_r, 6),
                                               round(cy + prim.major_radius * sin_r, 6)))
                        used_point_coords.add((round(cx - prim.major_radius * cos_r, 6),
                                               round(cy - prim.major_radius * sin_r, 6)))
                        # Minor axis endpoints (perpendicular to major axis)
                        used_point_coords.add((round(cx - prim.minor_radius * sin_r, 6),
                                               round(cy + prim.minor_radius * cos_r, 6)))
                        used_point_coords.add((round(cx + prim.minor_radius * sin_r, 6),
                                               round(cy - prim.minor_radius * cos_r, 6)))

            # Build set of ellipse centers for skipping ellipse arc segments
            ellipse_centers: set[tuple[float, float]] = set()
            for geom in self._segment_geometry_list:
                if geom['type'] in ('ellipse', 'elliptical_arc'):
                    ellipse_centers.add((round(geom['center'][0], 2), round(geom['center'][1], 2)))

            # Get all sketch segments
            # Note: In COM late binding, GetSketchSegments may be a property returning
            # a tuple rather than a callable method
            segments = self._get_com_result(sketch, "GetSketchSegments")
            if segments:
                for seg_idx, segment in enumerate(segments):
                    # Skip segments that are part of ellipse geometry
                    # SolidWorks may create ELLIPSE segments or decompose into ARC segments
                    if stored_ellipse_ids:
                        seg_type = self._get_com_result(segment, "GetType")
                        # Always skip ELLIPSE segments if we exported from stored geometry
                        if seg_type == SwSketchSegments.ELLIPSE:
                            continue
                        # Check if ARC segment is part of an ellipse
                        if self._is_ellipse_arc_segment(segment, ellipse_centers):
                            continue

                    if self._is_construction(segment):
                        construction = True
                    else:
                        construction = False

                    prim = self._export_segment(segment, construction, seg_idx)
                    if prim is not None:
                        doc.add_primitive(prim)
                        self._entity_to_id[id(segment)] = prim.id
                        self._id_to_entity[prim.id] = segment
                        # Track segment index to ID for constraint export
                        while len(self._segment_index_to_id) <= seg_idx:
                            self._segment_index_to_id.append(None)
                        self._segment_index_to_id[seg_idx] = prim.id

                        # Build property-based mappings for constraint entity matching
                        try:
                            length = segment.GetLength
                            if length is not None:
                                # Round to avoid floating point matching issues
                                length_key = round(length, 10)
                                self._length_to_id[length_key] = prim.id
                        except Exception:
                            pass
                        try:
                            radius = segment.GetRadius
                            if radius is not None:
                                radius_key = round(radius, 10)
                                self._radius_to_id[radius_key] = prim.id
                        except Exception:
                            pass

                        # Track coordinates used by this primitive
                        if isinstance(prim, Line):
                            start_coord = (round(prim.start.x, 6), round(prim.start.y, 6))
                            end_coord = (round(prim.end.x, 6), round(prim.end.y, 6))
                            used_point_coords.add(start_coord)
                            used_point_coords.add(end_coord)
                            # Also track point pair to avoid duplicate line matching
                            pair_key = (
                                (round(prim.start.x, 4), round(prim.start.y, 4)),
                                (round(prim.end.x, 4), round(prim.end.y, 4))
                            )
                            self._used_point_pairs.add(pair_key)
                            # Build midpoint mapping for unique line identification
                            midpoint_key = (
                                round((prim.start.x + prim.end.x) / 2, 6),
                                round((prim.start.y + prim.end.y) / 2, 6)
                            )
                            self._midpoint_to_id[midpoint_key] = prim.id
                        elif isinstance(prim, Arc):
                            used_point_coords.add((round(prim.start_point.x, 6), round(prim.start_point.y, 6)))
                            used_point_coords.add((round(prim.end_point.x, 6), round(prim.end_point.y, 6)))
                            used_point_coords.add((round(prim.center.x, 6), round(prim.center.y, 6)))
                        elif isinstance(prim, Circle):
                            used_point_coords.add((round(prim.center.x, 6), round(prim.center.y, 6)))
                        elif isinstance(prim, Ellipse):
                            used_point_coords.add((round(prim.center.x, 6), round(prim.center.y, 6)))
                        elif isinstance(prim, EllipticalArc):
                            used_point_coords.add((round(prim.center.x, 6), round(prim.center.y, 6)))
                            used_point_coords.add((round(prim.start_point.x, 6), round(prim.start_point.y, 6)))
                            used_point_coords.add((round(prim.end_point.x, 6), round(prim.end_point.y, 6)))
                        elif isinstance(prim, Spline):
                            for pt in prim.control_points:
                                used_point_coords.add((round(pt.x, 6), round(pt.y, 6)))

            # Export standalone points
            # First, export Points that we explicitly created (tracked in _standalone_point_ids)
            exported_point_coords: set[tuple[float, float]] = set()
            for point_id, point_entity in standalone_point_entities.items():
                if point_entity is not None:
                    prim = self._export_point(point_entity)
                    prim.id = point_id  # Preserve original ID
                    doc.add_primitive(prim)
                    point_coords = (round(prim.position.x, 6), round(prim.position.y, 6))
                    exported_point_coords.add(point_coords)
                    used_point_coords.add(point_coords)

            # Then export any other standalone points (skip points that are part of segments)
            points = self._get_com_result(sketch, "GetSketchPoints2")
            if points:
                for point in points:
                    # Skip points that are part of other geometry
                    if self._is_dependent_point(point):
                        continue

                    # Export the point
                    prim = self._export_point(point)

                    # Skip if this point's coordinates match a segment endpoint or already exported
                    point_coords = (round(prim.position.x, 6), round(prim.position.y, 6))
                    if point_coords in used_point_coords:
                        continue
                    if point_coords in exported_point_coords:
                        continue

                    # Skip points that lie on or inside ellipse/elliptical arc curves
                    # (SolidWorks creates internal vertex points on these curves)
                    if self._point_on_ellipse_curve(prim.position.x, prim.position.y):
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
        try:
            attr = getattr(obj, attr_name, None)
        except Exception:
            return None
        if attr is None:
            return None
        # If it's callable (a method), call it
        if callable(attr):
            try:
                return attr()
            except (TypeError, Exception):
                # If calling fails, it might be a property that looks callable
                # or a COM error - try returning the attribute itself
                try:
                    return attr
                except Exception:
                    return None
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

    def _point_on_ellipse_curve(self, px: float, py: float, tolerance: float = 0.5) -> bool:
        """Check if a point lies on or near any stored ellipse or elliptical arc curve.

        SolidWorks creates internal vertex/control points on ellipse curves that
        should not be exported as standalone Point primitives.

        Args:
            px: Point X coordinate in mm
            py: Point Y coordinate in mm
            tolerance: Distance tolerance in mm

        Returns:
            True if the point lies on or inside an ellipse/elliptical arc region
        """
        for geom in self._segment_geometry_list:
            if geom['type'] not in ('ellipse', 'elliptical_arc'):
                continue

            cx, cy = geom['center']
            major_r = geom['major_radius']
            minor_r = geom['minor_radius']
            rotation = geom.get('rotation', 0.0)

            # Transform point to ellipse-local coordinates
            dx = px - cx
            dy = py - cy

            # Check if point is within the bounding region of the ellipse
            dist_from_center = math.sqrt(dx * dx + dy * dy)
            if dist_from_center <= major_r + tolerance:
                # Rotate to align with ellipse axes
                cos_r = math.cos(-rotation)
                sin_r = math.sin(-rotation)
                local_x = dx * cos_r - dy * sin_r
                local_y = dx * sin_r + dy * cos_r

                if major_r > 0 and minor_r > 0:
                    # Check ellipse equation: (x/a)^2 + (y/b)^2 = 1
                    # Points on or inside the ellipse should be filtered
                    normalized_dist = (local_x / major_r) ** 2 + (local_y / minor_r) ** 2

                    # Filter points on the ellipse curve (normalized_dist ≈ 1)
                    # or inside the ellipse (normalized_dist < 1)
                    curve_tolerance = tolerance / min(major_r, minor_r)
                    if normalized_dist < 1.0 + curve_tolerance:
                        return True

        return False

    def _is_ellipse_arc_segment(self, segment: Any, ellipse_centers: set[tuple[float, float]]) -> bool:
        """Check if a segment is part of ellipse geometry.

        SolidWorks may decompose ellipses into multiple arc segments. This method
        checks if a segment's curve has ellipse parameters with a center matching
        one of the known ellipse centers.

        Args:
            segment: SolidWorks sketch segment
            ellipse_centers: Set of (x, y) tuples for known ellipse centers (in mm)

        Returns:
            True if segment appears to be part of an ellipse
        """
        if not ellipse_centers:
            return False

        try:
            # Check segment type - only arcs and ellipse segments could be ellipse parts
            seg_type = self._get_com_result(segment, "GetType")
            if seg_type not in (SwSketchSegments.ARC, SwSketchSegments.ELLIPSE):
                return False

            # Get the curve
            curve = self._get_com_result(segment, "GetCurve")
            if not curve:
                return False

            # Method 1: Try to get ellipse params directly (works even if IsEllipse is False)
            params = self._get_com_result(curve, "EllipseParams")
            if params and len(params) >= 3:
                center_x = params[0] * M_TO_MM
                center_y = params[1] * M_TO_MM
                center_key = (round(center_x, 2), round(center_y, 2))
                if center_key in ellipse_centers:
                    return True

            # Method 2: For arc segments, check if center matches using CircleParams
            # (Some ellipse arcs might report as circular arcs with center at ellipse center)
            if seg_type == SwSketchSegments.ARC:
                circle_params = self._get_com_result(curve, "CircleParams")
                if circle_params and len(circle_params) >= 3:
                    center_x = circle_params[0] * M_TO_MM
                    center_y = circle_params[1] * M_TO_MM
                    center_key = (round(center_x, 2), round(center_y, 2))
                    if center_key in ellipse_centers:
                        return True

        except Exception:
            pass

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
            elif isinstance(primitive, Ellipse):
                entity = self._add_ellipse(primitive)
            elif isinstance(primitive, EllipticalArc):
                entity = self._add_elliptical_arc(primitive)
            elif isinstance(primitive, Point):
                entity = self._add_point(primitive)
                self._standalone_point_ids.add(primitive.id)
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
                'element_id': line.id,
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
                'element_id': circle.id,
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
                'element_id': arc.id,
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

        # Store geometry data for export
        if segment is not None:
            self._segment_geometry_list.append({
                'type': 'spline',
                'element_id': spline.id,
                'control_points': [(pt.x, pt.y) for pt in spline.control_points],
                'degree': spline.degree,
                'construction': spline.construction
            })

        return segment

    def _add_ellipse(self, ellipse: Ellipse) -> Any:
        """Add an ellipse to the sketch."""
        assert self._sketch_manager is not None

        # SolidWorks CreateEllipse(Xc, Yc, Zc, Xmajor, Ymajor, Zmajor, Xminor, Yminor, Zminor)
        # We need to compute the major and minor axis endpoints from center, radii, and rotation
        cos_r = math.cos(ellipse.rotation)
        sin_r = math.sin(ellipse.rotation)

        # Major axis endpoint (from center along major axis direction)
        major_x = ellipse.center.x + ellipse.major_radius * cos_r
        major_y = ellipse.center.y + ellipse.major_radius * sin_r

        # Minor axis endpoint (from center, perpendicular to major axis)
        minor_x = ellipse.center.x - ellipse.minor_radius * sin_r
        minor_y = ellipse.center.y + ellipse.minor_radius * cos_r

        segment = self._sketch_manager.CreateEllipse(
            ellipse.center.x * MM_TO_M,
            ellipse.center.y * MM_TO_M,
            0,  # Zc
            major_x * MM_TO_M,
            major_y * MM_TO_M,
            0,  # Zmajor
            minor_x * MM_TO_M,
            minor_y * MM_TO_M,
            0   # Zminor
        )

        # Store geometry data for export - always store, even if segment is None
        # (CreateEllipse may return None in late binding even when successful)
        self._segment_geometry_list.append({
            'type': 'ellipse',
            'element_id': ellipse.id,
            'center': (ellipse.center.x, ellipse.center.y),
            'major_radius': ellipse.major_radius,
            'minor_radius': ellipse.minor_radius,
            'rotation': ellipse.rotation,
            'construction': ellipse.construction
        })

        return segment

    def _add_elliptical_arc(self, arc: EllipticalArc) -> Any:
        """Add an elliptical arc to the sketch."""
        assert self._sketch_manager is not None

        # SolidWorks CreateEllipticalArc takes 16 parameters:
        # (Xc, Yc, Zc, Xmajor, Ymajor, Zmajor, Xs, Ys, Zs, Xe, Ye, Ze, Xdir, Ydir, Zdir, Direction)
        # Direction: 1 = CCW, -1 = CW
        cos_r = math.cos(arc.rotation)
        sin_r = math.sin(arc.rotation)

        major_x = arc.center.x + arc.major_radius * cos_r
        major_y = arc.center.y + arc.major_radius * sin_r

        # Get start and end points from the arc parameters
        start_pt = arc.start_point
        end_pt = arc.end_point

        # Direction: 1 = CCW, -1 = CW
        direction = 1 if arc.ccw else -1

        segment = self._sketch_manager.CreateEllipticalArc(
            arc.center.x * MM_TO_M,
            arc.center.y * MM_TO_M,
            0,  # Zc
            major_x * MM_TO_M,
            major_y * MM_TO_M,
            0,  # Zmajor
            start_pt.x * MM_TO_M,
            start_pt.y * MM_TO_M,
            0,  # Zs
            end_pt.x * MM_TO_M,
            end_pt.y * MM_TO_M,
            0,  # Ze
            0,  # Xdir (direction vector, perpendicular to sketch plane)
            0,  # Ydir
            1,  # Zdir (Z-up for XY plane sketch)
            direction  # Arc direction: 1=CCW, -1=CW
        )

        # Store geometry data for export - always store, even if segment is None
        # (CreateEllipticalArc may return None in late binding even when successful)
        self._segment_geometry_list.append({
            'type': 'elliptical_arc',
            'element_id': arc.id,
            'center': (arc.center.x, arc.center.y),
            'major_radius': arc.major_radius,
            'minor_radius': arc.minor_radius,
            'rotation': arc.rotation,
            'start_param': arc.start_param,
            'end_param': arc.end_param,
            'ccw': arc.ccw,
            'construction': arc.construction
        })

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

            # Mark that constraints are being applied (geometry may change)
            self._constraints_applied = True

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
            elif ctype == ConstraintType.SYMMETRIC:
                return self._add_symmetric(model, refs)

            # Dimensional constraints - disable input dialog first
            elif ctype == ConstraintType.DISTANCE:
                return self._add_dimension_constraint(lambda: self._add_distance(model, refs, value))
            elif ctype == ConstraintType.RADIUS:
                return self._add_dimension_constraint(lambda: self._add_radius(model, refs, value))
            elif ctype == ConstraintType.DIAMETER:
                return self._add_dimension_constraint(lambda: self._add_diameter(model, refs, value))
            elif ctype == ConstraintType.ANGLE:
                return self._add_dimension_constraint(lambda: self._add_angle(model, refs, value))
            elif ctype == ConstraintType.LENGTH:
                return self._add_dimension_constraint(lambda: self._add_length(model, refs, value))
            elif ctype == ConstraintType.DISTANCE_X:
                return self._add_dimension_constraint(lambda: self._add_distance_x(model, refs, value))
            elif ctype == ConstraintType.DISTANCE_Y:
                return self._add_dimension_constraint(lambda: self._add_distance_y(model, refs, value))

            else:
                raise ConstraintError(f"Unsupported constraint type: {ctype}")

        except ConstraintError:
            raise
        except Exception as e:
            raise ConstraintError(f"Failed to add constraint: {e}") from e

    def _add_dimension_constraint(self, add_func) -> bool:
        """Wrapper to add dimensional constraints with dialog suppression.

        Temporarily disables the dimension input dialog to prevent blocking.
        """
        # swInputDimValOnCreate = 8 controls whether dimension dialog appears
        try:
            # Save current setting
            old_val = self._app.GetUserPreferenceToggle(8)
            # Disable the input dialog
            self._app.SetUserPreferenceToggle(8, False)
        except Exception:
            old_val = None

        try:
            return add_func()
        finally:
            # Restore setting
            if old_val is not None:
                try:
                    self._app.SetUserPreferenceToggle(8, old_val)
                except Exception:
                    pass

    def _select_entity(self, ref: str | PointRef, append: bool = False) -> bool:
        """Select an entity or point for constraint creation."""
        try:
            if isinstance(ref, PointRef):
                entity = self._id_to_entity.get(ref.element_id)
                if entity is None:
                    return False

                # Try to get point from entity directly
                point = get_sketch_point_from_entity(entity, ref.point_type)

                # If that failed, try finding the point by coordinates
                if point is None:
                    point = self._find_sketch_point_by_coords(ref.element_id, ref.point_type)

                if point is None:
                    return False

                # Select the point
                return bool(point.Select(append))
            else:
                entity = self._id_to_entity.get(ref)
                if entity is None:
                    return False
                # Use Select() instead of Select4() for COM compatibility
                return bool(entity.Select(append))
        except Exception:
            return False

    def _find_sketch_point_by_coords(self, element_id: str, point_type: PointType) -> Any:
        """Find a sketch point by looking up stored coordinates and matching."""
        if self._sketch is None:
            return None

        # Find the index of this element in the geometry list
        entity = self._id_to_entity.get(element_id)
        if entity is None:
            return None

        # Find the stored geometry for this element
        entity_index = self._entity_to_id.get(id(entity))
        if entity_index is None:
            # Try to find by searching through entities
            # (placeholder for future implementation)
            pass

        # Look up the geometry by element_id
        geom = None
        for g in self._segment_geometry_list:
            # Match by checking if the entity at this index is our entity
            # Since we can't compare COM objects reliably, use the stored coordinates
            if g.get('element_id') == element_id:
                geom = g
                break

        # If we don't have stored geometry with element_id, try matching by index
        if geom is None:
            # Find the entity index in the order we stored it
            for prim_id, _ent in self._id_to_entity.items():
                if prim_id == element_id:
                    # Find the index of this primitive
                    # We need to track which geometry belongs to which primitive
                    break

            # Can't reliably match without element_id
            return None

        # Get the target coordinates based on point type
        target_x, target_y = None, None
        if point_type == PointType.START:
            if 'start' in geom:
                target_x, target_y = geom['start']
        elif point_type == PointType.END:
            if 'end' in geom:
                target_x, target_y = geom['end']
        elif point_type == PointType.CENTER:
            if 'center' in geom:
                target_x, target_y = geom['center']

        if target_x is None:
            return None

        # Convert to meters for comparison
        target_x_m = target_x * MM_TO_M
        target_y_m = target_y * MM_TO_M

        # Get all sketch points and find the one at these coordinates
        points = self._get_com_result(self._sketch, "GetSketchPoints2")
        if not points:
            return None

        tolerance = 1e-6  # meters
        for pt in points:
            try:
                px = pt.X
                py = pt.Y
                if abs(px - target_x_m) < tolerance and abs(py - target_y_m) < tolerance:
                    return pt
            except Exception:
                continue

        return None

    def _add_coincident(self, model: Any, refs: list) -> bool:
        """Add a coincident constraint.

        For standalone Point primitives, we move the point to the target location
        since SolidWorks constraints may not always move geometry.
        """
        if len(refs) < 2:
            raise ConstraintError("Coincident requires 2 references")

        ref1, ref2 = refs[0], refs[1]

        # Check if first reference is a standalone Point that needs to be moved
        if isinstance(ref1, PointRef):
            point_id = ref1.element_id
            # Check if it's a standalone Point (not a segment endpoint)
            is_standalone = True
            for geom in self._segment_geometry_list:
                if geom.get('element_id') == point_id:
                    is_standalone = False
                    break

            if is_standalone:
                # Get target coordinates from second reference
                target_coords = None
                if isinstance(ref2, PointRef):
                    target_coords = self._get_point_coords(ref2)

                if target_coords is not None:
                    # Move the point to target location
                    point_entity = self._id_to_entity.get(point_id)
                    if point_entity is not None:
                        model.ClearSelection2(True)
                        point_entity.Select(False)
                        model.EditDelete()

                        assert self._sketch_manager is not None
                        new_point = self._sketch_manager.CreatePoint(
                            target_coords[0] * MM_TO_M,
                            target_coords[1] * MM_TO_M,
                            0
                        )

                        if new_point is not None:
                            self._id_to_entity[point_id] = new_point
                        return True

        # Default: apply SolidWorks constraint
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
        """Add a concentric constraint by moving the second entity's center.

        Note: SolidWorks API's sgCONCENTRIC and sgCOINCIDENT on center points
        don't reliably move geometry. Instead, we delete and recreate the second
        circle/arc at the first entity's center position.
        """
        if len(refs) < 2:
            raise ConstraintError("Concentric requires 2 references")

        entity1_id = refs[0]
        entity2_id = refs[1]

        # Get the center of the first entity
        center1_x, center1_y = None, None
        for geom in self._segment_geometry_list:
            if geom.get('element_id') == entity1_id:
                if geom['type'] in ('circle', 'arc'):
                    center1_x, center1_y = geom['center']
                break

        if center1_x is None:
            raise ConstraintError(f"Could not find center for first entity: {entity1_id}")

        # Get the second entity's geometry info
        geom2 = None
        for geom in self._segment_geometry_list:
            if geom.get('element_id') == entity2_id:
                geom2 = geom
                break

        if geom2 is None or geom2['type'] not in ('circle', 'arc'):
            raise ConstraintError(f"Second entity must be a circle or arc: {entity2_id}")

        # Get the second entity for deletion
        entity2 = self._id_to_entity.get(entity2_id)
        if entity2 is None:
            raise ConstraintError(f"Could not find second entity: {entity2_id}")

        # Delete the second entity
        model.ClearSelection2(True)
        entity2.Select(False)
        model.EditDelete()

        # Recreate the second entity at the first entity's center
        if geom2['type'] == 'circle':
            radius = geom2['radius']
            new_entity = self._sketch_manager.CreateCircle(
                center1_x * MM_TO_M, center1_y * MM_TO_M, 0,
                (center1_x + radius) * MM_TO_M, center1_y * MM_TO_M, 0
            )
        else:  # arc
            # For arcs, we need start/end angles and radius
            radius = geom2['radius']
            start_angle = geom2.get('start_angle', 0)
            end_angle = geom2.get('end_angle', math.pi)
            new_entity = self._sketch_manager.CreateArc(
                center1_x * MM_TO_M, center1_y * MM_TO_M, 0,
                (center1_x + radius * math.cos(start_angle)) * MM_TO_M,
                (center1_y + radius * math.sin(start_angle)) * MM_TO_M, 0,
                (center1_x + radius * math.cos(end_angle)) * MM_TO_M,
                (center1_y + radius * math.sin(end_angle)) * MM_TO_M, 0,
                1  # Direction
            )

        # Update mappings
        if new_entity is not None:
            self._id_to_entity[entity2_id] = new_entity
            # Update stored geometry
            geom2['center'] = (center1_x, center1_y)

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
        """Add a midpoint constraint by moving the point to the line's midpoint.

        Note: Using SketchAddConstraints may not always move the geometry.
        Instead, we calculate the midpoint and move the point directly.
        """
        if len(refs) < 2:
            raise ConstraintError("Midpoint requires 2 references")

        # First ref is the point, second is the line
        point_ref = refs[0]
        line_ref = refs[1]

        # Get line geometry to calculate midpoint
        line_id = line_ref.element_id if isinstance(line_ref, PointRef) else line_ref
        line_geom = None
        for geom in self._segment_geometry_list:
            if geom.get('element_id') == line_id and geom['type'] == 'line':
                line_geom = geom
                break

        if line_geom is None:
            # Try the traditional constraint approach
            model.ClearSelection2(True)
            if not self._select_entity(refs[0], False):
                raise ConstraintError("Could not select point")
            if not self._select_entity(refs[1], True):
                raise ConstraintError("Could not select line")
            model.SketchAddConstraints("sgATMIDDLE")
            return True

        # Calculate midpoint
        mid_x = (line_geom['start'][0] + line_geom['end'][0]) / 2
        mid_y = (line_geom['start'][1] + line_geom['end'][1]) / 2

        # Move the point to the midpoint
        if isinstance(point_ref, PointRef):
            point_id = point_ref.element_id
            # Check if this is a standalone Point primitive
            for geom in self._segment_geometry_list:
                if geom.get('element_id') == point_id:
                    # It's a segment point - use _move_point
                    return self._move_point(point_ref, mid_x, mid_y)

            # It's a standalone Point - find and recreate it
            point_entity = self._id_to_entity.get(point_id)
            if point_entity is not None:
                # Delete the point
                model.ClearSelection2(True)
                point_entity.Select(False)
                model.EditDelete()

                # Create new point at midpoint
                assert self._sketch_manager is not None
                new_point = self._sketch_manager.CreatePoint(
                    mid_x * MM_TO_M,
                    mid_y * MM_TO_M,
                    0
                )

                if new_point is not None:
                    self._id_to_entity[point_id] = new_point
                return True

        raise ConstraintError("Could not apply midpoint constraint")

    def _add_fixed(self, model: Any, refs: list) -> bool:
        """Add a fixed constraint."""
        if len(refs) < 1:
            raise ConstraintError("Fixed requires at least 1 reference")

        model.ClearSelection2(True)
        if not self._select_entity(refs[0], False):
            raise ConstraintError("Could not select entity")

        model.SketchAddConstraints("sgFIXED")
        return True

    def _add_symmetric(self, model: Any, refs: list) -> bool:
        """Add a symmetric constraint.

        The symmetric constraint makes two elements symmetric about a line.
        References: [element1, element2, symmetry_axis]

        For point symmetry: both element1 and element2 are PointRefs
        For line/entity symmetry: element1 and element2 are entity IDs
        The symmetry_axis is always a line entity ID.
        """
        if len(refs) < 3:
            raise ConstraintError("Symmetric requires 3 references: element1, element2, axis")

        ref1 = refs[0]
        ref2 = refs[1]
        axis_ref = refs[2]

        model.ClearSelection2(True)

        # Select first element (point or entity)
        if not self._select_entity(ref1, False):
            raise ConstraintError("Could not select first element")

        # Select second element (point or entity)
        if not self._select_entity(ref2, True):
            raise ConstraintError("Could not select second element")

        # Select the symmetry axis (always a line)
        if not self._select_entity(axis_ref, True):
            raise ConstraintError("Could not select symmetry axis")

        model.SketchAddConstraints("sgSYMMETRIC")
        return True

    def _add_distance(self, model: Any, refs: list, value: float | None) -> bool:
        """Add a distance constraint by modifying geometry.

        Note: Using AddDimension2 opens a blocking dialog in SolidWorks.
        Instead, we modify the geometry directly to achieve the target distance.
        """
        if value is None:
            raise ConstraintError("Distance requires a value")
        if len(refs) < 2:
            raise ConstraintError("Distance requires 2 references")

        ref1, ref2 = refs[0], refs[1]

        # Check if both references are PointRefs on the same element (line)
        if isinstance(ref1, PointRef) and isinstance(ref2, PointRef):
            if ref1.element_id == ref2.element_id:
                # Both points on same element - this is effectively a length constraint
                return self._add_length(model, [ref1.element_id], value)

            # Points on different elements - need to move one point to achieve distance
            # Get the coordinates of both points
            pt1_coords = self._get_point_coords(ref1)
            pt2_coords = self._get_point_coords(ref2)

            if pt1_coords is None or pt2_coords is None:
                raise ConstraintError("Could not find point coordinates")

            # Calculate current distance
            dx = pt2_coords[0] - pt1_coords[0]
            dy = pt2_coords[1] - pt1_coords[1]
            current_dist = math.sqrt(dx*dx + dy*dy)

            if current_dist < 1e-9:
                raise ConstraintError("Points are coincident")

            # Scale to target distance - move second point
            scale = value / current_dist
            new_x = pt1_coords[0] + dx * scale
            new_y = pt1_coords[1] + dy * scale

            # Update the geometry of the second element
            return self._move_point(ref2, new_x, new_y)

        # Distance constraint requires PointRef references for geometry recreation
        raise ConstraintError("Distance constraint requires PointRef references")

    def _get_point_coords(self, ref: PointRef) -> tuple[float, float] | None:
        """Get coordinates of a point reference from stored geometry."""
        for geom in self._segment_geometry_list:
            if geom.get('element_id') == ref.element_id:
                if ref.point_type == PointType.START and 'start' in geom:
                    return geom['start']
                elif ref.point_type == PointType.END and 'end' in geom:
                    return geom['end']
                elif ref.point_type == PointType.CENTER and 'center' in geom:
                    return geom['center']
        return None

    def _move_point(self, ref: PointRef, new_x: float, new_y: float) -> bool:
        """Move a point by recreating its parent geometry with the new position."""
        entity_id = ref.element_id
        entity = self._id_to_entity.get(entity_id)
        if entity is None:
            raise ConstraintError("Could not find entity")

        # Find the stored geometry
        geom = None
        for g in self._segment_geometry_list:
            if g.get('element_id') == entity_id:
                geom = g
                break

        if geom is None:
            raise ConstraintError("Could not find geometry data")

        model = self._document
        assert model is not None
        assert self._sketch_manager is not None

        if geom['type'] == 'line':
            # Get current line geometry
            start_x, start_y = geom['start']
            end_x, end_y = geom['end']

            # Update the appropriate point
            if ref.point_type == PointType.START:
                start_x, start_y = new_x, new_y
            elif ref.point_type == PointType.END:
                end_x, end_y = new_x, new_y
            else:
                raise ConstraintError("Invalid point type for line")

            # Delete the original entity
            model.ClearSelection2(True)
            entity.Select(False)
            model.EditDelete()

            # Create new line
            new_entity = self._sketch_manager.CreateLine(
                start_x * MM_TO_M, start_y * MM_TO_M, 0,
                end_x * MM_TO_M, end_y * MM_TO_M, 0
            )

            # Update mappings
            if new_entity is not None:
                self._id_to_entity[entity_id] = new_entity
                geom['start'] = (start_x, start_y)
                geom['end'] = (end_x, end_y)

            return True

        raise ConstraintError(f"Cannot move point on geometry type: {geom['type']}")

    def _add_radius(self, model: Any, refs: list, value: float | None) -> bool:
        """Add a radius constraint by recreating geometry with target radius.

        Note: Using AddDimension2 opens a blocking dialog in SolidWorks.
        Instead, we delete the original geometry and recreate it with the target radius.
        """
        if value is None:
            raise ConstraintError("Radius requires a value")
        if len(refs) < 1:
            raise ConstraintError("Radius requires 1 reference")

        entity_ref = refs[0]
        entity_id = entity_ref.element_id if isinstance(entity_ref, PointRef) else entity_ref
        entity = self._id_to_entity.get(entity_id)
        if entity is None:
            raise ConstraintError("Could not find entity")

        # Get the current center from stored geometry
        center_x, center_y = None, None
        for geom in self._segment_geometry_list:
            if geom.get('element_id') == entity_id:
                if geom['type'] == 'circle':
                    center_x, center_y = geom['center']
                elif geom['type'] == 'arc':
                    center_x, center_y = geom['center']
                break

        if center_x is None:
            raise ConstraintError("Could not find circle/arc center")

        # Delete the original entity
        model.ClearSelection2(True)
        entity.Select(False)
        model.EditDelete()

        # Create new circle with correct radius
        assert self._sketch_manager is not None
        new_entity = self._sketch_manager.CreateCircle(
            center_x * MM_TO_M,
            center_y * MM_TO_M,
            0,
            (center_x + value) * MM_TO_M,  # Point on circle at new radius
            center_y * MM_TO_M,
            0
        )

        # Update mappings
        if new_entity is not None:
            self._id_to_entity[entity_id] = new_entity
            # Update stored geometry
            for geom in self._segment_geometry_list:
                if geom.get('element_id') == entity_id:
                    geom['radius'] = value
                    break

        return True

    def _add_diameter(self, model: Any, refs: list, value: float | None) -> bool:
        """Add a diameter constraint."""
        if value is None:
            raise ConstraintError("Diameter requires a value")

        # SolidWorks uses radius, so convert diameter to radius
        return self._add_radius(model, refs, value / 2)

    def _add_angle(self, model: Any, refs: list, value: float | None) -> bool:
        """Add an angle constraint by rotating the second line.

        Note: Using AddDimension2 opens a blocking dialog in SolidWorks.
        Instead, we rotate the second line to achieve the target angle.
        """
        if value is None:
            raise ConstraintError("Angle requires a value")
        if len(refs) < 2:
            raise ConstraintError("Angle requires 2 references")

        # Get geometry for both lines
        line1_id = refs[0].element_id if isinstance(refs[0], PointRef) else refs[0]
        line2_id = refs[1].element_id if isinstance(refs[1], PointRef) else refs[1]

        line1_geom = None
        line2_geom = None
        for geom in self._segment_geometry_list:
            if geom.get('element_id') == line1_id and geom['type'] == 'line':
                line1_geom = geom
            elif geom.get('element_id') == line2_id and geom['type'] == 'line':
                line2_geom = geom

        if line1_geom is None or line2_geom is None:
            raise ConstraintError("Could not find line geometry for angle constraint")

        # Calculate direction vectors
        dx1 = line1_geom['end'][0] - line1_geom['start'][0]
        dy1 = line1_geom['end'][1] - line1_geom['start'][1]
        len1 = math.sqrt(dx1*dx1 + dy1*dy1)

        dx2 = line2_geom['end'][0] - line2_geom['start'][0]
        dy2 = line2_geom['end'][1] - line2_geom['start'][1]
        len2 = math.sqrt(dx2*dx2 + dy2*dy2)

        if len1 < 1e-9 or len2 < 1e-9:
            raise ConstraintError("Lines have zero length")

        # Calculate angle of line1 from horizontal
        angle1 = math.atan2(dy1, dx1)

        # Calculate new angle for line2 (line1_angle + target_angle)
        target_angle_rad = math.radians(value)
        new_angle2 = angle1 + target_angle_rad

        # Rotate line2 to the new angle, keeping its start point fixed
        start2_x, start2_y = line2_geom['start']
        new_end2_x = start2_x + len2 * math.cos(new_angle2)
        new_end2_y = start2_y + len2 * math.sin(new_angle2)

        # Delete and recreate line2
        entity2 = self._id_to_entity.get(line2_id)
        if entity2 is None:
            raise ConstraintError("Could not find second line entity")

        model.ClearSelection2(True)
        entity2.Select(False)
        model.EditDelete()

        assert self._sketch_manager is not None
        new_entity = self._sketch_manager.CreateLine(
            start2_x * MM_TO_M, start2_y * MM_TO_M, 0,
            new_end2_x * MM_TO_M, new_end2_y * MM_TO_M, 0
        )

        # Update mappings
        if new_entity is not None:
            self._id_to_entity[line2_id] = new_entity
            line2_geom['end'] = (new_end2_x, new_end2_y)

        return True

    def _add_length(self, model: Any, refs: list, value: float | None) -> bool:
        """Add a length constraint by recreating the line with target length.

        Note: Using AddDimension2 opens a blocking dialog in SolidWorks.
        Instead, we delete the original line and recreate it with the target length.
        """
        if value is None:
            raise ConstraintError("Length requires a value")
        if len(refs) < 1:
            raise ConstraintError("Length requires 1 reference")

        entity_ref = refs[0]
        entity_id = entity_ref.element_id if isinstance(entity_ref, PointRef) else entity_ref
        entity = self._id_to_entity.get(entity_id)
        if entity is None:
            raise ConstraintError("Could not find entity")

        # Try to get CURRENT line geometry from the COM object first
        # (in case other constraints have modified the geometry)
        start_x, start_y, end_x, end_y = None, None, None, None
        try:
            # Try to get current line endpoints from SolidWorks
            line_obj = self._export_line(entity, construction=False)
            start_x, start_y = line_obj.start.x, line_obj.start.y
            end_x, end_y = line_obj.end.x, line_obj.end.y
        except Exception:
            # Fall back to stored geometry
            for geom in self._segment_geometry_list:
                if geom.get('element_id') == entity_id and geom['type'] == 'line':
                    start_x, start_y = geom['start']
                    end_x, end_y = geom['end']
                    break

        if start_x is None:
            raise ConstraintError("Could not find line geometry")

        # Calculate new endpoint at target length (keep direction)
        dx = end_x - start_x
        dy = end_y - start_y
        current_length = math.sqrt(dx*dx + dy*dy)
        if current_length < 1e-9:
            raise ConstraintError("Line has zero length")

        # Scale to target length
        scale = value / current_length
        new_end_x = start_x + dx * scale
        new_end_y = start_y + dy * scale

        # Delete the original entity
        model.ClearSelection2(True)
        entity.Select(False)
        model.EditDelete()

        # Create new line with correct length
        assert self._sketch_manager is not None
        new_entity = self._sketch_manager.CreateLine(
            start_x * MM_TO_M,
            start_y * MM_TO_M,
            0,
            new_end_x * MM_TO_M,
            new_end_y * MM_TO_M,
            0
        )

        # Update mappings
        if new_entity is not None:
            self._id_to_entity[entity_id] = new_entity
            # Update stored geometry
            for geom in self._segment_geometry_list:
                if geom.get('element_id') == entity_id:
                    geom['end'] = (new_end_x, new_end_y)
                    break

        return True

    def _add_distance_x(self, model: Any, refs: list, value: float | None) -> bool:
        """Add a horizontal distance constraint by moving geometry.

        Note: Using AddDimension opens a blocking dialog in SolidWorks.
        Instead, we move the second point to achieve the target X distance.
        """
        if value is None:
            raise ConstraintError("DistanceX requires a value")
        if len(refs) < 2:
            raise ConstraintError("DistanceX requires 2 references")

        ref1, ref2 = refs[0], refs[1]

        if isinstance(ref1, PointRef) and isinstance(ref2, PointRef):
            pt1_coords = self._get_point_coords(ref1)
            pt2_coords = self._get_point_coords(ref2)

            if pt1_coords is None or pt2_coords is None:
                raise ConstraintError("Could not find point coordinates")

            # Calculate new X position for point 2 to achieve target distance
            new_x = pt1_coords[0] + value
            new_y = pt2_coords[1]  # Keep Y unchanged

            return self._move_point(ref2, new_x, new_y)

        raise ConstraintError("DistanceX requires point references")

    def _add_distance_y(self, model: Any, refs: list, value: float | None) -> bool:
        """Add a vertical distance constraint by moving geometry.

        Note: Using AddDimension opens a blocking dialog in SolidWorks.
        Instead, we move the second point to achieve the target Y distance.
        """
        if value is None:
            raise ConstraintError("DistanceY requires a value")
        if len(refs) < 2:
            raise ConstraintError("DistanceY requires 2 references")

        ref1, ref2 = refs[0], refs[1]

        if isinstance(ref1, PointRef) and isinstance(ref2, PointRef):
            pt1_coords = self._get_point_coords(ref1)
            pt2_coords = self._get_point_coords(ref2)

            if pt1_coords is None or pt2_coords is None:
                raise ConstraintError("Could not find point coordinates")

            # Calculate new Y position for point 2 to achieve target distance
            new_x = pt2_coords[0]  # Keep X unchanged
            new_y = pt1_coords[1] + value

            return self._move_point(ref2, new_x, new_y)

        raise ConstraintError("DistanceY requires point references")

    # =========================================================================
    # Export Methods
    # =========================================================================

    def _find_matching_stored_geometry(self, segment: Any, seg_type: int) -> dict | None:
        """Find stored geometry that matches a COM segment by type and properties.

        Args:
            segment: SolidWorks sketch segment COM object
            seg_type: Segment type from GetType()

        Returns:
            Matching stored geometry dict, or None if not found
        """
        # Map SolidWorks segment type to our type strings
        type_map = {
            SwSketchSegments.LINE: 'line',
            SwSketchSegments.ARC: ['arc', 'circle'],  # Arc can be arc or circle
            SwSketchSegments.ELLIPSE: ['ellipse', 'elliptical_arc'],  # Ellipse segment type covers both
            SwSketchSegments.SPLINE: 'spline',
        }

        expected_types = type_map.get(seg_type)
        if expected_types is None:
            return None

        if isinstance(expected_types, str):
            expected_types = [expected_types]

        # Track which stored geometries have already been matched
        if not hasattr(self, '_matched_geometry_ids'):
            self._matched_geometry_ids: set[str] = set()

        # Try to get COM segment properties for matching
        seg_length = None
        seg_radius = None
        try:
            seg_length = segment.GetLength  # in meters
        except Exception:
            pass
        try:
            seg_radius = segment.GetRadius  # in meters
            if seg_radius is not None:
                seg_radius = seg_radius * M_TO_MM  # convert to mm
        except Exception:
            pass

        # Find matching stored geometry
        for geom in self._segment_geometry_list:
            # Skip already matched geometries
            elem_id = geom.get('element_id')
            if elem_id and elem_id in self._matched_geometry_ids:
                continue

            # Check type match
            if geom['type'] not in expected_types:
                continue

            # For lines, try to match by length
            if geom['type'] == 'line' and seg_length is not None:
                stored_dx = geom['end'][0] - geom['start'][0]
                stored_dy = geom['end'][1] - geom['start'][1]
                stored_length_mm = math.sqrt(stored_dx**2 + stored_dy**2)
                stored_length_m = stored_length_mm * MM_TO_M
                if abs(stored_length_m - seg_length) < 1e-6:
                    if elem_id:
                        self._matched_geometry_ids.add(elem_id)
                    return geom

            # For arcs, match by radius
            elif geom['type'] == 'arc' and seg_radius is not None:
                # Calculate stored arc radius
                cx, cy = geom['center']
                sx, sy = geom['start']
                stored_radius = math.sqrt((sx - cx)**2 + (sy - cy)**2)
                if abs(stored_radius - seg_radius) < 0.01:
                    if elem_id:
                        self._matched_geometry_ids.add(elem_id)
                    return geom

            # For circles, match by radius
            elif geom['type'] == 'circle' and seg_radius is not None:
                if abs(geom['radius'] - seg_radius) < 0.01:
                    if elem_id:
                        self._matched_geometry_ids.add(elem_id)
                    return geom

            # For splines, just match by type (only one spline usually)
            elif geom['type'] == 'spline':
                if elem_id:
                    self._matched_geometry_ids.add(elem_id)
                return geom

            # For ellipses, match by major/minor radii ratio
            elif geom['type'] == 'ellipse':
                # Try to get ellipse params from COM
                try:
                    curve = self._get_com_result(segment, "GetCurve")
                    if curve:
                        params = self._get_com_result(curve, "EllipseParams")
                        if params and len(params) >= 10:
                            # Major and minor radii from axis vectors
                            major_r = math.sqrt(params[3]**2 + params[4]**2 + params[5]**2) * M_TO_MM
                            minor_r = math.sqrt(params[6]**2 + params[7]**2 + params[8]**2) * M_TO_MM
                            if (abs(geom['major_radius'] - major_r) < 0.1 and
                                abs(geom['minor_radius'] - minor_r) < 0.1):
                                if elem_id:
                                    self._matched_geometry_ids.add(elem_id)
                                return geom
                except Exception:
                    pass

            # For elliptical arcs, match by radii
            elif geom['type'] == 'elliptical_arc':
                try:
                    curve = self._get_com_result(segment, "GetCurve")
                    if curve:
                        params = self._get_com_result(curve, "EllipseParams")
                        if params and len(params) >= 10:
                            major_r = math.sqrt(params[3]**2 + params[4]**2 + params[5]**2) * M_TO_MM
                            minor_r = math.sqrt(params[6]**2 + params[7]**2 + params[8]**2) * M_TO_MM
                            if (abs(geom['major_radius'] - major_r) < 0.1 and
                                abs(geom['minor_radius'] - minor_r) < 0.1):
                                if elem_id:
                                    self._matched_geometry_ids.add(elem_id)
                                return geom
                except Exception:
                    pass

        # Fallback: return first unmatched geometry of matching type
        for geom in self._segment_geometry_list:
            elem_id = geom.get('element_id')
            if elem_id and elem_id in self._matched_geometry_ids:
                continue
            if geom['type'] in expected_types:
                if elem_id:
                    self._matched_geometry_ids.add(elem_id)
                return geom

        return None

    def _validate_stored_geometry(self, geom: dict) -> bool:
        """Check if stored geometry endpoints still exist in the sketch."""
        if self._sketch is None:
            return False

        try:
            points = self._get_com_result(self._sketch, "GetSketchPoints2")
            if not points:
                return False

            # Get actual sketch point coordinates
            actual_coords = set()
            for pt in points:
                actual_coords.add((round(pt.X * M_TO_MM, 2), round(pt.Y * M_TO_MM, 2)))

            # Check if stored geometry's key points exist in actual sketch
            if geom['type'] == 'line':
                start = (round(geom['start'][0], 2), round(geom['start'][1], 2))
                end = (round(geom['end'][0], 2), round(geom['end'][1], 2))
                return start in actual_coords and end in actual_coords
            elif geom['type'] == 'circle':
                center = (round(geom['center'][0], 2), round(geom['center'][1], 2))
                if center not in actual_coords:
                    return False
                # Also check if stored radius matches actual radius (Equal constraint changes radius)
                # For now, just return False if constraints applied - forces COM-based export
                if self._constraints_applied:
                    return False
                return True
            elif geom['type'] == 'arc':
                # Note: Arc centers are NOT exposed as sketch points in SolidWorks
                # Only check start and end points
                start = (round(geom['start'][0], 2), round(geom['start'][1], 2))
                end = (round(geom['end'][0], 2), round(geom['end'][1], 2))
                return start in actual_coords and end in actual_coords
            elif geom['type'] == 'spline':
                # For splines, check if control points exist
                for cp in geom['control_points']:
                    cp_rounded = (round(cp[0], 2), round(cp[1], 2))
                    if cp_rounded not in actual_coords:
                        return False
                return True

            return True
        except Exception:
            return False

    def _export_segment(self, segment: Any, construction: bool = False, segment_index: int = -1) -> SketchPrimitive | None:
        """Export a SolidWorks sketch segment to canonical format."""
        try:
            # Get the COM segment type first
            seg_type = self._get_com_result(segment, "GetType")

            # Try to find matching stored geometry by type and geometric properties
            # (Don't rely on segment_index as SolidWorks may return segments in different order)
            geom = self._find_matching_stored_geometry(segment, seg_type)

            if geom is not None:
                # Use stored geometry if constraints haven't been applied
                # OR if validation confirms stored points still exist
                if not self._constraints_applied or self._validate_stored_geometry(geom):
                    if geom['type'] == 'line':
                        return Line(
                            id=geom.get('element_id'),
                            start=Point2D(geom['start'][0], geom['start'][1]),
                            end=Point2D(geom['end'][0], geom['end'][1]),
                            construction=geom.get('construction', construction)
                        )
                    elif geom['type'] == 'circle':
                        return Circle(
                            id=geom.get('element_id'),
                            center=Point2D(geom['center'][0], geom['center'][1]),
                            radius=geom['radius'],
                            construction=geom.get('construction', construction)
                        )
                    elif geom['type'] == 'arc':
                        return Arc(
                            id=geom.get('element_id'),
                            center=Point2D(geom['center'][0], geom['center'][1]),
                            start_point=Point2D(geom['start'][0], geom['start'][1]),
                            end_point=Point2D(geom['end'][0], geom['end'][1]),
                            ccw=geom['ccw'],
                            construction=geom.get('construction', construction)
                        )
                    elif geom['type'] == 'spline':
                        return Spline(
                            id=geom.get('element_id'),
                            control_points=[Point2D(pt[0], pt[1]) for pt in geom['control_points']],
                            degree=geom.get('degree', 3),
                            construction=geom.get('construction', construction)
                        )
                    elif geom['type'] == 'ellipse':
                        return Ellipse(
                            id=geom.get('element_id'),
                            center=Point2D(geom['center'][0], geom['center'][1]),
                            major_radius=geom['major_radius'],
                            minor_radius=geom['minor_radius'],
                            rotation=geom.get('rotation', 0.0),
                            construction=geom.get('construction', construction)
                        )
                    elif geom['type'] == 'elliptical_arc':
                        return EllipticalArc(
                            id=geom.get('element_id'),
                            center=Point2D(geom['center'][0], geom['center'][1]),
                            major_radius=geom['major_radius'],
                            minor_radius=geom['minor_radius'],
                            rotation=geom.get('rotation', 0.0),
                            start_param=geom['start_param'],
                            end_param=geom['end_param'],
                            ccw=geom['ccw'],
                            construction=geom.get('construction', construction)
                        )

            # Fall back to COM-based export if no stored geometry

            if seg_type == SwSketchSegments.LINE:
                return self._export_line(segment, construction)
            elif seg_type == SwSketchSegments.ARC:
                return self._export_arc(segment, construction)
            elif seg_type == SwSketchSegments.SPLINE:
                return self._export_spline(segment, construction)
            elif seg_type == SwSketchSegments.ELLIPSE:
                return self._export_ellipse(segment, construction)
            # Circles are handled differently in SolidWorks
            # They may come as arcs or need special handling
            else:
                return None
        except Exception:
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
        except Exception:
            pass

        # Method 2: Try ISketchSegment interface
        if start_pt is None:
            try:
                sketch_seg = win32com.client.CastTo(segment, "ISketchSegment")
                start_pt = sketch_seg.GetStartPoint2()
                end_pt = sketch_seg.GetEndPoint2()
            except Exception:
                pass

        # Method 3: Match by segment length to find endpoint pair
        if start_pt is None and self._sketch is not None:
            try:
                # Get segment length and all sketch points
                seg_length = segment.GetLength  # in meters
                points = self._get_com_result(self._sketch, "GetSketchPoints2")
                if points and len(points) >= 2 and seg_length:
                    # Find the pair of points whose distance matches segment length
                    # Skip pairs that have already been used by other segments
                    point_coords = [(pt.X, pt.Y, pt) for pt in points]
                    tolerance = 1e-6  # meters
                    for i, (x1, y1, pt1) in enumerate(point_coords):
                        for j, (x2, y2, pt2) in enumerate(point_coords):
                            if i < j:
                                dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                                if abs(dist - seg_length) < tolerance:
                                    # Check if this pair has already been used
                                    pair_key = (
                                        (round(x1 * M_TO_MM, 4), round(y1 * M_TO_MM, 4)),
                                        (round(x2 * M_TO_MM, 4), round(y2 * M_TO_MM, 4))
                                    )
                                    pair_key_rev = (pair_key[1], pair_key[0])
                                    if hasattr(self, '_used_point_pairs'):
                                        if pair_key in self._used_point_pairs or pair_key_rev in self._used_point_pairs:
                                            continue  # Skip this pair, try next
                                    start_pt = pt1
                                    end_pt = pt2
                                    # Mark this pair as used
                                    if hasattr(self, '_used_point_pairs'):
                                        self._used_point_pairs.add(pair_key)
                                    break
                        if start_pt:
                            break
            except Exception:
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
        radius = None
        center_x = None
        center_y = None

        # Get radius from segment (this works reliably)
        try:
            r = segment.GetRadius
            if r is not None:
                radius = r * M_TO_MM
        except Exception:
            pass

        # Method 1: Try to get curve parameters which include center
        try:
            curve = self._get_com_result(segment, "GetCurve")
            if curve:
                is_circle = self._get_com_result(curve, "IsCircle")
                if is_circle:
                    # CircleParams returns [cx, cy, cz, ax, ay, az, radius]
                    params = self._get_com_result(curve, "CircleParams")
                    if params and len(params) >= 7:
                        center_x = params[0] * M_TO_MM
                        center_y = params[1] * M_TO_MM
                        if radius is None:
                            radius = params[6] * M_TO_MM
        except Exception:
            pass

        # Method 2: Check if this segment is a full circle by comparing arc length to circumference
        is_full_circle = False
        if radius is not None:
            try:
                arc_length = segment.GetLength * M_TO_MM
                circumference = 2 * math.pi * radius
                # If arc length is very close to circumference, it's a full circle
                if abs(arc_length - circumference) < 0.01:
                    is_full_circle = True
            except Exception:
                pass

        # Method 3: Find center from sketch points if not found yet
        if center_x is None and radius is not None and self._sketch is not None:
            try:
                points = self._get_com_result(self._sketch, "GetSketchPoints2")
                if points:
                    # For each point, check if it could be a center for this segment
                    # A center point is at radius distance from points on the arc
                    for pt in points:
                        px, py = pt.X * M_TO_MM, pt.Y * M_TO_MM
                        # Check if this point is a plausible center
                        # (There should be other points at exactly radius distance)
                        at_radius_count = 0
                        for other_pt in points:
                            if other_pt is not pt:
                                ox, oy = other_pt.X * M_TO_MM, other_pt.Y * M_TO_MM
                                dist = math.sqrt((px - ox)**2 + (py - oy)**2)
                                if abs(dist - radius) < 0.01:
                                    at_radius_count += 1
                        # For a circle, center has no other points at radius (just curve)
                        # For an arc, center has 2 points at radius (start and end)
                        if is_full_circle and at_radius_count == 0:
                            center_x, center_y = px, py
                            break
                        elif not is_full_circle and at_radius_count >= 2:
                            center_x, center_y = px, py
                            break
            except Exception:
                pass

        # If we have center and radius and it's a full circle, return Circle
        if center_x is not None and radius is not None and is_full_circle:
            return Circle(
                center=Point2D(center_x, center_y),
                radius=radius,
                construction=construction
            )

        # Otherwise try to export as an arc (original logic for arcs)
        if self._sketch is not None and radius is not None:
            try:
                points = self._get_com_result(self._sketch, "GetSketchPoints2")

                if points and len(points) >= 3:
                    # Arc: we have center, start, end points
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

            except Exception:
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

    def _export_ellipse(self, segment: Any, construction: bool = False) -> Ellipse | EllipticalArc:
        """Export a SolidWorks ellipse or elliptical arc to canonical format.

        SolidWorks uses segment type ELLIPSE (2) for both full ellipses and
        elliptical arcs. We need to determine which one it is based on whether
        it's a closed curve.
        """
        # Get ellipse parameters from the curve
        curve = self._get_com_result(segment, "GetCurve")

        center_x = 0.0
        center_y = 0.0
        major_radius = 1.0
        minor_radius = 0.5
        rotation = 0.0

        if curve:
            try:
                # EllipseParams returns [cx, cy, cz, major_ax, major_ay, major_az, minor_ax, minor_ay, minor_az, ratio]
                params = self._get_com_result(curve, "EllipseParams")
                if params and len(params) >= 10:
                    center_x = params[0] * M_TO_MM
                    center_y = params[1] * M_TO_MM

                    # Major axis direction
                    major_ax = params[3]
                    major_ay = params[4]

                    # Compute rotation from major axis direction
                    rotation = math.atan2(major_ay, major_ax)

                    # Major and minor radii are the lengths of the axis vectors
                    major_radius = math.sqrt(params[3]**2 + params[4]**2 + params[5]**2) * M_TO_MM
                    minor_radius = math.sqrt(params[6]**2 + params[7]**2 + params[8]**2) * M_TO_MM
            except Exception:
                pass

        # Check if this is a full ellipse or an elliptical arc
        is_full_ellipse = False
        try:
            # Compare arc length to expected full ellipse perimeter
            arc_length = segment.GetLength * M_TO_MM
            # Approximation of ellipse perimeter using Ramanujan's formula
            h = ((major_radius - minor_radius) / (major_radius + minor_radius)) ** 2
            perimeter = math.pi * (major_radius + minor_radius) * (1 + 3*h / (10 + math.sqrt(4 - 3*h)))
            if abs(arc_length - perimeter) < 0.01:
                is_full_ellipse = True
        except Exception:
            pass

        if is_full_ellipse:
            return Ellipse(
                center=Point2D(center_x, center_y),
                major_radius=major_radius,
                minor_radius=minor_radius,
                rotation=rotation,
                construction=construction
            )
        else:
            # It's an elliptical arc - get start and end points
            start_pt = None
            end_pt = None
            try:
                start_pt = segment.GetStartPoint2()
                end_pt = segment.GetEndPoint2()
            except Exception:
                pass

            if start_pt is not None and end_pt is not None:
                # Convert Cartesian points to parametric angles on the ellipse
                start_param = self._point_to_ellipse_param(
                    start_pt.X * M_TO_MM, start_pt.Y * M_TO_MM,
                    center_x, center_y, rotation
                )
                end_param = self._point_to_ellipse_param(
                    end_pt.X * M_TO_MM, end_pt.Y * M_TO_MM,
                    center_x, center_y, rotation
                )

                return EllipticalArc(
                    center=Point2D(center_x, center_y),
                    major_radius=major_radius,
                    minor_radius=minor_radius,
                    rotation=rotation,
                    start_param=start_param,
                    end_param=end_param,
                    ccw=True,  # Default to counter-clockwise
                    construction=construction
                )
            else:
                # Couldn't get start/end points, return as full ellipse
                return Ellipse(
                    center=Point2D(center_x, center_y),
                    major_radius=major_radius,
                    minor_radius=minor_radius,
                    rotation=rotation,
                    construction=construction
                )

    def _point_to_ellipse_param(self, px: float, py: float, cx: float, cy: float, rotation: float) -> float:
        """Convert a point on an ellipse to its parametric angle.

        Args:
            px, py: Point coordinates
            cx, cy: Ellipse center coordinates
            rotation: Ellipse rotation angle in radians

        Returns:
            Parametric angle in radians [0, 2*pi)
        """
        # Translate point relative to center
        dx = px - cx
        dy = py - cy

        # Rotate to align with ellipse axes
        cos_r = math.cos(-rotation)
        sin_r = math.sin(-rotation)
        local_x = dx * cos_r - dy * sin_r
        local_y = dx * sin_r + dy * cos_r

        # Compute parametric angle using atan2
        param = math.atan2(local_y, local_x)
        if param < 0:
            param += 2 * math.pi
        return param

    def _export_point(self, point: Any) -> Point:
        """Export a SolidWorks point to canonical format."""
        return Point(
            position=Point2D(point.X * M_TO_MM, point.Y * M_TO_MM)
        )

    def _export_constraints(self, doc: SketchDocument) -> None:
        """Export constraints from SolidWorks sketch.

        In late-bound COM, GetSketchRelations() on the sketch doesn't work.
        Instead, we iterate through segments and get relations from each segment.
        """
        if self._sketch is None:
            return

        # Track seen relations by their entities to avoid duplicates
        # (same relation appears on multiple segments it connects)
        seen_relations: set[tuple] = set()

        try:
            # Get all segments
            segments = self._get_com_result(self._sketch, "GetSketchSegments")
            if not segments:
                return

            for seg_idx, segment in enumerate(segments):
                try:
                    # Get the ID of this segment using index-based lookup
                    # (property-based matching fails when segments have same length)
                    segment_id = None
                    if seg_idx < len(self._segment_index_to_id):
                        segment_id = self._segment_index_to_id[seg_idx]
                    if not segment_id:
                        # Fallback to property-based matching
                        segment_id = self._get_entity_id_by_properties(segment)

                    # Get relations for this segment
                    relations = segment.GetRelations
                    if callable(relations):
                        relations = relations()

                    if not relations:
                        continue

                    for relation in relations:
                        # Get relation type
                        rel_type = relation.GetRelationType
                        if callable(rel_type):
                            rel_type = rel_type()

                        # Get entities count for deduplication key
                        entities_count = relation.GetEntitiesCount
                        if callable(entities_count):
                            entities_count = entities_count()

                        # Create a key to deduplicate relations
                        # Include segment_id to distinguish constraints on different entities
                        # (same relation type on different segments are different constraints)
                        dedup_key = (rel_type, segment_id)
                        if dedup_key in seen_relations:
                            # Skip duplicate
                            continue

                        seen_relations.add(dedup_key)

                        canonical = self._convert_relation(relation, segment_id)
                        if canonical is not None:
                            doc.constraints.append(canonical)

                except Exception:
                    continue

        except Exception:
            pass

    def _get_entity_id_by_properties(self, entity: Any) -> str | None:
        """Get entity ID by matching geometric properties (midpoint, length, radius).

        Uses midpoint for lines first (most unique), then falls back to length/radius.
        """
        # Try matching by midpoint (most unique for lines with same length)
        try:
            # Try to get start/end points to calculate midpoint
            start_pt = None
            end_pt = None
            for start_attr in ["GetStartPoint2", "StartPoint"]:
                try:
                    start_pt = getattr(entity, start_attr, None)
                    if callable(start_pt):
                        start_pt = start_pt()
                    if start_pt is not None:
                        break
                except Exception:
                    pass
            for end_attr in ["GetEndPoint2", "EndPoint"]:
                try:
                    end_pt = getattr(entity, end_attr, None)
                    if callable(end_pt):
                        end_pt = end_pt()
                    if end_pt is not None:
                        break
                except Exception:
                    pass

            if start_pt is not None and end_pt is not None:
                # Get coordinates (SolidWorks returns in meters)
                sx = self._get_com_result(start_pt, "X") or 0
                sy = self._get_com_result(start_pt, "Y") or 0
                ex = self._get_com_result(end_pt, "X") or 0
                ey = self._get_com_result(end_pt, "Y") or 0
                # Convert to mm and calculate midpoint
                midpoint_key = (
                    round(((sx + ex) / 2) * 1000, 6),
                    round(((sy + ey) / 2) * 1000, 6)
                )
                entity_id = self._midpoint_to_id.get(midpoint_key)
                if entity_id:
                    return entity_id
        except Exception:
            pass

        # Try matching by length (fallback for non-line entities)
        try:
            length = entity.GetLength
            if length is not None:
                length_key = round(length, 10)
                entity_id = self._length_to_id.get(length_key)
                if entity_id:
                    return entity_id
        except Exception:
            pass

        # Try matching by radius (for circles/arcs)
        try:
            radius = entity.GetRadius
            if radius is not None:
                radius_key = round(radius, 10)
                entity_id = self._radius_to_id.get(radius_key)
                if entity_id:
                    return entity_id
        except Exception:
            pass

        return None

    def _convert_relation(self, relation: Any, source_segment_id: str | None = None) -> SketchConstraint | None:
        """Convert a SolidWorks sketch relation to canonical constraint.

        Args:
            relation: The SolidWorks relation object
            source_segment_id: ID of the segment we found this relation on (for reference)
        """
        try:
            # Handle property/method access for late-bound COM
            rel_type = relation.GetRelationType
            if callable(rel_type):
                rel_type = rel_type()

            # Map SolidWorks relation types to canonical
            # Reference: swConstraintType_e from SolidWorks 2024 API
            type_map = {
                SwConstraintType.HORIZONTAL: ConstraintType.HORIZONTAL,        # 4
                SwConstraintType.VERTICAL: ConstraintType.VERTICAL,            # 5
                SwConstraintType.TANGENT: ConstraintType.TANGENT,              # 6
                SwConstraintType.PARALLEL: ConstraintType.PARALLEL,            # 7
                SwConstraintType.PERPENDICULAR: ConstraintType.PERPENDICULAR,  # 8
                SwConstraintType.COINCIDENT: ConstraintType.COINCIDENT,        # 9
                SwConstraintType.CONCENTRIC: ConstraintType.CONCENTRIC,        # 10
                SwConstraintType.SYMMETRIC: ConstraintType.SYMMETRIC,          # 11
                SwConstraintType.MIDPOINT: ConstraintType.MIDPOINT,            # 12
                SwConstraintType.EQUAL: ConstraintType.EQUAL,                  # 14
                SwConstraintType.FIX: ConstraintType.FIXED,                    # 17
                SwConstraintType.COLLINEAR: ConstraintType.COLLINEAR,          # 27
                SwConstraintType.CORADIAL: ConstraintType.EQUAL,               # 28 - equal radius
            }

            if rel_type not in type_map:
                return None

            ctype = type_map[rel_type]

            # For single-entity constraints (horizontal, vertical, fixed), the source
            # segment IS the constrained entity. Use source_segment_id directly to avoid
            # length-matching ambiguity when multiple segments have the same length.
            single_entity_types = {
                SwConstraintType.HORIZONTAL,
                SwConstraintType.VERTICAL,
                SwConstraintType.FIX,
            }

            refs: list[str | PointRef] = []

            if rel_type in single_entity_types and source_segment_id:
                # For single-entity constraints, use source segment directly
                refs.append(source_segment_id)
            else:
                # Get entities involved - handle property/method access
                entities = relation.GetEntities
                if callable(entities):
                    entities = entities()

                if entities:
                    for entity in entities:
                        # First try direct id() match (might work if same COM context)
                        entity_id = self._entity_to_id.get(id(entity))

                        # If that fails, try property-based matching
                        if not entity_id:
                            entity_id = self._get_entity_id_by_properties(entity)

                        if entity_id:
                            refs.append(entity_id)

                # If we have the source segment and it's not already in refs, add it
                # This handles cases where GetEntities doesn't return all involved entities
                if source_segment_id and source_segment_id not in refs:
                    refs.append(source_segment_id)

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

        status_val = None

        # Try multiple ways to get the constrained status
        # Method 1: Direct method call
        try:
            status_val = self._sketch.GetConstrainedStatus()
        except Exception:
            pass

        # Method 2: Property access
        if status_val is None:
            try:
                status_val = self._sketch.ConstrainedStatus
            except Exception:
                pass

        # Method 3: Try using _get_com_result helper
        if status_val is None:
            try:
                status_val = self._get_com_result(self._sketch, "GetConstrainedStatus")
            except Exception:
                pass

        # Method 4: Check if sketch has any underdefined geometry
        # by counting relations vs geometry DOF
        if status_val is None:
            try:
                # Get geometry
                segments = self._get_com_result(self._sketch, "GetSketchSegments")
                points = self._get_com_result(self._sketch, "GetSketchPoints2")

                # Count DOF from geometry
                geom_dof = 0
                if segments:
                    for seg in segments:
                        seg_type = self._get_com_result(seg, "GetType")
                        if seg_type == SwSketchSegments.LINE:
                            geom_dof += 4
                        elif seg_type == SwSketchSegments.ARC:
                            geom_dof += 5
                if points:
                    # Standalone points
                    geom_dof += len(points) * 2

                # Count constraints by iterating segments (GetSketchRelations doesn't work)
                constraint_dof = 0
                if segments:
                    seen_relations: set[tuple] = set()
                    for seg in segments:
                        try:
                            relations = seg.GetRelations
                            if callable(relations):
                                relations = relations()
                            if relations:
                                for rel in relations:
                                    rel_type = rel.GetRelationType
                                    if callable(rel_type):
                                        rel_type = rel_type()
                                    entities_count = rel.GetEntitiesCount
                                    if callable(entities_count):
                                        entities_count = entities_count()
                                    key = (rel_type, entities_count)
                                    if key not in seen_relations:
                                        seen_relations.add(key)
                                        constraint_dof += 1
                        except Exception:
                            continue

                # Estimate status
                remaining_dof = max(0, geom_dof - constraint_dof)
                if remaining_dof == 0:
                    return (SolverStatus.FULLY_CONSTRAINED, 0)
                else:
                    return (SolverStatus.UNDER_CONSTRAINED, remaining_dof)
            except Exception:
                pass

        # If we got a status value, interpret it
        if status_val is not None:
            # SolidWorks sketch states:
            # 1 = Under defined (blue)
            # 2 = Fully defined (black)
            # 3 = Over defined (red)
            if status_val == 2:
                return (SolverStatus.FULLY_CONSTRAINED, 0)
            elif status_val == 3:
                return (SolverStatus.OVER_CONSTRAINED, 0)
            else:
                dof = self._estimate_dof()
                return (SolverStatus.UNDER_CONSTRAINED, dof)

        # Fallback - return under constrained with estimated DOF
        dof = self._estimate_dof()
        if dof >= 0:
            return (SolverStatus.UNDER_CONSTRAINED, dof)

        return (SolverStatus.INCONSISTENT, -1)

    def _estimate_dof(self) -> int:
        """Estimate degrees of freedom (rough approximation)."""
        if self._sketch is None:
            return -1

        try:
            sketch = self._sketch
            dof = 0

            # Count geometry - handle property/method access for late-bound COM
            segments = self._get_com_result(sketch, "GetSketchSegments")
            if segments:
                for segment in segments:
                    seg_type = self._get_com_result(segment, "GetType")
                    if seg_type == SwSketchSegments.LINE:
                        dof += 4  # 2 points x 2 coords
                    elif seg_type == SwSketchSegments.ARC:
                        dof += 5  # center + radius + 2 angles
                    elif seg_type == SwSketchSegments.SPLINE:
                        points = self._get_com_result(segment, "GetPoints2")
                        if points:
                            dof += (len(points) // 3) * 2

            # Subtract for relations - iterate segments since GetSketchRelations doesn't work
            relation_count = 0
            if segments:
                seen_relations: set[tuple] = set()
                for segment in segments:
                    try:
                        relations = segment.GetRelations
                        if callable(relations):
                            relations = relations()
                        if relations:
                            for rel in relations:
                                rel_type = rel.GetRelationType
                                if callable(rel_type):
                                    rel_type = rel_type()
                                entities_count = rel.GetEntitiesCount
                                if callable(entities_count):
                                    entities_count = entities_count()
                                key = (rel_type, entities_count)
                                if key not in seen_relations:
                                    seen_relations.add(key)
                                    relation_count += 1
                    except Exception:
                        continue

            dof -= relation_count
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
