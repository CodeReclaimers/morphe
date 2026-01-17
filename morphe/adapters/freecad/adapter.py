"""
FreeCAD Sketch Adapter Implementation.

This adapter provides bidirectional translation between the canonical
sketch schema and FreeCAD's Sketcher workbench.

Key FreeCAD specifics:
- Native unit: millimeters (same as canonical)
- Arc representation: Uses three-point construction for reliable direction
- Tangent does NOT imply coincident - must add separately
- Constraint indexing uses (geometry_index, vertex_index)
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

from .vertex_map import VertexMap, get_point_type_from_vertex, get_vertex_index

# Try to import FreeCAD modules
FREECAD_AVAILABLE = False
try:
    import FreeCAD as App
    import Part
    import Sketcher
    FREECAD_AVAILABLE = True
except ImportError:
    App = None
    Part = None
    Sketcher = None


class FreeCADAdapter(SketchBackendAdapter):
    """
    FreeCAD Sketcher adapter.

    Translates between canonical sketch representation and FreeCAD's
    native Sketcher API.
    """

    # FreeCAD constraint type names
    FC_CONSTRAINT_NAMES = {
        ConstraintType.COINCIDENT: 'Coincident',
        ConstraintType.TANGENT: 'Tangent',
        ConstraintType.PERPENDICULAR: 'Perpendicular',
        ConstraintType.PARALLEL: 'Parallel',
        ConstraintType.EQUAL: 'Equal',
        ConstraintType.HORIZONTAL: 'Horizontal',
        ConstraintType.VERTICAL: 'Vertical',
        ConstraintType.FIXED: 'Block',  # FreeCAD uses 'Block' to lock geometry position
        ConstraintType.DISTANCE: 'Distance',
        ConstraintType.DISTANCE_X: 'DistanceX',
        ConstraintType.DISTANCE_Y: 'DistanceY',
        ConstraintType.RADIUS: 'Radius',
        ConstraintType.DIAMETER: 'Diameter',
        ConstraintType.ANGLE: 'Angle',
        ConstraintType.SYMMETRIC: 'Symmetric',
        ConstraintType.CONCENTRIC: 'Coincident',  # Concentric uses Coincident on centers
        ConstraintType.LENGTH: 'Distance',  # Length uses Distance on line endpoints
        ConstraintType.COLLINEAR: 'Tangent',  # Collinear uses Tangent on lines
        ConstraintType.MIDPOINT: 'Symmetric',  # Midpoint uses Symmetric with line endpoints
    }

    def __init__(self, document: Any | None = None):
        """
        Initialize the FreeCAD adapter.

        Args:
            document: Optional FreeCAD document. If None, uses ActiveDocument.
        """
        if not FREECAD_AVAILABLE:
            raise ImportError(
                "FreeCAD is not available. Please run this adapter within FreeCAD "
                "or ensure FreeCAD libraries are on the Python path."
            )

        self._document = document
        self._sketch = None
        self._sketch_doc: SketchDocument | None = None

        # ID to FreeCAD geometry index mapping
        self._id_to_index: dict[str, int] = {}
        self._index_to_id: dict[int, str] = {}

    def _get_document(self) -> Any:
        """Get the FreeCAD document, creating one if needed."""
        if self._document is not None:
            return self._document
        if App.ActiveDocument is None:
            App.newDocument("Sketch")
        return App.ActiveDocument

    def _get_active_sketch(self) -> Any:
        """Get the active sketch object."""
        if self._sketch is None:
            raise SketchCreationError("No active sketch. Call create_sketch() first.")
        return self._sketch

    def create_sketch(self, name: str, plane: Any | None = None) -> None:
        """
        Create a new empty sketch in FreeCAD.

        Args:
            name: Sketch name
            plane: Plane specification. Can be:
                   - None or "XY": XY plane (default)
                   - "XZ": XZ plane
                   - "YZ": YZ plane
                   - FreeCAD face/plane object for custom planes
        """
        doc = self._get_document()

        # Create sketch object
        self._sketch = doc.addObject('Sketcher::SketchObject', name)

        # Set the sketch plane based on specification
        if plane is None or plane == "XY":
            # XY plane is the default - no changes needed
            pass
        elif plane == "XZ":
            # XZ plane: rotate 90 degrees around X axis
            self._sketch.Placement = App.Placement(
                App.Vector(0, 0, 0),
                App.Rotation(App.Vector(1, 0, 0), 90)
            )
        elif plane == "YZ":
            # YZ plane: rotate 90 degrees around Y axis
            self._sketch.Placement = App.Placement(
                App.Vector(0, 0, 0),
                App.Rotation(App.Vector(0, 1, 0), -90)
            )
        elif hasattr(plane, 'Surface'):
            # It's a FreeCAD face - use as support
            self._sketch.Support = [(plane, '')]
        elif plane is not None:
            # Try to use as support directly
            try:
                self._sketch.Support = plane
            except Exception:
                pass  # Fall back to default XY

        # Clear mappings
        self._id_to_index.clear()
        self._index_to_id.clear()

        # Initialize sketch document for tracking
        self._sketch_doc = SketchDocument(name=name)

    def load_sketch(self, sketch: SketchDocument) -> None:
        """
        Load a canonical sketch into FreeCAD.

        Args:
            sketch: The canonical sketch document to load.
        """
        # Create the sketch if not already created
        if self._sketch is None:
            self.create_sketch(sketch.name)

        self._sketch_doc = sketch

        # Add all primitives
        for prim in sketch.primitives.values():
            self.add_primitive(prim)

        # Add all constraints
        for constraint in sketch.constraints:
            self.add_constraint(constraint)

        # Recompute
        self._get_document().recompute()

    def export_sketch(self) -> SketchDocument:
        """
        Export the current FreeCAD sketch to canonical form.

        Returns:
            SketchDocument containing the canonical representation.
        """
        sketch = self._get_active_sketch()
        doc = SketchDocument(name=sketch.Label)

        # Clear and rebuild mappings
        self._id_to_index.clear()
        self._index_to_id.clear()

        # Export geometry
        for i, geo in enumerate(sketch.Geometry):
            prim = self._geometry_to_primitive(geo, i)
            if prim is not None:
                doc.add_primitive(prim)
                self._index_to_id[i] = prim.id
                self._id_to_index[prim.id] = i

        # Export constraints
        for fc_constraint in sketch.Constraints:
            constraint = self._fc_constraint_to_canonical(fc_constraint)
            if constraint is not None:
                doc.constraints.append(constraint)

        # Get solver status
        status, dof = self.get_solver_status()
        doc.solver_status = status
        doc.degrees_of_freedom = dof

        return doc

    def add_primitive(self, primitive: SketchPrimitive) -> int:
        """
        Add a primitive to the FreeCAD sketch.

        Args:
            primitive: The canonical primitive to add.

        Returns:
            FreeCAD geometry index.
        """
        sketch = self._get_active_sketch()

        if isinstance(primitive, Line):
            idx = self._add_line(sketch, primitive)
        elif isinstance(primitive, Arc):
            idx = self._add_arc(sketch, primitive)
        elif isinstance(primitive, Circle):
            idx = self._add_circle(sketch, primitive)
        elif isinstance(primitive, Point):
            idx = self._add_point(sketch, primitive)
        elif isinstance(primitive, Spline):
            idx = self._add_spline(sketch, primitive)
        elif isinstance(primitive, Ellipse):
            idx = self._add_ellipse(sketch, primitive)
        elif isinstance(primitive, EllipticalArc):
            idx = self._add_elliptical_arc(sketch, primitive)
        else:
            raise GeometryError(f"Unsupported primitive type: {type(primitive).__name__}")

        # Store mapping
        self._id_to_index[primitive.id] = idx
        self._index_to_id[idx] = primitive.id

        return idx

    def _add_line(self, sketch: Any, line: Line) -> int:
        """Add a line to the sketch."""
        geo = Part.LineSegment(
            App.Vector(line.start.x, line.start.y, 0),
            App.Vector(line.end.x, line.end.y, 0)
        )
        return sketch.addGeometry(geo, line.construction)

    def _add_arc(self, sketch: Any, arc: Arc) -> int:
        """
        Add an arc to the sketch using three-point construction.

        Uses three-point construction for reliable arc direction,
        as recommended in the specification.
        """
        start, mid, end = arc.to_three_point()
        geo = Part.Arc(
            App.Vector(start.x, start.y, 0),
            App.Vector(mid.x, mid.y, 0),
            App.Vector(end.x, end.y, 0)
        )
        return sketch.addGeometry(geo, arc.construction)

    def _add_circle(self, sketch: Any, circle: Circle) -> int:
        """Add a circle to the sketch."""
        geo = Part.Circle(
            App.Vector(circle.center.x, circle.center.y, 0),
            App.Vector(0, 0, 1),  # Normal vector
            circle.radius
        )
        return sketch.addGeometry(geo, circle.construction)

    def _add_point(self, sketch: Any, point: Point) -> int:
        """Add a point to the sketch."""
        geo = Part.Point(App.Vector(point.position.x, point.position.y, 0))
        return sketch.addGeometry(geo, point.construction)

    def _add_spline(self, sketch: Any, spline: Spline) -> int:
        """Add a B-spline to the sketch."""
        # Convert control points to FreeCAD vectors
        poles = [App.Vector(p.x, p.y, 0) for p in spline.control_points]

        # Extract unique knots and multiplicities from expanded knot vector
        unique_knots, mults = self._extract_knots_and_mults(spline.knots)

        # Create B-spline curve
        bspline = Part.BSplineCurve()
        if spline.weights is not None:
            # Rational B-spline
            bspline.buildFromPolesMultsKnots(
                poles,
                mults,
                unique_knots,
                spline.periodic,
                spline.degree,
                spline.weights
            )
        else:
            # Non-rational B-spline
            bspline.buildFromPolesMultsKnots(
                poles,
                mults,
                unique_knots,
                spline.periodic,
                spline.degree
            )

        return sketch.addGeometry(bspline, spline.construction)

    def _add_ellipse(self, sketch: Any, ellipse: Ellipse) -> int:
        """Add an ellipse to the sketch."""
        center = App.Vector(ellipse.center.x, ellipse.center.y, 0)

        # Create ellipse with center and radii
        # Part.Ellipse(center, major_radius, minor_radius) creates ellipse in XY plane
        geo = Part.Ellipse(center, ellipse.major_radius, ellipse.minor_radius)

        # Apply rotation if needed by setting the XAxis direction
        if abs(ellipse.rotation) > 1e-10:
            # The XAxis defines the direction of the major axis
            geo.XAxis = App.Vector(
                math.cos(ellipse.rotation),
                math.sin(ellipse.rotation),
                0
            )

        return sketch.addGeometry(geo, ellipse.construction)

    def _add_elliptical_arc(self, sketch: Any, arc: EllipticalArc) -> int:
        """Add an elliptical arc to the sketch."""
        center = App.Vector(arc.center.x, arc.center.y, 0)

        # Create base ellipse with center and radii
        ellipse = Part.Ellipse(center, arc.major_radius, arc.minor_radius)

        # Apply rotation if needed by setting the XAxis direction
        if abs(arc.rotation) > 1e-10:
            ellipse.XAxis = App.Vector(
                math.cos(arc.rotation),
                math.sin(arc.rotation),
                0
            )

        # Adjust parameters for CW direction if needed
        start_param = arc.start_param
        end_param = arc.end_param
        if not arc.ccw:
            # Swap start and end for CW direction
            start_param, end_param = end_param, start_param

        # Create arc from ellipse
        geo = Part.ArcOfEllipse(ellipse, start_param, end_param)
        return sketch.addGeometry(geo, arc.construction)

    def _extract_knots_and_mults(self, knots: list[float]) -> tuple[list[float], list[int]]:
        """
        Extract unique knots and multiplicities from an expanded knot vector.

        Args:
            knots: Expanded knot vector (e.g., [0, 0, 0, 0, 1, 1, 1, 1])

        Returns:
            Tuple of (unique_knots, multiplicities)
            e.g., ([0, 1], [4, 4])
        """
        if not knots:
            return [], []

        unique_knots = []
        mults = []
        prev_knot = None
        count = 0

        for knot in knots:
            if prev_knot is None or abs(knot - prev_knot) < 1e-10:
                count += 1
            else:
                if count > 0:
                    unique_knots.append(prev_knot)
                    mults.append(count)
                count = 1
            prev_knot = knot

        if count > 0 and prev_knot is not None:
            unique_knots.append(prev_knot)
            mults.append(count)

        return unique_knots, mults

    def _compute_multiplicities(self, spline: Spline) -> list[int]:
        """Compute knot multiplicities from the knot vector."""
        _, mults = self._extract_knots_and_mults(spline.knots)
        return mults

    def add_constraint(self, constraint: SketchConstraint) -> bool:
        """
        Add a constraint to the FreeCAD sketch.

        Args:
            constraint: The canonical constraint to add.

        Returns:
            True if successful, False otherwise.
        """
        sketch = self._get_active_sketch()

        try:
            match constraint.constraint_type:
                case ConstraintType.COINCIDENT:
                    self._add_coincident(sketch, constraint)
                case ConstraintType.TANGENT:
                    self._add_tangent(sketch, constraint)
                case ConstraintType.PERPENDICULAR:
                    self._add_perpendicular(sketch, constraint)
                case ConstraintType.PARALLEL:
                    self._add_parallel(sketch, constraint)
                case ConstraintType.EQUAL:
                    self._add_equal(sketch, constraint)
                case ConstraintType.HORIZONTAL:
                    self._add_horizontal(sketch, constraint)
                case ConstraintType.VERTICAL:
                    self._add_vertical(sketch, constraint)
                case ConstraintType.FIXED:
                    self._add_fixed(sketch, constraint)
                case ConstraintType.CONCENTRIC:
                    self._add_concentric(sketch, constraint)
                case ConstraintType.DISTANCE:
                    self._add_distance(sketch, constraint)
                case ConstraintType.DISTANCE_X:
                    self._add_distance_x(sketch, constraint)
                case ConstraintType.DISTANCE_Y:
                    self._add_distance_y(sketch, constraint)
                case ConstraintType.RADIUS:
                    self._add_radius(sketch, constraint)
                case ConstraintType.DIAMETER:
                    self._add_diameter(sketch, constraint)
                case ConstraintType.ANGLE:
                    self._add_angle(sketch, constraint)
                case ConstraintType.SYMMETRIC:
                    self._add_symmetric(sketch, constraint)
                case ConstraintType.LENGTH:
                    self._add_length(sketch, constraint)
                case ConstraintType.COLLINEAR:
                    self._add_collinear(sketch, constraint)
                case ConstraintType.MIDPOINT:
                    self._add_midpoint(sketch, constraint)
                case _:
                    raise ConstraintError(
                        f"Unsupported constraint type: {constraint.constraint_type}"
                    )
            return True

        except Exception as e:
            raise ConstraintError(f"Failed to add constraint: {e}") from e

    def _point_ref_to_freecad(self, ref: PointRef) -> tuple[int, int]:
        """
        Convert a PointRef to FreeCAD (geometry_index, vertex_index).

        Args:
            ref: Canonical point reference

        Returns:
            Tuple of (geometry_index, vertex_index)
        """
        if ref.element_id not in self._id_to_index:
            raise ConstraintError(f"Unknown element ID: {ref.element_id}")

        geo_idx = self._id_to_index[ref.element_id]

        # Get primitive type to determine vertex mapping
        if self._sketch_doc and ref.element_id in self._sketch_doc.primitives:
            prim = self._sketch_doc.primitives[ref.element_id]
            vertex_idx = get_vertex_index(type(prim), ref.point_type)
            if vertex_idx is None:
                raise ConstraintError(
                    f"Invalid point type {ref.point_type} for {type(prim).__name__}"
                )
        else:
            # Fallback: infer from geometry
            geo = self._get_active_sketch().Geometry[geo_idx]
            vertex_idx = self._infer_vertex_index(geo, ref.point_type)

        return (geo_idx, vertex_idx)

    def _infer_vertex_index(self, geo: Any, point_type: PointType) -> int:
        """Infer vertex index from FreeCAD geometry object."""
        geo_type = type(geo).__name__

        if 'Line' in geo_type:
            return get_vertex_index(Line, point_type) or 1
        elif 'ArcOfEllipse' in geo_type:
            return get_vertex_index(EllipticalArc, point_type) or 1
        elif 'Arc' in geo_type:
            return get_vertex_index(Arc, point_type) or 1
        elif 'Ellipse' in geo_type:
            return get_vertex_index(Ellipse, point_type) or 3
        elif 'Circle' in geo_type:
            return get_vertex_index(Circle, point_type) or 3
        elif 'Point' in geo_type:
            return 1
        elif 'BSpline' in geo_type:
            return get_vertex_index(Spline, point_type) or 1
        else:
            return 1

    def _get_element_index(self, ref: str | PointRef) -> int:
        """Get the geometry index for an element reference."""
        if isinstance(ref, PointRef):
            element_id = ref.element_id
        else:
            element_id = ref

        if element_id not in self._id_to_index:
            raise ConstraintError(f"Unknown element ID: {element_id}")

        return self._id_to_index[element_id]

    # Constraint implementation methods

    def _add_coincident(self, sketch: Any, constraint: SketchConstraint) -> None:
        """Add a coincident constraint."""
        pt1 = self._point_ref_to_freecad(constraint.references[0])
        pt2 = self._point_ref_to_freecad(constraint.references[1])
        sketch.addConstraint(Sketcher.Constraint(
            'Coincident', pt1[0], pt1[1], pt2[0], pt2[1]
        ))

    def _add_tangent(self, sketch: Any, constraint: SketchConstraint) -> None:
        """Add a tangent constraint."""
        idx1 = self._get_element_index(constraint.references[0])
        idx2 = self._get_element_index(constraint.references[1])

        if constraint.connection_point:
            # Tangent at specific point
            pt = self._point_ref_to_freecad(constraint.connection_point)
            sketch.addConstraint(Sketcher.Constraint(
                'Tangent', idx1, pt[1], idx2, pt[1]
            ))
        else:
            # General tangent
            sketch.addConstraint(Sketcher.Constraint('Tangent', idx1, idx2))

    def _add_perpendicular(self, sketch: Any, constraint: SketchConstraint) -> None:
        """Add a perpendicular constraint."""
        idx1 = self._get_element_index(constraint.references[0])
        idx2 = self._get_element_index(constraint.references[1])
        sketch.addConstraint(Sketcher.Constraint('Perpendicular', idx1, idx2))

    def _add_parallel(self, sketch: Any, constraint: SketchConstraint) -> None:
        """Add a parallel constraint."""
        idx1 = self._get_element_index(constraint.references[0])
        idx2 = self._get_element_index(constraint.references[1])
        sketch.addConstraint(Sketcher.Constraint('Parallel', idx1, idx2))

    def _add_equal(self, sketch: Any, constraint: SketchConstraint) -> None:
        """Add equal constraints between multiple elements."""
        indices = [self._get_element_index(r) for r in constraint.references]
        # Chain equal constraints
        for i in range(len(indices) - 1):
            sketch.addConstraint(Sketcher.Constraint('Equal', indices[i], indices[i+1]))

    def _add_horizontal(self, sketch: Any, constraint: SketchConstraint) -> None:
        """Add a horizontal constraint."""
        idx = self._get_element_index(constraint.references[0])
        sketch.addConstraint(Sketcher.Constraint('Horizontal', idx))

    def _add_vertical(self, sketch: Any, constraint: SketchConstraint) -> None:
        """Add a vertical constraint."""
        idx = self._get_element_index(constraint.references[0])
        sketch.addConstraint(Sketcher.Constraint('Vertical', idx))

    def _add_fixed(self, sketch: Any, constraint: SketchConstraint) -> None:
        """Add a fixed (block) constraint to lock geometry position."""
        idx = self._get_element_index(constraint.references[0])
        sketch.addConstraint(Sketcher.Constraint('Block', idx))

    def _add_concentric(self, sketch: Any, constraint: SketchConstraint) -> None:
        """Add a concentric constraint (coincident centers)."""
        idx1 = self._get_element_index(constraint.references[0])
        idx2 = self._get_element_index(constraint.references[1])
        # Concentric = coincident centers
        sketch.addConstraint(Sketcher.Constraint(
            'Coincident', idx1, VertexMap.CIRCLE_CENTER, idx2, VertexMap.CIRCLE_CENTER
        ))

    def _add_distance(self, sketch: Any, constraint: SketchConstraint) -> None:
        """Add a distance constraint."""
        pt1 = self._point_ref_to_freecad(constraint.references[0])
        pt2 = self._point_ref_to_freecad(constraint.references[1])
        sketch.addConstraint(Sketcher.Constraint(
            'Distance', pt1[0], pt1[1], pt2[0], pt2[1], constraint.value
        ))

    def _add_distance_x(self, sketch: Any, constraint: SketchConstraint) -> None:
        """Add a horizontal distance constraint."""
        if len(constraint.references) == 1:
            # Distance from origin
            pt = self._point_ref_to_freecad(constraint.references[0])
            sketch.addConstraint(Sketcher.Constraint(
                'DistanceX', VertexMap.ORIGIN_GEO_INDEX, VertexMap.ORIGIN_VERTEX,
                pt[0], pt[1], constraint.value
            ))
        else:
            # Distance between two points
            pt1 = self._point_ref_to_freecad(constraint.references[0])
            pt2 = self._point_ref_to_freecad(constraint.references[1])
            sketch.addConstraint(Sketcher.Constraint(
                'DistanceX', pt1[0], pt1[1], pt2[0], pt2[1], constraint.value
            ))

    def _add_distance_y(self, sketch: Any, constraint: SketchConstraint) -> None:
        """Add a vertical distance constraint."""
        if len(constraint.references) == 1:
            # Distance from origin
            pt = self._point_ref_to_freecad(constraint.references[0])
            sketch.addConstraint(Sketcher.Constraint(
                'DistanceY', VertexMap.ORIGIN_GEO_INDEX, VertexMap.ORIGIN_VERTEX,
                pt[0], pt[1], constraint.value
            ))
        else:
            # Distance between two points
            pt1 = self._point_ref_to_freecad(constraint.references[0])
            pt2 = self._point_ref_to_freecad(constraint.references[1])
            sketch.addConstraint(Sketcher.Constraint(
                'DistanceY', pt1[0], pt1[1], pt2[0], pt2[1], constraint.value
            ))

    def _add_radius(self, sketch: Any, constraint: SketchConstraint) -> None:
        """Add a radius constraint."""
        idx = self._get_element_index(constraint.references[0])
        sketch.addConstraint(Sketcher.Constraint('Radius', idx, constraint.value))

    def _add_diameter(self, sketch: Any, constraint: SketchConstraint) -> None:
        """Add a diameter constraint."""
        idx = self._get_element_index(constraint.references[0])
        sketch.addConstraint(Sketcher.Constraint('Diameter', idx, constraint.value))

    def _add_angle(self, sketch: Any, constraint: SketchConstraint) -> None:
        """Add an angle constraint (value in degrees)."""
        idx1 = self._get_element_index(constraint.references[0])
        idx2 = self._get_element_index(constraint.references[1])
        # FreeCAD uses radians
        angle_rad = math.radians(constraint.value)
        sketch.addConstraint(Sketcher.Constraint('Angle', idx1, idx2, angle_rad))

    def _add_symmetric(self, sketch: Any, constraint: SketchConstraint) -> None:
        """Add a symmetric constraint."""
        # References: [element1, element2, symmetry_axis]
        if len(constraint.references) < 3:
            raise ConstraintError("Symmetric constraint requires 3 references")

        ref1 = constraint.references[0]
        ref2 = constraint.references[1]
        axis_id = constraint.references[2]

        if isinstance(ref1, PointRef) and isinstance(ref2, PointRef):
            # Point symmetry
            pt1 = self._point_ref_to_freecad(ref1)
            pt2 = self._point_ref_to_freecad(ref2)
            axis_idx = self._get_element_index(axis_id)
            sketch.addConstraint(Sketcher.Constraint(
                'Symmetric', pt1[0], pt1[1], pt2[0], pt2[1], axis_idx
            ))
        else:
            # Element symmetry
            idx1 = self._get_element_index(ref1)
            idx2 = self._get_element_index(ref2)
            axis_idx = self._get_element_index(axis_id)
            sketch.addConstraint(Sketcher.Constraint(
                'Symmetric', idx1, idx2, axis_idx
            ))

    def _add_length(self, sketch: Any, constraint: SketchConstraint) -> None:
        """Add a length constraint to a line.

        FreeCAD implements line length as Distance constraint between
        the line's start point (vertex 1) and end point (vertex 2).
        """
        if constraint.value is None:
            raise ConstraintError("Length constraint requires a value")
        idx = self._get_element_index(constraint.references[0])
        # Distance from start (vertex 1) to end (vertex 2) of the same line
        sketch.addConstraint(Sketcher.Constraint(
            'Distance', idx, VertexMap.LINE_START, idx, VertexMap.LINE_END, constraint.value
        ))

    def _add_collinear(self, sketch: Any, constraint: SketchConstraint) -> None:
        """Add a collinear constraint between lines.

        FreeCAD implements collinear as Tangent constraint on lines.
        When Tangent is applied to two lines (not curves), they become collinear.
        """
        if len(constraint.references) < 2:
            raise ConstraintError("Collinear constraint requires at least 2 references")

        # Apply collinear constraint pairwise
        first_idx = self._get_element_index(constraint.references[0])
        for i in range(1, len(constraint.references)):
            next_idx = self._get_element_index(constraint.references[i])
            sketch.addConstraint(Sketcher.Constraint('Tangent', first_idx, next_idx))

    def _add_midpoint(self, sketch: Any, constraint: SketchConstraint) -> None:
        """Add a midpoint constraint (point at midpoint of line).

        FreeCAD doesn't have a native midpoint constraint. We use the Symmetric
        constraint which makes the target point equidistant from the line's
        start and end points, effectively placing it at the midpoint.
        """
        if len(constraint.references) != 2:
            raise ConstraintError("Midpoint constraint requires exactly 2 references")

        ref0 = constraint.references[0]
        ref1 = constraint.references[1]

        # Determine which reference is the point and which is the line
        if isinstance(ref0, PointRef):
            point_ref = ref0
            line_id = ref1
        elif isinstance(ref1, PointRef):
            point_ref = ref1
            line_id = ref0
        else:
            raise ConstraintError(
                "Midpoint constraint requires one PointRef and one line reference"
            )

        pt = self._point_ref_to_freecad(point_ref)
        line_idx = self._get_element_index(line_id)

        # Use Symmetric constraint: makes the point equidistant from line's endpoints
        # Symmetric(line_start, line_end, symmetry_point)
        sketch.addConstraint(Sketcher.Constraint(
            'Symmetric',
            line_idx, VertexMap.LINE_START,  # First point (line start)
            line_idx, VertexMap.LINE_END,    # Second point (line end)
            pt[0], pt[1]                      # Point that will be at midpoint
        ))

    def get_solver_status(self) -> tuple[SolverStatus, int]:
        """Get the constraint solver status."""
        sketch = self._get_active_sketch()

        # FreeCAD solve method returns:
        # 0 = fully constrained
        # positive = degrees of freedom
        # negative = over-constrained or error
        result = sketch.solve()

        if result == 0:
            return (SolverStatus.FULLY_CONSTRAINED, 0)
        elif result > 0:
            return (SolverStatus.UNDER_CONSTRAINED, result)
        else:
            # Check for conflicts
            if hasattr(sketch, 'conflictingConstraints') and sketch.conflictingConstraints:
                return (SolverStatus.INCONSISTENT, -1)
            return (SolverStatus.OVER_CONSTRAINED, result)

    def capture_image(self, width: int, height: int) -> bytes:
        """
        Capture a visualization of the sketch.

        Note: This requires FreeCAD GUI to be available.
        """
        try:
            import FreeCADGui as Gui

            # Get the active view
            view = Gui.ActiveDocument.ActiveView

            # Set image size
            view.setImageSize(width, height)

            # Capture to temp file and read bytes
            import os
            import tempfile

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_path = f.name

            view.saveImage(temp_path, width, height, 'Current')

            with open(temp_path, 'rb') as f:
                data = f.read()

            os.unlink(temp_path)
            return data

        except ImportError:
            raise ExportError("FreeCADGui not available for image capture") from None
        except Exception as e:
            raise ExportError(f"Failed to capture image: {e}") from e

    def close_sketch(self) -> None:
        """Close the current sketch."""
        if self._sketch is not None:
            self._get_document().recompute()
        self._sketch = None
        self._sketch_doc = None
        self._id_to_index.clear()
        self._index_to_id.clear()

    def get_element_by_id(self, element_id: str) -> Any | None:
        """Get the FreeCAD geometry object for a canonical element ID."""
        if element_id not in self._id_to_index:
            return None
        idx = self._id_to_index[element_id]
        sketch = self._get_active_sketch()
        if idx < len(sketch.Geometry):
            return sketch.Geometry[idx]
        return None

    def supports_feature(self, feature: str) -> bool:
        """Check if a feature is supported."""
        supported = {
            "spline": True,
            "ellipse": True,
            "elliptical_arc": True,
            "three_point_arc": True,
            "image_capture": True,
            "solver_status": True,
            "construction_geometry": True,
        }
        return supported.get(feature, False)

    # Export helpers

    def _geometry_to_primitive(self, geo: Any, index: int) -> SketchPrimitive | None:
        """Convert FreeCAD geometry to canonical primitive."""
        geo_type = type(geo).__name__
        # Construction flag is stored on the sketch, not the geometry
        sketch = self._get_active_sketch()
        is_construction = sketch.getConstruction(index) if hasattr(sketch, 'getConstruction') else False

        if 'LineSegment' in geo_type:
            return Line(
                start=Point2D(geo.StartPoint.x, geo.StartPoint.y),
                end=Point2D(geo.EndPoint.x, geo.EndPoint.y),
                construction=is_construction
            )
        elif 'ArcOfCircle' in geo_type or geo_type == 'Arc':
            center = geo.Center if hasattr(geo, 'Center') else geo.Location
            # Determine CCW from angles
            start_angle = geo.FirstParameter if hasattr(geo, 'FirstParameter') else 0
            end_angle = geo.LastParameter if hasattr(geo, 'LastParameter') else math.pi
            ccw = end_angle > start_angle

            return Arc(
                center=Point2D(center.x, center.y),
                start_point=Point2D(geo.StartPoint.x, geo.StartPoint.y),
                end_point=Point2D(geo.EndPoint.x, geo.EndPoint.y),
                ccw=ccw,
                construction=is_construction
            )
        elif 'Circle' in geo_type and 'Arc' not in geo_type:
            return Circle(
                center=Point2D(geo.Center.x, geo.Center.y),
                radius=geo.Radius,
                construction=is_construction
            )
        elif 'Point' in geo_type:
            # FreeCAD Point uses uppercase X, Y, Z attributes
            if hasattr(geo, 'X'):
                return Point(
                    position=Point2D(geo.X, geo.Y),
                    construction=is_construction
                )
            elif hasattr(geo, 'Point'):
                pos = geo.Point
                return Point(
                    position=Point2D(pos.x, pos.y),
                    construction=is_construction
                )
            else:
                return None
        elif 'BSpline' in geo_type:
            poles = geo.getPoles()
            control_points = [Point2D(p.x, p.y) for p in poles]
            knots = list(geo.getKnots())
            weights = list(geo.getWeights()) if geo.isRational() else None

            # Expand knots with multiplicities
            mults = geo.getMultiplicities()
            full_knots = []
            for k, m in zip(knots, mults, strict=False):
                full_knots.extend([k] * m)

            return Spline(
                degree=geo.Degree,
                control_points=control_points,
                knots=full_knots,
                weights=weights,
                periodic=geo.isPeriodic(),
                construction=is_construction
            )
        elif 'ArcOfEllipse' in geo_type:
            # Elliptical arc - extract base ellipse properties
            ellipse = geo.Ellipse
            center = ellipse.Center
            major_radius = ellipse.MajorRadius
            minor_radius = ellipse.MinorRadius

            # Calculate rotation from the major axis direction
            # FreeCAD ellipse MajorAxis is a vector, we need the angle
            major_axis = ellipse.XAxis if hasattr(ellipse, 'XAxis') else App.Vector(1, 0, 0)
            rotation = math.atan2(major_axis.y, major_axis.x)

            # Get parameter range
            start_param = geo.FirstParameter
            end_param = geo.LastParameter

            # Determine CCW based on parameter order
            ccw = end_param > start_param

            return EllipticalArc(
                center=Point2D(center.x, center.y),
                major_radius=major_radius,
                minor_radius=minor_radius,
                rotation=rotation,
                start_param=start_param,
                end_param=end_param,
                ccw=ccw,
                construction=is_construction
            )
        elif 'Ellipse' in geo_type and 'Arc' not in geo_type:
            center = geo.Center
            major_radius = geo.MajorRadius
            minor_radius = geo.MinorRadius

            # Calculate rotation from the major axis direction
            major_axis = geo.XAxis if hasattr(geo, 'XAxis') else App.Vector(1, 0, 0)
            rotation = math.atan2(major_axis.y, major_axis.x)

            return Ellipse(
                center=Point2D(center.x, center.y),
                major_radius=major_radius,
                minor_radius=minor_radius,
                rotation=rotation,
                construction=is_construction
            )

        return None

    def _detect_constraint_type(
        self, fc_constraint: Any, fc_type: str, sketch: Any
    ) -> ConstraintType | None:
        """Detect the canonical constraint type from a FreeCAD constraint.

        Some FreeCAD constraint types map to different canonical types depending
        on context (e.g., Distance can be LENGTH when applied to a single line).
        """
        first = fc_constraint.First
        second = getattr(fc_constraint, 'Second', -1)
        first_pos = getattr(fc_constraint, 'FirstPos', 0)
        second_pos = getattr(fc_constraint, 'SecondPos', 0)

        # Distance constraint on same element with start/end → LENGTH
        if fc_type == 'Distance' and first == second and first >= 0:
            if first_pos == VertexMap.LINE_START and second_pos == VertexMap.LINE_END:
                return ConstraintType.LENGTH

        # Tangent on two lines → COLLINEAR
        if fc_type == 'Tangent' and first >= 0 and second >= 0:
            if first < len(sketch.Geometry) and second < len(sketch.Geometry):
                geo1 = sketch.Geometry[first]
                geo2 = sketch.Geometry[second]
                if 'Line' in type(geo1).__name__ and 'Line' in type(geo2).__name__:
                    return ConstraintType.COLLINEAR

        # Symmetric constraint where First==Second (same line) with start/end → MIDPOINT
        if fc_type == 'Symmetric':
            if (first == second and first >= 0 and
                    first_pos == VertexMap.LINE_START and
                    second_pos == VertexMap.LINE_END):
                # This is a midpoint constraint (point symmetric about line endpoints)
                if first < len(sketch.Geometry):
                    if 'Line' in type(sketch.Geometry[first]).__name__:
                        return ConstraintType.MIDPOINT

        # Check for concentric (both positions are center = 3 on arcs/circles)
        if fc_type == 'Coincident':
            if first >= 0 and second >= 0:
                if first_pos == VertexMap.CIRCLE_CENTER and second_pos == VertexMap.CIRCLE_CENTER:
                    if first < len(sketch.Geometry) and second < len(sketch.Geometry):
                        geo1_name = type(sketch.Geometry[first]).__name__
                        geo2_name = type(sketch.Geometry[second]).__name__
                        if (('Arc' in geo1_name or 'Circle' in geo1_name) and
                                ('Arc' in geo2_name or 'Circle' in geo2_name)):
                            return ConstraintType.CONCENTRIC

        # Standard type map for other cases
        type_map = {
            'Coincident': ConstraintType.COINCIDENT,
            'Tangent': ConstraintType.TANGENT,
            'Perpendicular': ConstraintType.PERPENDICULAR,
            'Parallel': ConstraintType.PARALLEL,
            'Equal': ConstraintType.EQUAL,
            'Horizontal': ConstraintType.HORIZONTAL,
            'Vertical': ConstraintType.VERTICAL,
            'Fixed': ConstraintType.FIXED,
            'Block': ConstraintType.FIXED,
            'Distance': ConstraintType.DISTANCE,
            'DistanceX': ConstraintType.DISTANCE_X,
            'DistanceY': ConstraintType.DISTANCE_Y,
            'Radius': ConstraintType.RADIUS,
            'Diameter': ConstraintType.DIAMETER,
            'Angle': ConstraintType.ANGLE,
            'Symmetric': ConstraintType.SYMMETRIC,
        }

        return type_map.get(fc_type)

    def _fc_constraint_to_canonical(self, fc_constraint: Any) -> SketchConstraint | None:
        """Convert FreeCAD constraint to canonical form."""
        fc_type = fc_constraint.Type
        sketch = self._get_active_sketch()

        # Detect special constraint types based on context
        constraint_type = self._detect_constraint_type(fc_constraint, fc_type, sketch)
        if constraint_type is None:
            return None

        # Build references based on constraint type
        references = self._extract_constraint_references(fc_constraint, constraint_type)
        if references is None:
            return None

        # Get value if applicable
        value = None
        if hasattr(fc_constraint, 'Value'):
            value = fc_constraint.Value
            # Convert radians to degrees for angle constraints
            if constraint_type == ConstraintType.ANGLE:
                value = math.degrees(value)

        return SketchConstraint(
            id=str(fc_constraint.Name) if hasattr(fc_constraint, 'Name') else "",
            constraint_type=constraint_type,
            references=references,
            value=value
        )

    def _extract_constraint_references(
        self, fc_constraint: Any, constraint_type: ConstraintType
    ) -> list[str | PointRef] | None:
        """Extract references from a FreeCAD constraint."""
        first = fc_constraint.First
        second = fc_constraint.Second if hasattr(fc_constraint, 'Second') else -1
        third = fc_constraint.Third if hasattr(fc_constraint, 'Third') else -1

        first_pos = fc_constraint.FirstPos if hasattr(fc_constraint, 'FirstPos') else 0
        second_pos = fc_constraint.SecondPos if hasattr(fc_constraint, 'SecondPos') else 0

        # Convert geometry indices to element IDs
        def idx_to_id(idx: int) -> str | None:
            return self._index_to_id.get(idx)

        def idx_to_point_ref(idx: int, pos: int) -> PointRef | None:
            elem_id = idx_to_id(idx)
            if elem_id is None:
                return None
            # Infer point type from position
            sketch = self._get_active_sketch()
            if idx < len(sketch.Geometry):
                geo = sketch.Geometry[idx]
                prim_type = self._geo_to_prim_type(geo)
                point_type = get_point_type_from_vertex(prim_type, pos)
                if point_type:
                    return PointRef(elem_id, point_type)
            return None

        # Point-based constraints
        if constraint_type == ConstraintType.COINCIDENT:
            ref1 = idx_to_point_ref(first, first_pos)
            ref2 = idx_to_point_ref(second, second_pos)
            if ref1 and ref2:
                return [ref1, ref2]

        # Element-based constraints
        elif constraint_type in (
            ConstraintType.TANGENT, ConstraintType.PERPENDICULAR,
            ConstraintType.PARALLEL, ConstraintType.EQUAL
        ):
            id1 = idx_to_id(first)
            id2 = idx_to_id(second)
            if id1 and id2:
                return [id1, id2]

        # Single-element constraints
        elif constraint_type in (
            ConstraintType.HORIZONTAL, ConstraintType.VERTICAL,
            ConstraintType.FIXED, ConstraintType.RADIUS, ConstraintType.DIAMETER
        ):
            id1 = idx_to_id(first)
            if id1:
                return [id1]

        # Distance constraints
        elif constraint_type == ConstraintType.DISTANCE:
            ref1 = idx_to_point_ref(first, first_pos)
            ref2 = idx_to_point_ref(second, second_pos)
            if ref1 and ref2:
                return [ref1, ref2]

        elif constraint_type in (ConstraintType.DISTANCE_X, ConstraintType.DISTANCE_Y):
            ref1 = idx_to_point_ref(first, first_pos)
            if ref1:
                if second >= 0:
                    ref2 = idx_to_point_ref(second, second_pos)
                    if ref2:
                        return [ref1, ref2]
                return [ref1]

        # Angle constraint
        elif constraint_type == ConstraintType.ANGLE:
            id1 = idx_to_id(first)
            id2 = idx_to_id(second)
            if id1 and id2:
                return [id1, id2]

        # Symmetric constraint
        elif constraint_type == ConstraintType.SYMMETRIC:
            ref1 = idx_to_point_ref(first, first_pos)
            ref2 = idx_to_point_ref(second, second_pos)
            axis_id = idx_to_id(third)
            if ref1 and ref2 and axis_id:
                return [ref1, ref2, axis_id]

        # Length constraint (Distance on same line's endpoints)
        elif constraint_type == ConstraintType.LENGTH:
            id1 = idx_to_id(first)
            if id1:
                return [id1]

        # Collinear constraint (Tangent on lines)
        elif constraint_type == ConstraintType.COLLINEAR:
            id1 = idx_to_id(first)
            id2 = idx_to_id(second)
            if id1 and id2:
                return [id1, id2]

        # Midpoint constraint (stored as Symmetric with line endpoints and point)
        elif constraint_type == ConstraintType.MIDPOINT:
            # Symmetric format: First=line, FirstPos=1, Second=line, SecondPos=2, Third=point
            third_pos = getattr(fc_constraint, 'ThirdPos', 0)
            line_id = idx_to_id(first)  # First and Second are the same line
            point_ref = idx_to_point_ref(third, third_pos)
            if line_id and point_ref:
                return [point_ref, line_id]

        # Concentric constraint (Coincident on arc/circle centers)
        elif constraint_type == ConstraintType.CONCENTRIC:
            id1 = idx_to_id(first)
            id2 = idx_to_id(second)
            if id1 and id2:
                return [id1, id2]

        return None

    def _geo_to_prim_type(self, geo: Any) -> type:
        """Get canonical primitive type from FreeCAD geometry."""
        geo_type = type(geo).__name__
        if 'Line' in geo_type:
            return Line
        elif 'ArcOfEllipse' in geo_type:
            return EllipticalArc
        elif 'Arc' in geo_type:
            return Arc
        elif 'Ellipse' in geo_type:
            return Ellipse
        elif 'Circle' in geo_type:
            return Circle
        elif 'Point' in geo_type:
            return Point
        elif 'BSpline' in geo_type:
            return Spline
        return Line  # Default
