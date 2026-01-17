"""Tests for the adapter interface and FreeCAD adapter."""

import math
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from morphe import (
    AdapterError,
    Angle,
    Arc,
    Circle,
    Coincident,
    ConstraintError,
    # Constraints
    ConstraintType,
    Diameter,
    Distance,
    DistanceX,
    DistanceY,
    ExportError,
    Fixed,
    GeometryError,
    Horizontal,
    Line,
    Parallel,
    Perpendicular,
    Point,
    # Types
    Point2D,
    PointRef,
    PointType,
    Radius,
    # Adapter interface
    SketchBackendAdapter,
    SketchConstraint,
    SketchCreationError,
    # Document
    SketchDocument,
    # Primitives
    SketchPrimitive,
    SolverStatus,
    Spline,
    Symmetric,
    Tangent,
    Vertical,
)

# Import vertex_map module (doesn't require FreeCAD)
from morphe.adapters.freecad.vertex_map import (
    VertexMap,
    get_point_type_from_vertex,
    get_vertex_index,
)

# =============================================================================
# Adapter Interface Tests
# =============================================================================

class ConcreteAdapter(SketchBackendAdapter):
    """Concrete implementation for testing the abstract base class."""

    def __init__(self):
        self._sketch_created = False
        self._primitives = []
        self._constraints = []

    def create_sketch(self, name: str, plane: Any | None = None) -> None:
        self._sketch_created = True
        self._name = name

    def load_sketch(self, sketch: SketchDocument) -> None:
        self._sketch_created = True
        for prim in sketch.primitives.values():
            self.add_primitive(prim)
        for constraint in sketch.constraints:
            self.add_constraint(constraint)

    def export_sketch(self) -> SketchDocument:
        return SketchDocument(name="exported")

    def add_primitive(self, primitive: SketchPrimitive) -> Any:
        self._primitives.append(primitive)
        return len(self._primitives) - 1

    def add_constraint(self, constraint: SketchConstraint) -> bool:
        self._constraints.append(constraint)
        return True

    def get_solver_status(self) -> tuple[SolverStatus, int]:
        return (SolverStatus.FULLY_CONSTRAINED, 0)

    def capture_image(self, width: int, height: int) -> bytes:
        return b'\x89PNG\r\n\x1a\n'  # Minimal PNG header


class TestSketchBackendAdapter:
    """Tests for the abstract base class interface."""

    def test_create_sketch(self):
        adapter = ConcreteAdapter()
        adapter.create_sketch("TestSketch")
        assert adapter._sketch_created
        assert adapter._name == "TestSketch"

    def test_create_sketch_with_plane(self):
        adapter = ConcreteAdapter()
        mock_plane = Mock()
        adapter.create_sketch("TestSketch", plane=mock_plane)
        assert adapter._sketch_created

    def test_load_sketch(self):
        adapter = ConcreteAdapter()
        sketch = SketchDocument(name="Test")
        sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
        sketch.add_constraint(Horizontal("L0"))

        adapter.load_sketch(sketch)
        assert len(adapter._primitives) == 1
        assert len(adapter._constraints) == 1

    def test_export_sketch(self):
        adapter = ConcreteAdapter()
        adapter.create_sketch("Test")
        result = adapter.export_sketch()
        assert isinstance(result, SketchDocument)
        assert result.name == "exported"

    def test_add_primitive(self):
        adapter = ConcreteAdapter()
        line = Line(start=Point2D(0, 0), end=Point2D(10, 0))
        result = adapter.add_primitive(line)
        assert result == 0
        assert adapter._primitives[0] is line

    def test_add_constraint(self):
        adapter = ConcreteAdapter()
        constraint = Horizontal("L0")
        result = adapter.add_constraint(constraint)
        assert result is True
        assert adapter._constraints[0] is constraint

    def test_get_solver_status(self):
        adapter = ConcreteAdapter()
        status, dof = adapter.get_solver_status()
        assert status == SolverStatus.FULLY_CONSTRAINED
        assert dof == 0

    def test_capture_image(self):
        adapter = ConcreteAdapter()
        img = adapter.capture_image(800, 600)
        assert isinstance(img, bytes)
        assert img.startswith(b'\x89PNG')

    def test_close_sketch_default(self):
        adapter = ConcreteAdapter()
        # Default implementation does nothing
        adapter.close_sketch()  # Should not raise

    def test_get_element_by_id_default(self):
        adapter = ConcreteAdapter()
        # Default implementation returns None
        assert adapter.get_element_by_id("L0") is None

    def test_supports_feature_default(self):
        adapter = ConcreteAdapter()
        # Default implementation returns False
        assert adapter.supports_feature("spline") is False
        assert adapter.supports_feature("unknown") is False


class TestAdapterExceptions:
    """Tests for adapter exception classes."""

    def test_adapter_error(self):
        err = AdapterError("Generic adapter error")
        assert str(err) == "Generic adapter error"
        assert isinstance(err, Exception)

    def test_sketch_creation_error(self):
        err = SketchCreationError("Failed to create sketch")
        assert str(err) == "Failed to create sketch"
        assert isinstance(err, AdapterError)

    def test_geometry_error(self):
        err = GeometryError("Invalid geometry")
        assert str(err) == "Invalid geometry"
        assert isinstance(err, AdapterError)

    def test_constraint_error(self):
        err = ConstraintError("Constraint conflict")
        assert str(err) == "Constraint conflict"
        assert isinstance(err, AdapterError)

    def test_export_error(self):
        err = ExportError("Export failed")
        assert str(err) == "Export failed"
        assert isinstance(err, AdapterError)


# =============================================================================
# Vertex Map Tests
# =============================================================================

class TestVertexMap:
    """Tests for the FreeCAD vertex mapping constants."""

    def test_line_vertex_constants(self):
        assert VertexMap.LINE_START == 1
        assert VertexMap.LINE_END == 2

    def test_arc_vertex_constants(self):
        assert VertexMap.ARC_START == 1
        assert VertexMap.ARC_END == 2
        assert VertexMap.ARC_CENTER == 3

    def test_circle_vertex_constants(self):
        assert VertexMap.CIRCLE_CENTER == 3

    def test_point_vertex_constants(self):
        assert VertexMap.POINT_CENTER == 1

    def test_spline_vertex_constants(self):
        assert VertexMap.SPLINE_START == 1
        assert VertexMap.SPLINE_END == 2

    def test_origin_constants(self):
        assert VertexMap.ORIGIN_GEO_INDEX == -1
        assert VertexMap.ORIGIN_VERTEX == 1

    def test_external_geo_base(self):
        assert VertexMap.EXTERNAL_GEO_BASE == -2


class TestGetVertexIndex:
    """Tests for get_vertex_index function."""

    def test_line_start(self):
        assert get_vertex_index(Line, PointType.START) == 1

    def test_line_end(self):
        assert get_vertex_index(Line, PointType.END) == 2

    def test_line_invalid_point_type(self):
        assert get_vertex_index(Line, PointType.CENTER) is None

    def test_arc_start(self):
        assert get_vertex_index(Arc, PointType.START) == 1

    def test_arc_end(self):
        assert get_vertex_index(Arc, PointType.END) == 2

    def test_arc_center(self):
        assert get_vertex_index(Arc, PointType.CENTER) == 3

    def test_arc_invalid_point_type(self):
        assert get_vertex_index(Arc, PointType.MIDPOINT) is None

    def test_circle_center(self):
        assert get_vertex_index(Circle, PointType.CENTER) == 3

    def test_circle_invalid_point_type(self):
        assert get_vertex_index(Circle, PointType.START) is None

    def test_point_center(self):
        assert get_vertex_index(Point, PointType.CENTER) == 1

    def test_point_invalid_point_type(self):
        assert get_vertex_index(Point, PointType.END) is None

    def test_spline_start(self):
        assert get_vertex_index(Spline, PointType.START) == 1

    def test_spline_end(self):
        assert get_vertex_index(Spline, PointType.END) == 2

    def test_spline_invalid_point_type(self):
        assert get_vertex_index(Spline, PointType.CENTER) is None

    def test_unknown_primitive_type(self):
        class UnknownPrimitive:
            pass
        assert get_vertex_index(UnknownPrimitive, PointType.START) is None


class TestGetPointTypeFromVertex:
    """Tests for get_point_type_from_vertex function."""

    def test_line_vertex_1(self):
        assert get_point_type_from_vertex(Line, 1) == PointType.START

    def test_line_vertex_2(self):
        assert get_point_type_from_vertex(Line, 2) == PointType.END

    def test_line_invalid_vertex(self):
        assert get_point_type_from_vertex(Line, 3) is None

    def test_arc_vertex_1(self):
        assert get_point_type_from_vertex(Arc, 1) == PointType.START

    def test_arc_vertex_2(self):
        assert get_point_type_from_vertex(Arc, 2) == PointType.END

    def test_arc_vertex_3(self):
        assert get_point_type_from_vertex(Arc, 3) == PointType.CENTER

    def test_arc_invalid_vertex(self):
        assert get_point_type_from_vertex(Arc, 4) is None

    def test_circle_vertex_3(self):
        assert get_point_type_from_vertex(Circle, 3) == PointType.CENTER

    def test_circle_invalid_vertex(self):
        assert get_point_type_from_vertex(Circle, 1) is None

    def test_point_vertex_1(self):
        assert get_point_type_from_vertex(Point, 1) == PointType.CENTER

    def test_point_invalid_vertex(self):
        assert get_point_type_from_vertex(Point, 2) is None

    def test_spline_vertex_1(self):
        assert get_point_type_from_vertex(Spline, 1) == PointType.START

    def test_spline_vertex_2(self):
        assert get_point_type_from_vertex(Spline, 2) == PointType.END

    def test_spline_invalid_vertex(self):
        assert get_point_type_from_vertex(Spline, 3) is None

    def test_unknown_primitive_type(self):
        class UnknownPrimitive:
            pass
        assert get_point_type_from_vertex(UnknownPrimitive, 1) is None


# =============================================================================
# FreeCAD Adapter Tests (Mocked)
# =============================================================================

class TestFreeCADAdapterImport:
    """Test FreeCAD adapter import behavior."""

    def test_freecad_available_flag(self):
        from morphe.adapters.freecad import FREECAD_AVAILABLE
        # In test environment, FreeCAD is not available
        assert FREECAD_AVAILABLE is False

    def test_import_adapter_without_freecad(self):
        """FreeCADAdapter can be imported even without FreeCAD."""
        from morphe.adapters.freecad import FreeCADAdapter
        assert FreeCADAdapter is not None

    def test_instantiation_requires_freecad(self):
        """Creating FreeCADAdapter instance requires FreeCAD."""
        from morphe.adapters.freecad import FreeCADAdapter
        with pytest.raises(ImportError) as exc_info:
            FreeCADAdapter()
        assert "FreeCAD is not available" in str(exc_info.value)


class TestFreeCADAdapterMocked:
    """Tests for FreeCADAdapter with mocked FreeCAD modules."""

    @pytest.fixture
    def mock_freecad(self):
        """Create mock FreeCAD modules."""
        # Create mock modules
        mock_app = MagicMock()
        mock_part = MagicMock()
        mock_sketcher = MagicMock()

        # Configure mock document
        mock_doc = MagicMock()
        mock_app.ActiveDocument = mock_doc

        # Configure mock sketch
        mock_sketch = MagicMock()
        mock_sketch.Geometry = []
        mock_sketch.Constraints = []
        mock_sketch.Label = "TestSketch"
        mock_sketch.solve.return_value = 0  # Fully constrained
        mock_doc.addObject.return_value = mock_sketch

        return {
            'FreeCAD': mock_app,
            'Part': mock_part,
            'Sketcher': mock_sketcher,
            'sketch': mock_sketch,
            'doc': mock_doc,
        }

    @pytest.fixture
    def adapter(self, mock_freecad):
        """Create adapter with mocked FreeCAD."""
        import morphe.adapters.freecad.adapter as adapter_module

        # Patch FREECAD_AVAILABLE and modules
        with patch.object(adapter_module, 'FREECAD_AVAILABLE', True), \
             patch.object(adapter_module, 'App', mock_freecad['FreeCAD']), \
             patch.object(adapter_module, 'Part', mock_freecad['Part']), \
             patch.object(adapter_module, 'Sketcher', mock_freecad['Sketcher']):

            adapter = adapter_module.FreeCADAdapter()
            yield adapter, mock_freecad

    def test_create_sketch(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("MySketch")

        mocks['doc'].addObject.assert_called_once_with(
            'Sketcher::SketchObject', 'MySketch'
        )

    def test_create_sketch_clears_mappings(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj._id_to_index['old'] = 0
        adapter_obj._index_to_id[0] = 'old'

        adapter_obj.create_sketch("NewSketch")

        assert len(adapter_obj._id_to_index) == 0
        assert len(adapter_obj._index_to_id) == 0

    def test_add_line(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        line = Line(start=Point2D(0, 0), end=Point2D(100, 0), id="L0")
        mocks['sketch'].addGeometry.return_value = 0

        idx = adapter_obj.add_primitive(line)

        assert idx == 0
        assert adapter_obj._id_to_index["L0"] == 0
        assert adapter_obj._index_to_id[0] == "L0"
        mocks['Part'].LineSegment.assert_called_once()

    def test_add_arc(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        arc = Arc(
            center=Point2D(0, 0),
            start_point=Point2D(10, 0),
            end_point=Point2D(0, 10),
            ccw=True,
            id="A0"
        )
        mocks['sketch'].addGeometry.return_value = 0

        idx = adapter_obj.add_primitive(arc)

        assert idx == 0
        mocks['Part'].Arc.assert_called_once()

    def test_add_circle(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        circle = Circle(center=Point2D(50, 50), radius=25, id="C0")
        mocks['sketch'].addGeometry.return_value = 0

        idx = adapter_obj.add_primitive(circle)

        assert idx == 0
        mocks['Part'].Circle.assert_called_once()

    def test_add_point(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        point = Point(position=Point2D(10, 20), id="P0")
        mocks['sketch'].addGeometry.return_value = 0

        idx = adapter_obj.add_primitive(point)

        assert idx == 0
        mocks['Part'].Point.assert_called_once()

    def test_add_spline(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        spline = Spline.create_uniform_bspline(
            control_points=[Point2D(0, 0), Point2D(10, 10), Point2D(20, 0), Point2D(30, 10)],
            degree=3,
        )
        spline = Spline(
            id="S0",
            degree=spline.degree,
            control_points=spline.control_points,
            knots=spline.knots,
            weights=spline.weights,
            periodic=spline.periodic,
        )
        mock_bspline = MagicMock()
        mocks['Part'].BSplineCurve.return_value = mock_bspline
        mocks['sketch'].addGeometry.return_value = 0

        idx = adapter_obj.add_primitive(spline)

        assert idx == 0
        mocks['Part'].BSplineCurve.assert_called_once()

    def test_add_unsupported_primitive(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        class UnsupportedPrimitive(SketchPrimitive):
            def get_point(self, point_type): return None
            def get_valid_point_types(self): return []

        with pytest.raises(GeometryError) as exc_info:
            adapter_obj.add_primitive(UnsupportedPrimitive(id="X0"))

        assert "Unsupported primitive type" in str(exc_info.value)

    def test_add_constraint_horizontal(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        # Add a line first
        line = Line(start=Point2D(0, 0), end=Point2D(100, 0), id="L0")
        mocks['sketch'].addGeometry.return_value = 0
        adapter_obj.add_primitive(line)
        adapter_obj._sketch_doc = SketchDocument(name="Test")
        adapter_obj._sketch_doc.primitives["L0"] = line

        constraint = Horizontal("L0")
        result = adapter_obj.add_constraint(constraint)

        assert result is True
        mocks['Sketcher'].Constraint.assert_called()

    def test_add_constraint_vertical(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        line = Line(start=Point2D(0, 0), end=Point2D(0, 100), id="L0")
        mocks['sketch'].addGeometry.return_value = 0
        adapter_obj.add_primitive(line)
        adapter_obj._sketch_doc = SketchDocument(name="Test")
        adapter_obj._sketch_doc.primitives["L0"] = line

        constraint = Vertical("L0")
        result = adapter_obj.add_constraint(constraint)

        assert result is True

    def test_add_constraint_fixed(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        line = Line(start=Point2D(0, 0), end=Point2D(100, 0), id="L0")
        mocks['sketch'].addGeometry.return_value = 0
        adapter_obj.add_primitive(line)
        adapter_obj._sketch_doc = SketchDocument(name="Test")
        adapter_obj._sketch_doc.primitives["L0"] = line

        constraint = Fixed("L0")
        result = adapter_obj.add_constraint(constraint)

        assert result is True

    def test_add_constraint_radius(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        circle = Circle(center=Point2D(0, 0), radius=50, id="C0")
        mocks['sketch'].addGeometry.return_value = 0
        adapter_obj.add_primitive(circle)
        adapter_obj._sketch_doc = SketchDocument(name="Test")
        adapter_obj._sketch_doc.primitives["C0"] = circle

        constraint = Radius("C0", 50)
        result = adapter_obj.add_constraint(constraint)

        assert result is True

    def test_add_constraint_diameter(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        circle = Circle(center=Point2D(0, 0), radius=25, id="C0")
        mocks['sketch'].addGeometry.return_value = 0
        adapter_obj.add_primitive(circle)
        adapter_obj._sketch_doc = SketchDocument(name="Test")
        adapter_obj._sketch_doc.primitives["C0"] = circle

        constraint = Diameter("C0", 50)
        result = adapter_obj.add_constraint(constraint)

        assert result is True

    def test_add_constraint_coincident(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        line1 = Line(start=Point2D(0, 0), end=Point2D(100, 0), id="L0")
        line2 = Line(start=Point2D(100, 0), end=Point2D(100, 100), id="L1")
        mocks['sketch'].addGeometry.side_effect = [0, 1]
        adapter_obj.add_primitive(line1)
        adapter_obj.add_primitive(line2)
        adapter_obj._sketch_doc = SketchDocument(name="Test")
        adapter_obj._sketch_doc.primitives["L0"] = line1
        adapter_obj._sketch_doc.primitives["L1"] = line2

        constraint = Coincident(
            PointRef("L0", PointType.END),
            PointRef("L1", PointType.START)
        )
        result = adapter_obj.add_constraint(constraint)

        assert result is True

    def test_add_constraint_tangent(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        line = Line(start=Point2D(0, 0), end=Point2D(100, 0), id="L0")
        circle = Circle(center=Point2D(50, 50), radius=50, id="C0")
        mocks['sketch'].addGeometry.side_effect = [0, 1]
        adapter_obj.add_primitive(line)
        adapter_obj.add_primitive(circle)
        adapter_obj._sketch_doc = SketchDocument(name="Test")
        adapter_obj._sketch_doc.primitives["L0"] = line
        adapter_obj._sketch_doc.primitives["C0"] = circle

        constraint = Tangent("L0", "C0")
        result = adapter_obj.add_constraint(constraint)

        assert result is True

    def test_add_constraint_perpendicular(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        line1 = Line(start=Point2D(0, 0), end=Point2D(100, 0), id="L0")
        line2 = Line(start=Point2D(0, 0), end=Point2D(0, 100), id="L1")
        mocks['sketch'].addGeometry.side_effect = [0, 1]
        adapter_obj.add_primitive(line1)
        adapter_obj.add_primitive(line2)
        adapter_obj._sketch_doc = SketchDocument(name="Test")
        adapter_obj._sketch_doc.primitives["L0"] = line1
        adapter_obj._sketch_doc.primitives["L1"] = line2

        constraint = Perpendicular("L0", "L1")
        result = adapter_obj.add_constraint(constraint)

        assert result is True

    def test_add_constraint_parallel(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        line1 = Line(start=Point2D(0, 0), end=Point2D(100, 0), id="L0")
        line2 = Line(start=Point2D(0, 50), end=Point2D(100, 50), id="L1")
        mocks['sketch'].addGeometry.side_effect = [0, 1]
        adapter_obj.add_primitive(line1)
        adapter_obj.add_primitive(line2)
        adapter_obj._sketch_doc = SketchDocument(name="Test")
        adapter_obj._sketch_doc.primitives["L0"] = line1
        adapter_obj._sketch_doc.primitives["L1"] = line2

        constraint = Parallel("L0", "L1")
        result = adapter_obj.add_constraint(constraint)

        assert result is True

    def test_add_constraint_distance(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        line = Line(start=Point2D(0, 0), end=Point2D(100, 0), id="L0")
        mocks['sketch'].addGeometry.return_value = 0
        adapter_obj.add_primitive(line)
        adapter_obj._sketch_doc = SketchDocument(name="Test")
        adapter_obj._sketch_doc.primitives["L0"] = line

        constraint = Distance(
            PointRef("L0", PointType.START),
            PointRef("L0", PointType.END),
            100
        )
        result = adapter_obj.add_constraint(constraint)

        assert result is True

    def test_add_constraint_distance_x_single(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        line = Line(start=Point2D(50, 0), end=Point2D(100, 0), id="L0")
        mocks['sketch'].addGeometry.return_value = 0
        adapter_obj.add_primitive(line)
        adapter_obj._sketch_doc = SketchDocument(name="Test")
        adapter_obj._sketch_doc.primitives["L0"] = line

        constraint = DistanceX(PointRef("L0", PointType.START), 50)
        result = adapter_obj.add_constraint(constraint)

        assert result is True

    def test_add_constraint_distance_x_double(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        line = Line(start=Point2D(0, 0), end=Point2D(50, 0), id="L0")
        mocks['sketch'].addGeometry.return_value = 0
        adapter_obj.add_primitive(line)
        adapter_obj._sketch_doc = SketchDocument(name="Test")
        adapter_obj._sketch_doc.primitives["L0"] = line

        # DistanceX signature: pt, value, pt2=None
        constraint = DistanceX(
            PointRef("L0", PointType.START),
            50,
            pt2=PointRef("L0", PointType.END)
        )
        result = adapter_obj.add_constraint(constraint)

        assert result is True

    def test_add_constraint_distance_y_single(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        line = Line(start=Point2D(0, 50), end=Point2D(0, 100), id="L0")
        mocks['sketch'].addGeometry.return_value = 0
        adapter_obj.add_primitive(line)
        adapter_obj._sketch_doc = SketchDocument(name="Test")
        adapter_obj._sketch_doc.primitives["L0"] = line

        constraint = DistanceY(PointRef("L0", PointType.START), 50)
        result = adapter_obj.add_constraint(constraint)

        assert result is True

    def test_add_constraint_distance_y_double(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        line = Line(start=Point2D(0, 0), end=Point2D(0, 50), id="L0")
        mocks['sketch'].addGeometry.return_value = 0
        adapter_obj.add_primitive(line)
        adapter_obj._sketch_doc = SketchDocument(name="Test")
        adapter_obj._sketch_doc.primitives["L0"] = line

        # DistanceY signature: pt, value, pt2=None
        constraint = DistanceY(
            PointRef("L0", PointType.START),
            50,
            pt2=PointRef("L0", PointType.END)
        )
        result = adapter_obj.add_constraint(constraint)

        assert result is True

    def test_add_constraint_angle(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        line1 = Line(start=Point2D(0, 0), end=Point2D(100, 0), id="L0")
        line2 = Line(start=Point2D(0, 0), end=Point2D(100, 100), id="L1")
        mocks['sketch'].addGeometry.side_effect = [0, 1]
        adapter_obj.add_primitive(line1)
        adapter_obj.add_primitive(line2)
        adapter_obj._sketch_doc = SketchDocument(name="Test")
        adapter_obj._sketch_doc.primitives["L0"] = line1
        adapter_obj._sketch_doc.primitives["L1"] = line2

        constraint = Angle("L0", "L1", 45)
        result = adapter_obj.add_constraint(constraint)

        assert result is True

    def test_add_constraint_symmetric_points(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        line1 = Line(start=Point2D(-50, 0), end=Point2D(0, 50), id="L0")
        line2 = Line(start=Point2D(50, 0), end=Point2D(0, 50), id="L1")
        axis = Line(start=Point2D(0, 0), end=Point2D(0, 100), id="L2")
        mocks['sketch'].addGeometry.side_effect = [0, 1, 2]
        adapter_obj.add_primitive(line1)
        adapter_obj.add_primitive(line2)
        adapter_obj.add_primitive(axis)
        adapter_obj._sketch_doc = SketchDocument(name="Test")
        adapter_obj._sketch_doc.primitives["L0"] = line1
        adapter_obj._sketch_doc.primitives["L1"] = line2
        adapter_obj._sketch_doc.primitives["L2"] = axis

        constraint = Symmetric(
            PointRef("L0", PointType.START),
            PointRef("L1", PointType.START),
            "L2"
        )
        result = adapter_obj.add_constraint(constraint)

        assert result is True

    def test_add_constraint_symmetric_elements(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        line1 = Line(start=Point2D(-50, 0), end=Point2D(-50, 50), id="L0")
        line2 = Line(start=Point2D(50, 0), end=Point2D(50, 50), id="L1")
        axis = Line(start=Point2D(0, 0), end=Point2D(0, 100), id="L2")
        mocks['sketch'].addGeometry.side_effect = [0, 1, 2]
        adapter_obj.add_primitive(line1)
        adapter_obj.add_primitive(line2)
        adapter_obj.add_primitive(axis)
        adapter_obj._sketch_doc = SketchDocument(name="Test")
        adapter_obj._sketch_doc.primitives["L0"] = line1
        adapter_obj._sketch_doc.primitives["L1"] = line2
        adapter_obj._sketch_doc.primitives["L2"] = axis

        # Element symmetry (strings instead of PointRefs)
        constraint = SketchConstraint(
            id="Sym0",
            constraint_type=ConstraintType.SYMMETRIC,
            references=["L0", "L1", "L2"]
        )
        result = adapter_obj.add_constraint(constraint)

        assert result is True

    def test_add_constraint_symmetric_insufficient_refs(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        line = Line(start=Point2D(0, 0), end=Point2D(100, 0), id="L0")
        mocks['sketch'].addGeometry.return_value = 0
        adapter_obj.add_primitive(line)
        adapter_obj._sketch_doc = SketchDocument(name="Test")
        adapter_obj._sketch_doc.primitives["L0"] = line

        constraint = SketchConstraint(
            id="Sym0",
            constraint_type=ConstraintType.SYMMETRIC,
            references=["L0"]  # Only 1 reference, need 3
        )

        with pytest.raises(ConstraintError) as exc_info:
            adapter_obj.add_constraint(constraint)

        assert "3 references" in str(exc_info.value)

    def test_add_constraint_equal(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        line1 = Line(start=Point2D(0, 0), end=Point2D(100, 0), id="L0")
        line2 = Line(start=Point2D(0, 50), end=Point2D(100, 50), id="L1")
        line3 = Line(start=Point2D(0, 100), end=Point2D(100, 100), id="L2")
        mocks['sketch'].addGeometry.side_effect = [0, 1, 2]
        adapter_obj.add_primitive(line1)
        adapter_obj.add_primitive(line2)
        adapter_obj.add_primitive(line3)
        adapter_obj._sketch_doc = SketchDocument(name="Test")
        adapter_obj._sketch_doc.primitives["L0"] = line1
        adapter_obj._sketch_doc.primitives["L1"] = line2
        adapter_obj._sketch_doc.primitives["L2"] = line3

        from morphe import Equal
        constraint = Equal("L0", "L1", "L2")
        result = adapter_obj.add_constraint(constraint)

        assert result is True
        # Equal should chain constraints: L0=L1, L1=L2
        assert mocks['sketch'].addConstraint.call_count >= 2

    def test_add_constraint_concentric(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        c1 = Circle(center=Point2D(50, 50), radius=25, id="C0")
        c2 = Circle(center=Point2D(50, 50), radius=50, id="C1")
        mocks['sketch'].addGeometry.side_effect = [0, 1]
        adapter_obj.add_primitive(c1)
        adapter_obj.add_primitive(c2)
        adapter_obj._sketch_doc = SketchDocument(name="Test")
        adapter_obj._sketch_doc.primitives["C0"] = c1
        adapter_obj._sketch_doc.primitives["C1"] = c2

        from morphe import Concentric
        constraint = Concentric("C0", "C1")
        result = adapter_obj.add_constraint(constraint)

        assert result is True

    def test_add_constraint_unknown_element(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")
        adapter_obj._sketch_doc = SketchDocument(name="Test")

        constraint = Horizontal("UnknownElement")

        with pytest.raises(ConstraintError) as exc_info:
            adapter_obj.add_constraint(constraint)

        assert "Unknown element ID" in str(exc_info.value)

    def test_get_solver_status_fully_constrained(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")
        mocks['sketch'].solve.return_value = 0

        status, dof = adapter_obj.get_solver_status()

        assert status == SolverStatus.FULLY_CONSTRAINED
        assert dof == 0

    def test_get_solver_status_under_constrained(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")
        mocks['sketch'].solve.return_value = 3

        status, dof = adapter_obj.get_solver_status()

        assert status == SolverStatus.UNDER_CONSTRAINED
        assert dof == 3

    def test_get_solver_status_over_constrained(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")
        mocks['sketch'].solve.return_value = -1
        mocks['sketch'].conflictingConstraints = []

        status, dof = adapter_obj.get_solver_status()

        assert status == SolverStatus.OVER_CONSTRAINED
        assert dof == -1

    def test_get_solver_status_inconsistent(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")
        mocks['sketch'].solve.return_value = -1
        mocks['sketch'].conflictingConstraints = [1, 2, 3]

        status, dof = adapter_obj.get_solver_status()

        assert status == SolverStatus.INCONSISTENT
        assert dof == -1

    def test_close_sketch(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")
        adapter_obj._id_to_index["L0"] = 0
        adapter_obj._index_to_id[0] = "L0"

        adapter_obj.close_sketch()

        assert adapter_obj._sketch is None
        assert adapter_obj._sketch_doc is None
        assert len(adapter_obj._id_to_index) == 0
        assert len(adapter_obj._index_to_id) == 0

    def test_get_element_by_id(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        line = Line(start=Point2D(0, 0), end=Point2D(100, 0), id="L0")
        mocks['sketch'].addGeometry.return_value = 0
        mock_geo = MagicMock()
        mocks['sketch'].Geometry = [mock_geo]

        adapter_obj.add_primitive(line)

        result = adapter_obj.get_element_by_id("L0")
        assert result is mock_geo

    def test_get_element_by_id_not_found(self, adapter):
        adapter_obj, mocks = adapter
        adapter_obj.create_sketch("Test")

        result = adapter_obj.get_element_by_id("NonExistent")
        assert result is None

    def test_supports_feature(self, adapter):
        adapter_obj, mocks = adapter

        assert adapter_obj.supports_feature("spline") is True
        assert adapter_obj.supports_feature("three_point_arc") is True
        assert adapter_obj.supports_feature("image_capture") is True
        assert adapter_obj.supports_feature("solver_status") is True
        assert adapter_obj.supports_feature("construction_geometry") is True
        assert adapter_obj.supports_feature("unknown_feature") is False

    def test_no_active_sketch_error(self, adapter):
        adapter_obj, mocks = adapter
        # Don't create a sketch

        line = Line(start=Point2D(0, 0), end=Point2D(100, 0), id="L0")

        with pytest.raises(SketchCreationError) as exc_info:
            adapter_obj.add_primitive(line)

        assert "No active sketch" in str(exc_info.value)


class TestFreeCADAdapterExport:
    """Tests for FreeCAD adapter export functionality."""

    @pytest.fixture
    def mock_freecad_for_export(self):
        """Create mock FreeCAD with geometry for export testing."""
        mock_app = MagicMock()
        mock_part = MagicMock()
        mock_sketcher = MagicMock()

        mock_doc = MagicMock()
        mock_app.ActiveDocument = mock_doc

        mock_sketch = MagicMock()
        mock_sketch.Label = "ExportTest"
        mock_sketch.solve.return_value = 0
        mock_doc.addObject.return_value = mock_sketch

        # Construction flags: line=False, circle=False, arc=True, point=False, spline=False
        construction_flags = {0: False, 1: False, 2: True, 3: False, 4: False}
        mock_sketch.getConstruction = lambda idx: construction_flags.get(idx, False)

        # Create mock geometries
        mock_line = MagicMock()
        mock_line.__class__.__name__ = 'LineSegment'
        mock_line.Construction = False
        mock_line.StartPoint.x = 0
        mock_line.StartPoint.y = 0
        mock_line.EndPoint.x = 100
        mock_line.EndPoint.y = 0

        mock_circle = MagicMock()
        mock_circle.__class__.__name__ = 'Circle'
        mock_circle.Construction = False
        mock_circle.Center.x = 50
        mock_circle.Center.y = 50
        mock_circle.Radius = 25

        mock_arc = MagicMock()
        mock_arc.__class__.__name__ = 'ArcOfCircle'
        mock_arc.Construction = True
        mock_arc.Center.x = 0
        mock_arc.Center.y = 0
        mock_arc.StartPoint.x = 10
        mock_arc.StartPoint.y = 0
        mock_arc.EndPoint.x = 0
        mock_arc.EndPoint.y = 10
        mock_arc.FirstParameter = 0
        mock_arc.LastParameter = math.pi / 2

        mock_point = MagicMock()
        mock_point.__class__.__name__ = 'Point'
        mock_point.Construction = False
        # FreeCAD Point uses uppercase X, Y, Z attributes
        mock_point.X = 25
        mock_point.Y = 25
        mock_point.Z = 0

        mock_bspline = MagicMock()
        mock_bspline.__class__.__name__ = 'BSplineCurve'
        mock_bspline.Construction = False
        mock_bspline.Degree = 3
        mock_bspline.isPeriodic.return_value = False
        mock_bspline.isRational.return_value = False
        mock_pole1 = MagicMock()
        mock_pole1.x, mock_pole1.y = 0, 0
        mock_pole2 = MagicMock()
        mock_pole2.x, mock_pole2.y = 10, 10
        mock_pole3 = MagicMock()
        mock_pole3.x, mock_pole3.y = 20, 0
        mock_pole4 = MagicMock()
        mock_pole4.x, mock_pole4.y = 30, 10
        mock_bspline.getPoles.return_value = [mock_pole1, mock_pole2, mock_pole3, mock_pole4]
        mock_bspline.getKnots.return_value = [0.0, 1.0]
        mock_bspline.getMultiplicities.return_value = [4, 4]

        mock_sketch.Geometry = [mock_line, mock_circle, mock_arc, mock_point, mock_bspline]
        mock_sketch.Constraints = []

        return {
            'FreeCAD': mock_app,
            'Part': mock_part,
            'Sketcher': mock_sketcher,
            'sketch': mock_sketch,
            'doc': mock_doc,
        }

    @pytest.fixture
    def adapter_for_export(self, mock_freecad_for_export):
        """Create adapter for export testing."""
        import morphe.adapters.freecad.adapter as adapter_module

        with patch.object(adapter_module, 'FREECAD_AVAILABLE', True), \
             patch.object(adapter_module, 'App', mock_freecad_for_export['FreeCAD']), \
             patch.object(adapter_module, 'Part', mock_freecad_for_export['Part']), \
             patch.object(adapter_module, 'Sketcher', mock_freecad_for_export['Sketcher']):

            adapter = adapter_module.FreeCADAdapter()
            adapter.create_sketch("Test")
            yield adapter, mock_freecad_for_export

    def test_export_sketch_with_geometry(self, adapter_for_export):
        adapter_obj, mocks = adapter_for_export

        result = adapter_obj.export_sketch()

        assert isinstance(result, SketchDocument)
        assert result.name == "ExportTest"
        # Should export 5 primitives
        assert len(result.primitives) == 5

    def test_export_line(self, adapter_for_export):
        adapter_obj, mocks = adapter_for_export

        result = adapter_obj.export_sketch()

        # First primitive should be a line
        line = list(result.primitives.values())[0]
        assert isinstance(line, Line)
        assert line.start.x == 0
        assert line.start.y == 0
        assert line.end.x == 100
        assert line.end.y == 0

    def test_export_circle(self, adapter_for_export):
        adapter_obj, mocks = adapter_for_export

        result = adapter_obj.export_sketch()

        # Second primitive should be a circle
        circle = list(result.primitives.values())[1]
        assert isinstance(circle, Circle)
        assert circle.center.x == 50
        assert circle.center.y == 50
        assert circle.radius == 25

    def test_export_arc(self, adapter_for_export):
        adapter_obj, mocks = adapter_for_export

        result = adapter_obj.export_sketch()

        # Third primitive should be an arc
        arc = list(result.primitives.values())[2]
        assert isinstance(arc, Arc)
        assert arc.construction is True

    def test_export_point(self, adapter_for_export):
        adapter_obj, mocks = adapter_for_export

        result = adapter_obj.export_sketch()

        # Fourth primitive should be a point
        point = list(result.primitives.values())[3]
        assert isinstance(point, Point)
        assert point.position.x == 25
        assert point.position.y == 25

    def test_export_spline(self, adapter_for_export):
        adapter_obj, mocks = adapter_for_export

        result = adapter_obj.export_sketch()

        # Fifth primitive should be a spline
        spline = list(result.primitives.values())[4]
        assert isinstance(spline, Spline)
        assert spline.degree == 3
        assert len(spline.control_points) == 4


class TestFreeCADAdapterConstraintExport:
    """Tests for FreeCAD constraint export."""

    @pytest.fixture
    def mock_freecad_with_constraints(self):
        """Create mock FreeCAD with constraints."""
        mock_app = MagicMock()
        mock_part = MagicMock()
        mock_sketcher = MagicMock()

        mock_doc = MagicMock()
        mock_app.ActiveDocument = mock_doc

        mock_sketch = MagicMock()
        mock_sketch.Label = "ConstraintTest"
        mock_sketch.solve.return_value = 0
        mock_doc.addObject.return_value = mock_sketch

        # Create mock geometry
        mock_line = MagicMock()
        mock_line.__class__.__name__ = 'LineSegment'
        mock_line.Construction = False
        mock_line.StartPoint.x = 0
        mock_line.StartPoint.y = 0
        mock_line.EndPoint.x = 100
        mock_line.EndPoint.y = 0

        mock_sketch.Geometry = [mock_line]

        # Create mock constraint
        mock_horizontal = MagicMock()
        mock_horizontal.Type = 'Horizontal'
        mock_horizontal.First = 0
        mock_horizontal.Name = 'H0'

        mock_radius = MagicMock()
        mock_radius.Type = 'Radius'
        mock_radius.First = 0
        mock_radius.Value = 50.0
        mock_radius.Name = 'R0'

        mock_unknown = MagicMock()
        mock_unknown.Type = 'UnknownConstraint'
        mock_unknown.First = 0

        mock_sketch.Constraints = [mock_horizontal, mock_radius, mock_unknown]

        return {
            'FreeCAD': mock_app,
            'Part': mock_part,
            'Sketcher': mock_sketcher,
            'sketch': mock_sketch,
            'doc': mock_doc,
        }

    @pytest.fixture
    def adapter_with_constraints(self, mock_freecad_with_constraints):
        """Create adapter with constraints."""
        import morphe.adapters.freecad.adapter as adapter_module

        with patch.object(adapter_module, 'FREECAD_AVAILABLE', True), \
             patch.object(adapter_module, 'App', mock_freecad_with_constraints['FreeCAD']), \
             patch.object(adapter_module, 'Part', mock_freecad_with_constraints['Part']), \
             patch.object(adapter_module, 'Sketcher', mock_freecad_with_constraints['Sketcher']):

            adapter = adapter_module.FreeCADAdapter()
            adapter.create_sketch("Test")
            yield adapter, mock_freecad_with_constraints

    def test_export_constraints(self, adapter_with_constraints):
        adapter_obj, mocks = adapter_with_constraints

        result = adapter_obj.export_sketch()

        # Should export known constraints, skip unknown
        assert len(result.constraints) >= 1


class TestInferVertexIndex:
    """Tests for _infer_vertex_index method."""

    @pytest.fixture
    def adapter(self):
        """Create mocked adapter."""
        import morphe.adapters.freecad.adapter as adapter_module

        mock_app = MagicMock()
        mock_app.ActiveDocument = MagicMock()
        mock_sketch = MagicMock()
        mock_app.ActiveDocument.addObject.return_value = mock_sketch

        with patch.object(adapter_module, 'FREECAD_AVAILABLE', True), \
             patch.object(adapter_module, 'App', mock_app), \
             patch.object(adapter_module, 'Part', MagicMock()), \
             patch.object(adapter_module, 'Sketcher', MagicMock()):

            adapter = adapter_module.FreeCADAdapter()
            adapter.create_sketch("Test")
            yield adapter

    def test_infer_line_vertex(self, adapter):
        mock_geo = MagicMock()
        mock_geo.__class__.__name__ = 'LineSegment'

        assert adapter._infer_vertex_index(mock_geo, PointType.START) == 1
        assert adapter._infer_vertex_index(mock_geo, PointType.END) == 2

    def test_infer_arc_vertex(self, adapter):
        mock_geo = MagicMock()
        mock_geo.__class__.__name__ = 'ArcOfCircle'

        assert adapter._infer_vertex_index(mock_geo, PointType.START) == 1
        assert adapter._infer_vertex_index(mock_geo, PointType.END) == 2
        assert adapter._infer_vertex_index(mock_geo, PointType.CENTER) == 3

    def test_infer_circle_vertex(self, adapter):
        mock_geo = MagicMock()
        mock_geo.__class__.__name__ = 'Circle'

        assert adapter._infer_vertex_index(mock_geo, PointType.CENTER) == 3

    def test_infer_point_vertex(self, adapter):
        mock_geo = MagicMock()
        mock_geo.__class__.__name__ = 'Point'

        assert adapter._infer_vertex_index(mock_geo, PointType.CENTER) == 1

    def test_infer_bspline_vertex(self, adapter):
        mock_geo = MagicMock()
        mock_geo.__class__.__name__ = 'BSplineCurve'

        assert adapter._infer_vertex_index(mock_geo, PointType.START) == 1
        assert adapter._infer_vertex_index(mock_geo, PointType.END) == 2

    def test_infer_unknown_vertex(self, adapter):
        mock_geo = MagicMock()
        mock_geo.__class__.__name__ = 'UnknownType'

        # Should default to 1
        assert adapter._infer_vertex_index(mock_geo, PointType.START) == 1


class TestComputeMultiplicities:
    """Tests for _compute_multiplicities method."""

    @pytest.fixture
    def adapter(self):
        """Create mocked adapter."""
        import morphe.adapters.freecad.adapter as adapter_module

        mock_app = MagicMock()
        mock_app.ActiveDocument = MagicMock()
        mock_sketch = MagicMock()
        mock_app.ActiveDocument.addObject.return_value = mock_sketch

        with patch.object(adapter_module, 'FREECAD_AVAILABLE', True), \
             patch.object(adapter_module, 'App', mock_app), \
             patch.object(adapter_module, 'Part', MagicMock()), \
             patch.object(adapter_module, 'Sketcher', MagicMock()):

            adapter = adapter_module.FreeCADAdapter()
            yield adapter

    def test_empty_knots(self, adapter):
        spline = Spline(degree=3, control_points=[], knots=[], id="S0")
        assert adapter._compute_multiplicities(spline) == []

    def test_uniform_knots(self, adapter):
        spline = Spline(
            degree=3,
            control_points=[Point2D(0, 0), Point2D(1, 1), Point2D(2, 0), Point2D(3, 1)],
            knots=[0, 0, 0, 0, 1, 1, 1, 1],
            id="S0"
        )
        mults = adapter._compute_multiplicities(spline)
        assert mults == [4, 4]

    def test_non_uniform_knots(self, adapter):
        spline = Spline(
            degree=2,
            control_points=[Point2D(0, 0), Point2D(1, 1), Point2D(2, 0)],
            knots=[0, 0, 0, 0.5, 1, 1, 1],
            id="S0"
        )
        mults = adapter._compute_multiplicities(spline)
        assert mults == [3, 1, 3]


class TestGeoToPrimType:
    """Tests for _geo_to_prim_type method."""

    @pytest.fixture
    def adapter(self):
        """Create mocked adapter."""
        import morphe.adapters.freecad.adapter as adapter_module

        mock_app = MagicMock()
        mock_app.ActiveDocument = MagicMock()
        mock_sketch = MagicMock()
        mock_app.ActiveDocument.addObject.return_value = mock_sketch

        with patch.object(adapter_module, 'FREECAD_AVAILABLE', True), \
             patch.object(adapter_module, 'App', mock_app), \
             patch.object(adapter_module, 'Part', MagicMock()), \
             patch.object(adapter_module, 'Sketcher', MagicMock()):

            adapter = adapter_module.FreeCADAdapter()
            yield adapter

    def test_line_type(self, adapter):
        mock_geo = MagicMock()
        mock_geo.__class__.__name__ = 'LineSegment'
        assert adapter._geo_to_prim_type(mock_geo) == Line

    def test_arc_type(self, adapter):
        mock_geo = MagicMock()
        mock_geo.__class__.__name__ = 'ArcOfCircle'
        assert adapter._geo_to_prim_type(mock_geo) == Arc

    def test_circle_type(self, adapter):
        mock_geo = MagicMock()
        mock_geo.__class__.__name__ = 'Circle'
        assert adapter._geo_to_prim_type(mock_geo) == Circle

    def test_point_type(self, adapter):
        mock_geo = MagicMock()
        mock_geo.__class__.__name__ = 'Point'
        assert adapter._geo_to_prim_type(mock_geo) == Point

    def test_bspline_type(self, adapter):
        mock_geo = MagicMock()
        mock_geo.__class__.__name__ = 'BSplineCurve'
        assert adapter._geo_to_prim_type(mock_geo) == Spline

    def test_unknown_defaults_to_line(self, adapter):
        mock_geo = MagicMock()
        mock_geo.__class__.__name__ = 'Unknown'
        assert adapter._geo_to_prim_type(mock_geo) == Line


class TestLoadSketch:
    """Tests for load_sketch functionality."""

    @pytest.fixture
    def adapter(self):
        """Create mocked adapter."""
        import morphe.adapters.freecad.adapter as adapter_module

        mock_app = MagicMock()
        mock_doc = MagicMock()
        mock_sketch = MagicMock()
        mock_sketch.Geometry = []
        mock_sketch.Constraints = []
        mock_app.ActiveDocument = mock_doc
        mock_doc.addObject.return_value = mock_sketch
        mock_sketch.addGeometry.side_effect = lambda g, c: len(mock_sketch.Geometry)

        with patch.object(adapter_module, 'FREECAD_AVAILABLE', True), \
             patch.object(adapter_module, 'App', mock_app), \
             patch.object(adapter_module, 'Part', MagicMock()), \
             patch.object(adapter_module, 'Sketcher', MagicMock()):

            adapter = adapter_module.FreeCADAdapter()
            yield adapter, mock_sketch, mock_doc

    def test_load_creates_sketch_if_needed(self, adapter):
        adapter_obj, mock_sketch, mock_doc = adapter

        sketch = SketchDocument(name="LoadTest")
        adapter_obj.load_sketch(sketch)

        mock_doc.addObject.assert_called()

    def test_load_adds_all_primitives(self, adapter):
        adapter_obj, mock_sketch, mock_doc = adapter

        sketch = SketchDocument(name="LoadTest")
        sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(100, 0)))
        sketch.add_primitive(Circle(center=Point2D(50, 50), radius=25))

        adapter_obj.load_sketch(sketch)

        # Should have called addGeometry twice
        assert mock_sketch.addGeometry.call_count == 2

    def test_load_triggers_recompute(self, adapter):
        adapter_obj, mock_sketch, mock_doc = adapter

        sketch = SketchDocument(name="LoadTest")
        adapter_obj.load_sketch(sketch)

        mock_doc.recompute.assert_called()


class TestCaptureImage:
    """Tests for capture_image functionality."""

    def test_capture_requires_gui(self):
        """capture_image requires FreeCADGui which won't be available."""
        import morphe.adapters.freecad.adapter as adapter_module

        mock_app = MagicMock()
        mock_sketch = MagicMock()
        mock_app.ActiveDocument.addObject.return_value = mock_sketch

        with patch.object(adapter_module, 'FREECAD_AVAILABLE', True), \
             patch.object(adapter_module, 'App', mock_app), \
             patch.object(adapter_module, 'Part', MagicMock()), \
             patch.object(adapter_module, 'Sketcher', MagicMock()):

            adapter = adapter_module.FreeCADAdapter()
            adapter.create_sketch("Test")

            with pytest.raises(ExportError) as exc_info:
                adapter.capture_image(800, 600)

            assert "FreeCADGui not available" in str(exc_info.value)


class TestTangentWithConnectionPoint:
    """Tests for tangent constraint with connection point."""

    @pytest.fixture
    def adapter(self):
        """Create mocked adapter."""
        import morphe.adapters.freecad.adapter as adapter_module

        mock_app = MagicMock()
        mock_sketch = MagicMock()
        mock_app.ActiveDocument = MagicMock()
        mock_app.ActiveDocument.addObject.return_value = mock_sketch
        mock_sketch.addGeometry.side_effect = [0, 1]

        with patch.object(adapter_module, 'FREECAD_AVAILABLE', True), \
             patch.object(adapter_module, 'App', mock_app), \
             patch.object(adapter_module, 'Part', MagicMock()), \
             patch.object(adapter_module, 'Sketcher', MagicMock()):

            adapter = adapter_module.FreeCADAdapter()
            adapter.create_sketch("Test")
            yield adapter, mock_sketch

    def test_tangent_at_point(self, adapter):
        adapter_obj, mock_sketch = adapter

        line = Line(start=Point2D(0, 0), end=Point2D(100, 0), id="L0")
        arc = Arc(
            center=Point2D(100, 50),
            start_point=Point2D(100, 0),
            end_point=Point2D(150, 50),
            ccw=True,
            id="A0"
        )
        adapter_obj.add_primitive(line)
        adapter_obj.add_primitive(arc)
        adapter_obj._sketch_doc = SketchDocument(name="Test")
        adapter_obj._sketch_doc.primitives["L0"] = line
        adapter_obj._sketch_doc.primitives["A0"] = arc

        constraint = SketchConstraint(
            id="T0",
            constraint_type=ConstraintType.TANGENT,
            references=["L0", "A0"],
            connection_point=PointRef("A0", PointType.START)
        )
        result = adapter_obj.add_constraint(constraint)

        assert result is True
