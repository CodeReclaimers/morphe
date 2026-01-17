"""Abstract adapter interface for CAD platform backends."""

from abc import ABC, abstractmethod
from typing import Any

from .constraints import SketchConstraint
from .document import SketchDocument, SolverStatus
from .primitives import SketchPrimitive


class SketchBackendAdapter(ABC):
    """
    Abstract interface for CAD platform adapters.

    Each adapter translates between the canonical sketch representation
    and a specific CAD platform's native API (FreeCAD, SolidWorks,
    Inventor, Fusion 360, etc.).

    Adapters handle:
    - Unit conversion (canonical mm to platform units)
    - Geometry translation (canonical primitives to platform entities)
    - Constraint translation (canonical constraints to platform constraints)
    - ID mapping (canonical IDs to platform-specific references)
    """

    @abstractmethod
    def create_sketch(self, name: str, plane: Any | None = None) -> None:
        """
        Create a new empty sketch.

        Args:
            name: Sketch name
            plane: Platform-specific plane/face reference.
                   If None, uses the default XY plane.
        """
        pass

    @abstractmethod
    def load_sketch(self, sketch: SketchDocument) -> None:
        """
        Load a canonical sketch into the CAD system.

        This creates all geometry and constraints from the canonical
        representation in the current sketch.

        Args:
            sketch: The canonical sketch document to load
        """
        pass

    @abstractmethod
    def export_sketch(self) -> SketchDocument:
        """
        Export the current CAD sketch to canonical form.

        Returns:
            A new SketchDocument containing the canonical representation
            of the current sketch.
        """
        pass

    @abstractmethod
    def add_primitive(self, primitive: SketchPrimitive) -> Any:
        """
        Add a single primitive to the sketch.

        Args:
            primitive: The canonical primitive to add

        Returns:
            Platform-specific entity reference that can be used for
            chaining or constraint application.
        """
        pass

    @abstractmethod
    def add_constraint(self, constraint: SketchConstraint) -> bool:
        """
        Add a constraint to the sketch.

        Args:
            constraint: The canonical constraint to add

        Returns:
            True if the constraint was successfully added, False otherwise.
        """
        pass

    @abstractmethod
    def get_solver_status(self) -> tuple[SolverStatus, int]:
        """
        Get the current solver status.

        Returns:
            Tuple of (SolverStatus, degrees_of_freedom).
            degrees_of_freedom is -1 if not computable.
        """
        pass

    @abstractmethod
    def capture_image(self, width: int, height: int) -> bytes:
        """
        Capture a visualization of the sketch.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            PNG image data as bytes
        """
        pass

    def close_sketch(self) -> None:
        """
        Close/finalize the current sketch.

        Override if the platform requires explicit sketch closing.
        Default implementation does nothing.
        """
        pass

    def get_element_by_id(self, element_id: str) -> Any | None:
        """
        Get the platform-specific entity for a canonical element ID.

        Args:
            element_id: Canonical element ID (e.g., "L0", "A1")

        Returns:
            Platform-specific entity, or None if not found.
        """
        return None

    def supports_feature(self, feature: str) -> bool:
        """
        Check if the adapter supports a specific feature.

        Common features:
        - "spline": B-spline/NURBS curves
        - "three_point_arc": Arc construction via three points
        - "image_capture": Screenshot capability
        - "solver_status": Constraint solver status reporting

        Args:
            feature: Feature name to check

        Returns:
            True if the feature is supported.
        """
        return False


class AdapterError(Exception):
    """Base exception for adapter errors."""
    pass


class SketchCreationError(AdapterError):
    """Error during sketch creation."""
    pass


class GeometryError(AdapterError):
    """Error adding geometry to the sketch."""
    pass


class ConstraintError(AdapterError):
    """Error adding a constraint to the sketch."""
    pass


class ExportError(AdapterError):
    """Error exporting the sketch."""
    pass
