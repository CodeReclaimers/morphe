"""Validation logic for the canonical sketch schema."""

from dataclasses import dataclass
from enum import Enum

from .constraints import CONSTRAINT_RULES, ConstraintType, SketchConstraint
from .document import SketchDocument
from .primitives import Arc, Circle, Line, Point, SketchPrimitive, Spline
from .types import PointRef, PointType


class ValidationSeverity(Enum):
    """Severity level for validation issues."""
    ERROR = "error"      # Invalid, will likely fail
    WARNING = "warning"  # Suspicious, may cause issues
    INFO = "info"        # Informational note


@dataclass
class ValidationIssue:
    """A single validation issue."""
    severity: ValidationSeverity
    element_id: str | None  # ID of the affected element, if applicable
    message: str
    code: str  # Machine-readable error code

    def __str__(self) -> str:
        prefix = f"[{self.element_id}] " if self.element_id else ""
        return f"{self.severity.value.upper()}: {prefix}{self.message}"


class ValidationResult:
    """Collection of validation issues."""

    def __init__(self):
        self.issues: list[ValidationIssue] = []

    def add_error(self, message: str, code: str, element_id: str | None = None):
        """Add an error-level issue."""
        self.issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            element_id=element_id,
            message=message,
            code=code
        ))

    def add_warning(self, message: str, code: str, element_id: str | None = None):
        """Add a warning-level issue."""
        self.issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            element_id=element_id,
            message=message,
            code=code
        ))

    def add_info(self, message: str, code: str, element_id: str | None = None):
        """Add an info-level issue."""
        self.issues.append(ValidationIssue(
            severity=ValidationSeverity.INFO,
            element_id=element_id,
            message=message,
            code=code
        ))

    @property
    def is_valid(self) -> bool:
        """Check if there are no errors."""
        return not any(i.severity == ValidationSeverity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(i.severity == ValidationSeverity.WARNING for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get all error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get all warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def __str__(self) -> str:
        if not self.issues:
            return "Validation passed (no issues)"
        lines = [f"Validation found {len(self.issues)} issue(s):"]
        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)

    def __bool__(self) -> bool:
        return self.is_valid


# Default tolerances
DEFAULT_TOLERANCE = 0.001  # 1 micron


def validate_sketch(sketch: SketchDocument, tolerance: float = DEFAULT_TOLERANCE) -> ValidationResult:
    """
    Validate a sketch for schema correctness.

    This performs:
    - Primitive validation (geometry consistency)
    - Constraint validation (reference validity, type checking)
    - Cross-reference validation (all referenced elements exist)

    Args:
        sketch: The sketch document to validate
        tolerance: Geometric tolerance for consistency checks (mm)

    Returns:
        ValidationResult containing all issues found
    """
    result = ValidationResult()

    # Validate primitives
    for prim in sketch.primitives.values():
        _validate_primitive(prim, result, tolerance)

    # Validate constraints
    for constraint in sketch.constraints:
        _validate_constraint(constraint, sketch, result)

    # Check for duplicate IDs (should not happen with proper add_primitive usage)
    _check_duplicate_ids(sketch, result)

    return result


def _validate_primitive(prim: SketchPrimitive, result: ValidationResult,
                        tolerance: float) -> None:
    """Validate a single primitive."""

    # Check for empty ID
    if not prim.id:
        result.add_error("Primitive has empty ID", "PRIM_EMPTY_ID", prim.id)

    # Check confidence range
    if not (0.0 <= prim.confidence <= 1.0):
        result.add_warning(
            f"Confidence {prim.confidence} outside [0,1] range",
            "PRIM_CONFIDENCE_RANGE",
            prim.id
        )

    # Type-specific validation
    if isinstance(prim, Line):
        _validate_line(prim, result, tolerance)
    elif isinstance(prim, Arc):
        _validate_arc(prim, result, tolerance)
    elif isinstance(prim, Circle):
        _validate_circle(prim, result, tolerance)
    elif isinstance(prim, Point):
        _validate_point(prim, result)
    elif isinstance(prim, Spline):
        _validate_spline(prim, result, tolerance)


def _validate_line(line: Line, result: ValidationResult, tolerance: float) -> None:
    """Validate a line primitive."""

    # Check for zero-length line
    if line.length < tolerance:
        result.add_error(
            f"Line has zero length (length={line.length:.6f})",
            "LINE_ZERO_LENGTH",
            line.id
        )

    # Check for NaN/inf coordinates
    for coord in [line.start.x, line.start.y, line.end.x, line.end.y]:
        if not _is_finite(coord):
            result.add_error(
                "Line has non-finite coordinates",
                "LINE_INVALID_COORDS",
                line.id
            )
            break


def _validate_arc(arc: Arc, result: ValidationResult, tolerance: float) -> None:
    """Validate an arc primitive."""

    # Check for NaN/inf coordinates
    coords = [
        arc.center.x, arc.center.y,
        arc.start_point.x, arc.start_point.y,
        arc.end_point.x, arc.end_point.y
    ]
    for coord in coords:
        if not _is_finite(coord):
            result.add_error(
                "Arc has non-finite coordinates",
                "ARC_INVALID_COORDS",
                arc.id
            )
            return

    # Check radius consistency (start and end should be same distance from center)
    r_start = arc.center.distance_to(arc.start_point)
    r_end = arc.center.distance_to(arc.end_point)

    if abs(r_start - r_end) > tolerance:
        result.add_error(
            f"Arc radius inconsistent (start={r_start:.4f}, end={r_end:.4f})",
            "ARC_RADIUS_INCONSISTENT",
            arc.id
        )

    # Check for zero radius
    if r_start < tolerance:
        result.add_error(
            "Arc has zero radius",
            "ARC_ZERO_RADIUS",
            arc.id
        )

    # Check for degenerate arc (start == end, which should be a circle)
    if arc.start_point.distance_to(arc.end_point) < tolerance:
        result.add_warning(
            "Arc start and end points are coincident (consider using Circle instead)",
            "ARC_DEGENERATE",
            arc.id
        )


def _validate_circle(circle: Circle, result: ValidationResult, tolerance: float) -> None:
    """Validate a circle primitive."""

    # Check for NaN/inf coordinates
    for coord in [circle.center.x, circle.center.y, circle.radius]:
        if not _is_finite(coord):
            result.add_error(
                "Circle has non-finite values",
                "CIRCLE_INVALID_VALUES",
                circle.id
            )
            return

    # Check for non-positive radius
    if circle.radius <= 0:
        result.add_error(
            f"Circle has non-positive radius ({circle.radius})",
            "CIRCLE_INVALID_RADIUS",
            circle.id
        )
    elif circle.radius < tolerance:
        result.add_warning(
            f"Circle has very small radius ({circle.radius})",
            "CIRCLE_TINY_RADIUS",
            circle.id
        )


def _validate_point(point: Point, result: ValidationResult) -> None:
    """Validate a point primitive."""

    # Check for NaN/inf coordinates
    for coord in [point.position.x, point.position.y]:
        if not _is_finite(coord):
            result.add_error(
                "Point has non-finite coordinates",
                "POINT_INVALID_COORDS",
                point.id
            )
            break


def _validate_spline(spline: Spline, result: ValidationResult, tolerance: float) -> None:
    """Validate a spline primitive."""

    # Check degree
    if spline.degree < 1:
        result.add_error(
            f"Spline degree must be at least 1 (got {spline.degree})",
            "SPLINE_INVALID_DEGREE",
            spline.id
        )

    # Check minimum control points for degree
    min_points = spline.degree + 1
    if len(spline.control_points) < min_points:
        result.add_error(
            f"Spline needs at least {min_points} control points for degree {spline.degree} "
            f"(got {len(spline.control_points)})",
            "SPLINE_INSUFFICIENT_POINTS",
            spline.id
        )

    # Check control point coordinates
    for i, cp in enumerate(spline.control_points):
        if not _is_finite(cp.x) or not _is_finite(cp.y):
            result.add_error(
                f"Spline control point {i} has non-finite coordinates",
                "SPLINE_INVALID_CONTROL_POINT",
                spline.id
            )

    # Check knot vector
    if spline.knots:
        # Knots should be non-decreasing
        for i in range(1, len(spline.knots)):
            if spline.knots[i] < spline.knots[i-1]:
                result.add_error(
                    f"Spline knot vector is not non-decreasing at index {i}",
                    "SPLINE_INVALID_KNOTS",
                    spline.id
                )
                break

        # Validate knot vector length
        if not spline.validate_knot_vector():
            expected = len(spline.control_points) + spline.order
            result.add_warning(
                f"Spline knot vector length ({len(spline.knots)}) doesn't match expected ({expected})",
                "SPLINE_KNOT_LENGTH_MISMATCH",
                spline.id
            )

    # Check weights if present
    if spline.weights is not None:
        if len(spline.weights) != len(spline.control_points):
            result.add_error(
                f"Spline weights count ({len(spline.weights)}) doesn't match "
                f"control points ({len(spline.control_points)})",
                "SPLINE_WEIGHT_COUNT_MISMATCH",
                spline.id
            )

        for i, w in enumerate(spline.weights):
            if w <= 0:
                result.add_error(
                    f"Spline weight {i} is non-positive ({w})",
                    "SPLINE_INVALID_WEIGHT",
                    spline.id
                )
            elif not _is_finite(w):
                result.add_error(
                    f"Spline weight {i} is non-finite",
                    "SPLINE_INVALID_WEIGHT",
                    spline.id
                )


def _validate_constraint(constraint: SketchConstraint, sketch: SketchDocument,
                          result: ValidationResult) -> None:
    """Validate a single constraint."""

    # Check for empty ID
    if not constraint.id:
        result.add_warning("Constraint has empty ID", "CONST_EMPTY_ID")

    # Get constraint rules
    rules = CONSTRAINT_RULES.get(constraint.constraint_type)
    if rules is None:
        result.add_error(
            f"Unknown constraint type: {constraint.constraint_type}",
            "CONST_UNKNOWN_TYPE"
        )
        return

    # Check reference count
    ref_count = len(constraint.references)
    if ref_count < rules["min_refs"]:
        result.add_error(
            f"{constraint.constraint_type.value}: Too few references "
            f"(need {rules['min_refs']}, got {ref_count})",
            "CONST_TOO_FEW_REFS"
        )
    if rules["max_refs"] is not None and ref_count > rules["max_refs"]:
        result.add_error(
            f"{constraint.constraint_type.value}: Too many references "
            f"(max {rules['max_refs']}, got {ref_count})",
            "CONST_TOO_MANY_REFS"
        )

    # Check value requirement
    if rules["value_required"] and constraint.value is None:
        result.add_error(
            f"{constraint.constraint_type.value}: Missing required value",
            "CONST_MISSING_VALUE"
        )

    # Check that value is finite if present
    if constraint.value is not None and not _is_finite(constraint.value):
        result.add_error(
            f"{constraint.constraint_type.value}: Value is non-finite",
            "CONST_INVALID_VALUE"
        )

    # Check dimensional constraint values
    if constraint.value is not None:
        if constraint.constraint_type in (ConstraintType.LENGTH, ConstraintType.RADIUS,
                                           ConstraintType.DIAMETER, ConstraintType.DISTANCE):
            if constraint.value < 0:
                result.add_error(
                    f"{constraint.constraint_type.value}: Value must be non-negative "
                    f"(got {constraint.value})",
                    "CONST_NEGATIVE_VALUE"
                )

    # Check that all referenced elements exist
    for ref in constraint.references:
        if isinstance(ref, str):
            if ref not in sketch.primitives:
                result.add_error(
                    f"Constraint references non-existent element '{ref}'",
                    "CONST_INVALID_REF"
                )
        elif isinstance(ref, PointRef):
            prim = sketch.primitives.get(ref.element_id)
            if prim is None:
                result.add_error(
                    f"Constraint references non-existent element '{ref.element_id}'",
                    "CONST_INVALID_REF"
                )
            else:
                # Check that the point type is valid for this primitive
                if ref.point_type not in prim.get_valid_point_types():
                    result.add_error(
                        f"Invalid point type {ref.point_type.value} for {ref.element_id} "
                        f"({type(prim).__name__})",
                        "CONST_INVALID_POINT_TYPE"
                    )

                # Check index for CONTROL point type
                if ref.point_type == PointType.CONTROL:
                    if ref.index is None:
                        result.add_error(
                            f"CONTROL point type requires index for {ref.element_id}",
                            "CONST_MISSING_INDEX"
                        )
                    elif isinstance(prim, Spline):
                        if ref.index < 0 or ref.index >= len(prim.control_points):
                            result.add_error(
                                f"Control point index {ref.index} out of range for {ref.element_id}",
                                "CONST_INDEX_OUT_OF_RANGE"
                            )

    # Check connection_point if present
    if constraint.connection_point is not None:
        cp = constraint.connection_point
        prim = sketch.primitives.get(cp.element_id)
        if prim is None:
            result.add_error(
                f"Connection point references non-existent element '{cp.element_id}'",
                "CONST_INVALID_CONNECTION_POINT"
            )
        elif cp.point_type not in prim.get_valid_point_types():
            result.add_error(
                f"Invalid connection point type {cp.point_type.value} for {cp.element_id}",
                "CONST_INVALID_CONNECTION_POINT_TYPE"
            )

    # Check confidence range
    if not (0.0 <= constraint.confidence <= 1.0):
        result.add_warning(
            f"Constraint confidence {constraint.confidence} outside [0,1] range",
            "CONST_CONFIDENCE_RANGE"
        )


def _check_duplicate_ids(sketch: SketchDocument, result: ValidationResult) -> None:
    """Check for duplicate element IDs (should not happen normally)."""
    # The dict structure prevents true duplicates, but check constraint IDs
    seen_constraint_ids = set()
    for c in sketch.constraints:
        if c.id in seen_constraint_ids:
            result.add_warning(
                f"Duplicate constraint ID: {c.id}",
                "CONST_DUPLICATE_ID"
            )
        seen_constraint_ids.add(c.id)


def _is_finite(value: float) -> bool:
    """Check if a value is finite (not NaN or infinity)."""
    import math
    return math.isfinite(value)


def validate_primitive(prim: SketchPrimitive, tolerance: float = DEFAULT_TOLERANCE) -> list[str]:
    """
    Validate a single primitive.

    Returns a list of error messages (empty if valid).
    This is a convenience function for simple validation.
    """
    result = ValidationResult()
    _validate_primitive(prim, result, tolerance)
    return [str(issue) for issue in result.errors]


def validate_constraint(constraint: SketchConstraint, sketch: SketchDocument) -> list[str]:
    """
    Validate a single constraint.

    Returns a list of error messages (empty if valid).
    This is a convenience function for simple validation.
    """
    result = ValidationResult()
    _validate_constraint(constraint, sketch, result)
    return [str(issue) for issue in result.errors]
