#!/usr/bin/env python3
"""Generate the cross-language conformance corpus.

This script builds a fixed set of SketchDocument fixtures using the
authoritative Python implementation and writes them as JSON files. The
same files are consumed by:

    - tests/test_conformance_corpus.py — Python pin check (byte-level
      round-trip stability through the Python encoder).
    - cpp/tests/test_conformance.cpp — C++ semantic round-trip + the
      §13.2 RoundedRect byte-parity check.

Run as:
    python tests/conformance/generate_corpus.py            # regenerate
    python tests/conformance/generate_corpus.py --check    # verify only

The --check mode regenerates each fixture in memory and diffs against
the on-disk version. CI uses --check to catch accidental drift.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Path setup so the script runs from a checkout without `pip install -e .`
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from morphe.constraints import (  # noqa: E402
    ConstraintStatus,
    ConstraintType,
    SketchConstraint,
)
from morphe.document import SketchDocument, SolverStatus  # noqa: E402
from morphe.primitives import (  # noqa: E402
    Arc,
    Circle,
    Ellipse,
    EllipticalArc,
    Line,
    Point,
    Spline,
)
from morphe.serialization import sketch_to_json  # noqa: E402
from morphe.types import Point2D as _Point2D_raw  # noqa: E402
from morphe.types import PointRef, PointType  # noqa: E402


def Point2D(x, y) -> _Point2D_raw:  # noqa: N802 — match upstream casing
    """Coerce coordinates to float so corpus JSON is uniformly typed.

    Python's `Point2D` is annotated `x: float, y: float` but does not
    coerce at construction time, so `Point2D(8, 0)` keeps ints and
    serializes as `[8, 0]` while `Point2D(10.0, 5.5)` serializes as
    `[10.0, 5.5]`. Coercing here keeps the corpus internally consistent
    (always floats) so the C++ implementation, which only has doubles,
    can produce byte-identical output.
    """
    return _Point2D_raw(float(x), float(y))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def fixture_empty() -> SketchDocument:
    """Default-constructed sketch — pins encoder behavior at zero state."""
    return SketchDocument(name="Empty")


def fixture_all_primitives() -> SketchDocument:
    """One of each primitive type, IDs auto-assigned in declaration order."""
    d = SketchDocument(name="AllPrimitives")
    d.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
    d.add_primitive(
        Arc(
            center=Point2D(10, 5),
            start_point=Point2D(10, 0),
            end_point=Point2D(15, 5),
            ccw=True,
        )
    )
    d.add_primitive(Circle(center=Point2D(0, 0), radius=5.0))
    d.add_primitive(Point(position=Point2D(1.5, 2.5)))
    d.add_primitive(
        Spline(
            degree=3,
            control_points=[
                Point2D(0, 0),
                Point2D(1, 1),
                Point2D(2, 0.5),
                Point2D(3, 1.5),
            ],
            knots=[0, 0, 0, 0, 1, 1, 1, 1],
        )
    )
    d.add_primitive(
        Ellipse(
            center=Point2D(5, 5),
            major_radius=3.0,
            minor_radius=1.5,
            rotation=0.5,
        )
    )
    d.add_primitive(
        EllipticalArc(
            center=Point2D(0, 0),
            major_radius=2.0,
            minor_radius=1.0,
            rotation=0.0,
            start_param=0.0,
            end_param=math.pi / 2,
            ccw=True,
        )
    )
    return d


def fixture_rounded_rect() -> SketchDocument:
    """The §13.2 RoundedRect example, with stable constraint IDs.

    The constraint IDs are user-assigned (c1..c6) rather than UUIDs so the
    fixture is byte-stable across regenerations. The shape and constraint
    set match SPECIFICATION.md §13.2.
    """
    d = SketchDocument(name="RoundedRect")
    d.add_primitive(Line(start=Point2D(8, 0), end=Point2D(52, 0)))  # L0
    d.add_primitive(
        Arc(
            center=Point2D(52, 8),
            start_point=Point2D(52, 0),
            end_point=Point2D(60, 8),
            ccw=True,
        )
    )  # A0
    d.add_primitive(Line(start=Point2D(60, 8), end=Point2D(60, 32)))  # L1
    d.add_primitive(Circle(center=Point2D(30, 20), radius=10.0))  # C0

    d.constraints.append(
        SketchConstraint(
            id="c1",
            constraint_type=ConstraintType.TANGENT,
            references=["L0", "A0"],
        )
    )
    d.constraints.append(
        SketchConstraint(
            id="c2",
            constraint_type=ConstraintType.TANGENT,
            references=["A0", "L1"],
        )
    )
    d.constraints.append(
        SketchConstraint(
            id="c3",
            constraint_type=ConstraintType.HORIZONTAL,
            references=["L0"],
        )
    )
    d.constraints.append(
        SketchConstraint(
            id="c4",
            constraint_type=ConstraintType.RADIUS,
            references=["A0"],
            value=8.0,
        )
    )
    d.constraints.append(
        SketchConstraint(
            id="c5",
            constraint_type=ConstraintType.COINCIDENT,
            references=[
                PointRef("L0", PointType.END),
                PointRef("A0", PointType.START),
            ],
        )
    )
    d.solver_status = SolverStatus.FULLY_CONSTRAINED
    d.degrees_of_freedom = 0
    return d


def fixture_spline_with_weights() -> SketchDocument:
    """Rational spline: weights present, encoder should emit the field."""
    d = SketchDocument(name="SplineWithWeights")
    d.add_primitive(
        Spline(
            degree=3,
            control_points=[
                Point2D(0, 0),
                Point2D(1, 1),
                Point2D(2, 0.5),
                Point2D(3, 1.5),
            ],
            knots=[0, 0, 0, 0, 1, 1, 1, 1],
            weights=[1.0, 2.0, 1.5, 1.0],
        )
    )
    return d


def fixture_spline_periodic() -> SketchDocument:
    """Periodic spline: knot-vector arity differs from non-periodic."""
    d = SketchDocument(name="SplinePeriodic")
    # Periodic knot count = n + 1
    d.add_primitive(
        Spline(
            degree=3,
            control_points=[
                Point2D(0, 0),
                Point2D(1, 1),
                Point2D(2, 0),
                Point2D(1, -1),
                Point2D(0, -1),
            ],
            knots=[0, 0.25, 0.5, 0.75, 0.9, 1.0],
            periodic=True,
        )
    )
    return d


def fixture_elliptical_arc_full() -> SketchDocument:
    """Non-trivial elliptical arc: rotation set, ccw=false, custom params."""
    d = SketchDocument(name="EllipticalArcFull")
    d.add_primitive(
        EllipticalArc(
            center=Point2D(2.5, 4.0),
            major_radius=5.0,
            minor_radius=2.0,
            rotation=math.pi / 6,  # 30 degrees
            start_param=0.25,
            end_param=2.5,
            ccw=False,
        )
    )
    return d


def fixture_inferred_constraints() -> SketchDocument:
    """Constraints with all the rare metadata: inferred, source, confidence, status."""
    d = SketchDocument(name="InferredConstraints")
    d.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))
    d.add_primitive(Line(start=Point2D(0, 5), end=Point2D(10, 5)))

    d.constraints.append(
        SketchConstraint(
            id="c-ai-1",
            constraint_type=ConstraintType.PARALLEL,
            references=["L0", "L1"],
            inferred=True,
            confidence=0.85,
            source="ai",
            status=ConstraintStatus.SATISFIED,
        )
    )
    d.constraints.append(
        SketchConstraint(
            id="c-tan",
            constraint_type=ConstraintType.HORIZONTAL,
            references=["L0"],
            connection_point=PointRef("L0", PointType.END),
            source="user",
        )
    )
    return d


def fixture_construction_geometry() -> SketchDocument:
    """Construction (reference) geometry mixed with profile geometry."""
    d = SketchDocument(name="ConstructionGeometry")
    profile = Line(start=Point2D(0, 0), end=Point2D(10, 0))
    d.add_primitive(profile)

    centerline = Line(
        start=Point2D(5, -1),
        end=Point2D(5, 11),
        construction=True,
    )
    d.add_primitive(centerline)

    construction_arc = Arc(
        center=Point2D(5, 5),
        start_point=Point2D(5, 0),
        end_point=Point2D(10, 5),
        ccw=True,
        construction=True,
    )
    d.add_primitive(construction_arc)
    return d


def fixture_point_refs_full() -> SketchDocument:
    """Exercise the PointType values that are actually valid on some primitive.

    Note: PointType.ON_CURVE is defined in the enum but is NOT in any
    primitive's get_valid_point_types() list — including Spline. So a
    PointRef with point_type=ON_CURVE will fail validation against any
    primitive in the current schema. This fixture intentionally omits
    ON_CURVE to keep the corpus validation-clean. ON_CURVE round-trip
    through (de)serialization is exercised separately by the C++ unit
    tests in test_serialization.cpp; this fixture is the validator's
    coverage check.
    """
    d = SketchDocument(name="PointRefsFull")
    d.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))  # L0
    d.add_primitive(
        Arc(
            center=Point2D(10, 5),
            start_point=Point2D(10, 0),
            end_point=Point2D(15, 5),
            ccw=True,
        )
    )  # A0
    d.add_primitive(Circle(center=Point2D(20, 5), radius=2.0))  # C0
    d.add_primitive(
        Spline(
            degree=3,
            control_points=[
                Point2D(0, 0),
                Point2D(1, 1),
                Point2D(2, 0.5),
                Point2D(3, 1.5),
            ],
            knots=[0, 0, 0, 0, 1, 1, 1, 1],
        )
    )  # S0
    d.add_primitive(Point(position=Point2D(50, 50)))  # P0

    # Coincidences exercising START, END, CENTER, MIDPOINT, CONTROL — the
    # five PointTypes that are valid on some primitive. ON_CURVE is omitted
    # (see docstring).
    refs = [
        ("start", PointRef("L0", PointType.START)),
        ("end", PointRef("L0", PointType.END)),
        ("center", PointRef("A0", PointType.CENTER)),
        ("midpoint", PointRef("L0", PointType.MIDPOINT)),
        ("control", PointRef("S0", PointType.CONTROL, index=2)),
    ]
    for name, r in refs:
        d.constraints.append(
            SketchConstraint(
                id=f"c-{name}",
                constraint_type=ConstraintType.COINCIDENT,
                references=[r, PointRef("P0", PointType.CENTER)],
            )
        )
    return d


def fixture_all_constraint_types() -> SketchDocument:
    """One constraint of every ConstraintType (20 in total).

    Some constraint types require specific primitive shapes for their
    references to make sense; we allocate enough primitives to satisfy
    every constraint's reference rules from CONSTRAINT_RULES.
    """
    d = SketchDocument(name="AllConstraintTypes")
    # Lines for orientation / line constraints
    d.add_primitive(Line(start=Point2D(0, 0), end=Point2D(10, 0)))  # L0
    d.add_primitive(Line(start=Point2D(0, 5), end=Point2D(10, 5)))  # L1
    d.add_primitive(Line(start=Point2D(2, -2), end=Point2D(2, 7)))  # L2 (axis)
    # Arcs / circles for arc/circle constraints. The arc's center must be
    # equidistant from start_point and end_point (validation requires this).
    d.add_primitive(
        Arc(
            center=Point2D(10, 5),
            start_point=Point2D(10, 0),
            end_point=Point2D(15, 5),
            ccw=True,
        )
    )  # A0
    d.add_primitive(Circle(center=Point2D(30, 0), radius=4.0))  # C0
    # A point for midpoint / coincident
    d.add_primitive(Point(position=Point2D(5, 0)))  # P0
    d.add_primitive(Point(position=Point2D(7, 0)))  # P1

    p0 = PointRef("P0", PointType.CENTER)
    p1 = PointRef("P1", PointType.CENTER)
    l0_end = PointRef("L0", PointType.END)
    a0_start = PointRef("A0", PointType.START)

    d.constraints.append(
        SketchConstraint(id="c-coincident", constraint_type=ConstraintType.COINCIDENT,
                         references=[l0_end, a0_start])
    )
    d.constraints.append(
        SketchConstraint(id="c-tangent", constraint_type=ConstraintType.TANGENT,
                         references=["L0", "A0"])
    )
    d.constraints.append(
        SketchConstraint(id="c-perp", constraint_type=ConstraintType.PERPENDICULAR,
                         references=["L0", "L2"])
    )
    d.constraints.append(
        SketchConstraint(id="c-parallel", constraint_type=ConstraintType.PARALLEL,
                         references=["L0", "L1"])
    )
    d.constraints.append(
        SketchConstraint(id="c-concentric", constraint_type=ConstraintType.CONCENTRIC,
                         references=["A0", "C0"])
    )
    d.constraints.append(
        SketchConstraint(id="c-equal", constraint_type=ConstraintType.EQUAL,
                         references=["L0", "L1"])
    )
    d.constraints.append(
        SketchConstraint(id="c-collinear", constraint_type=ConstraintType.COLLINEAR,
                         references=["L0", "L1"])
    )
    d.constraints.append(
        SketchConstraint(id="c-horizontal", constraint_type=ConstraintType.HORIZONTAL,
                         references=["L0"])
    )
    d.constraints.append(
        SketchConstraint(id="c-vertical", constraint_type=ConstraintType.VERTICAL,
                         references=["L2"])
    )
    d.constraints.append(
        SketchConstraint(id="c-fixed", constraint_type=ConstraintType.FIXED,
                         references=["P0"])
    )
    d.constraints.append(
        SketchConstraint(id="c-distance", constraint_type=ConstraintType.DISTANCE,
                         references=[p0, p1], value=2.0)
    )
    d.constraints.append(
        SketchConstraint(id="c-distx", constraint_type=ConstraintType.DISTANCE_X,
                         references=[p0, p1], value=2.0)
    )
    d.constraints.append(
        SketchConstraint(id="c-disty", constraint_type=ConstraintType.DISTANCE_Y,
                         references=[p0, p1], value=0.0)
    )
    d.constraints.append(
        SketchConstraint(id="c-length", constraint_type=ConstraintType.LENGTH,
                         references=["L0"], value=10.0)
    )
    d.constraints.append(
        SketchConstraint(id="c-radius", constraint_type=ConstraintType.RADIUS,
                         references=["C0"], value=4.0)
    )
    d.constraints.append(
        SketchConstraint(id="c-diameter", constraint_type=ConstraintType.DIAMETER,
                         references=["C0"], value=8.0)
    )
    d.constraints.append(
        SketchConstraint(id="c-angle", constraint_type=ConstraintType.ANGLE,
                         references=["L0", "L2"], value=90.0)
    )
    d.constraints.append(
        SketchConstraint(id="c-symmetric", constraint_type=ConstraintType.SYMMETRIC,
                         references=["L0", "L1", "L2"])
    )
    d.constraints.append(
        SketchConstraint(id="c-midpoint", constraint_type=ConstraintType.MIDPOINT,
                         references=[p0, "L0"])
    )
    return d


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

# Order matters: corpus README and the C++ test enumerate fixtures in this list.
FIXTURES: list[tuple[str, callable]] = [
    ("empty", fixture_empty),
    ("all_primitives", fixture_all_primitives),
    ("rounded_rect", fixture_rounded_rect),
    ("spline_with_weights", fixture_spline_with_weights),
    ("spline_periodic", fixture_spline_periodic),
    ("elliptical_arc_full", fixture_elliptical_arc_full),
    ("inferred_constraints", fixture_inferred_constraints),
    ("construction_geometry", fixture_construction_geometry),
    ("point_refs_full", fixture_point_refs_full),
    ("all_constraint_types", fixture_all_constraint_types),
]


def render(name: str, builder) -> str:
    sketch = builder()
    return sketch_to_json(sketch, indent=2) + "\n"


def coverage_audit() -> None:
    """Confirm the corpus exercises every primitive type and every ConstraintType."""
    seen_prim_types: set[str] = set()
    seen_constraint_types: set[ConstraintType] = set()
    for _, builder in FIXTURES:
        sketch = builder()
        for p in sketch.primitives.values():
            seen_prim_types.add(type(p).__name__.lower())
        for c in sketch.constraints:
            seen_constraint_types.add(c.constraint_type)

    expected_prims = {"line", "arc", "circle", "point", "spline", "ellipse", "ellipticalarc"}
    missing_prims = expected_prims - seen_prim_types
    if missing_prims:
        raise AssertionError(f"Corpus missing primitive types: {sorted(missing_prims)}")

    missing_cts = set(ConstraintType) - seen_constraint_types
    if missing_cts:
        names = sorted(t.value for t in missing_cts)
        raise AssertionError(f"Corpus missing constraint types: {names}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify on-disk fixtures match what would be generated; exit 1 on drift.",
    )
    args = parser.parse_args()

    coverage_audit()

    drift_count = 0
    for name, builder in FIXTURES:
        path = HERE / f"{name}.json"
        new_text = render(name, builder)
        if args.check:
            if not path.exists():
                print(f"[DRIFT] missing: {path.name}")
                drift_count += 1
                continue
            old_text = path.read_text()
            if old_text != new_text:
                print(f"[DRIFT] {path.name} differs from generator output")
                drift_count += 1
            else:
                print(f"[OK] {path.name}")
        else:
            path.write_text(new_text)
            print(f"wrote {path}")

    if args.check:
        if drift_count:
            print(f"\n{drift_count} fixture(s) drifted. Run without --check to regenerate.",
                  file=sys.stderr)
            return 1
        print(f"\n{len(FIXTURES)} fixtures verified, no drift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
