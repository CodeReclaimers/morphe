# Cross-language conformance corpus

This directory holds canonical JSON fixtures consumed by both the Python
and C++ Morphe implementations. They pin the wire format: any change to
either implementation that affects the JSON output of one of these
documents will fail CI.

## Fixtures

| File | Coverage |
|---|---|
| `empty.json` | Default-constructed `SketchDocument` — pins encoder behavior at zero state. |
| `all_primitives.json` | One of each primitive: Line, Arc, Circle, Point, Spline, Ellipse, EllipticalArc. |
| `rounded_rect.json` | The §13.2 RoundedRect example, byte-stable (constraint IDs `c1..c5`). C++ tests construct it programmatically and check parity. |
| `spline_with_weights.json` | Rational spline — exercises the optional `weights` field. |
| `spline_periodic.json` | Periodic spline — exercises the periodic knot-vector arity. |
| `elliptical_arc_full.json` | Non-trivial elliptical arc with rotation, ccw=false, custom params. |
| `inferred_constraints.json` | Constraints with `inferred=true`, `confidence!=1.0`, `source` set, `connection_point` set, `status=satisfied`. |
| `construction_geometry.json` | Mix of profile and `construction=true` reference geometry. |
| `point_refs_full.json` | Coincidence constraints exercising START, END, CENTER, MIDPOINT, CONTROL PointTypes. |
| `all_constraint_types.json` | One constraint of every `ConstraintType` enum value (19 in total). |

## Regenerating

```bash
python tests/conformance/generate_corpus.py
```

This rewrites every `*.json` file from the Python fixture builders.

## Verifying (CI)

```bash
python tests/conformance/generate_corpus.py --check
```

Returns exit code 1 if any on-disk file diverges from what the generator
would produce now.

## Consumers

- **Python**: `tests/test_conformance_corpus.py` runs three checks per
  fixture: parses cleanly, is a fixed point of `save_sketch(load_sketch(f))`
  byte-for-byte, and is valid JSON.
- **C++**: `cpp/tests/test_conformance.cpp` loads every fixture via
  `morphe::load`, round-trips it through `morphe::to_json`, and verifies
  semantic equivalence (parsed-tree compare with float tolerance). It
  also re-builds `rounded_rect.json` programmatically in C++ and
  semantically compares against the on-disk file — the strongest
  cross-language parity claim.

## Notes on the schema

- **`PointType.ON_CURVE` is reserved but not currently supported by
  any primitive** — see `SPECIFICATION.md` §3.1 ("ON_CURVE status").
  The corpus intentionally does not exercise it: a `PointRef` with
  `point_type=ON_CURVE` fails validation against every primitive in
  the current schema. Round-trip through (de)serialization is
  exercised separately by the C++ unit tests in
  `cpp/tests/test_serialization.cpp`.
