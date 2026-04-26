# Morphe C++17 Reference Implementation

A standalone C++17 reference implementation of the Morphe sketch schema:
data structures, JSON (de)serialization, and validation. Mirrors the
Python implementation in `../morphe/` but does not depend on it.

The two implementations agree on the wire format documented in
`../SPECIFICATION.md` §13 and pinned by the conformance corpus under
`../tests/conformance/`.

## Building

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Requires CMake 3.16+, a C++17 compiler (gcc 9+, clang 9+, MSVC 2019+).

## Consuming from another CMake project

```cmake
add_subdirectory(path/to/morphe/cpp)
target_link_libraries(your_target PRIVATE morphe::morphe)
```

`morphe` is a header-only `INTERFACE` library at this stage; linking it
brings in the include path for `<morphe/...>` headers and the vendored
`<nlohmann/json.hpp>`.

## Vendored dependencies

- `third_party/nlohmann/json.hpp` — [nlohmann/json](https://github.com/nlohmann/json) v3.11.3 (MIT)
- `third_party/doctest/doctest.h` — [doctest/doctest](https://github.com/doctest/doctest) v2.4.11 (MIT, tests only)

Both are checked in as single headers so the project builds offline with
no CMake `FetchContent` dance.

## Scope

The reference implementation covers:

- All core types (`Point2D`, `Vector2D`, `PointType`, `PointRef`).
- All seven primitive types (`Line`, `Arc`, `Circle`, `Point`, `Spline`,
  `Ellipse`, `EllipticalArc`) modeled as a `std::variant`.
- All 19 `ConstraintType` values plus the `CONSTRAINT_RULES` table.
- `SketchDocument` with insertion-ordered primitives.
- JSON (de)serialization matching the Python encoder byte-for-byte on
  shared corpus fixtures.
- Schema validation with the same severity-tagged issue codes as the
  Python validator (e.g., `LINE_ZERO_LENGTH`, `ARC_RADIUS_INCONSISTENT`,
  `CONST_TOO_FEW_REFS`), and the same 1-micron default tolerance.

Out of scope (matches the Python core): a constraint solver, CAD
adapters (FreeCAD / SolidWorks / Inventor / Fusion), or computational
helpers (B-spline evaluation, ellipse parametric, etc.). These belong
above the schema layer.

## Cross-language conformance

The shared corpus under `../tests/conformance/*.json` is the wire-format
contract. It is generated from the Python implementation
(`tests/conformance/generate_corpus.py`) and consumed by both:

- Python: `tests/test_conformance_corpus.py` checks that
  `save_sketch(load_sketch(f)) == f` byte-for-byte.
- C++: `tests/test_conformance.cpp` loads each fixture, round-trips
  through `morphe::to_json`, and compares parsed JSON trees with float
  tolerance. The `rounded_rect.json` fixture is also rebuilt in C++ and
  semantically compared, pinning the exact wire format.

CI (`.github/workflows/cpp.yml`) runs both checks on Ubuntu (gcc-13,
clang-18), macOS, and Windows.

## Decoder leniency

The C++ decoder accepts both the omit-when-default form (the
canonical wire format described in `SPECIFICATION.md` §13.1.1) and
the verbose form with explicit defaults (`"value": null,
"inferred": false, "confidence": 1.0`). Documents from older or
third-party encoders that emit defaults explicitly therefore load
correctly. Round-tripping through `morphe::to_json` normalizes them
to the minimal form.
