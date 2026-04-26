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

## Status

| Phase | Scope | Status |
|---|---|---|
| 1 | Skeleton, CMake, vendored deps, smoke test | in progress |
| 2 | Core types, primitives, document | pending |
| 3 | JSON (de)serialization | pending |
| 4 | Validation port | pending |
| 5 | Cross-language conformance corpus + CI | pending |

See `/home/alan/.claude/plans/i-m-thinking-about-adding-bubbly-wave.md`
for the full design (until the plan is migrated into this repo).
