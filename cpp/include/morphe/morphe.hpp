#pragma once

// Morphe C++17 reference implementation.
//
// This header is the public umbrella. Subsequent phases populate it with
// types, primitives, constraints, document, serialization, and validation.
//
// The implementation is standalone: it does not link against, embed, or
// communicate with the Python `morphe` package. The two implementations
// agree only on the JSON wire format described in SPECIFICATION.md §13
// and in the conformance corpus under tests/conformance/.

namespace morphe {

inline constexpr int version_major = 0;
inline constexpr int version_minor = 1;
inline constexpr int version_patch = 0;

}  // namespace morphe
