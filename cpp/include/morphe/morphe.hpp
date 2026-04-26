#pragma once

// Morphe C++17 reference implementation.
//
// Public umbrella header. The implementation is standalone: it does not link
// against, embed, or communicate with the Python `morphe` package. The two
// implementations agree only on the JSON wire format described in
// SPECIFICATION.md §13 and pinned by the conformance corpus under
// tests/conformance/.

#include "morphe/constraints.hpp"
#include "morphe/document.hpp"
#include "morphe/primitives.hpp"
#include "morphe/types.hpp"

namespace morphe {

inline constexpr int version_major = 0;
inline constexpr int version_minor = 1;
inline constexpr int version_patch = 0;

}  // namespace morphe
