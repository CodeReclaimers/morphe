#pragma once

#include <filesystem>
#include <string>
#include <string_view>

#include "morphe/document.hpp"

namespace morphe {

// Serialize a sketch document to a JSON string.
//
// Encoding policy (matches morphe/serialization.py):
//   - Top-level key order: name, primitives, constraints, solver_status, degrees_of_freedom
//   - Primitive type tag is lowercase ("line", "arc", ..., "ellipticalarc" — one word)
//   - Point2D encoded as a 2-element array [x, y]
//   - PointRef encoded as {element, point, [parameter], [index]}
//   - Optional fields are omitted at default values, NOT emitted as null:
//     * Primitive: source omitted when null; confidence omitted when 1.0
//     * Constraint: value, connection_point, source omitted when null;
//       inferred omitted when false; confidence omitted when 1.0; status
//       omitted when Unknown
//     * Spline: weights omitted when null
//     * PointRef: parameter, index each omitted when null
//
// `indent` is passed to nlohmann::json::dump (use -1 for compact output).
std::string to_json(const SketchDocument& doc, int indent = 2);

// Parse a sketch document from a JSON string.
//
// Decoder policy:
//   - Permissive on missing optional fields (defaults applied per the
//     Python implementation; e.g., missing point coordinates default to 0.0)
//   - Strict on unknown enum string values: PointType, ConstraintType,
//     ConstraintStatus, SolverStatus all throw std::invalid_argument
//   - Unknown primitive `type` strings throw std::invalid_argument
SketchDocument from_json(std::string_view json_text);

// File I/O convenience.
void save(const SketchDocument& doc,
          const std::filesystem::path& path,
          int indent = 2);
SketchDocument load(const std::filesystem::path& path);

}  // namespace morphe
