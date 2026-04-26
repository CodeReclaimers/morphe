#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "morphe/document.hpp"

namespace morphe {

// Validation tolerance, in millimeters. Mirrors morphe.validation.DEFAULT_TOLERANCE.
inline constexpr double default_tolerance = 0.001;  // 1 micron

enum class ValidationSeverity {
    Error,
    Warning,
    Info,
};

inline std::string_view to_string(ValidationSeverity s) {
    switch (s) {
        case ValidationSeverity::Error:   return "error";
        case ValidationSeverity::Warning: return "warning";
        case ValidationSeverity::Info:    return "info";
    }
    return "?";
}

// A single validation issue. `code` is a stable machine-readable identifier
// (e.g., "ARC_RADIUS_INCONSISTENT") matching the codes used by
// morphe.validation. `element_id` is the affected primitive ID, or empty for
// constraint-level issues.
struct ValidationIssue {
    ValidationSeverity severity = ValidationSeverity::Error;
    std::string element_id;
    std::string message;
    std::string code;
};

class ValidationResult {
public:
    std::vector<ValidationIssue> issues;

    void add_error(std::string message, std::string code, std::string element_id = {}) {
        issues.push_back({ValidationSeverity::Error,
                          std::move(element_id), std::move(message), std::move(code)});
    }
    void add_warning(std::string message, std::string code, std::string element_id = {}) {
        issues.push_back({ValidationSeverity::Warning,
                          std::move(element_id), std::move(message), std::move(code)});
    }
    void add_info(std::string message, std::string code, std::string element_id = {}) {
        issues.push_back({ValidationSeverity::Info,
                          std::move(element_id), std::move(message), std::move(code)});
    }

    bool is_valid() const {
        for (const auto& i : issues) {
            if (i.severity == ValidationSeverity::Error) return false;
        }
        return true;
    }

    bool has_warnings() const {
        for (const auto& i : issues) {
            if (i.severity == ValidationSeverity::Warning) return true;
        }
        return false;
    }

    std::vector<ValidationIssue> errors() const {
        std::vector<ValidationIssue> out;
        for (const auto& i : issues) {
            if (i.severity == ValidationSeverity::Error) out.push_back(i);
        }
        return out;
    }

    std::vector<ValidationIssue> warnings() const {
        std::vector<ValidationIssue> out;
        for (const auto& i : issues) {
            if (i.severity == ValidationSeverity::Warning) out.push_back(i);
        }
        return out;
    }
};

// Validate a sketch document against the schema rules ported from
// morphe.validation. Performs primitive-level geometry checks, constraint
// reference and reference-count checks, dimensional value sign checks, and
// duplicate constraint ID detection.
//
// NOTE: Like the Python validator, this does NOT enforce that referenced
// primitive types match the `ref_kinds` field of CONSTRAINT_RULES (e.g., a
// Horizontal constraint on a Circle currently passes). That gap is shared
// with the Python implementation by design — fixing it would be a spec
// change, not an implementation change.
ValidationResult validate(const SketchDocument& doc, double tolerance = default_tolerance);

}  // namespace morphe
