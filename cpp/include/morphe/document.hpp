#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "morphe/constraints.hpp"
#include "morphe/primitives.hpp"

namespace morphe {

enum class SolverStatus {
    Dirty,
    UnderConstrained,
    FullyConstrained,
    OverConstrained,
    Inconsistent,
};

inline std::string_view to_string(SolverStatus s) {
    switch (s) {
        case SolverStatus::Dirty:             return "dirty";
        case SolverStatus::UnderConstrained:  return "under_constrained";
        case SolverStatus::FullyConstrained:  return "fully_constrained";
        case SolverStatus::OverConstrained:   return "over_constrained";
        case SolverStatus::Inconsistent:      return "inconsistent";
    }
    throw std::logic_error("unhandled SolverStatus in to_string");
}

inline SolverStatus solver_status_from_string(std::string_view s) {
    if (s == "dirty")              return SolverStatus::Dirty;
    if (s == "under_constrained")  return SolverStatus::UnderConstrained;
    if (s == "fully_constrained")  return SolverStatus::FullyConstrained;
    if (s == "over_constrained")   return SolverStatus::OverConstrained;
    if (s == "inconsistent")       return SolverStatus::Inconsistent;
    throw std::invalid_argument(std::string{"unknown SolverStatus value: "} + std::string{s});
}

// Container for a 2D sketch.
//
// Insertion order of `primitives` is significant: it is preserved across
// (de)serialization to match Python's dict iteration order, so that a
// Python-saved document and a C++-saved document of the same sketch produce
// identical primitive arrays in JSON.
class SketchDocument {
public:
    std::string name = "Untitled";
    std::vector<Primitive> primitives;
    std::vector<SketchConstraint> constraints;
    SolverStatus solver_status = SolverStatus::Dirty;
    int degrees_of_freedom = -1;

    // Auto-assign an ID using the prefix counter and append. Marks the document
    // dirty. Returns the assigned ID. Mirrors SketchDocument.add_primitive.
    std::string add_primitive(Primitive p) {
        const char prefix = id_prefix(p);
        int& counter = next_index_[prefix];
        std::string id;
        id.reserve(8);
        id.push_back(prefix);
        id += std::to_string(counter++);
        meta_of(p).id = id;
        primitives.push_back(std::move(p));
        solver_status = SolverStatus::Dirty;
        return id;
    }

    // Append with a caller-supplied ID (e.g., during deserialization). Throws
    // std::invalid_argument if the ID already exists or if its prefix does
    // not match the primitive type. Updates the prefix counter so subsequent
    // auto-assigns do not collide.
    std::string add_primitive_with_id(Primitive p, std::string id) {
        if (find_primitive(id) != nullptr) {
            throw std::invalid_argument("element ID already exists: " + id);
        }
        const char expected_prefix = id_prefix(p);
        if (!id.empty() && id.front() != expected_prefix) {
            throw std::invalid_argument(
                std::string{"Element ID '"} + id + "' has prefix '" + id.front() +
                "' but expected '" + expected_prefix + "' for " +
                std::string{type_tag(p)});
        }
        meta_of(p).id = id;
        primitives.push_back(std::move(p));

        if (id.size() >= 2) {
            try {
                const int idx = std::stoi(id.substr(1));
                int& counter = next_index_[expected_prefix];
                if (idx + 1 > counter) counter = idx + 1;
            } catch (const std::exception&) {
                // Non-numeric suffix: skip counter update silently, matching
                // the Python implementation's `except ValueError: pass`.
            }
        }
        solver_status = SolverStatus::Dirty;
        return id;
    }

    const Primitive* find_primitive(std::string_view id) const {
        for (const auto& p : primitives) {
            if (id_of(p) == id) return &p;
        }
        return nullptr;
    }
    Primitive* find_primitive(std::string_view id) {
        for (auto& p : primitives) {
            if (id_of(p) == id) return &p;
        }
        return nullptr;
    }

    void add_constraint(SketchConstraint c) {
        // Validate that all referenced elements exist. Matches
        // SketchDocument.add_constraint in Python.
        for (const auto& elem_id : c.get_element_ids()) {
            if (find_primitive(elem_id) == nullptr) {
                throw std::out_of_range(
                    "constraint references non-existent element '" + elem_id + "'");
            }
        }
        constraints.push_back(std::move(c));
        solver_status = SolverStatus::Dirty;
    }

    friend bool operator==(const SketchDocument& a, const SketchDocument& b) {
        return a.name == b.name
            && a.primitives == b.primitives
            && a.constraints == b.constraints
            && a.solver_status == b.solver_status
            && a.degrees_of_freedom == b.degrees_of_freedom;
    }
    friend bool operator!=(const SketchDocument& a, const SketchDocument& b) { return !(a == b); }

private:
    std::unordered_map<char, int> next_index_;
};

}  // namespace morphe
