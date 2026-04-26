#include "morphe/constraints.hpp"

#include <stdexcept>
#include <unordered_map>

namespace morphe {

namespace {

// Position-by-position reference kind list. For chainable constraints
// (max_refs == nullopt), the list has size 1 and applies to every reference.
// For SYMMETRIC, the list has size 3: {Any, Any, Line}.
//
// Source: morphe/constraints.py CONSTRAINT_RULES table.
const std::unordered_map<ConstraintType, ConstraintRules>& table() {
    static const std::unordered_map<ConstraintType, ConstraintRules> rules = {
        {ConstraintType::Coincident,    {2, 2,            {RefKind::Point},             false}},
        {ConstraintType::Tangent,       {2, 2,            {RefKind::Curve},             false}},
        {ConstraintType::Perpendicular, {2, 2,            {RefKind::Line},              false}},
        {ConstraintType::Parallel,      {2, 2,            {RefKind::Line},              false}},
        {ConstraintType::Concentric,    {2, 2,            {RefKind::ArcOrCircle},       false}},
        {ConstraintType::Equal,         {2, std::nullopt, {RefKind::LineOrArcOrCircle}, false}},
        {ConstraintType::Collinear,     {2, std::nullopt, {RefKind::Line},              false}},
        {ConstraintType::Horizontal,    {1, 1,            {RefKind::Line},              false}},
        {ConstraintType::Vertical,      {1, 1,            {RefKind::Line},              false}},
        {ConstraintType::Fixed,         {1, 1,            {RefKind::Any},               false}},
        {ConstraintType::Distance,      {2, 2,            {RefKind::Point},             true}},
        {ConstraintType::DistanceX,     {1, 2,            {RefKind::Point},             true}},
        {ConstraintType::DistanceY,     {1, 2,            {RefKind::Point},             true}},
        {ConstraintType::Length,        {1, 1,            {RefKind::Line},              true}},
        {ConstraintType::Radius,        {1, 1,            {RefKind::ArcOrCircle},       true}},
        {ConstraintType::Diameter,      {1, 1,            {RefKind::ArcOrCircle},       true}},
        {ConstraintType::Angle,         {2, 2,            {RefKind::Line},              true}},
        {ConstraintType::Symmetric,     {3, 3,            {RefKind::Any, RefKind::Any, RefKind::Line}, false}},
        {ConstraintType::Midpoint,      {2, 2,            {RefKind::Point, RefKind::Line}, false}},
    };
    return rules;
}

}  // namespace

const ConstraintRules& rules_for(ConstraintType t) {
    const auto& tbl = table();
    auto it = tbl.find(t);
    if (it == tbl.end()) {
        throw std::logic_error("missing rules entry for ConstraintType");
    }
    return it->second;
}

}  // namespace morphe
