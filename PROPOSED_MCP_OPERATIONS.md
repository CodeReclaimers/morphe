# Sketch Reconstruction MCP Operations

This document describes the operations available for manipulating 2D sketch reconstructions. These operations are designed for an LLM-assisted workflow where sketches are iteratively refined from point cloud data.

## Overview

The sketch reconstruction system works with two primitive types:
- **Line Segments**: Defined by start and end points
- **Arcs**: Defined by center, radius, start angle, and end angle

Elements are labeled for reference:
- Lines: `L1`, `L2`, `L3`, ...
- Arcs: `A1`, `A2`, `A3`, ...

Labels are persistent - they don't change when other elements are deleted.

---

## Implemented Operations

### Delete Operations

#### `delete`
Remove one or more elements from the sketch.

**Arguments:** `<target1> [target2] [target3] ...`

**Example:**
```
delete L1 L2 A3
```

**Use cases:**
- Remove spurious elements that don't correspond to real geometry
- Clean up incorrectly detected features
- Remove redundant overlapping elements

---

### Merge Operations

#### `merge_to_line`
Combine multiple collinear segments (or nearly-collinear arcs) into a single line segment.

**Arguments:** `<target1> <target2> [target3] ...`

**Example:**
```
merge_to_line L1 L2 L3
```

**Behavior:**
- Fits a new line through all support points from the merged elements
- Deletes the original elements
- Creates one new line segment

**Use cases:**
- Combine fragmented line detections
- Simplify over-segmented straight edges

#### `merge_to_arc`
Combine multiple elements that form an arc into a single arc.

**Arguments:** `<target1> <target2> [target3] ...`

**Example:**
```
merge_to_arc L1 L2 L3 L4
```

**Behavior:**
- Fits a new arc through all support points from the merged elements
- Deletes the original elements
- Creates one new arc

**Use cases:**
- Combine line segments that approximate a curve
- Merge fragmented arc detections

---

### Convert Operations

#### `convert_to_circle`
Convert one or more arcs to full circles (360° arcs).

**Arguments:** `<target1> [target2] ...`

**Example:**
```
convert_to_circle A1 A2
```

**Behavior:**
- Preserves center and radius
- Sets angular span to full circle (0 to 2π)

**Use cases:**
- Complete partial circle detections
- Fix arcs that should be full circles

#### `convert_to_line` *(not yet implemented)*
Convert an arc to a line segment.

#### `convert_to_arc` *(not yet implemented)*
Convert a line segment to an arc.

---

### Snap/Constraint Operations

#### `make_coincident`
Snap the closest endpoints of two elements together.

**Arguments:** `<target1> <target2>`

**Example:**
```
make_coincident L1 L2
```

**Behavior:**
- Finds the closest pair of endpoints between the two elements
- Moves both endpoints to their midpoint
- Records a coincident constraint

**Use cases:**
- Close small gaps at junctions
- Ensure connected elements share exact endpoints

#### `make_tangent`
Make two elements tangent at their closest junction point.

**Arguments:** `<target1> <target2>`

**Example:**
```
make_tangent L1 A1
```

**Behavior:**
- Identifies the junction point (closest endpoints)
- Adjusts geometry so tangent directions match
- For line-arc: rotates the line to be tangent to the arc
- For arc-arc: adjusts one arc's angle
- Records a tangent constraint

**Use cases:**
- Smooth transitions between lines and arcs
- Fix G1 continuity at junctions

#### `make_horizontal`
Snap near-horizontal line(s) to be exactly horizontal.

**Arguments:** `<target1> [target2] ...`

**Example:**
```
make_horizontal L1 L3
```

**Behavior:**
- Adjusts endpoints so the line has zero slope
- Preserves the line's midpoint and length
- Only works on lines within 15° of horizontal
- Records a horizontal constraint

**Use cases:**
- Clean up lines that should be horizontal
- Enforce axis-aligned geometry

#### `make_vertical`
Snap near-vertical line(s) to be exactly vertical.

**Arguments:** `<target1> [target2] ...`

**Example:**
```
make_vertical L2 L4
```

**Behavior:**
- Adjusts endpoints so the line is exactly vertical
- Preserves the line's midpoint and length
- Only works on lines within 15° of vertical
- Records a vertical constraint

**Use cases:**
- Clean up lines that should be vertical
- Enforce axis-aligned geometry

---

### Fillet Operations

#### `make_fillet`
Create a fillet (rounded corner) arc between two intersecting lines.

**Arguments:** `<radius> <line1> <line2>`

**Example:**
```
make_fillet 0.05 L1 L2
```

**Behavior:**
- Computes the intersection of the two lines
- Creates an arc of the specified radius tangent to both lines
- Trims the lines to the tangent points
- Records tangent constraints

**Use cases:**
- Add rounded corners to sharp intersections
- Model filleted edges

#### `replace_with_fillet`
Delete existing arc(s) and replace with a properly computed fillet.

**Arguments:** `<radius> <line1> <line2> [arc1] [arc2] ...`

**Example:**
```
replace_with_fillet 0.05 L1 L4 A1
```

**Behavior:**
- Deletes the specified arcs
- Creates a new fillet arc between the two lines
- Trims the lines to the tangent points

**Use cases:**
- Replace poorly-detected corner arcs with clean fillets
- Fix incorrect fillet radius

---

### Add Operations

#### `add_line`
Add a new line segment at specified coordinates.

**Arguments:** `<x1> <y1> <x2> <y2>`

**Example:**
```
add_line 0.0 0.0 1.0 0.0
```

**Behavior:**
- Creates a new line segment from (x1, y1) to (x2, y2)
- Finds and assigns support points from the point cloud
- Assigns a new label

**Use cases:**
- Add missing line segments
- Manually specify geometry not detected by fitting

#### `add_arc`
Add a new arc through three points (two endpoints and one interior point).

**Arguments:** `<x1> <y1> <x2> <y2> <x3> <y3>`

**Example:**
```
add_arc 0.0 0.0 0.5 0.25 1.0 0.0
```

**Behavior:**
- Computes arc center and radius from three points
- Points 1 and 3 are endpoints, point 2 is on the arc interior
- Finds and assigns support points
- Assigns a new label

**Use cases:**
- Add missing arc segments
- Manually specify curved geometry

---

### Solver Operations

#### `solve_constraints`
Run the constraint solver to satisfy all recorded constraints.

**Arguments:** `[point_fit_weight] [timeout]`
- `point_fit_weight`: Weight for point cloud fitting term (default: 1.0)
- `timeout`: Maximum solve time in seconds (default: 5.0)

**Example:**
```
solve_constraints 1.0 10.0
```

**Behavior:**
- Optimizes element geometry to satisfy constraints
- Balances constraint satisfaction with point cloud fit
- Uses numerical optimization (scipy)

**Use cases:**
- Apply accumulated constraints
- Refine geometry after multiple snap operations

---

### Control Operations

#### `accept`
Mark the current sketch as complete/acceptable.

**Arguments:** None

**Example:**
```
accept
```

**Behavior:**
- Signals that the sketch is satisfactory
- Used by LLM to indicate no further changes needed

---

## Not Yet Implemented

The following operations are defined but not yet implemented:

| Operation | Description |
|-----------|-------------|
| `convert_to_arc` | Convert line segment to arc |
| `convert_to_line` | Convert arc to line segment |
| `mark_outliers` | Flag noisy/outlier points |
| `split` | Split a segment at a problem region |

---

## MCP Server Design Considerations

### Tool Structure

Each operation could be exposed as an MCP tool with:
1. **Name**: Operation name (e.g., `sketch_delete`, `sketch_make_fillet`)
2. **Description**: What the operation does
3. **Input Schema**: JSON schema for arguments
4. **Output**: Success/failure status and message

### Example Tool Definitions

```json
{
  "name": "sketch_delete",
  "description": "Remove one or more elements from the sketch",
  "inputSchema": {
    "type": "object",
    "properties": {
      "targets": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Element labels to delete (e.g., ['L1', 'A2'])"
      }
    },
    "required": ["targets"]
  }
}
```

```json
{
  "name": "sketch_make_fillet",
  "description": "Create a fillet arc between two intersecting lines",
  "inputSchema": {
    "type": "object",
    "properties": {
      "radius": {
        "type": "number",
        "description": "Fillet radius"
      },
      "line1": {
        "type": "string",
        "description": "First line label"
      },
      "line2": {
        "type": "string",
        "description": "Second line label"
      }
    },
    "required": ["radius", "line1", "line2"]
  }
}
```

### State Management

The MCP server should maintain:
- Current sketch state (elements, constraints)
- Point cloud data
- Undo history

### Recommended Additional Tools

For a complete MCP server, consider adding:

1. **`sketch_load`**: Load point cloud data from file or array
2. **`sketch_fit`**: Run initial fitting pipeline
3. **`sketch_get_state`**: Return current elements, metrics, constraints
4. **`sketch_render`**: Generate visualization images
5. **`sketch_undo`**: Revert last operation
6. **`sketch_export`**: Export sketch to JSON/SVG format

### Error Handling

Operations return structured results:
```json
{
  "success": true,
  "message": "Deleted 2 elements: L1, L2",
  "modified_elements": [],
  "added_elements": [],
  "removed_elements": ["L1", "L2"]
}
```

Failed operations include descriptive error messages:
```json
{
  "success": false,
  "message": "Element 'L99' not found in sketch"
}
```
