# Backporting Canonical Sketch Tests to FreeCAD

This document provides instructions for backporting the Fusion 360 test suite to FreeCAD. The tests validate round-trip behavior of the canonical sketch format through the FreeCAD adapter.

## Overview

- **Source tests**: `cs_test.py` (73 tests for Fusion 360)
- **Target adapter**: `sketch_adapter_freecad`
- **FreeCAD adapter location**: `F:\canonical_sketch\sketch_adapter_freecad\adapter.py`

## Key Differences Between Adapters

### 1. Unit System
| Adapter | Internal Units | Conversion Needed |
|---------|---------------|-------------------|
| Fusion 360 | Centimeters | Yes (mm ↔ cm) |
| FreeCAD | Millimeters | No (matches canonical) |

### 2. Constraint API Patterns

**Fusion 360:**
```python
sketch.geometricConstraints.addHorizontal(line)
sketch.geometricConstraints.addCoincident(point1, point2)
sketch.sketchDimensions.addDistanceDimension(...)
```

**FreeCAD:**
```python
sketch.addConstraint(Sketcher.Constraint('Horizontal', geo_index))
sketch.addConstraint(Sketcher.Constraint('Coincident', geo1, vtx1, geo2, vtx2))
sketch.addConstraint(Sketcher.Constraint('Distance', geo1, vtx1, geo2, vtx2, value))
```

### 3. Vertex Indexing

**Fusion 360** uses property names:
- `line.startSketchPoint`, `line.endSketchPoint`
- `arc.centerSketchPoint`

**FreeCAD** uses numeric indices:
```python
# Line vertices
LINE_START = 1
LINE_END = 2

# Arc vertices
ARC_START = 1
ARC_END = 2
ARC_CENTER = 3

# Circle vertices
CIRCLE_CENTER = 3

# Point vertices
POINT_CENTER = 1

# Spline vertices
SPLINE_START = 1
SPLINE_END = 2
```

### 4. Fixed Constraint

**Fusion 360:**
```python
entity.isFixed = True  # Property assignment
```

**FreeCAD:**
```python
sketch.addConstraint(Sketcher.Constraint('Block', geo_index))
```

### 5. Concentric Constraint

**Fusion 360:**
```python
sketch.geometricConstraints.addConcentric(circle1, circle2)
```

**FreeCAD:** Uses center-point coincident:
```python
sketch.addConstraint(Sketcher.Constraint(
    'Coincident',
    circle1_idx, 3,  # 3 = CIRCLE_CENTER
    circle2_idx, 3
))
```

### 6. Solver Status

**Fusion 360:**
```python
sketch.isFullyConstrained  # Boolean property
# DOF must be estimated from geometry
```

**FreeCAD:**
```python
result = sketch.solve()
# Returns: 0 = fully constrained, >0 = DOF count, <0 = over-constrained
sketch.conflictingConstraints  # List of conflicting constraint indices
```

### 7. Angle Constraint Units

**Fusion 360:** Radians internally
**FreeCAD:** Radians (adapter handles conversion from canonical degrees)

## Test File Structure

Create a new test file `cs_test_freecad.py` with this structure:

```python
"""
Round-trip tests for FreeCAD adapter.

Run from FreeCAD's Python console or as a macro.
"""

import math
import sys
import traceback
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(r"F:\canonical_sketch")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import FreeCAD as App
import FreeCADGui as Gui
import Sketcher
import Part

from sketch_adapter_freecad import FreeCADAdapter

from sketch_canonical import (
    Arc, Circle, Line, Point, Point2D,
    PointRef, PointType, SketchDocument, Spline,
)
from sketch_canonical.constraints import (
    Angle, Coincident, Collinear, Concentric, Diameter, Distance,
    DistanceX, DistanceY, Equal, Fixed, Horizontal, Length,
    MidpointConstraint, Parallel, Perpendicular, Radius, Symmetric,
    Tangent, Vertical,
)
from sketch_canonical.document import SolverStatus


class FreeCADTestRunner:
    """Test runner for FreeCAD round-trip tests."""

    def __init__(self):
        self.results = []
        self._test_doc = None
        self._adapter = None

    def setup(self):
        """Set up test environment - create a new document."""
        self._test_doc = App.newDocument("TestDocument")
        self._adapter = FreeCADAdapter(self._test_doc)

    def teardown(self):
        """Clean up test environment."""
        if self._test_doc:
            App.closeDocument(self._test_doc.Name)
        self._test_doc = None
        self._adapter = None

    # ... test methods go here ...
```

## Tests That Should Work Unchanged

These tests use only the adapter's public interface and should work with minimal changes:

### Primitive Tests
- `test_single_line`
- `test_single_arc`
- `test_single_circle`
- `test_single_point`
- `test_diagonal_line`
- `test_rectangle`
- `test_negative_coordinates`
- `test_geometry_at_origin`
- `test_small_geometry`
- `test_large_geometry`
- `test_construction_geometry`
- `test_construction_arc`
- `test_simple_bspline`
- `test_quadratic_bspline`
- `test_higher_degree_spline`
- `test_many_control_points_spline`
- `test_weighted_spline`

### Constraint Tests
- `test_horizontal_constraint`
- `test_vertical_constraint`
- `test_parallel_constraint`
- `test_perpendicular_constraint`
- `test_coincident_constraint`
- `test_tangent_arc_line`
- `test_tangent_line_circle`
- `test_collinear_constraint`
- `test_equal_constraint`
- `test_concentric_constraint`
- `test_length_constraint`
- `test_radius_constraint`
- `test_diameter_constraint`
- `test_distance_constraint`
- `test_distance_x_constraint`
- `test_distance_y_constraint`
- `test_angle_constraint`
- `test_symmetric_constraint`
- `test_midpoint_constraint`

### Complex Tests
- `test_closed_profile`
- `test_slot_profile`
- `test_mixed_geometry`
- `test_nested_geometry`
- `test_fully_constrained_rectangle`
- `test_concentric_circles`
- `test_equal_circles`
- `test_arc_clockwise`
- `test_arc_large_angle`

## Tests Requiring Modification

### 1. `test_fixed_constraint`

The Fixed constraint export may differ. FreeCAD uses 'Block' constraint:

```python
def test_fixed_constraint(self):
    """Test fixed constraint locks geometry in place."""
    sketch = SketchDocument(name="FixedTest")
    line_id = sketch.add_primitive(Line(
        start=Point2D(10, 20),
        end=Point2D(50, 60)
    ))
    sketch.add_constraint(Fixed(line_id))

    self._adapter.create_sketch(sketch.name)
    self._adapter.load_sketch(sketch)
    exported = self._adapter.export_sketch()

    line = list(exported.primitives.values())[0]
    # Geometry should maintain exact position
    assert abs(line.start.x - 10) < 0.01
    assert abs(line.start.y - 20) < 0.01
    assert abs(line.end.x - 50) < 0.01
    assert abs(line.end.y - 60) < 0.01
```

### 2. `test_solver_status_fullyconstrained`

FreeCAD has better DOF reporting:

```python
def test_solver_status_fullyconstrained(self):
    """Test that fully constrained sketch reports zero DOF."""
    sketch = SketchDocument(name="FullyConstrainedTest")
    line_id = sketch.add_primitive(Line(
        start=Point2D(0, 0),
        end=Point2D(100, 0)
    ))
    sketch.add_constraint(Fixed(line_id))

    self._adapter.create_sketch(sketch.name)
    self._adapter.load_sketch(sketch)

    status, dof = self._adapter.get_solver_status()

    assert status == SolverStatus.FULLY_CONSTRAINED, \
        f"Expected FULLY_CONSTRAINED, got {status}"
    assert dof == 0, f"Expected DOF = 0, got {dof}"
```

### 3. `test_solver_status_underconstrained`

```python
def test_solver_status_underconstrained(self):
    """Test that unconstrained sketch reports positive DOF."""
    sketch = SketchDocument(name="UnderConstrainedTest")
    sketch.add_primitive(Line(
        start=Point2D(0, 0),
        end=Point2D(100, 0)
    ))
    # No constraints added

    self._adapter.create_sketch(sketch.name)
    self._adapter.load_sketch(sketch)

    status, dof = self._adapter.get_solver_status()

    assert status == SolverStatus.UNDER_CONSTRAINED, \
        f"Expected UNDER_CONSTRAINED, got {status}"
    assert dof > 0, f"Expected DOF > 0, got {dof}"
```

### 4. `test_periodic_spline`

FreeCAD may handle periodic splines better than Fusion:

```python
def test_periodic_spline(self):
    """Test closed/periodic spline round-trip."""
    # FreeCAD's Part.BSplineCurve supports periodic=True
    control_points = [
        Point2D(0, 50),
        Point2D(50, 100),
        Point2D(100, 50),
        Point2D(50, 0),
    ]
    # Try true periodic spline first
    knots = [0.0, 0.25, 0.5, 0.75, 1.0]

    sketch = SketchDocument(name="PeriodicSplineTest")
    sketch.add_primitive(Spline(
        control_points=control_points,
        degree=3,
        knots=knots,
        periodic=True
    ))

    self._adapter.create_sketch(sketch.name)
    self._adapter.load_sketch(sketch)
    exported = self._adapter.export_sketch()

    spline = list(exported.primitives.values())[0]
    assert isinstance(spline, Spline), "Expected Spline primitive"
    assert len(spline.control_points) >= 4, "Should have control points"
```

### 5. `test_empty_sketch`

Should work, but verify FreeCAD handles empty sketches:

```python
def test_empty_sketch(self):
    """Test that empty sketch exports correctly."""
    sketch = SketchDocument(name="EmptySketchTest")

    self._adapter.create_sketch(sketch.name)
    self._adapter.load_sketch(sketch)
    exported = self._adapter.export_sketch()

    assert len(exported.primitives) == 0, \
        f"Empty sketch should have no primitives, got {len(exported.primitives)}"
```

## New Tests to Add for FreeCAD

### 1. Over-Constrained Detection

FreeCAD can detect conflicting constraints:

```python
def test_overconstrained_detection(self):
    """Test that conflicting constraints are detected."""
    sketch = SketchDocument(name="OverConstrainedTest")
    line_id = sketch.add_primitive(Line(
        start=Point2D(0, 0),
        end=Point2D(100, 0)
    ))
    # Add conflicting constraints
    sketch.add_constraint(Horizontal(line_id))
    sketch.add_constraint(Vertical(line_id))  # Conflicts with horizontal!

    self._adapter.create_sketch(sketch.name)
    self._adapter.load_sketch(sketch)

    status, dof = self._adapter.get_solver_status()

    # Should detect the conflict
    assert status in [SolverStatus.OVER_CONSTRAINED, SolverStatus.INCONSISTENT], \
        f"Expected conflict detection, got {status}"
```

### 2. Conflicting Constraint List

```python
def test_conflicting_constraints_list(self):
    """Test that conflicting constraints can be identified."""
    sketch = SketchDocument(name="ConflictListTest")
    line_id = sketch.add_primitive(Line(
        start=Point2D(0, 0),
        end=Point2D(100, 50)
    ))
    sketch.add_constraint(Length(line_id, 50))
    sketch.add_constraint(Length(line_id, 100))  # Conflicts!

    self._adapter.create_sketch(sketch.name)
    self._adapter.load_sketch(sketch)

    # Check if adapter exposes conflicting constraints
    # This is FreeCAD-specific functionality
```

## Running Tests in FreeCAD

### Option 1: FreeCAD Macro

Save the test file as a macro and run from FreeCAD:
1. Open FreeCAD
2. Macro > Macros > Navigate to test file
3. Execute

### Option 2: FreeCAD Python Console

```python
exec(open(r"path\to\cs_test_freecad.py").read())
run(None)
```

### Option 3: Command Line (FreeCAD CLI)

```bash
freecadcmd -c "exec(open('cs_test_freecad.py').read()); run(None)"
```

## Test Output

The test runner should output to FreeCAD's Report View:
- Use `FreeCAD.Console.PrintMessage()` for normal output
- Use `FreeCAD.Console.PrintWarning()` for warnings
- Use `FreeCAD.Console.PrintError()` for errors

```python
def report_results(self):
    """Report test results to FreeCAD console."""
    passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
    failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
    errors = sum(1 for r in self.results if r.status == TestStatus.ERROR)

    App.Console.PrintMessage(f"\nTest Results: {passed}/{len(self.results)} passed")
    if failed > 0:
        App.Console.PrintWarning(f", {failed} failed")
    if errors > 0:
        App.Console.PrintError(f", {errors} errors")
    App.Console.PrintMessage("\n")
```

## Recommended Backporting Order

1. **Phase 1: Basic Primitives** (get adapter working)
   - `test_single_line`
   - `test_single_circle`
   - `test_single_arc`
   - `test_single_point`
   - `test_empty_sketch`

2. **Phase 2: Simple Constraints**
   - `test_horizontal_constraint`
   - `test_vertical_constraint`
   - `test_length_constraint`
   - `test_radius_constraint`

3. **Phase 3: Point Constraints**
   - `test_coincident_constraint`
   - `test_distance_constraint`
   - `test_midpoint_constraint`

4. **Phase 4: Curve Constraints**
   - `test_parallel_constraint`
   - `test_perpendicular_constraint`
   - `test_tangent_arc_line`
   - `test_concentric_constraint`

5. **Phase 5: Complex Tests**
   - `test_closed_profile`
   - `test_fully_constrained_rectangle`
   - `test_solver_status_*`

6. **Phase 6: Splines**
   - `test_simple_bspline`
   - `test_weighted_spline`
   - `test_periodic_spline`

7. **Phase 7: Edge Cases & Precision**
   - All remaining tests

## Troubleshooting

### Common Issues

1. **"No module named 'FreeCAD'"**
   - Must run inside FreeCAD environment
   - Or set up FreeCAD Python path externally

2. **Geometry index mismatches**
   - FreeCAD geometry indices start at 0
   - External geometry uses negative indices (-1, -2, etc.)

3. **Constraint failures**
   - Check that geometry exists before adding constraints
   - Verify vertex indices match geometry type

4. **Spline knot errors**
   - FreeCAD uses multiplicities format
   - Adapter should handle conversion automatically

### Debug Tips

```python
# Print geometry count
print(f"Geometry count: {sketch.GeometryCount}")

# Print all constraints
for i, c in enumerate(sketch.Constraints):
    print(f"Constraint {i}: {c.Type}")

# Check solver state
result = sketch.solve()
print(f"Solver result: {result}")
print(f"Conflicts: {sketch.conflictingConstraints}")
```

## Summary

The FreeCAD adapter uses a different API pattern but supports all the same canonical sketch features. Most tests should work with minimal changes - primarily:

1. Replace `FusionAdapter` with `FreeCADAdapter`
2. Update document creation to use `App.newDocument()`
3. Update console output to use `App.Console.PrintMessage()`
4. Leverage FreeCAD's better solver status reporting

The test logic and assertions remain the same since both adapters implement the same `SketchBackendAdapter` interface.
