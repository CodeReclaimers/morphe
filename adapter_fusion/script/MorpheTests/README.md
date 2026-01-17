# Fusion 360 Round-Trip Tests

This directory contains test scripts designed to run inside Autodesk Fusion 360.
These tests verify the round-trip behavior of the Fusion 360 adapter - ensuring that
sketches can be loaded into Fusion and exported back without loss of essential information.

## Prerequisites

1. Autodesk Fusion 360 installed
2. The `canonical_sketch` package accessible from Fusion's Python environment

## Setup

### Option 1: Add to Python Path (Recommended for Development)

The test script automatically adds the project root to the Python path. If you're
running from the default location within the canonical_sketch project, no additional
setup is required.

### Option 2: Install the Package

```bash
cd /path/to/canonical_sketch
pip install -e .
```

Note: Fusion 360 uses its own Python environment, so you may need to install the
package specifically for Fusion's Python.

## Running the Tests

1. Open Autodesk Fusion 360
2. Go to **Utilities** > **Add-Ins** (or press Shift+S)
3. In the Scripts tab, click the green **+** button (Create Script)
4. Navigate to this directory and select `test_roundtrip.py`
5. Click **Run**

Alternatively, you can add the script to your Scripts folder:
- Windows: `%appdata%\Autodesk\Autodesk Fusion 360\API\Scripts\`
- macOS: `~/Library/Application Support/Autodesk/Autodesk Fusion 360/API/Scripts/`

## Test Output

The tests will:
1. Display a start message
2. Create test sketches in a temporary document
3. Run all tests and capture results
4. Show a summary message box
5. Output detailed results to the **Text Commands** palette

To view the Text Commands palette: **View** > **Show Text Commands** (or Ctrl+Alt+C)

## Test Categories

### Basic Geometry Tests
- `test_single_line` - Single line round-trip
- `test_single_circle` - Single circle round-trip
- `test_single_arc` - Single arc round-trip
- `test_single_point` - Single point round-trip

### Complex Geometry Tests
- `test_rectangle` - Four connected lines
- `test_mixed_geometry` - Line, arc, circle, and point together
- `test_construction_geometry` - Construction flag preservation

### Constraint Tests
- `test_horizontal_constraint` - Horizontal line constraint
- `test_vertical_constraint` - Vertical line constraint
- `test_radius_constraint` - Circle/arc radius constraint
- `test_diameter_constraint` - Circle/arc diameter constraint
- `test_coincident_constraint` - Point coincidence
- `test_parallel_constraint` - Parallel lines
- `test_perpendicular_constraint` - Perpendicular lines
- `test_equal_constraint` - Equal length lines
- `test_concentric_constraint` - Concentric circles
- `test_length_constraint` - Line length constraint
- `test_angle_constraint` - Angle between lines

### Spline Tests
- `test_simple_bspline` - Cubic B-spline (degree 3)
- `test_quadratic_bspline` - Quadratic B-spline (degree 2)

### Integration Tests
- `test_fully_constrained_rectangle` - Rectangle with full constraint set

## Troubleshooting

### "Module not found" Errors

If you see import errors for `sketch_canonical` or `sketch_adapter_fusion`:

1. Check that the project root path is correct in the test script
2. Verify the package structure is intact
3. Try installing the package to Fusion's Python environment

### Test Failures

If tests fail:
1. Check the Text Commands palette for detailed error messages
2. Verify you have a valid Fusion document open
3. Some constraints may behave differently depending on Fusion version

### Performance

The tests create and delete sketches rapidly. If Fusion becomes slow:
1. Close and reopen Fusion
2. Run tests in smaller batches by commenting out test methods

## Adding New Tests

To add new tests, add methods to the `FusionTestRunner` class that:
1. Start with `test_`
2. Create a `SketchDocument` with primitives and/or constraints
3. Load it using `self._adapter.load_sketch()`
4. Export using `self._adapter.export_sketch()`
5. Use `assert` statements to verify the results

Example:
```python
def test_my_new_feature(self):
    """Test description."""
    sketch = SketchDocument(name="MyTest")
    sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(100, 0)))

    self._adapter.create_sketch(sketch.name)
    self._adapter.load_sketch(sketch)
    exported = self._adapter.export_sketch()

    assert len(exported.primitives) == 1
    # More assertions...
```
