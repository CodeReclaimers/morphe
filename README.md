# Canonical Sketch

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/codereclaimers/canonical_sketch/actions/workflows/test.yml/badge.svg)](https://github.com/codereclaimers/canonical_sketch/actions/workflows/test.yml)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

A CAD-agnostic 2D sketch geometry and constraint representation with adapter support for FreeCAD and Fusion 360.

## Overview

This project provides:

- **`sketch_canonical`**: Platform-independent schema for 2D sketch geometry and constraints
- **`sketch_adapter_freecad`**: Adapter for FreeCAD's Sketcher workbench
- **`sketch_adapter_fusion`**: Adapter for Autodesk Fusion 360

The canonical format enables constrained sketches to be stored, transferred, and manipulated independently of any specific CAD system.

See [SPECIFICATION.md](SPECIFICATION.md) for the complete technical specification, including supported geometry types, constraints, JSON schema format, and platform-specific adapter details.

## Installation

```bash
pip install -e .
```

For FreeCAD integration, ensure FreeCAD is installed and accessible (via snap, package manager, or `PYTHONPATH`).

For Fusion 360 integration, the adapter must be run from within Fusion 360's Python environment (as a script or add-in).

## Quick Start

```python
from sketch_canonical import (
    SketchDocument, Point2D, Line, Circle, Horizontal, Radius,
    save_sketch, load_sketch
)

# Create a sketch
sketch = SketchDocument(name="MySketch")
line_id = sketch.add_primitive(Line(start=Point2D(0, 0), end=Point2D(100, 0)))
circle_id = sketch.add_primitive(Circle(center=Point2D(50, 50), radius=20))

sketch.add_constraint(Horizontal(line_id))
sketch.add_constraint(Radius(circle_id, 20))

save_sketch(sketch, "my_sketch.json")

# Load from file
sketch = load_sketch("my_sketch.json")
```

## FreeCAD Integration

```python
from sketch_canonical import load_sketch
from sketch_adapter_freecad import FreeCADAdapter

sketch = load_sketch("my_sketch.json")
adapter = FreeCADAdapter()
adapter.create_sketch(sketch.name)
adapter.load_sketch(sketch)

status, dof = adapter.get_solver_status()
print(f"Status: {status.name}, DOF: {dof}")

adapter.close_sketch()
```

## Fusion 360 Integration

The Fusion 360 adapter runs as a script or add-in within Fusion 360:

```python
# Run this inside Fusion 360's Scripts environment
import sys
sys.path.insert(0, r"path/to/canonical_sketch")

from sketch_canonical import load_sketch
from sketch_adapter_fusion import FusionAdapter

def run(context):
    sketch = load_sketch("my_sketch.json")
    adapter = FusionAdapter()
    adapter.create_sketch(sketch.name)
    adapter.load_sketch(sketch)

    # Export back to canonical format
    exported = adapter.export_sketch()
    print(f"Exported {len(exported.primitives)} primitives")

    status, dof = adapter.get_solver_status()
    print(f"Status: {status.name}, DOF: {dof}")
```

**Supported Features:**
- All primitive types: Line, Arc, Circle, Point, Spline (NURBS)
- All geometric constraints: Coincident, Tangent, Parallel, Perpendicular, etc.
- All dimensional constraints: Length, Radius, Diameter, Angle, Distance
- Construction geometry
- Solver status and DOF reporting

## Running Tests

**Core tests (no CAD software required):**
```bash
pytest tests/                    # All tests
pytest tests/ --cov              # With coverage
```

**FreeCAD adapter tests (requires FreeCAD):**
```bash
pytest tests/test_freecad_roundtrip.py -v
```

**Fusion 360 adapter tests (run from within Fusion 360):**

The Fusion 360 test suite (73 tests) must be run as a script inside Fusion 360:
1. Open Fusion 360
2. Go to Utilities > Add-Ins > Scripts
3. Add and run the test script from `sketch_adapter_fusion/tests/`

Tests cover all primitives, constraints, solver status, and edge cases.

## License

MIT - see [LICENSE](LICENSE) for details.
