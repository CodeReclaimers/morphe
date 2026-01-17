# Morphe

*Pronounced "mor-FAY" (from Greek [μορφή](https://en.wiktionary.org/wiki/%CE%BC%CE%BF%CF%81%CF%86%CE%AE), meaning "form" or "shape")*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/codereclaimers/morphe/actions/workflows/test.yml/badge.svg)](https://github.com/codereclaimers/morphe/actions/workflows/test.yml)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

A CAD-agnostic 2D sketch geometry and constraint representation with adapter support for FreeCAD, Fusion 360, SolidWorks, and Autodesk Inventor.

## Overview

This project provides:

- **`morphe`**: Platform-independent schema for 2D sketch geometry and constraints
- **`morphe.adapters.freecad`**: Adapter for FreeCAD's Sketcher workbench
- **`morphe.adapters.fusion`**: Adapter for Autodesk Fusion 360
- **`morphe.adapters.solidworks`**: Adapter for SolidWorks (Windows only, via COM)
- **`morphe.adapters.inventor`**: Adapter for Autodesk Inventor (Windows only, via COM)

The canonical format enables constrained sketches to be stored, transferred, and manipulated independently of any specific CAD system.

See [SPECIFICATION.md](SPECIFICATION.md) for the complete technical specification, including supported geometry types, constraints, JSON schema format, and platform-specific adapter details.

## Installation

```bash
pip install -e .
```

### Platform-Specific Requirements

| Adapter | Platform | Requirements |
|---------|----------|--------------|
| FreeCAD | Linux, macOS, Windows | FreeCAD installed and accessible via `PYTHONPATH` |
| Fusion 360 | Windows, macOS | Run from within Fusion 360's Python environment |
| SolidWorks | Windows only | SolidWorks installed, `pywin32` package |
| Inventor | Windows only | Autodesk Inventor installed, `pywin32` package |

For Windows COM-based adapters (SolidWorks, Inventor):
```bash
pip install pywin32
```

## Quick Start

```python
from morphe import (
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
from morphe import load_sketch
from morphe.adapters.freecad import FreeCADAdapter

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
sys.path.insert(0, r"path/to/morphe-repo")

from morphe import load_sketch
from morphe.adapters.fusion import FusionAdapter

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

## SolidWorks Integration

The SolidWorks adapter uses COM automation via `pywin32` (Windows only):

```python
from morphe import load_sketch
from morphe.adapters.solidworks import SolidWorksAdapter, SOLIDWORKS_AVAILABLE

if SOLIDWORKS_AVAILABLE:
    sketch = load_sketch("my_sketch.json")
    adapter = SolidWorksAdapter()
    adapter.create_sketch(sketch.name)
    adapter.load_sketch(sketch)

    # Export back to canonical format
    exported = adapter.export_sketch()
    print(f"Exported {len(exported.primitives)} primitives")

    status, dof = adapter.get_solver_status()
    print(f"Status: {status.name}, DOF: {dof}")
```

**Notes:**
- Requires SolidWorks to be installed and running (or will launch automatically)
- The adapter connects to an existing SolidWorks instance or starts a new one
- Sketches are created on the Front Plane by default (XY plane)
- Dimensional constraints use geometry recreation to avoid blocking dialogs

**Supported Features:**
- All primitive types: Line, Arc, Circle, Point, Spline, Ellipse, EllipticalArc
- All geometric constraints: Coincident, Tangent, Parallel, Perpendicular, Horizontal, Vertical, Equal, Concentric, Collinear, Midpoint, Fixed, Symmetric
- All dimensional constraints: Length, Radius, Diameter, Angle, Distance, DistanceX, DistanceY
- Construction geometry
- Solver status and DOF reporting

## Inventor Integration

The Inventor adapter uses COM automation via `pywin32` (Windows only):

```python
from morphe import load_sketch
from morphe.adapters.inventor import InventorAdapter, INVENTOR_AVAILABLE

if INVENTOR_AVAILABLE:
    sketch = load_sketch("my_sketch.json")
    adapter = InventorAdapter()
    adapter.create_sketch(sketch.name)
    adapter.load_sketch(sketch)

    # Export back to canonical format
    exported = adapter.export_sketch()
    print(f"Exported {len(exported.primitives)} primitives")

    status, dof = adapter.get_solver_status()
    print(f"Status: {status.name}, DOF: {dof}")
```

**Notes:**
- Requires Autodesk Inventor to be installed and running (or will launch automatically)
- The adapter connects to an existing Inventor instance or starts a new one
- Sketches are created on the XY plane by default

**Supported Features:**
- All primitive types: Line, Arc, Circle, Point, Spline, Ellipse, EllipticalArc
- All geometric constraints: Coincident, Tangent, Parallel, Perpendicular, Horizontal, Vertical, Equal, Concentric, Collinear, Midpoint, Fixed, Symmetric
- All dimensional constraints: Length, Radius, Diameter, Angle, Distance, DistanceX, DistanceY
- Construction geometry
- Solver status and DOF reporting

## RPC Server Mode

For GUI tools or external scripts that need to communicate with CAD applications, each adapter provides an RPC server/client pair.

### Starting Servers

**FreeCAD** (run in FreeCAD's Python console):
```python
from morphe.adapters.freecad import start_server
start_server()  # localhost:9876
```

**Fusion 360** (install the MorpheServer add-in from `morphe/adapters/fusion/addin/MorpheServer/`):
```bash
cd morphe/adapters/fusion/addin
python setup_addin.py install
```
The server starts automatically when Fusion 360 launches (localhost:9879).

**SolidWorks** (run with SolidWorks open):
```python
from morphe.adapters.solidworks import start_server
start_server()  # localhost:9878
```

**Inventor** (run with Inventor open):
```python
from morphe.adapters.inventor import start_server
start_server()  # localhost:9877
```

### Using Clients

All clients share the same interface:

```python
from morphe.adapters.freecad import FreeCADClient
# or: from morphe.adapters.fusion import FusionClient
# or: from morphe.adapters.solidworks import SolidWorksClient
# or: from morphe.adapters.inventor import InventorClient

client = FreeCADClient()
if client.connect():
    # List available sketches
    sketches = client.list_sketches()

    # Export a sketch to canonical format
    sketch = client.export_sketch("Sketch")

    # Import a sketch into the CAD application
    client.import_sketch(my_sketch, name="NewSketch")

    # Get solver status
    status, dof = client.get_solver_status("Sketch")
```

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
3. Add and run the test script from `morphe/adapters/fusion/script/MorpheTests/`

**SolidWorks adapter tests (requires SolidWorks on Windows):**
```bash
pytest tests/test_solidworks_roundtrip.py -v
```

The SolidWorks test suite includes 80 tests covering:
- Basic and complex geometry primitives
- All constraint types including symmetric constraints
- Solver status detection
- Precision and edge cases
- Arc variations and tangent constraints

**Inventor adapter tests (requires Inventor on Windows):**
```bash
pytest tests/test_inventor_roundtrip.py -v
```

Tests cover all primitives, constraints, solver status, and edge cases.

## License

MIT - see [LICENSE](LICENSE) for details.
