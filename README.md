# Canonical Sketch

A CAD-agnostic 2D sketch geometry and constraint representation with adapter support for FreeCAD.

## Overview

This project provides:

- **`sketch_canonical`**: Platform-independent schema for 2D sketch geometry and constraints
- **`sketch_adapter_freecad`**: Adapter for FreeCAD's Sketcher workbench

The canonical format enables sketches to be stored, transferred, and manipulated independently of any specific CAD system.

See [SPECIFICATION.md](SPECIFICATION.md) for the complete technical specification, including supported geometry types, constraints, JSON schema format, and platform-specific adapter details.

## Installation

```bash
pip install -e .
```

For FreeCAD integration, ensure FreeCAD is installed and accessible (via snap, package manager, or `PYTHONPATH`).

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

## Running Tests

```bash
pytest tests/                    # All tests
pytest tests/ --cov              # With coverage
pytest tests/test_freecad_roundtrip.py -v  # FreeCAD tests (requires FreeCAD)
```

## License

MIT - see [LICENSE](LICENSE) for details.
