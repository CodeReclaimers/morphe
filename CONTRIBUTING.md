# Contributing to Canonical Sketch

Thank you for your interest in contributing to Canonical Sketch! This document provides guidelines and information for contributors.

## Ways to Contribute

- **Add CAD adapters**: Implement adapters for other CAD systems (Onshape, CATIA, etc.)
- **Improve existing adapters**: Enhance FreeCAD, Fusion 360, SolidWorks, or Inventor adapters
- **Improve test coverage**: Add tests for edge cases and new functionality
- **Report bugs**: Open an issue with a clear description and reproduction steps
- **Suggest features**: Propose new geometry types, constraints, or capabilities
- **Improve documentation**: Fix typos, clarify explanations, or add examples

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/codereclaimers/canonical_sketch.git
   cd canonical_sketch
   ```

2. Install in development mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Run the tests:
   ```bash
   pytest tests/
   ```

## Code Style

This project uses automated tools to maintain consistent code style:

- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **Mypy** for type checking

Before submitting a pull request, please run:

```bash
black .
ruff check .
mypy sketch_canonical
```

## Adding a New CAD Adapter

All adapters are implemented in Python. To add support for a new CAD system:

1. Create a new package: `sketch_adapter_<cadname>/`
2. Implement the `SketchBackendAdapter` abstract base class from `sketch_adapter_common`
3. Add tests in `tests/test_<cadname>_adapter.py` or within the adapter package
4. Update the README to mention the new adapter

See `sketch_adapter_freecad/` for a reference implementation. For Windows COM-based CAD systems (like SolidWorks or Inventor), see `sketch_adapter_solidworks/` for patterns using `pywin32`.

## Pull Request Guidelines

- Keep changes focused and atomic
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass before submitting

## Questions?

Open an issue on GitHub if you have questions or need help getting started.
