"""Pin-test the conformance corpus on the Python side.

The corpus under tests/conformance/*.json is the cross-language wire-
format pin between the Python and C++ Morphe implementations. This file
verifies, on the Python side:

  1. Every fixture parses cleanly via load_sketch.
  2. save_sketch(load_sketch(f)) == on-disk text — byte-for-byte. This
     guarantees the corpus is a fixed point of the Python encoder, so
     any drift in encoder behavior (e.g., a future change that adds an
     emit-when-default field) breaks CI immediately.
  3. The generator's --check mode reports no drift (the corpus on disk
     matches what generate_corpus.py would produce right now).

The C++ side has its own conformance test in cpp/tests/test_conformance.cpp.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from morphe.serialization import load_sketch, sketch_to_json

CORPUS_DIR = Path(__file__).resolve().parent / "conformance"
GENERATOR = CORPUS_DIR / "generate_corpus.py"

FIXTURES = sorted(p.name for p in CORPUS_DIR.glob("*.json"))


def test_corpus_directory_is_populated():
    assert len(FIXTURES) > 0, "no fixtures found in tests/conformance/"


@pytest.mark.parametrize("fixture", FIXTURES)
def test_fixture_loads(fixture: str):
    path = CORPUS_DIR / fixture
    sketch = load_sketch(str(path))
    assert sketch.name  # any non-empty name


@pytest.mark.parametrize("fixture", FIXTURES)
def test_fixture_is_a_fixed_point_of_the_encoder(fixture: str):
    """save(load(f)) == f, byte-for-byte. Catches encoder drift."""
    path = CORPUS_DIR / fixture
    on_disk = path.read_text()
    sketch = load_sketch(str(path))
    re_emitted = sketch_to_json(sketch, indent=2) + "\n"
    assert re_emitted == on_disk, (
        f"{fixture} drifted from canonical encoding "
        f"(check encoder behavior or regenerate corpus)"
    )


@pytest.mark.parametrize("fixture", FIXTURES)
def test_fixture_is_valid_json(fixture: str):
    """Sanity check: each file must parse as JSON irrespective of schema."""
    path = CORPUS_DIR / fixture
    json.loads(path.read_text())


def test_generator_check_mode_reports_no_drift():
    """`generate_corpus.py --check` must succeed against the on-disk corpus.

    This is what CI runs as a guard — any future change that alters the
    Python encoder, modifies a fixture builder, or adds a fixture without
    regenerating the JSON files will fail here.
    """
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"generate_corpus.py --check exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
