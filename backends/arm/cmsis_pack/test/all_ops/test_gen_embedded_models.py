# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Host tests for the embedded-models generator.

Verify that gen_embedded_models.build() emits a well-formed stub when no
manifest is present (build/link coverage), and correct .incbin / table sources
with project-relative paths when models are present (so the same generated file
resolves on the host and inside the Docker build container).

"""

import json
import sys
from pathlib import Path

ALL_OPS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ALL_OPS_DIR))

import gen_embedded_models  # type: ignore[import-not-found]  # noqa: E402


def test_stub_when_no_manifest(tmp_path):
    blob_s, cpp, header, n_models, n_bytes = gen_embedded_models.build(
        tmp_path / "models", tmp_path
    )
    assert n_models == 0 and n_bytes == 0
    assert ".incbin" not in blob_s
    assert "g_embedded_models_count = 0" in cpp
    # Well-formed: a 1-element dummy array (C++ has no zero-length arrays).
    assert "g_embedded_models[]" in cpp
    assert "struct EmbeddedModel" in header


def _make_models(root: Path) -> Path:
    """Two ops, one with a single input, one with two inputs."""
    models = root / "models"
    a = models / "portable__abs"
    a.mkdir(parents=True)
    (a / "model.pte").write_bytes(b"\x01\x02\x03\x04")
    (a / "input_0.bin").write_bytes(b"\x00" * 8)
    (a / "expected_0.bin").write_bytes(b"\x00" * 8)
    b = models / "portable__add"
    b.mkdir(parents=True)
    (b / "model.pte").write_bytes(b"\x05\x06")
    (b / "input_0.bin").write_bytes(b"\x00" * 4)
    (b / "input_1.bin").write_bytes(b"\x00" * 4)
    (b / "expected_0.bin").write_bytes(b"\x00" * 4)
    manifest = [
        {
            "op": "abs",
            "dir": "portable__abs",
            "atol": 0.001,
            "rtol": 0.002,
            "inputs": [{"file": "input_0.bin"}],
            "outputs": [{"file": "expected_0.bin"}],
        },
        {
            "op": "add",
            "dir": "portable__add",
            "atol": 0.001,
            "rtol": 0.001,
            "inputs": [{"file": "input_0.bin"}, {"file": "input_1.bin"}],
            "outputs": [{"file": "expected_0.bin"}],
        },
    ]
    (models / "manifest.json").write_text(json.dumps(manifest))
    return models


def test_real_models_emit_relative_incbin(tmp_path):
    models = _make_models(tmp_path)
    blob_s, cpp, header, n_models, n_bytes = gen_embedded_models.build(models, tmp_path)

    assert n_models == 2
    assert n_bytes == (4 + 8 + 8) + (2 + 4 + 4 + 4)
    # Paths are project-relative (resolve via -I<project-dir>), never absolute.
    assert '.incbin "models/portable__abs/model.pte"' in blob_s
    assert '.incbin "models/portable__add/input_1.bin"' in blob_s
    assert "/tmp" not in blob_s and str(tmp_path) not in blob_s  # nosec B108
    # Table carries op name + tolerances and both ops.
    assert '"abs", "portable__abs"' in cpp
    assert "0.002f" in cpp
    assert "kInputs_portable__add, 2" in cpp
    assert cpp.count("g_embedded_models[]") == 1


def test_count_is_sizeof_based(tmp_path):
    _, cpp, _, _, _ = gen_embedded_models.build(_make_models(tmp_path), tmp_path)
    assert (
        "g_embedded_models_count = sizeof(g_embedded_models) / "
        "sizeof(g_embedded_models[0])" in cpp
    )
