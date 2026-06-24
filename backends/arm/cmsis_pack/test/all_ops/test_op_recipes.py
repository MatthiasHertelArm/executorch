# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Host tests for the all-ops execution recipes and export engine.

The key invariant is that every operator component the pack ships has either an
execution recipe or a documented skip reason -- no silent coverage gaps. A few
representative exports also confirm the engine produces loadable .pte files.

"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
ALL_OPS_DIR = Path(__file__).resolve().parent
SCRIPTS = REPO_ROOT / "backends" / "arm" / "cmsis_pack" / "scripts"
sys.path.insert(0, str(ALL_OPS_DIR))
sys.path.insert(0, str(SCRIPTS))

import op_recipes  # type: ignore[import-not-found]  # noqa: E402
from op_guards import (  # type: ignore[import-not-found]  # noqa: E402
    discover_components,
)


def _component_keys():
    if not (REPO_ROOT / "kernels" / "portable" / "cpu").is_dir():
        pytest.skip("ExecuTorch kernel sources not available")
    return {(c.category, c.name) for c in discover_components(REPO_ROOT)}


def test_no_silent_coverage_gaps():
    """Every shipped operator must have a recipe or an explicit skip reason."""
    gaps = sorted(_component_keys() - op_recipes.all_keys())
    assert not gaps, f"operators with neither recipe nor skip: {gaps}"


def test_no_stale_recipes():
    """Every recipe / skip must correspond to a real shipped component."""
    stale = sorted(op_recipes.all_keys() - _component_keys())
    assert not stale, f"recipes/skips for non-existent components: {stale}"


def test_skips_have_reasons():
    assert all(reason for reason in op_recipes.SKIPS.values())


@pytest.mark.parametrize("op", ["add", "native_layer_norm", "cat", "max"])
def test_representative_portable_exports(op):
    """The engine produces a non-empty, runnable .pte for varied op shapes
    (binary, structured, multi-input, multi-output).
    """
    import generate_test_models as gen  # type: ignore[import-not-found]

    model, inputs = op_recipes.RECIPES[("Portable", op)].make()
    reference = gen._flatten_outputs(model.eval()(*inputs))
    assert reference
    pte = gen._export_portable(model, inputs)
    assert isinstance(pte, bytes) and len(pte) > 0

    from executorch.runtime import Runtime

    program = Runtime.get().load_program(pte)
    program.load_method("forward")  # raises if the kernels/signatures are wrong
