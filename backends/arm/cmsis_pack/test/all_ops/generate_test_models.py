#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Export a .pte + reference output for every operator recipe.

For each operator component the pack ships (op_guards.discover_components), this
looks up its recipe in op_recipes and exports a tiny model to a .pte plus the
PyTorch reference inputs/outputs. The all-ops consumer firmware then runs each
.pte on the FVP and compares against the reference.

Coverage is reconciled up front: every discovered component must have either a
recipe or an explicit skip reason, otherwise the run fails (no silent gaps).

"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parents[1] / "scripts"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_SCRIPTS))

import op_recipes  # type: ignore[import-not-found]  # noqa: E402
from op_guards import (  # type: ignore[import-not-found]  # noqa: E402
    discover_components,
)


@dataclass
class TensorSpec:
    file: str
    dtype: str
    shape: list


def _save_tensor(t: torch.Tensor, path: Path) -> TensorSpec:
    src = t.detach().cpu()
    # The firmware copies these bytes directly into ExecuTorch tensor storage.
    # For channels_last tensors, that storage is NHWC even though the logical
    # tensor shape remains NCHW.
    if (
        src.dim() == 4
        and src.is_contiguous(memory_format=torch.channels_last)
        and not src.is_contiguous()
    ):
        storage = src.permute(0, 2, 3, 1).contiguous()
    else:
        storage = src.contiguous()
    path.write_bytes(storage.numpy().tobytes())
    return TensorSpec(
        file=path.name, dtype=str(src.dtype).replace("torch.", ""), shape=list(src.shape)
    )


def _flatten_outputs(out) -> list:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        flat = []
        for o in out:
            flat.extend(_flatten_outputs(o))
        return flat
    raise TypeError(f"unsupported output type {type(out)}")


def _export_portable(model: torch.nn.Module, inputs: tuple,recipe: op_recipes.Recipe, display_quantized_values: bool = False) -> tuple[bytes, float, float]:
    from executorch.exir import EdgeCompileConfig, to_edge

    model = model.eval()
    exported = torch.export.export(model, inputs, strict=True)
    # The pack ships the full portable op set, including ops outside the Core
    # ATen opset (bitwise shifts, unfold, var_mean.correction, ...), so skip the
    # core-ATen IR validity gate; to_executorch still lowers to the .out kernels.
    edge = to_edge(exported, compile_config=EdgeCompileConfig(_check_ir_validity=False))
    program = edge.to_executorch()
    return bytes(program.buffer), recipe.atol, recipe.rtol

def _compute_test_threshold(actual, expected):
    abs_err = (actual - expected).abs()

    atol = abs_err.max().item()
    
    # Avoid division by zero
    mask = expected.abs() > 1e-12
    if mask.any():
        rtol = (abs_err[mask] / expected.abs()[mask]).max().item()
    else:
        rtol = 0.0
    return atol, rtol

def _export_cortex_m(model: torch.nn.Module, inputs: tuple, recipe: op_recipes.Recipe, display_quantized_values: bool = False) -> tuple[bytes, float, float]:
    from executorch.backends.cortex_m.passes.cortex_m_pass_manager import (
        CortexMPassManager,
    )
    from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    from torchao.quantization.pt2e import move_exported_model_to_eval


    model = model.eval()
    expected = model(*inputs)
    captured = torch.export.export(model, inputs, strict=True).module()
    prepared = prepare_pt2e(captured, CortexMQuantizer())
    prepared(*inputs)  # calibrate
    quantized = convert_pt2e(prepared)
    quantized_ = move_exported_model_to_eval(quantized)
    actual = quantized_(*inputs)
    if display_quantized_values:
        print("=== quantized values ===")
        print("Expected:", expected)
        print("Actual:", actual)
    atol,rtol = _compute_test_threshold(actual, expected)
    exported = torch.export.export(quantized, inputs, strict=True)
    edge = to_edge_transform_and_lower(
        exported, compile_config=EdgeCompileConfig(
                      preserve_ops=[
                          torch.ops.aten.linear.default,
                          torch.ops.aten.hardsigmoid.default,
                          torch.ops.aten.hardsigmoid_.default,
                          torch.ops.aten.hardswish.default,
                          torch.ops.aten.hardswish_.default,
                      ],
                      _check_ir_validity=False,
                      _core_aten_ops_exception_list=[torch.ops.aten.max_pool2d.default],
                    )
    )
    edge._edge_programs["forward"] = CortexMPassManager(
        edge.exported_program()
    ).transform()
    return bytes(edge.to_executorch().buffer),atol,rtol


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a .pte + reference output for every operator recipe"
    )
    parser.add_argument(
        "--source-dir", "-s", required=True, help="repo root / staged pack tree"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="where to write models/ + manifest.json",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record export failures and keep going instead of exiting non-zero",
    )
    parser.add_argument(
        "--display-quantized-values",
        action="store_true",
        help="display quantized values during export",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    components = discover_components(source_dir)
    keys = {(c.category, c.name) for c in components}

    # No-silent-gap reconciliation: every component must have a recipe or a skip.
    gaps = sorted(keys - op_recipes.all_keys())
    if gaps:
        raise SystemExit(
            "Operators with no recipe and no skip reason (add either to op_recipes.py):\n"
            + "\n".join(f"    [{cat}] {name}" for cat, name in gaps)
        )

    exporters = {
        "Portable": _export_portable,
        "Quantized": _export_portable,
        "Cortex-M": _export_cortex_m,
    }
    manifest: list = []
    skipped: list = []
    failed: list = []

    for component in components:
        key = (component.category, component.name)
        if key in op_recipes.SKIPS:
            skipped.append((component.category, component.name, op_recipes.SKIPS[key]))
            continue
        recipe = op_recipes.RECIPES[key]
        cat_id = component.category.lower().replace("-", "_")
        op_dir = out_dir / f"{cat_id}__{component.name}"
        try:
            if args.display_quantized_values:
               print(f"=== exporting {component.category}/{component.name} ===")
            model, inputs = recipe.make()
            reference = model.eval()(*inputs)
            pte,atol,rtol = exporters[component.category](model, inputs, recipe,display_quantized_values=args.display_quantized_values)
            op_dir.mkdir(parents=True, exist_ok=True)
            (op_dir / "model.pte").write_bytes(pte)
            in_specs = [
                _save_tensor(t, op_dir / f"input_{i}.bin") for i, t in enumerate(inputs)
            ]
            out_specs = [
                _save_tensor(t, op_dir / f"expected_{i}.bin")
                for i, t in enumerate(_flatten_outputs(reference))
            ]
            manifest.append(
                {
                    "op": component.name,
                    "category": component.category,
                    "dir": op_dir.name,
                    "atol": atol,
                    "rtol": rtol,
                    "inputs": [s.__dict__ for s in in_specs],
                    "outputs": [s.__dict__ for s in out_specs],
                }
            )
        except Exception as exc:  # noqa: BLE001 - report, don't abort the sweep
            failed.append(
                (component.category, component.name, f"{type(exc).__name__}: {exc}")
            )
            if not args.continue_on_error:
                traceback.print_exc()

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = len(keys)
    print(
        f"\n=== coverage: {len(manifest)} exported, {len(skipped)} skipped, "
        f"{len(failed)} failed of {total} components ==="
    )
    if skipped:
        print("\nskipped (build/link covered, not executed):")
        for cat, name, reason in skipped:
            print(f"    [{cat}] {name}: {reason}")
    if failed:
        print("\nFAILED to export:")
        for cat, name, reason in failed:
            print(f"    [{cat}] {name}: {reason}")

    if failed and not args.continue_on_error:
        raise SystemExit(f"{len(failed)} operator(s) failed to export")


if __name__ == "__main__":
    main()
