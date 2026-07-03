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



def _get_number_of_outputs(outputs) -> int:
    if isinstance(outputs, torch.Tensor):
        return 1
    elif isinstance(outputs, (tuple, list)):
        return len(outputs)
    else:
        raise TypeError(f"unsupported output type {type(outputs)}")


def _mk_metadata(inputs: tuple, outputs: tuple | torch.Tensor, atol: float, rtol: float) -> dict:
    metadata = {}
    metadata["nb_inputs"] = len(inputs)
    metadata["nb_outputs"] = _get_number_of_outputs(outputs)
    metadata["atol"] = atol
    metadata["rtol"] = rtol
    
    channel_last = False 
    # We assume that when one input tensor is channel_last, all others too
    # This assumption is true for the operators tested
    for i, t in enumerate(inputs):
        metadata[f"input_{i}"] = t
        if t.is_contiguous(memory_format=torch.channels_last):
            channel_last = True
    if isinstance(outputs, torch.Tensor):
        metadata["output_0"] = outputs
    else:
        for i,t in enumerate(outputs):
           metadata[f"output_{i}"] = t

    # Input / outputs are exported as channel_first.
    # So we need to store the memory format information in the metadata so that
    # we can change the input / output tensors to channel_last in the C++
    # tests.
    # It is a workaround for what looks like a bug in the to_edge_transform_and_lower 
    # function, which does not preserve the memory format of the input / output tensors
    # exported as constant_methods.
    metadata["channel_last"] = channel_last
    
    return metadata

def _export_portable(model: torch.nn.Module, inputs: tuple,recipe: op_recipes.Recipe, display_quantized_values: bool = False, display_metadata: bool = False) -> bytes:
    from executorch.exir import EdgeCompileConfig, to_edge

    model = model.eval()
    exported = torch.export.export(model, inputs, strict=True)
    
    expected = model(*inputs)
    metadata = _mk_metadata(inputs, expected,recipe.atol, recipe.rtol)
    if display_metadata:
        print(metadata)
    # The pack ships the full portable op set, including ops outside the Core
    # ATen opset (bitwise shifts, unfold, var_mean.correction, ...), so skip the
    # core-ATen IR validity gate; to_executorch still lowers to the .out kernels.
    edge = to_edge(exported, compile_config=EdgeCompileConfig(_check_ir_validity=False),
                   constant_methods=metadata)
    program = edge.to_executorch()
    return bytes(program.buffer)

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

def _export_cortex_m(model: torch.nn.Module, inputs: tuple, recipe: op_recipes.Recipe, display_quantized_values: bool = False, display_metadata: bool = False) -> bytes:
    from debug_cortex_m.passes.cortex_m_pass_manager import (
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
    move_exported_model_to_eval(quantized)
    actual = quantized(*inputs)
    if display_quantized_values:
        print("=== quantized values ===")
        print("Expected:", expected,expected.shape)
        print("Actual:", actual,actual.shape)
    atol,rtol = _compute_test_threshold(actual, expected)
    
    metadata = _mk_metadata(inputs, expected,atol, rtol)
    if display_metadata:
       print(metadata)
    
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
                    ),
        constant_methods=metadata
    )
    edge._edge_programs["forward"] = CortexMPassManager(
        edge.exported_program()
    ).transform()
    return bytes(edge.to_executorch().buffer)


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
    parser.add_argument(
        "--display-metadata",
        action="store_true",
        help="display metadata during export",
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
        #if component.name != "quantized_max_pool2d" and component.name != "quantized_conv2d" and component.name != "quantized_avg_pool2d":
        #    skipped.append((component.category, component.name, "For debug"))
        #    continue

        if key in op_recipes.SKIPS:
            skipped.append((component.category, component.name, op_recipes.SKIPS[key]))
            continue
        recipe = op_recipes.RECIPES[key]
        cat_id = component.category.lower().replace("-", "_")
        op_name = out_dir / f"{cat_id}__{component.name}"
        op_pte = op_name.with_suffix(".pte")
        try:
            if args.display_quantized_values or args.display_metadata:
               print(f"=== exporting {component.category}/{component.name} ===")
            model, inputs = recipe.make()
            
            pte = exporters[component.category](model, inputs, recipe,display_quantized_values=args.display_quantized_values,display_metadata=args.display_metadata)
            op_pte.write_bytes(pte)
            
            manifest.append(
                {
                    "op": component.name,
                    "category": component.category,
                    "name": op_name.name
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
