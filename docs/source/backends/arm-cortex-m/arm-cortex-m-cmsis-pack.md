# CMSIS-Pack Integration

:::{note}
`PyTorch::ExecuTorch` is a **source pack**. It ships the ExecuTorch runtime and kernel
sources as CMSIS components rather than a prebuilt library, so `cbuild` compiles them with
your project's own toolchain and flags.
:::

The ExecuTorch CMSIS-Pack packages the runtime, the portable and quantized kernel libraries,
the CMSIS-NN accelerated Arm&reg; Cortex&reg;-M operators, and the Ethos&trade;-U delegate for use from a
[CMSIS project](https://open-cmsis-pack.github.io/Open-CMSIS-Pack-Spec/main/html/index.html).
Every operator is a separately selectable component, so an image contains only the
kernels the application's models use.

This page covers integrating the pack into a CMSIS-Toolbox project. For exporting a model to
`.pte`, see the [Tutorial](arm-cortex-m-overview.md#tutorial) on the backend overview page.

## Prerequisites

The pack is consumed like any other CMSIS-Pack, so it works from Arm Keil Studio for
VS&nbsp;Code, Keil MDK, or the [CMSIS-Toolbox](https://github.com/Open-CMSIS-Pack/cmsis-toolbox)
command line — or any other CMSIS-compliant toolkit.

If you do not already have a project, Arm's guide to
[creating a new solution](https://mdk-packs.github.io/vscode-cmsis-solution-docs/create_app.html)
walks through choosing a target board or device and adding software components from the
**Software Components** view. The rest of this page describes the same selections in
`*.csolution.yml` and `*.cproject.yml` terms, so they apply whichever toolkit you use.

In addition to the toolkit you need:

- **CMSIS-Toolbox 2.14.1 or later** — provides `cpackget`, `csolution` and `cbuild`.
- **`ARM::CMSIS` 6.0.0 or later** — the pack's only declared hard dependency.
- **A device family pack (DFP) for your target** — `ARM::Cortex_DFP` covers the generic
  Cortex-M devices.
- **`ARM::CMSIS-NN`** — only when selecting CMSIS-NN accelerated operators.
- **`ARM::ethos-u-core-driver`** — only when selecting the Ethos-U delegate.

Components are gated on the processor core rather than on a specific device, and accept
Arm Cortex-M0, M0+, M3, M4, M7, M23, M33, M35P, M52, M55 and M85, and the STAR-MC1 and
STAR-MC3 cores.

## Install the Pack

```bash
cpackget add PyTorch::ExecuTorch
```

## Add the Pack to a Project

Declare the pack in the `*.csolution.yml`, alongside CMSIS and a DFP for the device:

```yaml
solution:
  created-for: CMSIS-Toolbox@2.14.1
  cdefault:
  compiler: GCC

  packs:
    - pack: ARM::CMSIS@>=6.0.0
    - pack: ARM::Cortex_DFP@>=1.1.0
    - pack: PyTorch::ExecuTorch

  target-types:
    - type: ARMCM55
      device: ARM::ARMCM55

  build-types:
    - type: Debug
      debug: on
      optimize: debug

  projects:
    - project: ./my_project.cproject.yml
```

Then select components in the `*.cproject.yml`. The smallest configuration that links is the
core runtime plus a single operator:

```yaml
project:
  packs:
    - pack: PyTorch::ExecuTorch

  components:
    - component: ARM::CMSIS:CORE
    - component: ARM::Device:Startup&C Startup
    - component: Machine Learning:ExecuTorch:Runtime
    - component: Machine Learning:ExecuTorch:Kernel Utils
    - component: Machine Learning:ExecuTorch:Kernel Registration
    - component: Machine Learning:ExecuTorch Operators:Portable add

  groups:
    - group: App
      files:
        - file: ./main.cpp
```

:::{note}
Component identifiers use a **single** colon between `Cclass`, `Cgroup` and `Csub`, and a
space inside the `Csub` name — `Machine Learning:ExecuTorch Operators:Portable add`, not
`Machine Learning::ExecuTorch Operators::Portable::add`.
:::

The
[smoke project](https://github.com/pytorch/executorch/tree/main/backends/arm/cmsis_pack/test/smoke)
in the ExecuTorch repository builds this configuration plus one CMSIS-NN operator
(`Cortex-M quantized_add` with `ARM::CMSIS:NN Lib`), and is a usable starting point for a
new project.

## Core Components

| Component | Requires | Purpose |
|-----------|----------|---------|
| `Machine Learning:ExecuTorch:Runtime` | A Cortex-M core | Program loading, method execution, the operator registry and the bare-metal platform layer. Required by everything else. |
| `Machine Learning:ExecuTorch:Kernel Utils` | `Runtime` | Shared kernel helpers — broadcasting, dtype dispatch, shape inference. Required by every operator component. |
| `Machine Learning:ExecuTorch:Kernel Registration` | `Kernel Utils` | The generated registration table that binds the selected operators into the registry. |
| `Machine Learning:ExecuTorch:Extension Tensor` | `Runtime` | Optional. `TensorPtr` / `make_tensor_ptr` helpers for constructing tensors. |
| `Machine Learning:ExecuTorch:Backend EthosU` | An Ethos-U driver component | Optional. Ethos-U NPU delegate for a bare-metal Cortex-M host. |

`Runtime`, `Kernel Utils` and `Kernel Registration` are always required. Omitting
`Kernel Registration` still links, but no kernel is visible to the runtime and every method
fails to load.

## Operator Components

Kernels are exposed one component per operator, under the `ExecuTorch Operators` group, in
three families:

| Family | Example | Description |
|--------|---------|-------------|
| `Portable <op>` | `Machine Learning:ExecuTorch Operators:Portable add` | Reference C++ implementations covering the full portable operator set. |
| `Quantized <op>` | `Machine Learning:ExecuTorch Operators:Quantized dequantize` | Quantization primitives and quantized embedding / matmul kernels. |
| `Cortex-M <op>` | `Machine Learning:ExecuTorch Operators:Cortex-M quantized_conv2d` | CMSIS-NN accelerated INT8 kernels. |

Selecting an operator component defines a matching `RTE_ML_EXECUTORCH_OP_<FAMILY>_<NAME>`
symbol in the generated `RTE_Components.h`. The registration source contributed by
`Kernel Registration` is guarded on those symbols, so an unselected operator is neither
compiled nor registered and costs nothing in flash.

Select the operators your exported model uses. A model whose graph references an operator
whose component is not selected loads no method and reports a missing-kernel error at
runtime.

### Listing the Components a Model Needs

The exported program carries the exact set of operators the runtime will look up. Add the
following to the script that produces the `.pte`, where `executorch_program` is the result of
`to_executorch()`, and it prints a `components:` block ready to paste into the
`*.cproject.yml`:

```python
FAMILIES = {
    "aten": "Portable", "dim_order_ops": "Portable", "et_copy": "Portable",
    "quantized_decomposed": "Quantized", "cortex_m": "Cortex-M",
}
QUANTIZED_ALIASES = {  # several quantized_decomposed operators share one component
    "quantize_per_tensor": "quantize", "quantize_per_channel": "quantize",
    "quantize_per_token": "quantize", "dequantize_per_tensor": "dequantize",
    "dequantize_per_channel": "dequantize", "dequantize_per_token": "dequantize",
    "choose_qparams_per_token_asymmetric": "choose_qparams",
    "embedding_byte": "embedding", "embedding_2bit": "embedding2b",
    "embedding_4bit": "embedding4b",
}
PORTABLE_ALIASES = {  # operators implemented by a differently named kernel file
    "native_batch_norm_legit": "native_batch_norm",
    "native_batch_norm_legit_no_training": "native_batch_norm",
    "d2h_copy": "device_copy", "h2d_copy": "device_copy",
}
ABBREVIATIONS = {  # names the 32-character component-name limit shortens
    "max_pool2d_with_indices_backward": "max_pool2d_indices_bwd",
    "quantized_depthwise_conv2d": "quantized_dw_conv2d",
    "quantized_transpose_conv2d": "quantized_tr_conv2d",
}


def component_for(operator_name):
    namespace, _, name = operator_name.partition("::")
    family = FAMILIES[namespace]
    name = name.strip("_")  # aten::_softmax -> softmax, aten::copy_ -> copy
    if family == "Quantized":
        name = QUANTIZED_ALIASES.get(name, name)
    elif family == "Cortex-M":
        name = name.removesuffix("_nhwc")  # explicit-layout kernels share the component
    else:
        name = PORTABLE_ALIASES.get(name, name)
    name = ABBREVIATIONS.get(name, name)
    return f"Machine Learning:ExecuTorch Operators:{family} {name}"


operators = set()
for plan in executorch_program.executorch_program.execution_plan:
    for operator in plan.operators:
        operators.add(component_for(operator.name))

print("  components:")
for component in [
    "Machine Learning:ExecuTorch:Runtime",
    "Machine Learning:ExecuTorch:Kernel Utils",
    "Machine Learning:ExecuTorch:Kernel Registration",
] + sorted(operators):
    print(f"    - component: {component}")
```

For an INT8 quantized linear model this prints:

```yaml
  components:
    - component: Machine Learning:ExecuTorch:Runtime
    - component: Machine Learning:ExecuTorch:Kernel Utils
    - component: Machine Learning:ExecuTorch:Kernel Registration
    - component: Machine Learning:ExecuTorch Operators:Portable addmm
    - component: Machine Learning:ExecuTorch Operators:Portable permute_copy
    - component: Machine Learning:ExecuTorch Operators:Portable relu
    - component: Machine Learning:ExecuTorch Operators:Quantized dequantize
    - component: Machine Learning:ExecuTorch Operators:Quantized quantize
```

A model that is fully delegated to Ethos-U lists only the quantize and dequantize operators
at the delegate boundary, because the delegated subgraph is executed by the backend rather
than by registered kernels. Add `Machine Learning:ExecuTorch:Backend EthosU` alongside them.

If you only have the `.pte`, the same operator names can be read out of it — they are stored
as plain strings — and mapped by hand using the rules above:

```bash
strings model.pte | grep -oE '(aten|dim_order_ops|et_copy|quantized_decomposed|cortex_m)::[a-z0-9_]+' | sort -u
```

### Quantized Operators

`Quantized add`, `Quantized choose_qparams`, `Quantized dequantize`, `Quantized embedding`,
`Quantized embedding2b`, `Quantized embedding4b`, `Quantized mixed_linear`,
`Quantized mixed_mm`, `Quantized quantize`.

### CMSIS-NN Accelerated Operators

These are the kernels the Cortex-M backend lowers to. See
[CMSIS-NN Supported Operators](arm-cortex-m-overview.md#cmsis-nn-supported-operators) for the
ATen operators each one implements.

`Cortex-M dequantize_per_tensor`, `Cortex-M maximum`, `Cortex-M minimum`, `Cortex-M pad`,
`Cortex-M quantize_per_tensor`, `Cortex-M quantized_activation`, `Cortex-M quantized_add`,
`Cortex-M quantized_avg_pool2d`, `Cortex-M quantized_batch_matmul`,
`Cortex-M quantized_conv2d`, `Cortex-M quantized_div`, `Cortex-M quantized_dw_conv2d`
(depthwise), `Cortex-M quantized_linear`, `Cortex-M quantized_max_pool2d`,
`Cortex-M quantized_mul`,
`Cortex-M quantized_tr_conv2d` (transpose), `Cortex-M softmax`, `Cortex-M transpose`.
Component names are capped at 32 characters by the CMSIS-Pack schema, hence the
abbreviated depthwise/transpose convolution names. The experimental explicit-layout
`cortex_m::*_nhwc` kernels are built into the same components as their channels-last
counterparts.

All of them except `quantize_per_tensor` and `dequantize_per_tensor` call into CMSIS-NN, so
they additionally require the `ARM::CMSIS-NN` pack and its `ARM::CMSIS:NN Lib` component:

```yaml
  packs:
    - pack: ARM::CMSIS-NN

  components:
    - component: ARM::CMSIS:NN Lib
    - component: Machine Learning:ExecuTorch Operators:Cortex-M quantized_conv2d
    - component: Machine Learning:ExecuTorch Operators:Cortex-M quantized_linear
```

:::{note}
`ARM::CMSIS-NN` 8.0.0 does not compile in its default configuration for cores with Helium
floating point (Cortex-M55, Cortex-M85): its element-wise float sources fail with
`implicit declaration of function 'arm_nn_clamp_mve_f32'`. Until a fixed release is
available, add to the `*.csolution.yml` or `*.cproject.yml` of any project that selects
CMSIS-NN accelerated operators on such a core:

```yaml
  define:
    - ARM_NN_ENABLE_F32: 1
    - ARM_NN_ENABLE_F16: 1
```
:::

Selecting one without `ARM::CMSIS:NN Lib` fails component resolution during
`csolution convert`, not at link time.

### Ethos-U Delegate

`Machine Learning:ExecuTorch:Backend EthosU` runs Ethos-U delegated subgraphs from a
bare-metal Cortex-M host. It requires an NPU driver component supplied by the
`ARM::ethos-u-core-driver` pack, and a `.pte` whose subgraphs were compiled for the matching
NPU configuration. See the {doc}`/backends/arm-ethos-u/arm-ethos-u-overview` for the
ahead-of-time side of that flow.

## Using the Runtime

The pack contributes an include directory, so runtime headers resolve as they do in a CMake
build:

```cpp
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/platform/runtime.h>
```

See {doc}`/running-a-model-cpp-tutorial` for the load-and-execute API itself. Two differences
apply on bare metal:

- Call `executorch::runtime::runtime_init()` before loading a program.
- The pack does not ship `extension/module`. Load the program with the header-only
  `executorch::extension::BufferDataLoader` (`extension/data_loader` is on the include path)
  over the buffer holding the model, and back the `MemoryManager` with statically allocated
  pools rather than the heap.

## Providing the Model

The pack contains no ahead-of-time tooling. Produce the `.pte` with the normal ExecuTorch
export flow described in the [Tutorial](arm-cortex-m-overview.md#tutorial), using
`ExecutorchBackendConfig(extract_delegate_segments=False)` so the program is self-contained
and can be loaded straight from flash.

With no filesystem on the target, the `.pte` is normally linked into the image. Either
convert it to a C array in a generated header, or place it in `.rodata` with a short
assembler file:

```asm
    .section .rodata.model,"a",%progbits
    .balign 16
    .global model_pte_start
model_pte_start:
    .incbin "model.pte"
    .global model_pte_end
model_pte_end:
```

:::{note}
`add-path` applies to C and C++ only. An assembler file using a relative `.incbin` path also
needs the directory on `add-path-asm`, otherwise the path resolves against the build working
directory:

```yaml
  add-path-asm:
    - .
```
:::

## Toolchain Support

The pack is built and validated with `arm-none-eabi-gcc` 14.3 and Arm Compiler for Embedded 6
(6.23) on Cortex-M85; only the GCC build runs in the pack's CI. Arm Compiler 6 projects must
add `-ffp-mode=full` to the C/C++ options: armclang otherwise assumes finite math, and
operators that rely on IEEE inf/NaN behaviour (`isinf`, `isnan`, `gelu`) miscompute. The pack
ships a `sys/types.h` shim used only under Arm Compiler 6. LLVM Embedded Toolchain for Arm
is accepted by the component conditions but has not been exercised.

## Reference

- [backends/arm/cmsis_pack/](https://github.com/pytorch/executorch/tree/main/backends/arm/cmsis_pack) — pack sources, and instructions for building the pack locally
- [Open-CMSIS-Pack specification](https://open-cmsis-pack.github.io/Open-CMSIS-Pack-Spec/main/html/index.html)
- [CMSIS-Toolbox documentation](https://open-cmsis-pack.github.io/cmsis-toolbox/)
- {doc}`/backends/arm-cortex-m/arm-cortex-m-overview` — backend overview, quantization and export
- {doc}`/backends/arm-ethos-u/arm-ethos-u-overview` — Ethos-U backend

```
Arm is a registered trademark of Arm Limited (or its subsidiaries or affiliates).
```
