# ExecuTorch CMSIS Pack

## Overview

ExecuTorch is the PyTorch Edge Runtime, enabling efficient on-device AI inference. This pack is the Cortex-M / bare-metal distribution.

This pack provides:
- **Core Runtime**: Program loading and execution
- **Portable Operators**: Platform-independent operator implementations
- **Quantized Operators**: Optimized quantized inference
- **Ethos-U Backend**: Hardware acceleration for Arm Ethos-U NPU
- **Cortex-M Operators**: CMSIS-NN optimized INT8 kernels

## Getting Started

### Basic Usage

1. Add the pack to your csolution:
   ```yaml
   packs:
     - pack: PyTorch::ExecuTorch
   ```

2. Add required components. CMSIS component IDs use a single colon
   between Cclass / Cgroup / Csub, with a space (not "::") inside the
   Csub name.
   ```yaml
   components:
     # Core runtime (always required)
     - component: Machine Learning:ExecuTorch:Runtime
     - component: Machine Learning:ExecuTorch:Kernel Utils
     - component: Machine Learning:ExecuTorch:Kernel Registration

     # Optional: TensorPtr / make_tensor_ptr helpers
     - component: Machine Learning:ExecuTorch:Extension Tensor

     # Optional: Ethos-U NPU delegate. Also requires an NPU driver
     # component from the ARM::ethos-u-core-driver pack.
     - component: Machine Learning:ExecuTorch:Backend EthosU
   ```

   CMSIS-NN acceleration is not a backend component: select the
   individual `Cortex-M <op>` operator components (see below). All of
   them except `quantize_per_tensor` and `dequantize_per_tensor`
   additionally require `ARM::CMSIS:NN Lib`.

3. Include ExecuTorch headers in your code:
   ```cpp
   #include <executorch/runtime/executor/program.h>
   #include <executorch/runtime/executor/method.h>
   ```

### Model Integration

Operators are included as individual CMSIS components, one per
`op_*.cpp`. Add the ones your model uses:

```yaml
components:
  - component: Machine Learning:ExecuTorch Operators:Portable add
  - component: Machine Learning:ExecuTorch Operators:Portable mul
  # ... other operators used by your model
```

You can also generate this list automatically: a Model Pack produced
from your `.pte` file declares dependencies on the operators the model
needs, and the toolchain resolves them against this pack's components.

### Renamed Components

The CMSIS-Pack schema limits component names to 32 characters. Since
pack version 1.4.1 three operator components therefore carry shortened
names; projects written against the 1.4.0 pack must update them:

| Up to 1.4.0 | From 1.4.1 |
|-------------|------------|
| `Portable max_pool2d_with_indices_backward` | `Portable max_pool2d_indices_bwd` |
| `Cortex-M quantized_depthwise_conv2d` | `Cortex-M quantized_dw_conv2d` |
| `Cortex-M quantized_transpose_conv2d` | `Cortex-M quantized_tr_conv2d` |

## Memory Requirements

| Component | Flash (approx) | RAM (approx) |
|-----------|----------------|--------------|
| Runtime | 50-100 KB | 4-8 KB |
| Per Operator | 1-10 KB | minimal |
| Ethos-U Backend | 20-40 KB | 2-4 KB |

Actual requirements depend on:
- Selected operators
- Model complexity
- Tensor sizes

## Compiler Support

Tested with `arm-none-eabi-gcc` 14.3 and Arm Compiler for Embedded (AC6)
6.23 on Cortex-M85. LLVM Embedded Toolchain for Arm is expected to work
but was not exercised for this release.

AC6 projects must build with `-ffp-mode=full`. armclang otherwise assumes
finite math, which breaks operators that depend on IEEE inf/NaN
behaviour (`isinf`, `isnan`, `gelu`).

## Dependencies

- ARM::CMSIS (core headers)
- ARM::CMSIS-NN (for the `Cortex-M <op>` CMSIS-NN accelerated operators)
- ARM::ethos-u-core-driver (for Ethos-U backend)

### CMSIS-NN 8.0.0 on Cortex-M55 and Cortex-M85

The `ARM::CMSIS-NN` 8.0.0 pack does not compile in its default
configuration for cores with Helium floating-point support (MVE-F), such
as Cortex-M55 and Cortex-M85. Four of its element-wise sources fail with
`implicit declaration of function 'arm_nn_clamp_mve_f32'` (or `..._f16`).

Until a fixed CMSIS-NN release is available, enable CMSIS-NN's float
support in any project that selects the Cortex-M operators on such a core:

```yaml
# in the *.csolution.yml or *.cproject.yml
define:
  - ARM_NN_ENABLE_F32: 1
  - ARM_NN_ENABLE_F16: 1
```

## Resources

- [ExecuTorch Documentation](https://pytorch.org/executorch/)
- [CMSIS-Pack Specification](https://open-cmsis-pack.github.io/Open-CMSIS-Pack-Spec/main/html/index.html)
- [Issue Tracker](https://github.com/pytorch/executorch/issues)
