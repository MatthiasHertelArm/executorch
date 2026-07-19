/*
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cortex_m_ops_common.h"

namespace cortex_m {
namespace native {

using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

// cppcheck-suppress unusedFunction
Tensor& quantized_avg_pool2d_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Int64ArrayRef kernel_size,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const int64_t zero_point,
    const int64_t multiplier,
    const int64_t shift,
    const Tensor& scratch,
    Tensor& out) {
  constexpr int32_t activation_min = std::numeric_limits<int8_t>::min();
  constexpr int32_t activation_max = std::numeric_limits<int8_t>::max();

  const int64_t dilation_values[2] = {1, 1};
  const Int64ArrayRef dilation(dilation_values, 2);
  CmsisPool2DConfig pool_config;
  if (!prepare_cmsis_pool2d_config(
          context,
          "quantized_avg_pool2d_out",
          input,
          out,
          kernel_size,
          stride,
          padding,
          dilation,
          false,
          activation_min,
          activation_max,
          pool_config)) {
    return out;
  }

  cmsis_nn_context cmsis_ctx;
  cmsis_ctx.buf = nullptr;
  cmsis_ctx.size = scratch.nbytes();
  if (cmsis_ctx.size > 0) {
    cmsis_ctx.buf = scratch.mutable_data_ptr<int8_t>();
  }

  // The AoT-exported scratch is sized for the export target's CMSIS-NN path
  // (MVE needs 0 bytes here); the running core may need more (the DSP path
  // needs ch * sizeof(int32_t)). Top up from the kernel temp allocator when
  // short so one .pte runs on every M-profile core.
  const int32_t required_buffer_bytes = arm_avgpool_s8_get_buffer_size(
      pool_config.output_dims.w, pool_config.input_dims.c);
  if (required_buffer_bytes > 0 &&
      cmsis_ctx.size < static_cast<size_t>(required_buffer_bytes)) {
    auto temp = context.allocate_temp(required_buffer_bytes);
    if (!temp.ok()) {
      ET_LOG(
          Error,
          "quantized_avg_pool2d_out: scratch %d B < required %d B and temp"
          " allocation failed",
          static_cast<int>(cmsis_ctx.size),
          static_cast<int>(required_buffer_bytes));
      context.fail(temp.error());
      return out;
    }
    cmsis_ctx.buf = temp.get();
    cmsis_ctx.size = static_cast<size_t>(required_buffer_bytes);
  }

#ifdef CORTEX_M_ENABLE_RUNTIME_CHECKS
  const int32_t runtime_buffer_bytes = arm_avgpool_s8_get_buffer_size(
      pool_config.output_dims.w, pool_config.input_dims.c);
  if (scratch.nbytes() != static_cast<size_t>(runtime_buffer_bytes)) {
    ET_LOG(
        Error,
        "quantized_avg_pool2d_out: scratch buffer size incorrect - actual: (%d) needed: (%d)",
        static_cast<int>(scratch.nbytes()),
        static_cast<int>(runtime_buffer_bytes));
    context.fail(Error::Internal);
    return out;
  }
#endif

  const int8_t* input_data = input.const_data_ptr<int8_t>();
  int8_t* output_data = out.mutable_data_ptr<int8_t>();

  const arm_cmsis_nn_status status = arm_avgpool_s8(
      &cmsis_ctx,
      &pool_config.pool_params,
      &pool_config.input_dims,
      input_data,
      &pool_config.filter_dims,
      &pool_config.output_dims,
      output_data);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "quantized_avg_pool2d_out: arm_avgpool_s8 failed with status [%d]",
        status);
    context.fail(Error::Internal);
  }

  (void)zero_point;
  (void)multiplier;
  (void)shift;

  return out;
}

} // namespace native
} // namespace cortex_m
