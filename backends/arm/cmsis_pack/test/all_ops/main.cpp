// Copyright 2026 Arm Limited and/or its affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// All-ops consumer runner, built entirely from the CMSIS Pack's ExecuTorch
// runtime + kernel components.
//
// Dual purpose:
//   * Build/link coverage -- all_ops.cproject.yml selects every operator
//     component, so linking this image exercises every op's generated
//     registration, forward declaration and kernel source.
//   * Execution coverage -- every model produced by generate_test_models.py
//     is embedded directly into the ELF (.rodata) via embedded_models.S, so
//     this image self-tests on bare metal without semihosting access to the
//     host filesystem. Iterates over every model.pte / input_*.bin /
//     expected_*.bin, prints "PASS"/"FAIL" per op, and a final aggregate.
//
// API usage mirrors examples/arm/executor_runner/arm_executor_runner.cpp.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

#include "embedded_models.h"

using executorch::aten::Tensor;
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

namespace {

constexpr size_t kMethodPoolSize = 16 * 1024 * 1024;
constexpr size_t kTempPoolSize = 4 * 1024 * 1024;

alignas(16) uint8_t g_method_pool[kMethodPoolSize];
alignas(16) uint8_t g_temp_pool[kTempPoolSize];

// Cortex-M55 DWT cycle counter, sampled around method->execute() to report a
// rough inference cost per op.
inline volatile uint32_t& reg32(uintptr_t addr) {
  return *reinterpret_cast<volatile uint32_t*>(addr);
}
constexpr uintptr_t kDemcr = 0xE000EDFC; // CoreDebug DEMCR, TRCENA = bit 24
constexpr uintptr_t kDwtCtrl = 0xE0001000; // DWT_CTRL, CYCCNTENA = bit 0
constexpr uintptr_t kDwtCyccnt = 0xE0001004;

// Per-op stats accumulated during the run and dumped as a table at the end.
struct RowStat {
  const EmbeddedModel* m;
  bool pass;
  size_t in_bytes;
  size_t out_bytes;
  uint32_t cycles;
};
constexpr size_t kMaxRows = 512;
RowStat g_rows[kMaxRows];

// Minimal in-memory DataLoader so we do not depend on the extension/ tree
// (the pack ships only the runtime/ + kernels).
class BufferLoader final : public DataLoader {
 public:
  BufferLoader(const void* data, size_t size) : data_(data), size_(size) {}

  Result<FreeableBuffer> load(
      size_t offset,
      size_t size,
      const DataLoader::SegmentInfo&) const override {
    if (offset + size > size_) {
      return Error::InvalidArgument;
    }
    return FreeableBuffer(
        static_cast<const uint8_t*>(data_) + offset, size, nullptr);
  }

  Result<size_t> size() const override {
    return size_;
  }

 private:
  const void* data_;
  size_t size_;
};

bool tensors_match(
    const Tensor& got,
    const void* expected,
    size_t expected_bytes,
    float atol,
    float rtol) {
  if (got.nbytes() != expected_bytes) {
    printf(
        "  size mismatch: got %u vs expected %u bytes\n",
        static_cast<unsigned>(got.nbytes()),
        static_cast<unsigned>(expected_bytes));
    return false;
  }
  const auto dtype = got.scalar_type();
  if (dtype == executorch::aten::ScalarType::Float) {
    const float* a = got.const_data_ptr<float>();
    const float* b = static_cast<const float*>(expected);
    size_t n = expected_bytes / sizeof(float);
    float max_abs = 0.f;
    size_t worst = 0;
    bool ok = true;
    for (size_t i = 0; i < n; ++i) {
      float d = std::fabs(a[i] - b[i]);
      if (d > max_abs) {
        max_abs = d;
        worst = i;
      }
      if (d > atol + rtol * std::fabs(b[i])) {
        ok = false;
      }
    }
    printf(
        "  [cmp] %u float(s): max|err|=%g at [%u] (got %f vs exp %f),"
        " atol=%g rtol=%g -> %s\n",
        static_cast<unsigned>(n),
        max_abs,
        static_cast<unsigned>(worst),
        a[worst],
        b[worst],
        atol,
        rtol,
        ok ? "within tol" : "OUT OF TOL");
    return ok;
  }
  bool ok = std::memcmp(got.const_data_ptr(), expected, expected_bytes) == 0;
  printf(
      "  [cmp] %u byte(s) exact compare -> %s\n",
      static_cast<unsigned>(expected_bytes),
      ok ? "match" : "MISMATCH");
  return ok;
}

// Runs one embedded model end-to-end, returning true if it produced outputs
// matching the embedded expected_*.bin within (atol, rtol). Prints
// "Test_result: <op> PASS" / "FAIL (<stage>)" so a host log parser can
// aggregate results across the full run.
bool run_one_model(const EmbeddedModel& m, RowStat& row) {
  row.m = &m;
  row.pass = false;
  row.in_bytes = 0;
  row.out_bytes = 0;
  row.cycles = 0;
  printf(
      "Test_exec: %s (dir=%s) pte=%u bytes, %u input(s), %u expected\n",
      m.op,
      m.dir,
      static_cast<unsigned>(m.pte_size),
      static_cast<unsigned>(m.num_inputs),
      static_cast<unsigned>(m.num_outputs));
  BufferLoader loader(m.pte_data, m.pte_size);
  Result<Program> program = Program::load(&loader);
  if (!program.ok()) {
    printf(
        "Test_result: %s FAIL (Program::load err=%u)\n",
        m.op,
        static_cast<unsigned>(program.error()));
    return false;
  }

  const char* method_name = "forward";
  Result<MethodMeta> meta = program->method_meta(method_name);
  if (!meta.ok()) {
    printf("Test_result: %s FAIL (method_meta)\n", m.op);
    return false;
  }

  MemoryAllocator method_allocator(kMethodPoolSize, g_method_pool);
  MemoryAllocator temp_allocator(kTempPoolSize, g_temp_pool);

  size_t num_planned = meta->num_memory_planned_buffers();
  Span<uint8_t> planned_spans[8];
  for (size_t i = 0; i < num_planned && i < 8; ++i) {
    size_t sz = meta->memory_planned_buffer_size(i).get();
    uint8_t* buf = static_cast<uint8_t*>(method_allocator.allocate(sz, 16));
    planned_spans[i] = {buf, sz};
  }
  HierarchicalAllocator planned_memory({planned_spans, num_planned});
  MemoryManager memory_manager(
      &method_allocator, &planned_memory, &temp_allocator);

  Result<Method> method = program->load_method(method_name, &memory_manager);
  if (!method.ok()) {
    printf("Test_result: %s FAIL (load_method)\n", m.op);
    return false;
  }

  size_t num_inputs = method->inputs_size();
  for (size_t i = 0; i < num_inputs; ++i) {
    EValue in = method->get_input(i);
    if (!in.isTensor()) {
      continue;
    }
    if (i >= m.num_inputs) {
      printf(
          "Test_result: %s FAIL (missing embedded input %u)\n",
          m.op,
          static_cast<unsigned>(i));
      return false;
    }
    const EmbeddedBuffer& src = m.inputs[i];
    Tensor t = in.toTensor();
    if (src.size != t.nbytes()) {
      printf(
          "Test_result: %s FAIL (input %u size %u vs %u)\n",
          m.op,
          static_cast<unsigned>(i),
          static_cast<unsigned>(src.size),
          static_cast<unsigned>(t.nbytes()));
      return false;
    }
    std::memcpy(t.mutable_data_ptr(), src.data, src.size);
    row.in_bytes += src.size;
    printf(
        "  [in %u] %u bytes\n",
        static_cast<unsigned>(i),
        static_cast<unsigned>(src.size));
  }

  printf("  [run] %s execute() ...\n", m.op);
  uint32_t c0 = reg32(kDwtCyccnt);
  Error exec_err = method->execute();
  row.cycles = reg32(kDwtCyccnt) - c0;
  if (exec_err != Error::Ok) {
    printf("Test_result: %s FAIL (execute)\n", m.op);
    return false;
  }

  size_t num_outputs = method->outputs_size();
  printf(
      "  [run] %s execute() ok, %u output(s)\n",
      m.op,
      static_cast<unsigned>(num_outputs));
  bool pass = true;
  for (size_t i = 0; i < num_outputs; ++i) {
    EValue out = method->get_output(i);
    if (!out.isTensor()) {
      continue;
    }
    if (i >= m.num_outputs) {
      printf(
          "Test_result: %s FAIL (missing embedded expected %u)\n",
          m.op,
          static_cast<unsigned>(i));
      return false;
    }
    const EmbeddedBuffer& exp = m.expected[i];
    row.out_bytes += out.toTensor().nbytes();
    if (!tensors_match(out.toTensor(), exp.data, exp.size, m.atol, m.rtol)) {
      printf("  output %u mismatch\n", static_cast<unsigned>(i));
      pass = false;
    }
  }

  row.pass = pass;
  printf("Test_result: %s %s\n", m.op, pass ? "PASS" : "FAIL");
  return pass;
}

} // namespace

extern "C" int main(void) {
  // Unbuffered stdout: the bare-metal startup may loop after main() returns
  // (never calling exit()), so block-buffered output would never flush. Make
  // every printf reach the semihosting console immediately.
  setvbuf(stdout, nullptr, _IONBF, 0);

  executorch::runtime::runtime_init();

  // Enable the DWT cycle counter so run_one_model can time each inference.
  reg32(kDemcr) |= (1u << 24);
  reg32(kDwtCyccnt) = 0;
  reg32(kDwtCtrl) |= 1u;

  size_t passed = 0;
  const size_t total = g_embedded_models_count;
  for (size_t mi = 0; mi < total; ++mi) {
    RowStat& row = g_rows[mi < kMaxRows ? mi : kMaxRows - 1];
    if (run_one_model(g_embedded_models[mi], row)) {
      ++passed;
    }
  }

  printf(
      "\n==== Per-op results (%u models) ====\n", static_cast<unsigned>(total));
  printf(
      "%-30s %-6s %10s %8s %8s %12s\n",
      "op",
      "result",
      "model(B)",
      "in(B)",
      "out(B)",
      "cycles");
  for (size_t mi = 0; mi < total && mi < kMaxRows; ++mi) {
    const RowStat& r = g_rows[mi];
    printf(
        "%-30s %-6s %10u %8u %8u %12u\n",
        r.m->op,
        r.pass ? "PASS" : "FAIL",
        static_cast<unsigned>(r.m->pte_size),
        static_cast<unsigned>(r.in_bytes),
        static_cast<unsigned>(r.out_bytes),
        static_cast<unsigned>(r.cycles));
  }

  printf(
      "Test_result: SUMMARY %u/%u PASS\n",
      static_cast<unsigned>(passed),
      static_cast<unsigned>(total));
  return passed == total ? 0 : 1;
}
