#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Build the CMSIS Pack and exercise it with a consumer project that enables
# EVERY operator component, so a single cbuild links the whole operator surface
# (build/link coverage). Phase 2 will add per-op .pte execution on the FVP.
#
# Mirrors smoke/run.sh: csolution + cbuild against an in-tree project, run
# inside the AVH-MLOps Docker image. The only addition is generating the
# all-ops cproject (every operator component) before the build.
#
# Prerequisites / environment overrides: see smoke/run.sh.

set -euo pipefail

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ET_ROOT="$(cd "${TEST_DIR}/../../../../.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${ET_ROOT}/arm_test/cmake-out}"
OUTPUT_DIR="${OUTPUT_DIR:-${ET_ROOT}/arm_test/cmsis-pack-output}"
BASE_VER="$(sed 's/a0$//' "${ET_ROOT}/version.txt")"
PACK_VERSION="${PACK_VERSION:-${BASE_VER}-stage}"
DOCKER_IMAGE="${DOCKER_IMAGE:-ghcr.io/arm-software/avh-mlops/arm-mlops-docker-licensed-community:latest-arm64}"

echo "=== Build pack ${PACK_VERSION} ==="
"${ET_ROOT}/backends/arm/cmsis_pack/scripts/build_pack.sh" \
    --executorch-root "${ET_ROOT}" \
    --build-dir       "${BUILD_DIR}" \
    --version         "${PACK_VERSION}" \
    --output-dir      "${OUTPUT_DIR}"

PACK_BASENAME="PyTorch.ExecuTorch.${PACK_VERSION}.pack"
PACK_FILE="${OUTPUT_DIR}/${PACK_BASENAME}"
[[ -f "${PACK_FILE}" ]] || { echo "Pack file not found: ${PACK_FILE}"; exit 1; }

echo
echo "=== Validate pack structure ==="
python3 "${TEST_DIR}/../validate_pack.py" "${PACK_FILE}"

echo
echo "=== Generate all-ops project (every operator component) ==="
python3 "${TEST_DIR}/gen_cproject.py" \
    --source-dir "${ET_ROOT}" \
    --output     "${TEST_DIR}/all_ops.cproject.yml"

# -------------------------------------------------------------------------
# Embed the per-op test models into the image (.incbin in .rodata) so the
# firmware self-tests every op in one boot -- no semihosting filesystem access
# -- and reports a per-op DWT cycle count. Models must exist before the build.
#
# RUN_FVP=1 exports a fresh .pte + reference I/O for every op (needs the torch
# export env: examples/arm/setup.sh, incl. cmsis_nn for the Cortex-M models).
# Without it, gen_embedded_models.py emits an empty stub, so the image still
# links and boots (build/link coverage only).
# -------------------------------------------------------------------------
MODELS_DIR="${TEST_DIR}/models"
if [[ "${RUN_FVP:-0}" == "1" ]]; then
    echo
    echo "=== Export per-op .pte + reference outputs ==="
    python3 "${TEST_DIR}/generate_test_models.py" \
        --source-dir "${ET_ROOT}" \
        --output-dir "${MODELS_DIR}" \
        --continue-on-error
fi
echo
echo "=== Embed models into firmware ==="
python3 "${TEST_DIR}/gen_embedded_models.py" \
    --models-dir "${MODELS_DIR}" \
    --output-dir "${TEST_DIR}"

echo
echo "=== Consumer build (${DOCKER_IMAGE}) ==="
docker run --rm \
    -e PACK_BASENAME="${PACK_BASENAME}" \
    -v "${TEST_DIR}:/workspace" \
    -v "${OUTPUT_DIR}:/pack-output:ro" \
    "${DOCKER_IMAGE}" \
    bash -lc '
set -euo pipefail

# Docker Desktop for Mac bind mounts fail the csolution --update-rte device-file
# copy (gRPC-FUSE / std::filesystem). Build on a container-local path, then copy
# the outputs back to the bind-mounted project so the host FVP step finds the
# ELF. Stale cbuild locks/outputs are dropped so packs re-resolve cleanly.
rm -rf /tmp/build
cp -r /workspace /tmp/build
cd /tmp/build
rm -rf out tmp RTE *.cbuild-idx.yml *.cbuild-pack.yml *.cbuild-set.yml

# Acquire the toolchain set declared in vcpkg-configuration.json.
export Z_VCPKG_POSTSCRIPT="$(mktemp /tmp/vcpkg.XXXXXX.sh)"
vcpkg activate
source "${Z_VCPKG_POSTSCRIPT}"

# Always reinstall the freshly built pack into a clean container-local pack
# root, isolated from any host pack store. --packs pulls ARM::CMSIS-NN (needed
# by the Cortex-M CMSIS-NN ops) from the public index.
export CMSIS_PACK_ROOT=/tmp/cmsis-pack-root
cpackget init https://www.keil.com/pack/index.pidx
cpackget add --agree-embedded-license -F "/pack-output/${PACK_BASENAME}"

cbuild all_ops.csolution.yml --packs --update-rte --context all_ops.Debug+ARMCM55

# Publish build outputs back to the bind-mounted project for the host FVP step.
rm -rf /workspace/out
cp -r /tmp/build/out /workspace/out
'
echo
echo "=== Build/link coverage PASS (every operator component linked) ==="

# -------------------------------------------------------------------------
# Execution + per-op cycles (RUN_FVP=1): the image embeds every model (above),
# so a SINGLE FVP boot self-tests all ops and prints a per-op DWT cycle table
# plus a final "Test_result: SUMMARY <pass>/<total> PASS" line.
#
# Requires the Corstone FVP on PATH (examples/arm/setup.sh --enable-fvps).
# -------------------------------------------------------------------------
if [[ "${RUN_FVP:-0}" != "1" ]]; then
    echo "(set RUN_FVP=1 to export + embed models and run on the FVP)"
    exit 0
fi

FVP="${FVP_BIN:-FVP_Corstone_SSE-300_Ethos-U55}"
ELF="$(find "${TEST_DIR}/out" -name '*.elf' | head -1)"
[[ -n "${ELF}" ]] || { echo "No firmware ELF found under out/"; exit 1; }

# The all-ops image is linked into DDR @0x70000000 (ARMCM55_large.ld) because it
# is far too large for the on-chip ITCM/SRAM. Point the CPU's reset vector table
# at DDR so reset fetches SP/PC from there (default is 0x10000000 = ITCM).
BOOT_BASE=0x70000000

echo
echo "=== Run embedded all-ops image on ${FVP} ==="
log="$("${FVP}" \
    -C cpu0.semihosting-enable=1 \
    -C "cpu0.INITSVTOR=${BOOT_BASE}" \
    -C "mps3_board.sse300.iotss3_systemcontrol.INITSVTOR_RST=${BOOT_BASE}" \
    -a "${ELF}" --timelimit 600 2>&1 || true)"
echo "${log}"

# The per-op cycle table + SUMMARY are the artifact. Individual ops can differ
# legitimately per toolchain (RNG seed; CLANG MVE.fp quantized rounding), so we
# fail only if the image never reached SUMMARY (crash / hang / boot failure).
if ! grep -qE "Test_result: SUMMARY [0-9]+/[0-9]+ PASS" <<<"${log}"; then
    echo "=== Execution FAILED: image did not reach SUMMARY ==="
    exit 1
fi
echo
echo "=== PASS (build/link + execution): $(grep -oE "SUMMARY [0-9]+/[0-9]+" <<<"${log}" | tail -1) ==="
