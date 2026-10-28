# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: BSD-3-Clause
"""Manifest-to-Zephyr contract tests; no Zephyr build or SDK is required."""

import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import generate_components as components  # noqa: E402
import generate_zephyr_module as zephyr  # noqa: E402


@pytest.fixture
def pack(tmp_path):
    root = tmp_path / "pack output" / "PyTorch.ExecuTorch.test"
    for source in (
        "runtime/core/runtime.cpp",
        "runtime/platform/default/minimal.cpp",
        "runtime/platform/default/posix.cpp",
        "runtime/platform/default/zephyr.cpp",
        "runtime/core/testing_util/not_for_consumers.cpp",
        "kernels/portable/cpu/op_add.cpp",
        "kernels/portable/cpu/op_mul.cpp",
        "kernels/portable/cpu/util/broadcast.cpp",
        "kernels/quantized/cpu/op_add.cpp",
        "backends/cortex_m/ops/op_quantized_add.cpp",
        "backends/arm/runtime/EthosUBackend_CoreDriver.cpp",
        "backends/arm/runtime/EthosUBackend_LinuxDriver.cpp",
        "extension/tensor/tensor_ptr.cpp",
        "registration/RegisterAllKernels.cpp",
    ):
        path = root / "src" / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('#include <arm_nnfunctions.h>\n' if "cortex_m" in source else "")
    (root / "include").mkdir()
    subprocess.run([
        sys.executable, str(SCRIPTS / "generate_components.py"),
        "--source-dir", str(root), "--template",
        str(SCRIPTS.parent / "templates" / "PyTorch.ExecuTorch.pdsc.tpl"),
        "--pdsc-output", str(root / "PyTorch.ExecuTorch.pdsc"),
        "--output", str(root.parent / "components.xml"),
    ], check=True, capture_output=True, text=True)
    return root


def run_cmake(module, tmp_path, settings, conflicting_target=None):
    cmake = shutil.which("cmake")
    if not cmake:
        pytest.skip("CMake is required for configuration contract tests")
    trace = tmp_path / "trace.txt"
    trace.write_text("")
    driver = tmp_path / ("CMakeLists.txt" if conflicting_target else "check.cmake")
    prelude = ""
    if conflicting_target:
        prelude = (
            "cmake_minimum_required(VERSION 3.20)\nproject(check NONE)\n"
            f"add_library({conflicting_target} INTERFACE)\n"
        )
    driver.write_text(
        prelude + f'set(CMAKE_CURRENT_BINARY_DIR "{tmp_path.as_posix()}")\n'
        + "\n".join(f"set({key} {value})" for key, value in settings.items())
        + "\n"
        + "\n".join(
            f"function({command})\n"
            f'  file(APPEND "{trace.as_posix()}" "{command}:${{ARGV}}\\n")\n'
            "endfunction()"
            for command in ("zephyr_library_named", "target_compile_features",
                            "zephyr_library_sources", "zephyr_include_directories",
                            "zephyr_compile_definitions")
        )
        + f'\ninclude("{(module / "CMakeLists.txt").as_posix()}")\n'
    )
    args = ["-S", str(tmp_path), "-B", str(tmp_path / "configure")] if conflicting_target else ["-P", str(driver)]
    result = subprocess.run([cmake, *args], capture_output=True, text=True)
    return result, trace.read_text()


def test_defines_do_not_consume_next_line():
    assert [d.strip() for d in zephyr.defines(
        "#define EMPTY\n#define VALUE 1024\n#define COMMENT /* hello */\n"
    )] == ["EMPTY", "VALUE 1024", "COMMENT"]


def test_manifest_platform_exclusions(pack):
    sources = components.generate_runtime_files(pack)
    assert "minimal.cpp" in sources
    assert "posix.cpp" not in sources
    assert "zephyr.cpp" not in sources
    assert "not_for_consumers.cpp" not in sources
    assert "\\" not in sources


@pytest.mark.parametrize("embedded", [False, True])
def test_relocated_module_matches_manifest(pack, tmp_path, embedded):
    module = (pack if embedded else pack.parent) / "zephyr"
    zephyr.generate(pack / "PyTorch.ExecuTorch.pdsc", module)
    assert {p.name for p in module.iterdir()} == {"module.yml", "CMakeLists.txt", "Kconfig"}
    relocated = tmp_path / "external consumer"
    shutil.move(str(pack.parent), relocated)
    module = (relocated / pack.name if embedded else relocated) / "zephyr"
    root = relocated / pack.name
    kconfig = (module / "Kconfig").read_text()
    cmake = (module / "CMakeLists.txt").read_text()
    manifest = ET.parse(root / "PyTorch.ExecuTorch.pdsc").getroot()
    for component in manifest.findall("./components/component"):
        assert f"config {zephyr.symbol(component)}\n" in kconfig
        for file in component.findall("./files/file"):
            if file.get("condition") == "AC6":
                continue
            assert f'"${{CMSIS_EXECUTORCH_PACK_ROOT}}/{file.attrib["name"]}"' in cmake
    assert str(SCRIPTS.parents[3]) not in cmake
    result, trace = run_cmake(module, tmp_path, {
        "CONFIG_CMSIS_EXECUTORCH": "y", "CONFIG_CPU_CORTEX_M": "y",
        "CONFIG_CMSIS_EXECUTORCH_RUNTIME": "y",
        "CONFIG_CMSIS_EXECUTORCH_KERNEL_UTILS": "y",
        "CONFIG_CMSIS_EXECUTORCH_KERNEL_REGISTRATION": "y",
        "CONFIG_CMSIS_EXECUTORCH_OP_PORTABLE_ADD": "y",
    })
    assert result.returncode == 0, result.stderr
    assert f"{root.as_posix()}/src/kernels/portable/cpu/op_add.cpp" in trace
    assert "op_mul.cpp" not in trace
    assert "kernels/quantized/cpu/op_add.cpp" not in trace
    assert "RegisterAllKernels.cpp" in trace
    assert "C10_USING_CUSTOM_GENERATED_MACROS\n" in trace
    assert "FLATBUFFERS_MAX_ALIGNMENT=1024" in trace
    assert "posix.cpp" not in trace
    rte = (tmp_path / "include" / "RTE_Components.h").read_text()
    assert "#define RTE_ML_EXECUTORCH_OP_PORTABLE_ADD\n" in rte
    assert "RTE_ML_EXECUTORCH_OP_PORTABLE_MUL" not in rte


@pytest.mark.parametrize("target", [None, "executorch", "executorch_core", "libexecutorch"])
def test_reject_meta_module(pack, tmp_path, target):
    module = pack.parent / "zephyr"
    zephyr.generate(pack / "PyTorch.ExecuTorch.pdsc", module)
    settings = {"CONFIG_CMSIS_EXECUTORCH": "y"}
    if not target:
        settings["CONFIG_EXECUTORCH"] = "y"
    result, _ = run_cmake(module, tmp_path, settings, conflicting_target=target)
    assert result.returncode != 0
    assert "replaces Meta's executorch" in result.stderr


def test_disabled_module_is_inert(pack, tmp_path):
    module = pack.parent / "zephyr"
    zephyr.generate(pack / "PyTorch.ExecuTorch.pdsc", module)
    result, trace = run_cmake(module, tmp_path, {"CONFIG_EXECUTORCH": "y"})
    assert result.returncode == 0, result.stderr
    assert trace == ""


def test_missing_sources_fail_generation(pack):
    (pack / "src/kernels/portable/cpu/op_add.cpp").unlink()
    with pytest.raises(ValueError, match="Missing or external pack file"):
        zephyr.generate(pack / "PyTorch.ExecuTorch.pdsc", pack.parent / "zephyr")


def test_cmsis_nn_requires_external_library(pack, tmp_path):
    module = pack.parent / "zephyr"
    zephyr.generate(pack / "PyTorch.ExecuTorch.pdsc", module)
    result, _ = run_cmake(module, tmp_path, {
        "CONFIG_CMSIS_EXECUTORCH": "y", "CONFIG_CPU_CORTEX_M": "y",
        "CONFIG_CMSIS_EXECUTORCH_RUNTIME": "y", "CONFIG_CMSIS_EXECUTORCH_KERNEL_UTILS": "y",
        "CONFIG_CMSIS_EXECUTORCH_OP_CORTEX_M_QUANTIZED_ADD": "y",
    })
    assert result.returncode != 0
    assert "requires CMSIS_NN" in result.stderr


def test_kconfig_dependencies_follow_pack(pack):
    module = pack.parent / "zephyr"
    zephyr.generate(pack / "PyTorch.ExecuTorch.pdsc", module)
    text = (module / "Kconfig").read_text()
    configs = dict(re.findall(r"config (\w+)\n(.*?)(?=\nconfig |\Z)", text, re.S))
    assert "select CMSIS_EXECUTORCH_KERNEL_UTILS" in configs["CMSIS_EXECUTORCH_OP_PORTABLE_ADD"]
    assert "select CMSIS_EXECUTORCH_RUNTIME" in configs["CMSIS_EXECUTORCH_KERNEL_UTILS"]
    assert "depends on CMSIS_NN" in configs["CMSIS_EXECUTORCH_OP_CORTEX_M_QUANTIZED_ADD"]
    assert "depends on (ARM_ETHOS_U || ETHOS_U)" in configs["CMSIS_EXECUTORCH_BACKEND_ETHOS_U"]


@pytest.mark.parametrize("driver", ["CONFIG_ARM_ETHOS_U", "CONFIG_ETHOS_U"])
def test_ethos_u_cmake_accepts_enabled_driver(pack, tmp_path, driver):
    module = pack.parent / "zephyr"
    zephyr.generate(pack / "PyTorch.ExecuTorch.pdsc", module)
    result, trace = run_cmake(module, tmp_path, {
        "CONFIG_CMSIS_EXECUTORCH": "y", "CONFIG_CPU_CORTEX_M": "y",
        "CONFIG_CMSIS_EXECUTORCH_RUNTIME": "y",
        "CONFIG_CMSIS_EXECUTORCH_BACKEND_ETHOS_U": "y", driver: "y",
    })
    assert result.returncode == 0, result.stderr
    assert "EthosUBackend_CoreDriver.cpp" in trace
    assert "EthosUBackend_LinuxDriver.cpp" not in trace
    assert "EXECUTORCH_BUILD_ARM_BAREMETAL=1" in trace


def test_kconfig_resolves_selections(pack, tmp_path):
    kconfiglib = pytest.importorskip("kconfiglib")
    module = pack.parent / "zephyr"
    zephyr.generate(pack / "PyTorch.ExecuTorch.pdsc", module)
    external = (
        "CPU_CORTEX_M CPP REQUIRES_FULL_LIBCPP CMSIS_NN ARM_ETHOS_U ETHOS_U "
        "CMSIS_NN_BASICMATH CMSIS_NN_CONVOLUTION CMSIS_NN_FULLYCONNECTED "
        "CMSIS_NN_NNSUPPORT CMSIS_NN_POOLING CMSIS_NN_SOFTMAX CMSIS_NN_PAD CMSIS_NN_TRANSPOSE"
    ).split()
    driver = tmp_path / "Kconfig"
    driver.write_text(
        "\n".join(f'config {name}\n\tbool "{name}"\n' for name in external)
        + f'\nsource "{(module / "Kconfig").as_posix()}"\n'
    )
    config = kconfiglib.Kconfig(str(driver))
    for name in ("CPU_CORTEX_M", "CMSIS_EXECUTORCH", "CMSIS_EXECUTORCH_OP_PORTABLE_ADD"):
        config.syms[name].set_value("y")
    for name in (
        "CPP", "REQUIRES_FULL_LIBCPP", "CMSIS_EXECUTORCH_KERNEL_UTILS",
        "CMSIS_EXECUTORCH_RUNTIME",
    ):
        assert config.syms[name].str_value == "y"
    assert config.syms["CMSIS_EXECUTORCH_OP_PORTABLE_MUL"].str_value == "n"
    nn_op = config.syms["CMSIS_EXECUTORCH_OP_CORTEX_M_QUANTIZED_ADD"]
    nn_op.set_value("y")
    assert nn_op.str_value == "n"
    config.syms["CMSIS_NN"].set_value("y")
    assert nn_op.str_value == "y"
    assert config.syms["CMSIS_NN_BASICMATH"].str_value == "y"
    backend = config.syms["CMSIS_EXECUTORCH_BACKEND_ETHOS_U"]
    backend.set_value("y")
    assert backend.str_value == "n"
    config.syms["ARM_ETHOS_U"].set_value("y")
    assert backend.str_value == "y"
    assert not config.warnings
