#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: BSD-3-Clause
"""Generate a relocatable Zephyr module from the populated pack manifest."""

import argparse
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def defines(text: str) -> list[str]:
    return re.findall(r"^[ \t]*#define[ \t]+([A-Za-z_]\w*(?:[ \t]+[^/\n]+)?)", text, re.M)


def symbol(component: ET.Element) -> str:
    guards = defines(component.findtext("RTE_Components_h", ""))
    if len(guards) != 1 or not guards[0].strip().startswith("RTE_ML_EXECUTORCH_"):
        raise ValueError(f"Expected one ExecuTorch component guard: {component.attrib}")
    return guards[0].strip().replace("RTE_ML_", "CMSIS_", 1)


def generate(pdsc: Path, output: Path) -> None:
    root = ET.parse(pdsc).getroot()
    pack_root = pdsc.resolve().parent
    components = root.findall("./components/component")
    by_name = {c.attrib["Csub"]: c for c in components}
    conditions = {c.attrib["id"]: c for c in root.findall("./conditions/condition")}

    def requirements(condition: str) -> tuple[set[str], set[str]]:
        if condition == "Cortex-M":
            return set(), {"CPU_CORTEX_M"}
        selections, dependencies = set(), set()
        for req in conditions[condition].findall("require"):
            if "condition" in req.attrib:
                sel, dep = requirements(req.attrib["condition"])
                selections.update(sel)
                dependencies.update(dep)
            elif req.get("Cgroup") == "NN Lib":
                dependencies.add("CMSIS_NN")
                selections.add("CMSIS_EXECUTORCH_CMSIS_NN")
            elif req.get("Cgroup") == "NPU Support":
                dependencies.add("(ARM_ETHOS_U || ETHOS_U)")
            elif req.get("Csub") in by_name:
                selections.add(symbol(by_name[req.attrib["Csub"]]))
            else:
                raise ValueError(f"Unsupported pack requirement: {req.attrib}")
        return selections, dependencies

    kconfig_groups = {name: [] for name in ("RUNTIME", "PORTABLE", "QUANTIZED", "CORTEX_M")}
    operator_groups = {"Portable": "PORTABLE", "Quantized": "QUANTIZED", "Cortex-M": "CORTEX_M"}
    cmake = []
    seen_symbols = set()
    for component in components:
        group = "RUNTIME"
        if component.get("Cgroup") == "ExecuTorch Operators":
            group = operator_groups[component.attrib["Csub"].split()[0]]
        kconfig = kconfig_groups[group]
        config = symbol(component)
        if config in seen_symbols:
            raise ValueError(f"Duplicate Kconfig symbol: {config}")
        seen_symbols.add(config)
        selections, dependencies = requirements(component.attrib["condition"])
        # Unlike a CMSIS component resolver, Kconfig does not infer this from
        # selecting Backend EthosU's driver dependency.
        if config == "CMSIS_EXECUTORCH_BACKEND_ETHOS_U":
            selections.add("CMSIS_EXECUTORCH_RUNTIME")
        kconfig.extend([f"config {config}", f'\tbool "{component.attrib["Csub"]}"'])
        if config == "CMSIS_EXECUTORCH_RUNTIME":
            kconfig.append("\tdefault y")
        for dep in sorted(dependencies):
            kconfig.append(f"\tdepends on {dep}")
        for selection in sorted(selections):
            kconfig.append(f"\tselect {selection}")
        kconfig.extend(["\thelp", "\t  " + component.findtext("description", "").strip(), ""])

        cmake.append(f"if(CONFIG_{config})")
        # Guard against manually supplied CMake configurations as well.
        for dep in sorted(selections - {"CMSIS_EXECUTORCH_CMSIS_NN"}):
            cmake.extend([
                f"  if(NOT CONFIG_{dep})",
                f'    message(FATAL_ERROR "{config} requires {dep}")',
                "  endif()",
            ])
        for dep in sorted(dependencies):
            expr = re.sub(r"\b[A-Z][A-Z_0-9]*\b", lambda m: "CONFIG_" + m[0], dep)
            expr = expr.replace("||", "OR").replace("&&", "AND")
            cmake.extend([
                f"  if(NOT ({expr}))",
                f'    message(FATAL_ERROR "{config} requires {dep}")',
                "  endif()",
            ])
        for define in defines(component.findtext("RTE_Components_h", "")):
            cmake.append(f'  string(APPEND _cmsis_executorch_rte "#define {define.strip()}\\n")')
        for define in defines(component.findtext("Pre_Include_Global_h", "")):
            parts = define.split()
            cmake.append(f'  zephyr_compile_definitions({"=".join(parts)})')
        sources, includes = [], []
        for file in component.findall("./files/file"):
            # The pack's AC6 sysroot shim is inappropriate for Zephyr SDKs.
            if file.get("condition") == "AC6":
                continue
            if file.get("condition"):
                raise ValueError(f"Unsupported file condition: {file.attrib}")
            name = file.attrib["name"].replace("\\", "/")
            path = (pack_root / name).resolve()
            if not path.is_relative_to(pack_root) or not path.exists():
                raise ValueError(f"Missing or external pack file: {name}")
            category = file.attrib["category"]
            if category in ("sourceCpp", "sourceC", "source"):
                sources.append(name)
            elif category == "include":
                includes.append(name)
        for command, paths in (
            ("zephyr_library_sources", sources),
            ("zephyr_include_directories", includes),
        ):
            if paths:
                cmake.append(f"  {command}(")
                cmake.extend(f'    "${{CMSIS_EXECUTORCH_PACK_ROOT}}/{p}"' for p in paths)
                cmake.append("  )")
        cmake.extend(["endif()", ""])

    templates = Path(__file__).resolve().parents[1] / "contributions" / "zephyr"
    output.mkdir(parents=True, exist_ok=True)
    relative_root = os.path.relpath(pack_root, output.resolve()).replace("\\", "/")
    replacements = {
        "PACK_ROOT": relative_root,
        "COMPONENTS": "\n".join(cmake),
        **{f"{group}_COMPONENTS": "\n".join(lines) for group, lines in kconfig_groups.items()},
    }
    for name in ("Kconfig", "CMakeLists.txt"):
        text = (templates / (name + ".tpl")).read_text(encoding="utf-8")
        for placeholder, value in replacements.items():
            text = text.replace(f"%{{{placeholder}}}%", value)
        (output / name).write_text(text, encoding="utf-8", newline="\n")
    (output / "module.yml").write_text(
        (templates / "module.yml").read_text(encoding="utf-8"),
        encoding="utf-8", newline="\n",
    )
    print(f"Generated {output}: {len(components)} selectable pack components")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdsc", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    generate(args.pdsc, args.output_dir)


if __name__ == "__main__":
    main()
