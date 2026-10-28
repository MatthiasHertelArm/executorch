# Generated from PyTorch.ExecuTorch.pdsc. Do not edit.
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: BSD-3-Clause

if(NOT CONFIG_CMSIS_EXECUTORCH)
  return()
endif()

# All module paths are known before Zephyr evaluates module CMakeLists files.
# Checking CONFIG_EXECUTORCH also catches the case where Meta's module has
# not created its targets yet.
if(CONFIG_EXECUTORCH OR TARGET executorch OR TARGET executorch_core OR TARGET libexecutorch)
  message(FATAL_ERROR
    "cmsis-executorch replaces Meta's executorch Zephyr module. "
    "Disable CONFIG_EXECUTORCH or remove the executorch module from ZEPHYR_MODULES.")
endif()
if(DEFINED CONFIG_STD_CPP_VERSION AND CONFIG_STD_CPP_VERSION LESS 201703)
  message(FATAL_ERROR "cmsis-executorch requires C++17 or newer: set CONFIG_STD_CPP17=y.")
endif()

get_filename_component(CMSIS_EXECUTORCH_PACK_ROOT
  "${CMAKE_CURRENT_LIST_DIR}/%{PACK_ROOT}%" ABSOLUTE)
if(NOT EXISTS "${CMSIS_EXECUTORCH_PACK_ROOT}/PyTorch.ExecuTorch.pdsc")
  message(FATAL_ERROR "cmsis-executorch: pack sources are missing at ${CMSIS_EXECUTORCH_PACK_ROOT}")
endif()

zephyr_library_named(cmsis_executorch)
target_compile_features(cmsis_executorch PRIVATE cxx_std_17)

# Zephyr links its libraries with --whole-archive. Keep registration and backend
# static initializers in this library so they survive the final link.
set(_cmsis_executorch_rte "#pragma once\n")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include")
zephyr_include_directories("${CMAKE_CURRENT_BINARY_DIR}/include")

%{COMPONENTS}%

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/include/RTE_Components.h"
  "${_cmsis_executorch_rte}")
