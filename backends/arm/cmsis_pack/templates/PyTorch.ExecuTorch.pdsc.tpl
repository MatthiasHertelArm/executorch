<?xml version="1.0" encoding="UTF-8"?>
<!--
  Copyright 2026 Arm Limited and/or its affiliates.

  This source code is licensed under the BSD-style license found in the
  LICENSE file in the root directory of this source tree.
-->
<package schemaVersion="1.7"
  xmlns:xs="http://www.w3.org/2001/XMLSchema-instance" 
  xs:noNamespaceSchemaLocation="PACK.xsd">
  
  <vendor>PyTorch</vendor>
  <name>ExecuTorch</name>
  <description overview="Documentation/README.md">ExecuTorch: PyTorch Edge Runtime for on-device AI inference on Arm Cortex-M processors.</description>
  
  <url>https://github.com/pytorch/executorch/releases/download/v%{RELEASE_VERSION}%/</url>
  <license>LICENSE</license>
  <licenseSets>
    <licenseSet id="all" default="true" gating="false">
      <license name="LICENSE" title="BSD 3-Clause (ExecuTorch)" spdx="BSD-3-Clause"/>
    </licenseSet>
  </licenseSets>
  <repository type="git">https://github.com/pytorch/executorch.git</repository>

  <releases>
    <release version="%{RELEASE_VERSION}%" date="%{RELEASE_DATE}%"%{RELEASE_ATTRS}%>
      Add Zephyr integration folder; use generation script to generate the files.
      Apply fix for AC6 crash in op_quantize.cpp.
      Apply fix &lt;stddef.h&gt;/&lt;cstddef&gt; in tag.h; remove the -include stddef.h workaround.
    </release>
    %{HISTORY}%
  </releases>

  <requirements>
    <packages>
      <package name="CMSIS" vendor="ARM" version="6.0.0:6.99.99"/>
    </packages>
  </requirements>

  <taxonomy>
    <description Cclass="Machine Learning">Software Components for Machine Learning</description>
    <description Cclass="Machine Learning" Cgroup="ExecuTorch">ExecuTorch PyTorch Edge Runtime</description>
    <description Cclass="Machine Learning" Cgroup="ExecuTorch Operators">ExecuTorch Operator Components</description>
  </taxonomy>

  <conditions>
    <!-- Toolchain conditions -->
    <condition id="Cortex-M">
      <description>Cortex-M / STAR-MC processor</description>
      <accept Dcore="Cortex-M0"/>
      <accept Dcore="Cortex-M0+"/>
      <accept Dcore="Cortex-M3"/>
      <accept Dcore="Cortex-M4"/>
      <accept Dcore="Cortex-M7"/>
      <accept Dcore="Cortex-M23"/>
      <accept Dcore="Cortex-M33"/>
      <accept Dcore="Cortex-M35P"/>
      <accept Dcore="Cortex-M52"/>
      <accept Dcore="Cortex-M55"/>
      <accept Dcore="Cortex-M85"/>
      <accept Dcore="Star-MC1"/>
      <accept Dcore="Star-MC3"/>
    </condition>

    <condition id="GCC">
      <description>GNU Compiler</description>
      <require Tcompiler="GCC"/>
    </condition>
    
    <condition id="AC6">
      <description>Arm Compiler 6</description>
      <require Tcompiler="ARMCC" Toptions="AC6"/>
    </condition>
    
    <condition id="CLANG">
      <description>LLVM/Clang Compiler</description>
      <require Tcompiler="CLANG"/>
    </condition>

    <!-- Component dependency conditions -->
    <condition id="Runtime">
      <description>ExecuTorch Runtime required</description>
      <require condition="Cortex-M"/>
      <require Cclass="Machine Learning" Cgroup="ExecuTorch" Csub="Runtime"/>
    </condition>
    
    <condition id="Kernel Utils">
      <description>Kernel registration utilities required</description>
      <require Cclass="Machine Learning" Cgroup="ExecuTorch" Csub="Kernel Utils"/>
    </condition>
    
    <condition id="CMSIS-NN">
      <description>CMSIS-NN optimized backend</description>
      <require condition="Runtime"/>
      <require Cclass="CMSIS" Cgroup="NN Lib"/>
    </condition>
    
    <condition id="Ethos-U">
      <description>Ethos-U NPU backend (Cortex-M host, bare-metal)</description>
      <require condition="Cortex-M"/>
      <require Cclass="Machine Learning" Cgroup="NPU Support" Csub="Ethos-U Driver"/>
    </condition>

    <condition id="Extension Tensor">
      <description>ExecuTorch Tensor extension (TensorPtr utilities)</description>
      <require condition="Runtime"/>
    </condition>

    <!-- Operator conditions - each operator requires Kernel Utils -->
%{OPERATOR_CONDITIONS}%
  </conditions>

  <components>
    <!-- ==================== Core Runtime ==================== -->
    <component Cclass="Machine Learning" Cgroup="ExecuTorch" Csub="Runtime" Cversion="%{RELEASE_VERSION}%" condition="Cortex-M">
      <description>ExecuTorch Core Runtime - Required for all ExecuTorch applications</description>
      <RTE_Components_h>
        #define RTE_ML_EXECUTORCH_RUNTIME     /* ExecuTorch Runtime */
      </RTE_Components_h>
      <Pre_Include_Global_h>
        /* ExecuTorch global configuration */
        #define C10_USING_CUSTOM_GENERATED_MACROS
        #define FLATBUFFERS_MAX_ALIGNMENT 1024
        #define FLATBUFFERS_LOCALE_INDEPENDENT 0
      </Pre_Include_Global_h>
      <files>
%{RUNTIME_FILES}%
        <file category="include" name="include/"/>
        <file category="include" name="armclang_shims/" condition="AC6"/>
      </files>
    </component>

    <!-- ==================== Kernel Utils ==================== -->
    <component Cclass="Machine Learning" Cgroup="ExecuTorch" Csub="Kernel Utils" Cversion="%{RELEASE_VERSION}%" condition="Runtime">
      <description>ExecuTorch Kernel Registration Utilities</description>
      <RTE_Components_h>
        #define RTE_ML_EXECUTORCH_KERNEL_UTILS     /* ExecuTorch Kernel Utils */
      </RTE_Components_h>
      <files>
%{KERNEL_UTILS_FILES}%
      </files>
    </component>

    <!-- ==================== Extensions ==================== -->
    <component Cclass="Machine Learning" Cgroup="ExecuTorch" Csub="Extension Tensor" Cversion="%{RELEASE_VERSION}%" condition="Extension Tensor">
      <description>ExecuTorch Tensor Extension - TensorPtr / make_tensor_ptr utilities</description>
      <RTE_Components_h>
        #define RTE_ML_EXECUTORCH_EXTENSION_TENSOR     /* ExecuTorch Tensor Extension */
      </RTE_Components_h>
      <files>
%{TENSOR_EXTENSION_FILES}%
      </files>
    </component>

    <!-- ==================== Kernel Registration ==================== -->
    <component Cclass="Machine Learning" Cgroup="ExecuTorch" Csub="Kernel Registration" Cversion="%{RELEASE_VERSION}%" condition="Kernel Utils">
      <description>ExecuTorch Kernel Registration - Registers all selected operators</description>
      <RTE_Components_h>
        #define RTE_ML_EXECUTORCH_KERNEL_REGISTRATION     /* ExecuTorch Kernel Registration */
      </RTE_Components_h>
      <files>
        <file category="sourceCpp" name="src/registration/RegisterAllKernels.cpp"/>
      </files>
    </component>

    <!-- ==================== Portable Operators ==================== -->
%{PORTABLE_OPERATOR_COMPONENTS}%

    <!-- ==================== Quantized Operators ==================== -->
%{QUANTIZED_OPERATOR_COMPONENTS}%

    <!-- ==================== Cortex-M Operators ==================== -->
%{CORTEX_M_OPERATOR_COMPONENTS}%

    <!-- ==================== Backends ==================== -->

    <!-- Ethos-U Backend (core driver, bare-metal) -->
    <component Cclass="Machine Learning" Cgroup="ExecuTorch" Csub="Backend EthosU" Cversion="%{RELEASE_VERSION}%" condition="Ethos-U">
      <description>ExecuTorch Ethos-U NPU Backend - Cortex-M host (bare-metal). Hardware acceleration for Ethos-U55/U65/U85</description>
      <RTE_Components_h>
        #define RTE_ML_EXECUTORCH_BACKEND_ETHOS_U     /* ExecuTorch Ethos-U Backend (Cortex-M host) */
      </RTE_Components_h>
      <Pre_Include_Global_h>
        #define EXECUTORCH_BUILD_ARM_BAREMETAL 1
        #define ET_USE_ETHOS_U_BACKEND 1
      </Pre_Include_Global_h>
      <files>
%{ETHOS_U_BACKEND_CORE_DRIVER_FILES}%
      </files>
    </component>

  </components>

</package>
