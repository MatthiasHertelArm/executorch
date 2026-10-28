# Generated from PyTorch.ExecuTorch.pdsc. Do not edit.
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: BSD-3-Clause

menuconfig CMSIS_EXECUTORCH
	bool "CMSIS-Pack ExecuTorch"
	depends on CPU_CORTEX_M
	select CPP
	select REQUIRES_FULL_LIBCPP
	help
	  Build ExecuTorch using only the sources shipped in this pack.
	  Replaces Meta's executorch module; disable CONFIG_EXECUTORCH.
	  Select C++17 or newer in the application's configuration.

if CMSIS_EXECUTORCH

menu "ExecuTorch runtime settings"

%{RUNTIME_COMPONENTS}%

endmenu

menu "Portable Operators"

%{PORTABLE_COMPONENTS}%

endmenu

menu "Quantized Operators"

%{QUANTIZED_COMPONENTS}%

endmenu

menu "Cortex-M operators"

%{CORTEX_M_COMPONENTS}%

endmenu

config CMSIS_EXECUTORCH_CMSIS_NN
	bool
	depends on CMSIS_NN
	select CMSIS_NN_BASICMATH
	select CMSIS_NN_CONVOLUTION
	select CMSIS_NN_FULLYCONNECTED
	select CMSIS_NN_NNSUPPORT
	select CMSIS_NN_POOLING
	select CMSIS_NN_SOFTMAX
	select CMSIS_NN_PAD
	select CMSIS_NN_TRANSPOSE
	help
	  Enable the CMSIS-NN implementation families required by pack operators.
	  The pack depends on NN Lib as a whole rather than individual functions.

endif # CMSIS_EXECUTORCH
