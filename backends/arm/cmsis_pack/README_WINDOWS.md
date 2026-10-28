1. Open “Developer PowerShell for VS 2022”.

   Building flatc.exe requires a Windows C++ compiler. Use the Visual Studio C++ development environment for that; your Arm GCC will be used later for Arm code.

   Run:

   Set-Location $HOME\executorch

$env:VIRTUAL_ENV = "$PWD\.venv"
$env:Path = "$env:VIRTUAL_ENV\Scripts;$env:Path"

python -c "import sys; print(sys.executable)"

# Make your existing Arm compiler available in this terminal.
$armGccRoot = "$HOME\.vcpkg\artifacts\2139c4c6\compilers.arm.arm.none.eabi.gcc\15.2.1"
$env:Path = "$armGccRoot\bin;$env:Path"

Get-Command cmake,ninja,python,cl,arm-none-eabi-gcc

   These changes affect only this terminal. CMake and Ninja are already installed, and your .venv already contains the Python codegen dependencies.
2. Initialize the FlatBuffers sources.

   Your third-party/flatbuffers directory is currently empty:

   git submodule update --init --recursive third-party/flatbuffers

   This retrieves repository source code. It doesn’t install a compiler, FVP, or other Arm tools.
3. Build and install Windows-native flatc using those sources.

   Install it inside the directory layout that the existing pack script already searches:

   $flatcPrefix = "$PWD/cmake-out-arm/third-party/flatc_ep"

cmake -S third-party/flatbuffers -B cmake-out-flatc-host -G Ninja `
  -DCMAKE_C_COMPILER=cl `
  -DCMAKE_CXX_COMPILER=cl `
  -DCMAKE_BUILD_TYPE=Release `
  -DFLATBUFFERS_BUILD_FLATC=ON `
  -DFLATBUFFERS_BUILD_TESTS=OFF `
  -DFLATBUFFERS_BUILD_FLATLIB=OFF `
  -DFLATBUFFERS_BUILD_FLATHASH=OFF `
  -DFLATBUFFERS_INSTALL=ON `
  "-DCMAKE_INSTALL_PREFIX=$flatcPrefix"

cmake --build cmake-out-flatc-host --target install --parallel 8

   In PowerShell, the continuation character is a backtick, with no trailing spaces.
4. Generate the schema headers directly.

   This uses the same flatc options as the repository’s schema/CMakeLists.txt (schema/CMakeLists.txt:29):

   $schemaOutput = "$PWD/cmake-out-arm/schema/include/executorch/schema"
New-Item -ItemType Directory -Force $schemaOutput | Out-Null

& "$flatcPrefix/bin/flatc.exe" `
  --cpp --cpp-std c++11 --gen-mutable --scoped-enums `
  -o $schemaOutput `
  schema/scalar_type.fbs schema/program.fbs

   Verify the three inputs required by the pack script:

   Test-Path "$schemaOutput/program_generated.h"
Test-Path "$schemaOutput/scalar_type_generated.h"
Test-Path "$flatcPrefix/include/flatbuffers/flatbuffers.h"

   All three should return True. You can then use cmake-out-arm as the pack script’s --build-dir.

There are two relevant problems with the README’s original CMake command:

- EXECUTORCH_BUILD_FLATC appears only in that README; it isn’t a recognized build option in this checkout.
- The automatic host-tool setup checks WIN32, which describes the target platform. Your toolchain targets Generic, so it selects a Unix Makefiles host build even though you’re running Windows. This follows CMake’s definition of WIN32 (https://cmake.org/cmake/help/latest/variable/WIN32.html).

The steps above avoid that host-build problem and produce the headers and FlatBuffers includes expected by copy_sources.sh (backends/arm/cmsis_pack/scripts/copy_sources.sh:130). No Arm setup script or FVP is needed. The later .sh packaging step requires a Bash environment with utilities such as patch, plus python3 resolving to your Python environment.

The windows .venv is naming the python3 executable python.
To be able to use the original build_pack.sh, in a git bash shell do:

python3() { python "$@"; }
export -f python3
