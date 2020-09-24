cmake_minimum_required(VERSION 3.16)

project(Hydrogen CXX)

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif ()

# Setup version information
find_program(__GIT_EXECUTABLE git)
mark_as_advanced(__GIT_EXECUTABLE)
if (__GIT_EXECUTABLE)

  execute_process(
    COMMAND ${__GIT_EXECUTABLE} rev-parse --is-inside-work-tree
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE __BUILDING_FROM_GIT_SOURCES
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if (__BUILDING_FROM_GIT_SOURCES)
    execute_process(
      COMMAND ${__GIT_EXECUTABLE} rev-parse --show-toplevel
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      OUTPUT_VARIABLE __GIT_TOPLEVEL_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND ${__GIT_EXECUTABLE} rev-parse --git-dir
      WORKING_DIRECTORY "${__GIT_TOPLEVEL_DIR}"
      OUTPUT_VARIABLE __GIT_GIT_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND ${__GIT_EXECUTABLE} --git-dir "${__GIT_GIT_DIR}" describe
      --abbrev=7 --always --dirty --tags
      WORKING_DIRECTORY "${__GIT_TOPLEVEL_DIR}"
      OUTPUT_VARIABLE __GIT_DESCRIBE_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND ${__GIT_EXECUTABLE} rev-list HEAD --max-count=1 --date-order
      WORKING_DIRECTORY "${__GIT_TOPLEVEL_DIR}"
      OUTPUT_VARIABLE __GIT_LATEST_COMMIT
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND ${__GIT_EXECUTABLE} rev-list hydrogen --max-count=1 --date-order
      WORKING_DIRECTORY "${__GIT_TOPLEVEL_DIR}"
      OUTPUT_VARIABLE __GIT_LATEST_HYDROGEN_COMMIT
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(HYDROGEN_GIT_VERSION "${__GIT_DESCRIBE_VERSION}")
    set(HYDROGEN_LATEST_SHA "${__GIT_LATEST_COMMIT}")
    set(HYDROGEN_LATEST_HYDROGEN_SHA "${__GIT_LATEST_HYDROGEN_COMMIT}")
  endif (__BUILDING_FROM_GIT_SOURCES)
endif (__GIT_EXECUTABLE)

# This must be set because version tags
set(HYDROGEN_VERSION_MAJOR 1)
set(HYDROGEN_VERSION_MINOR 5)
set(HYDROGEN_VERSION_PATCH 0)
set(HYDROGEN_VERSION_MAJOR_MINOR
    "${HYDROGEN_VERSION_MAJOR}.${HYDROGEN_VERSION_MINOR}")
set(HYDROGEN_VERSION
  "${HYDROGEN_VERSION_MAJOR_MINOR}.${HYDROGEN_VERSION_PATCH}")

# Back-compatibility, until it's all sorted out
set(EL_VERSION_MAJOR ${HYDROGEN_VERSION_MAJOR})
set(EL_VERSION_MINOR ${HYDROGEN_VERSION_MINOR})
set(EL_VERSION_PATCH ${HYDROGEN_VERSION_PATCH})

string(TOUPPER "${PROJECT_NAME}" UPPER_PROJECT_NAME)

if (NOT CMAKE_BUILD_TYPE MATCHES "Debug")
  set(HYDROGEN_RELEASE_BUILD TRUE)
endif ()

# Prevent in-source builds
if (PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR
    "In-source build attempted; please clean the CMake cache and then "
    "switch to an out-of-source build, e.g.,\n"
    "rm CMakeCache.txt && rm -Rf CMakeFiles/\nmkdir build/ && "
    "cd build/ && cmake ..\n")
endif ()

# Set the module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")

# Add warning flags
include(CheckCXXCompilerFlag)
macro (add_cxx_flag_as_library FLAG_LIST_VAR FLAG)
  LIST(APPEND ${FLAG_LIST_VAR} $<$<COMPILE_LANGUAGE:CXX>:${FLAG}>)
endmacro()

LIST(APPEND INITIAL_CXX_FLAGS
  "-Wall" "-Wextra" "-Wno-unused-parameter")

# Check all the CXX flags
foreach (flag IN LISTS INITIAL_CXX_FLAGS)
  check_cxx_compiler_flag("${flag}" CXX_COMPILER_HAS_FLAG${flag})
  if (CXX_COMPILER_HAS_FLAG${flag})
    add_cxx_flag_as_library(EXTRA_CXX_FLAGS "${flag}")
  endif ()
endforeach ()

# No sense having both, but some compilers support both. Probably overkill.
check_cxx_compiler_flag("-pedantic" CXX_COMPILER_HAS_PEDANTIC_FLAG)
if (CXX_COMPILER_HAS_PEDANTIC_FLAG)
  add_cxx_flag_as_library(EXTRA_CXX_FLAGS "-pedantic")
else ()
  check_cxx_compiler_flag("-Wpedantic" CXX_COMPILER_HAS_WPEDANTIC_FLAG)
  if (CXX_COMPILER_HAS_WPEDANTIC_FLAG)
    add_cxx_flag_as_library(EXTRA_CXX_FLAGS "-Wpedantic")
  endif ()
endif ()

# Configuration options
option(Hydrogen_EXPORT_PACKAGE_REGISTRY
  "Export the build directory to the user's CMake package registry." OFF)
if (NOT Hydrogen_EXPORT_PACKAGE_REGISTRY)
  set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON)
endif ()

option(CMAKE_POSITION_INDEPENDENT_CODE
  "Build using position-independent code" ON)

option(CMAKE_INTERPROCEDURAL_OPTIMIZATION
  "Build Hydrogen with interprocedural optimization enabled" OFF)

option(Hydrogen_ENABLE_BOUNDS_CHECKING
  "Do bounds checking in various routines."
  OFF)
set(HYDROGEN_DO_BOUNDS_CHECKING ${Hydrogen_ENABLE_BOUNDS_CHECKING})

option(Hydrogen_ENABLE_TESTING
  "Build the test suite in the tests directory." ON)

option(Hydrogen_ENABLE_UNIT_TESTS
  "Build the Catch2-based unit tests." OFF)

option(Hydrogen_ENABLE_QUADMATH
  "Search for quadmath library and enable related features if found." OFF)

option(Hydrogen_ENABLE_QD
  "Search for QD library and enable related features if found." OFF)

option(Hydrogen_ENABLE_MPC
  "Search for MPC(+MPFR+GMP) library and enable related features if found."
  OFF)

option(Hydrogen_ENABLE_ALUMINUM
  "Enable the Aluminum package for improved Allreduces."
  OFF)

option(Hydrogen_ENABLE_CUDA
  "Search for CUDA support and enable related features if found."
  OFF)

option(Hydrogen_ENABLE_ROCM
  "Search for ROCm/HIP support and enable related features if found."
  OFF)

if (Hydrogen_ENABLE_CUDA OR Hydrogen_ENABLE_ROCM)
  option(Hydrogen_ENABLE_CUB
    "Search for CUB support and enable related features if found."
    ON)

  option(Hydrogen_ENABLE_GPU_TENSOR_MATH
    "Use the GPU tensor operations when available."
    OFF)

  option(Hydrogen_ENABLE_GPU_FP16
    "Enable FP16 arithmetic in GPU code."
    ON)
endif ()

if (Hydrogen_ENABLE_ROCM AND Hydrogen_ENABLE_CUDA)
  message(FATAL_ERROR
    "ROCm and CUDA code paths are mutually exclusive. "
    "Please enable the one that corresponds to your hardware. "
    "If you have mixed hardware, please contact the Hydrogen developers "
    "as this would be of great interest.")
endif ()

#
# MEMORY-RELATED OPTIONS
#

option(Hydrogen_ENABLE_HALF
  "Enable the use of \"half\" library." OFF)
# This refers to the half.sourceforge.net type "half"

option(Hydrogen_ENABLE_BFLOAT16
  "Enable the use of Intel's bfloat16 type." OFF)
if (Hydrogen_ENABLE_BFLOAT16)
  message(WARNING
    "At time of writing, we are unaware of any implementation "
    "of the \"bfloat16\". This is a placeholder option. If our "
    "information is out of date, please open an issue and point "
    "us to an implementation.")
endif ()

option(Hydrogen_USE_64BIT_INTS
  "Use 64-bit integers for internal indexing" OFF)
option(Hydrogen_USE_64BIT_BLAS_INTS
  "Use 64-bit integers for BLAS/LAPACK" OFF)
if (Hydrogen_USE_64BIT_BLAS_INTS)
  set(EL_USE_64BIT_BLAS_INTS ON)
  set(Hydrogen_USE_64BIT_INTS ON)
endif ()

if (Hydrogen_USE_64BIT_INTS)
  set(EL_USE_64BIT_INTS ON)
endif ()

option(Hydrogen_ZERO_INIT "Initialize buffers to zero by default?" OFF)
mark_as_advanced(Hydrogen_ZERO_INIT)
if (Hydrogen_ZERO_INIT)
  set(EL_ZERO_INIT ${Hydrogen_ZERO_INIT})
endif ()

option(Hydrogen_ENABLE_VALGRIND
  "Search for valgrind and enable related features" OFF)
mark_as_advanced(Hydrogen_ENABLE_VALGRIND)
if (Hydrogen_ENABLE_VALGRIND)
  set(EL_ENABLE_VALGRIND ${Hydrogen_ENABLE_VALGRIND})
endif ()

#
# Profiling
#

option(Hydrogen_ENABLE_NVPROF
  "Search for NVProf and enable related features" OFF)
option(Hydrogen_ENABLE_VTUNE
  "Search for VTune and enable related features" OFF)
option(Hydrogen_ENABLE_SYNCHRONOUS_PROFILING
  "Perform compute-stream synchronization before beginning profile regions"
  OFF)
if (Hydrogen_ENABLE_SYNCHRONOUS_PROFILING)
  set(HYDROGEN_DEFAULT_SYNC_PROFILING TRUE)
endif ()

#
# OpenMP
#

option(Hydrogen_ENABLE_OPENMP
  "Enable OpenMP support in Hydrogen" OFF)

if (Hydrogen_ENABLE_OPENMP)
  option(Hydrogen_ENABLE_OMP_TASKLOOP
    "Use taskloops instead of parallel for" OFF)
endif ()

#
# MPI
#

# CUDA-aware MPI can provide a benefit, if it works. However, it is
# often buggy. We can avoid it by setting this to ON
option(Hydrogen_AVOID_CUDA_AWARE_MPI "Avoid CUDA-aware MPI" OFF)

option(Hydrogen_USE_CUSTOM_ALLTOALLV
  "Avoid MPI_Alltoallv for performance reasons" OFF)
mark_as_advanced(Hydrogen_USE_CUSTOM_ALLTOALLV)
if (Hydrogen_USE_CUSTOM_ALLTOALLV)
  set(EL_USE_CUSTOM_ALLTOALLV ${Hydrogen_USE_CUSTOM_ALLTOALLV})
endif ()

# Since it is surprisingly common for MPI libraries to have bugs in
# their support for complex data, the following option forces
# Elemental to cast all possible MPI communications in terms of twice
# as many real units of data.
option(Hydrogen_AVOID_COMPLEX_MPI
  "Avoid potentially buggy complex MPI routines" OFF)
mark_as_advanced(Hydrogen_AVOID_COMPLEX_MPI)
if (Hydrogen_AVOID_COMPLEX_MPI)
  set(EL_AVOID_COMPLEX_MPI ${Hydrogen_AVOID_COMPLEX_MPI})
endif ()

# Due to a subtle flaw in the Blue Gene/P extensions for MPICH2,
# treating floating-point data as a collection of byte-sized objects
# results in a better algorithm being chosen for MPI_Allgather. This
# should not effect performance on most machines.
option(Hydrogen_USE_BYTE_ALLGATHERS
  "Avoid BG/P allgather performance bug." OFF)
mark_as_advanced(Hydrogen_USE_BYTE_ALLGATHERS)
if (Hydrogen_USE_BYTE_ALLGATHERS)
  set(EL_USE_BYTE_ALLGATHERS ${Hydrogen_USE_BYTE_ALLGATHERS})
endif ()

#
# Runtime warnings
#

# Print a warning any time a redistribution is performed which unpacks
# a large amount of data with a non-unit stride
option(Hydrogen_CACHE_WARNINGS
  "Warns when using cache-unfriendly routines" OFF)
mark_as_advanced(Hydrogen_CACHE_WARNINGS)
if (Hydrogen_CACHE_WARNINGS)
  set(EL_CACHE_WARNINGS ${Hydrogen_CACHE_WARNINGS})
endif ()

# Print a warning when an improperly aligned redistribution is
# performed, i.e., if an unnecessary permutation communication stage
# must take place
option(Hydrogen_UNALIGNED_WARNINGS
  "Warn when performing unaligned redistributions" OFF)
mark_as_advanced(Hydrogen_UNALIGNED_WARNINGS)
if (Hydrogen_CACHE_WARNINGS)
  set(EL_CACHE_WARNINGS ${Hydrogen_UNALIGNED_WARNINGS})
endif ()

# Print a warning if an opportunity was missed to implement a
# redistribution approach specifically for vectors (instead of
# matrices)
option(Hydrogen_VECTOR_WARNINGS
  "Warn when vector redistribution chances are missed" OFF)
mark_as_advanced(Hydrogen_VECTOR_WARNINGS)
if (Hydrogen_VECTOR_WARNINGS)
  set(EL_VECTOR_WARNINGS ${Hydrogen_VECTOR_WARNINGS})
endif ()

#
# Check the compiler features
#

if (Hydrogen_ENABLE_HALF)
  find_package(HALF REQUIRED)
  set(HYDROGEN_HAVE_HALF TRUE)
endif ()

if (Hydrogen_ENABLE_CUDA)

  include(CheckLanguage)
  check_language(CUDA)

  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    if (NOT CMAKE_CUDA_STANDARD OR CMAKE_CUDA_STANDARD EQUAL 98)
      set(CMAKE_CUDA_STANDARD 11)
      set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
    endif ()

    find_package(CUDA REQUIRED) # Enable all the macros
    find_package(NVML REQUIRED)

    if (Hydrogen_ENABLE_GPU_TENSOR_MATH)
      set(HYDROGEN_GPU_USE_TENSOR_OP_MATH TRUE)
    endif ()

    if (Hydrogen_ENABLE_CUB)
      find_package(CUB REQUIRED)
      set(HYDROGEN_HAVE_CUB TRUE)
    else ()
      set(HYDROGEN_HAVE_CUB FALSE)
    endif ()

    if (Hydrogen_ENABLE_NVPROF)
      find_package(NVTX REQUIRED)
      set(HYDROGEN_HAVE_NVPROF TRUE)
    else ()
      set(HYDROGEN_HAVE_NVPROF FALSE)
    endif ()

    if (NOT TARGET cuda::toolkit)

      add_library(cuda::toolkit INTERFACE IMPORTED)

      foreach (lib IN LISTS CUDA_CUBLAS_LIBRARIES CUDA_LIBRARIES
          CUDA_CUDA_LIBRARY CUB_LIBRARIES NVML_LIBRARIES
          CUDA_cusolver_LIBRARY)

        if (lib)
          list(APPEND _CUDA_TOOLKIT_LIBS ${lib})
        endif ()

      endforeach ()

      set_property(TARGET cuda::toolkit PROPERTY
        INTERFACE_LINK_LIBRARIES "${_CUDA_TOOLKIT_LIBS}")

      set_property(TARGET cuda::toolkit PROPERTY
        INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_60>)

      set_property(TARGET cuda::toolkit PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}")
    endif ()
    set(HYDROGEN_HAVE_CUDA TRUE)
  else ()

    message(FATAL_ERROR
      "CUDA support not found. Disabling CUDA the features.")
    set(Hydrogen_ENABLE_CUDA FALSE)
    set(HYDROGEN_HAVE_CUDA FALSE)

  endif ()
endif (Hydrogen_ENABLE_CUDA)

if (Hydrogen_ENABLE_ROCM)
  set(CMAKE_MODULE_PATH "/opt/rocm/hip/cmake" ${CMAKE_MODULE_PATH})
  find_package(HIP REQUIRED)

  if (Hydrogen_ENABLE_CUB)
    set(CMAKE_PREFIX_PATH "/opt/rocm/hip" ${CMAKE_PREFIX_PATH})
    set(HIP_FOUND FALSE)
    find_package(HIP CONFIG REQUIRED)
    find_package(rocPRIM REQUIRED)
    find_package(hipCUB REQUIRED)
    set(HYDROGEN_HAVE_CUB TRUE)
  else ()
    set(HYDROGEN_HAVE_CUB FALSE)
  endif ()

  if (HIP_FOUND)
    set(CMAKE_CXX_EXTENSIONS FALSE)
    find_package(rocblas CONFIG REQUIRED)
    find_package(rocsolver CONFIG REQUIRED)
    set(HYDROGEN_HAVE_ROCM TRUE)
    message(STATUS "Found ROCm/HIP toolchain. Using HIP/ROCm.")
  else ()
    message(FATAL_ERROR "ROCm requested but not found.")
  endif ()
endif (Hydrogen_ENABLE_ROCM)

if (HYDROGEN_HAVE_CUDA OR HYDROGEN_HAVE_ROCM)
  set(HYDROGEN_HAVE_GPU TRUE)

  if (Hydrogen_ENABLE_GPU_FP16)
    set(HYDROGEN_GPU_USE_FP16 TRUE)
  endif ()
endif ()

if (Hydrogen_ENABLE_ALUMINUM)
  find_package(Aluminum 0.5.0 NO_MODULE
    HINTS ${Aluminum_DIR} ${ALUMINUM_DIR} ${AL_DIR}
    $ENV{Aluminum_DIR} $ENV{ALUMINUM_DIR} $ENV{AL_DIR}
    PATH_SUFFIXES lib64/cmake/aluminum lib/cmake/aluminum
    NO_DEFAULT_PATH)
  if (NOT Aluminum_FOUND)
    find_package(Aluminum 0.5.0 NO_MODULE)
  endif ()

  if (Aluminum_FOUND)
    set(HYDROGEN_HAVE_ALUMINUM TRUE)
    message(STATUS "Found Aluminum: ${Aluminum_DIR}")

    if (HYDROGEN_HAVE_GPU AND AL_HAS_NCCL)
      set(HYDROGEN_HAVE_NCCL2 TRUE)
      message(STATUS "Aluminum detected with NCCL2 backend support.")
    else ()
      set(HYDROGEN_HAVE_NCCL2 FALSE)
    endif (HYDROGEN_HAVE_GPU AND AL_HAS_NCCL)

    if (HYDROGEN_HAVE_GPU AND AL_HAS_HOST_TRANSFER)
      set(HYDROGEN_HAVE_AL_HOST_XFER TRUE)
      message(STATUS "Aluminum detected with HostTransfer backend support.")
    else ()
      set(HYDROGEN_HAVE_AL_HOST_XFER FALSE)
    endif (HYDROGEN_HAVE_GPU AND AL_HAS_HOST_TRANSFER)

    if (HYDROGEN_HAVE_GPU AND AL_HAS_MPI_CUDA)
      set(HYDROGEN_HAVE_AL_MPI_CUDA TRUE)
      message(STATUS "Aluminum detected with MPI-CUDA backend support.")
    else ()
      set(HYDROGEN_HAVE_AL_MPI_CUDA FALSE)
    endif (HYDROGEN_HAVE_GPU AND AL_HAS_MPI_CUDA)
  else ()
    set(HYDROGEN_HAVE_ALUMINUM FALSE)
    set(HYDROGEN_HAVE_NCCL2 FALSE)
    set(HYDROGEN_HAVE_AL_MPI_CUDA FALSE)
    set(Hydrogen_ENABLE_ALUMINUM FALSE)
    message(FATAL_ERROR "Aluminum support requested but not found. "
      "Please set Aluminum_DIR to point to the installation prefix "
      "for Aluminum.")
  endif (Aluminum_FOUND)
endif (Hydrogen_ENABLE_ALUMINUM)

include(detect/CXX)

#
# Find third-party libraries
#

if (Hydrogen_ENABLE_VTUNE)
  find_package(VTUNE REQUIRED)
  set(HYDROGEN_HAVE_VTUNE TRUE)
else ()
  set(HYDROGEN_HAVE_VTUNE FALSE)
endif ()

if (Hydrogen_ENABLE_OPENMP)
  include(FindAndVerifyOpenMP)
  if (OpenMP_FOUND)
    set(EL_HYBRID ON)
  else ()
    message(WARNING "Requested OpenMP support but OpenMP support was either "
      "not found or not functional.")
    set(EL_HYBRID OFF)
    set(Hydrogen_ENABLE_OPENMP OFF)
  endif ()
endif (Hydrogen_ENABLE_OPENMP)

include(FindAndVerifyMPI)
include(FindAndVerifyLAPACK)
include(FindAndVerifyExtendedPrecision)

# Catch2
if (Hydrogen_ENABLE_UNIT_TESTS)
  find_package(Catch2 2.0.0 CONFIG QUIET
    HINTS ${CATCH2_DIR} $ENV{CATCH2_DIR} ${CATCH_DIR} $ENV{CATCH_DIR}
    PATH_SUFFIXES lib64/cmake/Catch2 lib/cmake/Catch2
    NO_DEFAULT_PATH)
  if (NOT Catch2_FOUND)
    find_package(Catch2 2.0.0 CONFIG QUIET REQUIRED)
  endif ()
  message(STATUS "Found Catch2: ${Catch2_DIR}")

  include(Catch)
endif ()

# Macro for setting up full paths
macro(set_full_path VAR)
  unset(__tmp_names)
  foreach(filename ${ARGN})
    unset(__name)
    get_filename_component(__name "${filename}" ABSOLUTE)
    list(APPEND __tmp_names "${__name}")
  endforeach()
  set(${VAR} "${__tmp_names}")
endmacro()

set(HYDROGEN_HEADERS)
set(HYDROGEN_SOURCES)
add_subdirectory(include)
add_subdirectory(src)

configure_file("${PROJECT_SOURCE_DIR}/cmake/configure_files/config.h.in"
  "${PROJECT_BINARY_DIR}/include/El/config.h")
configure_file("${PROJECT_SOURCE_DIR}/cmake/configure_files/hydrogen_config.h.in"
  "${PROJECT_BINARY_DIR}/include/El/hydrogen_config.h")

configure_file("${PROJECT_SOURCE_DIR}/doxy/Doxyfile.in"
  "${PROJECT_BINARY_DIR}/doxy/Doxyfile")

if (HYDROGEN_HAVE_ROCM)
  hip_add_library(Hydrogen_CXX "${HYDROGEN_SOURCES}" "${HYDROGEN_HEADERS}")
else ()
  add_library(Hydrogen_CXX "${HYDROGEN_SOURCES}" "${HYDROGEN_HEADERS}")
endif ()

target_include_directories(Hydrogen_CXX PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/El>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# These flags don't need to be propagated. Just because Hydrogen_CXX is
# built with "-Werror", for instance, doesn't mean downstreams should
# be forced to build with that (even though they maybe should)...
target_compile_options(Hydrogen_CXX PRIVATE ${EXTRA_CXX_FLAGS})

target_link_libraries(
  Hydrogen_CXX PUBLIC
  ${Aluminum_LIBRARIES}
  ${HALF_LIBRARIES}
  ${VTUNE_LIBRARIES}
  ${NVTX_LIBRARIES}
  $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
  $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
  $<TARGET_NAME_IF_EXISTS:LAPACK::lapack>
  $<TARGET_NAME_IF_EXISTS:EP::extended_precision>
  $<TARGET_NAME_IF_EXISTS:cuda::toolkit>
  $<TARGET_NAME_IF_EXISTS:hip::rocprim_hip>
  $<TARGET_NAME_IF_EXISTS:hip::hipcub>
  $<TARGET_NAME_IF_EXISTS:roc::rocblas>
  $<TARGET_NAME_IF_EXISTS:roc::rocsolver>
  )

# Add the CXX library to "Hydrogen"
set(HYDROGEN_LIBRARIES Hydrogen_CXX)

if (HYDROGEN_HAVE_CUDA)
  add_library(Hydrogen_CUDA "${HYDROGEN_GPU_SOURCES}")
  target_include_directories(Hydrogen_CUDA PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)

  target_link_libraries(
    Hydrogen_CUDA PUBLIC
    ${HALF_LIBRARIES}
    ${NVTX_LIBRARIES}
    $<TARGET_NAME_IF_EXISTS:cuda::toolkit>
    )

  target_link_libraries(Hydrogen_CXX PUBLIC Hydrogen_CUDA)
  list(APPEND HYDROGEN_LIBRARIES Hydrogen_CUDA)
endif ()

if (HYDROGEN_HAVE_ROCM)
  hip_add_library(Hydrogen_ROCM STATIC "${HYDROGEN_GPU_SOURCES}")
  target_include_directories(Hydrogen_ROCM PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
    )

  target_link_libraries(Hydrogen_ROCM PUBLIC
    ${HALF_LIBRARIES}
    ${ROCBLAS_LIBRARIES}
    )

  #set_target_properties(Hydrogen_ROCM PROPERTIES LINKER_LANGUAGE CXX)
  list(APPEND HYDROGEN_LIBRARIES Hydrogen_ROCM)
endif ()

# Setup the tests
if (Hydrogen_ENABLE_TESTING OR Hydrogen_ENABLE_UNIT_TESTS)
  include(CTest)
endif ()

if (Hydrogen_ENABLE_TESTING)
  add_subdirectory(tests)
endif ()

if (Hydrogen_ENABLE_UNIT_TESTS)
  add_subdirectory(unit_test)
endif ()

# Setup the library install
install(TARGETS ${HYDROGEN_LIBRARIES}
  EXPORT HydrogenTargets
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include/El
  )

# Install target stuff
include (CMakePackageConfigHelpers)

# Write the version file for both the build and the install tree
configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/configure_files/HydrogenConfigVersion.cmake.in"
  "${CMAKE_BINARY_DIR}/HydrogenConfigVersion.cmake"
  @ONLY)

# Write the configure file for the build tree
set(INCLUDE_INSTALL_DIRS "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_SOURCE_DIR}/include/El"
  "${CMAKE_BINARY_DIR}/include/El")
set(LIB_INSTALL_DIR "${CMAKE_BINARY_DIR}")
set(EXTRA_CMAKE_MODULE_DIR "${CMAKE_SOURCE_DIR}/cmake/modules")
configure_package_config_file(cmake/configure_files/HydrogenConfig.cmake.in
  "${CMAKE_BINARY_DIR}/HydrogenConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_BINARY_DIR}"
  PATH_VARS INCLUDE_INSTALL_DIRS LIB_INSTALL_DIR)

# Build tree export
export(EXPORT HydrogenTargets NAMESPACE H:: FILE HydrogenTargets.cmake)

# Write the configure file for the install tree
set(INCLUDE_INSTALL_DIRS include)
set(LIB_INSTALL_DIR lib)
set(CMAKE_INSTALL_DIR ${LIB_INSTALL_DIR}/cmake/hydrogen)
set(EXTRA_CMAKE_MODULE_DIR)
configure_package_config_file(
  cmake/configure_files/HydrogenConfig.cmake.in
  "${CMAKE_BINARY_DIR}/cmake/HydrogenConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_DIR}
  PATH_VARS INCLUDE_INSTALL_DIRS LIB_INSTALL_DIR)

# Install the configuration headers
install(FILES
  "${PROJECT_BINARY_DIR}/include/El/config.h"
  "${PROJECT_BINARY_DIR}/include/El/hydrogen_config.h"
  DESTINATION "include/El")

# Install tree export
install(EXPORT HydrogenTargets
  NAMESPACE H::
  DESTINATION ${CMAKE_INSTALL_DIR})

# Install the headers
install(DIRECTORY include DESTINATION .
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")

# Install the cmake stuff
install(FILES
  "${PROJECT_BINARY_DIR}/cmake/HydrogenConfig.cmake"
  "${PROJECT_BINARY_DIR}/HydrogenConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_DIR})
install(DIRECTORY cmake/modules DESTINATION ${CMAKE_INSTALL_DIR}
  FILES_MATCHING PATTERN "*.cmake")

# Summarize this configuration
if (NOT __dont_print_summary)
  include(PrintHydrogenSummary)
  print_full_hydrogen_summary(
    VERSION_VARIABLES
    HYDROGEN_VERSION
    HYDROGEN_GIT_VERSION
    HYDROGEN_LATEST_SHA
    HYDROGEN_LATEST_HYDROGEN_SHA

    PATH_VARIABLES
    CMAKE_CXX_COMPILER
    CMAKE_CUDA_COMPILER
    CMAKE_CUDA_HOST_COMPILER
    MPI_CXX_COMPILER
    CMAKE_INSTALL_PREFIX
    CMAKE_SOURCE_DIR
    CMAKE_BINARY_DIR

    STRING_VARIABLES
    CMAKE_BUILD_TYPE
    CMAKE_CUDA_FLAGS
    CMAKE_CXX_FLAGS
    HYDROGEN_BLAS_SUFFIX
    HYDROGEN_LAPACK_SUFFIX

    BOOLEAN_VARIABLES
    BUILD_SHARED_LIBS
    Hydrogen_ENABLE_TESTING
    HYDROGEN_HAVE_QUADMATH
    HYDROGEN_HAVE_QD
    HYDROGEN_HAVE_GMP
    HYDROGEN_HAVE_MPFR
    HYDROGEN_HAVE_MPC
    HYDROGEN_HAVE_ALUMINUM
    HYDROGEN_HAVE_NCCL2
    HYDROGEN_HAVE_AL_MPI_CUDA
    HYDROGEN_HAVE_CUDA
    HYDROGEN_HAVE_CUB
    HYDROGEN_HAVE_OMP_TASKLOOP
    HYDROGEN_HAVE_CUDA_AWARE_MPI
    HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    HYDROGEN_HAVE_OPENBLAS
    HYDROGEN_HAVE_LAPACK
    HYDROGEN_HAVE_MKL
    HYDROGEN_HAVE_MKL_GEMMT
    EL_USE_64BIT_INTS
    EL_USE_64BIT_BLAS_INTS
    EL_ZERO_INIT
    EL_HAVE_VALGRIND
    EL_HYBRID
    EL_HAVE_OPENMP
    EL_HAVE_OMP_COLLAPSE
    EL_HAVE_OMP_SIMD
    EL_USE_CUSTOM_ALLTOALLV
    EL_AVOID_COMPLEX_MPI
    EL_USE_BYTE_ALLGATHERS
    EL_CACHE_WARNINGS
    EL_VECTOR_WARNINGS

    PRINT_EMPTY_VARIABLES
    )
  set(__dont_print_summary ON CACHE INTERNAL "" FORCE)
  # Only print the configuration on initial configuration
endif ()
