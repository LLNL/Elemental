cmake_minimum_required(VERSION 3.9)

project(Hydrogen C CXX)

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif ()

set(HYDROGEN_VERSION_MAJOR 0)
set(HYDROGEN_VERSION_MINOR 1)

# Back-compatibility, until it's all sorted out
set(EL_VERSION_MAJOR ${HYDROGEN_VERSION_MAJOR})
set(EL_VERSION_MINOR ${HYDROGEN_VERSION_MINOR})

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
check_cxx_compiler_flag("-pedantic" CXX_COMPILER_HAS_PEDANTIC_FLAG)
check_cxx_compiler_flag("-Wpedantic" CXX_COMPILER_HAS_WPEDANTIC_FLAG)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")
if (CXX_COMPILER_HAS_PEDANTIC_FLAG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
elseif (CXX_COMPILER_HAS_WPEDANTIC_FLAG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wpedantic")
endif ()

if(USE_ALUMINUM)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHYDROGEN_USES_ALUMINUM")
  if(USE_NCCL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHYDROGEN_USES_NCCL2")
  endif ()
endif ()
  
# Configuration options
option(${PROJECT_NAME}_EXPORT_PACKAGE_REGISTRY
  "Export the build directory to the user's CMake package registry." OFF)
if (NOT ${PROJECT_NAME}_EXPORT_PACKAGE_REGISTRY)
  set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON)
endif ()

option(CMAKE_POSITION_INDEPENDENT_CODE
  "Build using position-independent code" ON)

option(${PROJECT_NAME}_ENABLE_TESTING "Build the test suite." ON)

option(${PROJECT_NAME}_ENABLE_QUADMATH
  "Search for quadmath library and enable related features if found." OFF)

option(${PROJECT_NAME}_ENABLE_QD
  "Search for QD library and enable related features if found." OFF)

option(${PROJECT_NAME}_ENABLE_MPC
  "Search for MPC(+MPFR+GMP) library and enable related features if found."
  OFF)

option(${PROJECT_NAME}_ENABLE_CUDA
  "Search for CUDA support and enable related features if found."
  ON)

if (${PROJECT_NAME}_ENABLE_CUDA)
  option(${PROJECT_NAME}_ENABLE_CUB
    "Search for CUB support and enable related features if found."
    ON)
endif ()

# Legacy options, with EL_ prefix intact
option(BINARY_SUBDIRECTORIES "Install binaries into tree based on type" ON)

#
# MEMORY-RELATED OPTIONS
#

option(${PROJECT_NAME}_USE_64BIT_INTS
  "Use 64-bit integers for internal indexing" OFF)
option(${PROJECT_NAME}_USE_64BIT_BLAS_INTS
  "Use 64-bit integers for BLAS/LAPACK" OFF)
if (${PROJECT_NAME}_USE_64BIT_BLAS_INTS)
  set(EL_USE_64BIT_BLAS_INTS ON)
  set(${PROJECT_NAME}_USE_64BIT_INTS ON)
endif ()

if (${PROJECT_NAME}_USE_64BIT_INTS)
  set(EL_USE_64BIT_INTS ON)
endif ()

option(${PROJECT_NAME}_ZERO_INIT "Initialize buffers to zero by default?" OFF)
mark_as_advanced(${PROJECT_NAME}_ZERO_INIT)

option(${PROJECT_NAME}_ENABLE_VALGRIND
  "Search for valgrind and enable related features" OFF)
mark_as_advanced(${PROJECT_NAME}_ENABLE_VALGRIND)

#
# OpenMP
#

option(${PROJECT_NAME}_ENABLE_OPENMP
  "Enable OpenMP support in Hydrogen" OFF)

# At one point, a bug was found in IBM's C++ compiler for Blue Gene/P,
# where OpenMP statements of the form a[i] += alpha b[i], with complex data,
# would segfault and/or return incorrect results
option(${PROJECT_NAME}_AVOID_OMP_FMA "Avoid a bug in the IBM compilers." OFF)
mark_as_advanced(${PROJECT_NAME}_AVOID_OMP_FMA)

#
# MPI
#

option(${PROJECT_NAME}_USE_CUSTOM_ALLTOALLV "Avoid MPI_Alltoallv for performance reasons" ON)
mark_as_advanced(${PROJECT_NAME}_USE_CUSTOM_ALLTOALLV)

# Since it is surprisingly common for MPI libraries to have bugs in their
# support for complex data, the following option forces Elemental to cast
# all possible MPI communications in terms of twice as many real units of data.
option(${PROJECT_NAME}_AVOID_COMPLEX_MPI "Avoid potentially buggy complex MPI routines" ON)
mark_as_advanced(${PROJECT_NAME}_AVOID_COMPLEX_MPI)

# Due to a subtle flaw in the Blue Gene/P extensions for MPICH2, treating
# floating-point data as a collection of byte-sized objects results in a
# better algorithm being chosen for MPI_Allgather. This should not effect
# performance on most machines.
option(${PROJECT_NAME}_USE_BYTE_ALLGATHERS "Avoid BG/P allgather performance bug." ON)
mark_as_advanced(${PROJECT_NAME}_USE_BYTE_ALLGATHERS)

# If MPI_Reduce_scatter_block doesn't exist, perform it by composing
# MPI_Allreduce and std::memcpy rather than MPI_Reduce and MPI_Scatter
option(${PROJECT_NAME}_REDUCE_SCATTER_BLOCK_VIA_ALLREDUCE
       "AllReduce based block MPI_Reduce_scatter" OFF)
mark_as_advanced(${PROJECT_NAME}_REDUCE_SCATTER_BLOCK_VIA_ALLREDUCE)

#
# Runtime warnings
#

# Print a warning any time a redistribution is performed which unpacks a
# large amount of data with a non-unit stride
option(${PROJECT_NAME}_CACHE_WARNINGS "Warns when using cache-unfriendly routines" OFF)
mark_as_advanced(${PROJECT_NAME}_CACHE_WARNINGS)

# Print a warning when an improperly aligned redistribution is performed,
# i.e., if an unnecessary permutation communication stage must take place
option(${PROJECT_NAME}_UNALIGNED_WARNINGS
       "Warn when performing unaligned redistributions" OFF)
mark_as_advanced(${PROJECT_NAME}_UNALIGNED_WARNINGS)

# Print a warning if an opportunity was missed to implement a redistribution
# approach specifically for vectors (instead of matrices)
option(${PROJECT_NAME}_VECTOR_WARNINGS
       "Warn when vector redistribution chances are missed" OFF)
mark_as_advanced(${PROJECT_NAME}_VECTOR_WARNINGS)

#
# Check the compiler features
#

if (${PROJECT_NAME}_ENABLE_CUDA)

  include(CheckLanguage)
  check_language(CUDA)

  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    if (NOT CMAKE_CUDA_STANDARD OR CMAKE_CUDA_STANDAR EQUAL 98)
      set(CMAKE_CUDA_STANDARD 11)
      set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
    endif ()

    find_package(CUDA) # Enable all the macros

    if (${PROJECT_NAME}_ENABLE_CUB)
      find_package(CUB REQUIRED)
      set(HYDROGEN_HAVE_CUB TRUE)
    else ()
      set(HYDROGEN_HAVE_CUB FALSE)
    endif ()

    if (NOT TARGET cuda::toolkit)

      add_library(cuda::toolkit INTERFACE IMPORTED)

      set_property(TARGET cuda::toolkit PROPERTY
        INTERFACE_LINK_LIBRARIES
        "${CUDA_CUBLAS_LIBRARIES}" "${CUDA_LIBRARIES}" "${CUB_LIBRARIES}")

      set_property(TARGET cuda::toolkit PROPERTY
        INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_30>)

      set_property(TARGET cuda::toolkit PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}")
    endif ()
    set(HYDROGEN_HAVE_CUDA TRUE)

  else ()

    message(FATAL_ERROR "CUDA support not found. Disabling CUDA the features.")
    set(${PROJECT_NAME}_ENABLE_CUDA FALSE)
    set(HYDROGEN_HAVE_CUDA FALSE)

  endif ()

endif (${PROJECT_NAME}_ENABLE_CUDA)

# FIXME: GET RID OF FORTRAN
include(detect/CXX)
include(detect/Fortran)

#
# Find third-party libraries
#

if (${PROJECT_NAME}_ENABLE_OPENMP)
  include(FindAndVerifyOpenMP)
  if (OpenMP_FOUND)
    set(EL_HYBRID ON)
    set(EL_HAVE_OPENMP TRUE)
  else ()
    message(WARNING "Requested OpenMP support but OpenMP support was either "
      "not found or not functional.")
    set(EL_HYBRID OFF)
    set(EL_HAVE_OPENMP FALSE)
    set(${PROJECT_NAME}_ENABLE_OPENMP OFF)
  endif ()
endif (${PROJECT_NAME}_ENABLE_OPENMP)

include(FindAndVerifyMPI)
include(FindAndVerifyLAPACK)
include(FindAndVerifyExtendedPrecision)

# External projects build internally
# TODO Investigate why
add_subdirectory(external/pmrrr)

# Add SuiteSparse (TODO: Why bundled here?)
add_subdirectory(external/suite_sparse)
include_directories(external/suite_sparse/include)

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

add_library(${PROJECT_NAME} "${HYDROGEN_SOURCES}" "${HYDROGEN_HEADERS}")
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/El>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

set(ALUMINUM_HOME /usr/workspace/wsb/brain/aluminum)
set(ALUMINUM_INC ${ALUMINUM_HOME}/include)
set(ALUMINUM_LIB ${ALUMINUM_HOME}/lib)
include_directories(${ALUMINUM_INC})

set(NCCLINC ${NCCL_HOME}/include)
set(NCCLLIB ${NCCL_HOME}/lib)
include_directories(${NCCLINC})

#link_directories(/usr/workspace/wsb/brain/aluminum/lib)
# Your-external "mylib", add GLOBAL if the imported library is located in directories above the current.
add_library( Al SHARED IMPORTED )
# You can define two import-locations: one for debug and one for release.
set_target_properties( Al  PROPERTIES IMPORTED_LOCATION ${ALUMINUM_LIB}/libAl.so)

target_link_libraries(${PROJECT_NAME} PRIVATE Al )

if (TARGET OpenMP::OpenMP_CXX)
  target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif ()
target_link_libraries(${PROJECT_NAME} PUBLIC pmrrr)
target_link_libraries(${PROJECT_NAME} PUBLIC MPI::MPI_CXX)
target_link_libraries(${PROJECT_NAME} PUBLIC LAPACK::lapack)
target_link_libraries(${PROJECT_NAME} PUBLIC EP::extended_precision)
if (HYDROGEN_HAVE_CUDA)
  target_link_libraries(${PROJECT_NAME} PUBLIC cuda::toolkit)
endif ()

if (BUILD_SHARED_LIBS)
  if (APPLE)
    target_link_libraries(${PROJECT_NAME} PUBLIC "-Wl,-undefined,error")
  else ()
    target_link_libraries(${PROJECT_NAME} PUBLIC "-Wl,-no-undefined")
  endif ()
endif ()

# Setup the tests
if (${PROJECT_NAME}_ENABLE_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif ()

# Setup the library install
install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include/El
  )

# Install target stuff
include (CMakePackageConfigHelpers)

# Write the version file for both the build and the install tree
write_basic_package_version_file(
  "${CMAKE_BINARY_DIR}/HydrogenConfigVersion.cmake"
  VERSION "${HYDROGEN_VERSION_MAJOR}.${HYDROGEN_VERSION_MINOR}"
  COMPATIBILITY AnyNewerVersion
  )

# Write the configure file for the build tree
set(INCLUDE_INSTALL_DIRS "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_SOURCE_DIR}/include/El"
  "${CMAKE_BINARY_DIR}/include/El")
set(LIB_INSTALL_DIR "${CMAKE_BINARY_DIR}")
configure_package_config_file(cmake/configure_files/HydrogenConfig.cmake.in
  "${CMAKE_BINARY_DIR}/HydrogenConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_BINARY_DIR}"
  PATH_VARS INCLUDE_INSTALL_DIRS LIB_INSTALL_DIR)

# Build tree export
export(EXPORT HydrogenTargets NAMESPACE H:: FILE HydrogenTargets.cmake)

# Write the configure file for the install tree
set(INCLUDE_INSTALL_DIRS include)
set(LIB_INSTALL_DIR lib)
set(CMAKE_INSTALL_DIR lib/cmake/hydrogen)

configure_package_config_file(cmake/configure_files/HydrogenConfig.cmake.in
  "${CMAKE_BINARY_DIR}/cmake/HydrogenConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_DIR}
  PATH_VARS INCLUDE_INSTALL_DIRS LIB_INSTALL_DIR)

# Install the configuration headers
install(FILES
  "${PROJECT_BINARY_DIR}/include/El/config.h"
  "${PROJECT_BINARY_DIR}/include/El/hydrogen_config.h"
  DESTINATION "include/El")

# Install tree export
install(EXPORT ${PROJECT_NAME}Targets
  NAMESPACE H::
  DESTINATION lib/cmake/hydrogen)

# Install the headers
install(DIRECTORY include DESTINATION .
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")

# Install the cmake stuff
install(FILES
  "${PROJECT_BINARY_DIR}/cmake/HydrogenConfig.cmake"
  "${PROJECT_BINARY_DIR}/HydrogenConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_DIR})
install(DIRECTORY cmake/modules DESTINATION lib/cmake/hydrogen
  FILES_MATCHING PATTERN "*.cmake")
