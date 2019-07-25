# Find rocBLAS library and supporting header
#
#   rocBLAS_DIR or ROCBLAS_DIR[in]: The prefix for rocBLAS
#
#   ROCBLAS_INCLUDE_PATH[out,cache]: The include path for rocBLAS
#   ROCBLAS_LIBRARY[out,cache]: The rocBLAS library
#
#   ROCBLAS_LIBRARIES[out]: The thing to link to for rocBLAS
#   ROCBLAS_FOUND[out]: Variable indicating whether rocBLAS has been found
#
#   rocm::rocblas: Imported library for rocBLAS
#

find_path(ROCBLAS_INCLUDE_PATH rocblas.h
  HINTS ${rocBLAS_DIR} $ENV{rocBLAS_DIR} ${ROCBLAS_DIR} $ENV{ROCBLAS_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "The rocBLAS include path.")
find_path(ROCBLAS_INCLUDE_PATH rocblas.h)

find_library(ROCBLAS_LIBRARY rocblas
  HINTS ${rocBLAS_DIR} $ENV{rocBLAS_DIR} ${ROCBLAS_DIR} $ENV{ROCBLAS_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The rocBLAS library.")
find_library(ROCBLAS_LIBRARY rocblas)
  
# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Rocblas
  REQUIRED_VARS ROCBLAS_LIBRARY ROCBLAS_INCLUDE_PATH)

if (NOT TARGET rocblas::rocblas)
  add_library(rocblas::rocblas INTERFACE IMPORTED)
endif ()

if (ROCBLAS_INCLUDE_PATH AND ROCBLAS_LIBRARY)
  set_target_properties(rocblas::rocblas PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ROCBLAS_INCLUDE_PATH}"
    INTERFACE_LINK_LIBRARIES "${ROCBLAS_LIBRARY}")
endif ()

set(ROCBLAS_LIBRARIES rocblas::rocblas)
mark_as_advanced(ROCBLAS_INCLUDE_PATH)
mark_as_advanced(ROCBLAS_LIBRARY)
