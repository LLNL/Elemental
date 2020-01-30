# Exports the following variables
#
#   HALF_FOUND
#   HALF_INCLUDE_PATH
#   HALF_LIBRARIES
#
# Also adds the following imported target:
#
#   half::half
#

find_path(HALF_INCLUDE_PATH half.hpp
  HINTS ${HALF_DIR} $ENV{HALF_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "The HALF header directory."
  )
find_path(HALF_INCLUDE_PATH half.hpp)

include(CheckCXXSourceCompiles)
set(_half_verify_code
  "#ifndef HALF_ENABLE_F16C_INTRINSICS
#define HALF_ENABLE_F16C_INTRINSICS __F16C__
#endif

#include <half.hpp>
int main(int, char**)
{
  half_float::half x
    = half_float::half_cast<half_float::half>(9.0);
}")
set(CMAKE_REQUIRED_INCLUDES ${HALF_INCLUDE_PATH})
check_cxx_source_compiles(
  "${_half_verify_code}" HALF_HEADER_OK)
set(CMAKE_REQUIRED_INCLUDES)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HALF
  DEFAULT_MSG HALF_INCLUDE_PATH HALF_HEADER_OK)

# Setup the imported target
if (HALF_FOUND)
  if (NOT TARGET half::half)
    add_library(half::half INTERFACE IMPORTED)
  endif (NOT TARGET half::half)

  # Set the include directories for the target
  set_property(TARGET half::half
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${HALF_INCLUDE_PATH})

  # Set the libraries
  set(HALF_LIBRARIES half::half)
endif (HALF_FOUND)

#
# Cleanup
#

# Set the include directories
mark_as_advanced(FORCE HALF_INCLUDE_PATH)
