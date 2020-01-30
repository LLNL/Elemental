# This will handle the logic surrounding BLAS/LAPACK
#
# Options that affect the choices made here:
#
#   BLA_VENDOR -- If set, use this value and ignore all other options.
#   Hydrogen_USE_MKL -- If set, look for MKL installations.
#   Hydrogen_USE_OpenBLAS -- If set, look for OpenBLAS implementation.
#   Hydrogen_USE_ACCELERATE -- If set, skip other searches in favor of
#       Apple's accelerate framework
#   Hydrogen_USE_GENERIC_LAPACK -- If set, skip other searches in
#       favor of a generic LAPACK library
#   Hydrogen_GENERAL_LAPACK_FALLBACK -- If set and no other LAPACK
#       library has been found, this will call "find_package(LAPACK)"
#       with all influential variables to that module cleared.
#
# In addition to the variables set by the standard FindBLAS and
# FindLAPACK modules, this module also defines the IMPORTED library
# "LAPACK::lapack" and might output the variable
#
#   HYDROGEN_BLAS_SUFFIX -- Only defined if BLAS requires a suffix, usually "_".
#   HYDROGEN_LAPACK_SUFFIX -- Only defined if LAPACK requires a suffix, usually "_".
#
# This will prioritize BLA_VENDOR, MKL, OpenBLAS, Apple, Generic, and
# setting multiple options above will cause a short-circuit if any are
# found in that order.
#
# Ultimately, this boils down to repeated calls to
# "find_package(LAPACK)" with changes to the environment. These calls,
# of course, can be avoided if LAPACK_LIBRARIES and related variables
# are set on the command line.

if (NOT UPPER_PROJECT_NAME)
  string(TOUPPER "${PROJECT_NAME}" UPPER_PROJECT_NAME)
endif ()

# Check straight-away if BLA_VENDOR is already set
if (BLA_VENDOR)
  find_package(LAPACK QUIET)
endif (BLA_VENDOR)

# Check for MKL
# TODO (trb 10.26.2017): add LP model support if 64bit indices
if (${PROJECT_NAME}_USE_MKL AND NOT LAPACK_FOUND)
  set(BLA_VENDOR "Intel10_64")
  find_package(LAPACK QUIET)
endif (${PROJECT_NAME}_USE_MKL AND NOT LAPACK_FOUND)

# Check for OpenBLAS
if (${PROJECT_NAME}_USE_OpenBLAS AND NOT LAPACK_FOUND)
  set(BLA_VENDOR "OpenBLAS")
  set(LAPACK_DIR "${OpenBLAS_DIR}")
  set(BLAS_DIR "${OpenBLAS_DIR}")
  find_package(LAPACK)

  # Build OpenBLAS if requested
  if (NOT LAPACK_FOUND)

    find_package(OpenBLAS NO_MODULE REQUIRED
      PATH_SUFFIXES lib/cmake/openblas)

    # FIXME: So it looks like this might be the only way to trick the
    # FindLAPACK module?? Another possibility is that we just say that
    # LAPACK_FOUND=BLAS_FOUND=OpenBLAS_FOUND,
    # LAPACK_LIBRARIES=BLAS_LIBRARIES=OpenBLAS_LIBRARIES, and no
    # {BLAS,LAPACK}_LINK_FLAGS.
    get_filename_component(_openblas_lib_dir "${OpenBLAS_LIBRARIES}" DIRECTORY)
    if (APPLE)
      set(_tmp_ld_path $ENV{DYLD_LIBRARY_PATH})
      set(ENV{DYLD_LIBRARY_PATH} "${_openblas_lib_dir}:$ENV{DYLD_LIBRARY_PATH}")
      find_package(LAPACK REQUIRED)
      set(ENV{DYLD_LIBRARY_PATH} "${_tmp_ld_path}")
    else ()
      set(_tmp_ld_path $ENV{LD_LIBRARY_PATH})
      set(ENV{LD_LIBRARY_PATH} "${_openblas_lib_dir}:$ENV{LD_LIBRARY_PATH}")
      find_package(LAPACK REQUIRED)
      set(ENV{LD_LIBRARY_PATH} "${_tmp_ld_path}")
    endif ()
  endif()
  set(${UPPER_PROJECT_NAME}_HAVE_OPENBLAS ${LAPACK_FOUND})
endif (${PROJECT_NAME}_USE_OpenBLAS AND NOT LAPACK_FOUND)

# Set the flag here
set(EL_HAVE_OPENBLAS ${${UPPER_PROJECT_NAME}_HAVE_OPENBLAS})

# Check for Accelerate
if (APPLE AND ${PROJECT_NAME}_USE_ACCELERATE AND NOT LAPACK_FOUND)
  set(BLA_VENDOR "Apple")
  find_package(LAPACK QUIET)
endif (APPLE AND ${PROJECT_NAME}_USE_ACCELERATE AND NOT LAPACK_FOUND)

# Check for a generic lapack build
if (${PROJECT_NAME}_USE_GENERIC_LAPACK AND NOT LAPACK_FOUND)
  set(BLA_VENDOR "Generic")
  find_package(LAPACK QUIET)
endif (${PROJECT_NAME}_USE_GENERIC_LAPACK AND NOT LAPACK_FOUND)

# Check for a fallback if requested
if (${PROJECT_NAME}_GENERAL_LAPACK_FALLBACK AND NOT LAPACK_FOUND)
  set(BLA_VENDOR "All")
  find_package(LAPACK QUIET)
endif (${PROJECT_NAME}_GENERAL_LAPACK_FALLBACK AND NOT LAPACK_FOUND)

find_package(LAPACK)

# Wrap-up
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HydrogenLAPACK
  DEFAULT_MSG
  LAPACK_FOUND LAPACK_LIBRARIES)

if (LAPACK_FOUND)
  message(STATUS "Found LAPACK: ${LAPACK_LIBRARIES}")
else ()
  message(FATAL_ERROR "No LAPACK library found!")
endif ()

# Define the imported target
if (NOT TARGET LAPACK::lapack)
  add_library(LAPACK::lapack INTERFACE IMPORTED)

  # This only uses the LAPACK_* variables. LAPACK_LIBRARIES _should_
  # include BLAS_LIBRARIES and LAPACK_LINKER_FLAGS _should_ include
  # BLAS_LINKER_FLAGS. If it is ever discovered that this isn't true,
  # it can be fixed.
  if (LAPACK_LINKER_FLAGS)
    if (UNIX)# Or we could force CMake 3.9 and use NATIVE_COMMAND!
      separate_arguments(__link_flags UNIX_COMMAND "${LAPACK_LINKER_FLAGS}")
    else ()
      separate_arguments(__link_flags WINDOWS_COMMAND "${LAPACK_LINKER_FLAGS}")
    endif ()
  endif ()
  list(APPEND __link_flags "${LAPACK_LIBRARIES}")

  set_property(TARGET LAPACK::lapack PROPERTY
    INTERFACE_LINK_LIBRARIES "${__link_flags}")

endif (NOT TARGET LAPACK::lapack)

# Detect the suffix
include(CheckFunctionExists)

set(CMAKE_REQUIRED_LIBRARIES "${BLAS_LINKER_FLAGS}" "${BLAS_LIBRARIES}")
check_function_exists(dgemm BLAS_NO_USE_UNDERSCORE)
check_function_exists(dgemm_ BLAS_USE_UNDERSCORE)

set(CMAKE_REQUIRED_LIBRARIES "${LAPACK_LINKER_FLAGS}" "${LAPACK_LIBRARIES}")
check_function_exists(dlacpy LAPACK_NO_USE_UNDERSCORE)
check_function_exists(dlacpy_ LAPACK_USE_UNDERSCORE)

# If both dgemm and dgemm_ are found, don't use the suffix
if (BLAS_NO_USE_UNDERSCORE)
  unset(${UPPER_PROJECT_NAME}_BLAS_SUFFIX)
  message(STATUS "Using BLAS with no symbol mangling.")
elseif (BLAS_USE_UNDERSCORE)
  set(${UPPER_PROJECT_NAME}_BLAS_SUFFIX "_")
  message(STATUS "Using BLAS with trailing underscore.")
else ()
  message(FATAL_ERROR "Could not determine BLAS suffix!")
endif ()

set(${UPPER_PROJECT_NAME}_HAVE_LAPACK ON)
if (LAPACK_NO_USE_UNDERSCORE)
  unset(${UPPER_PROJECT_NAME}_LAPACK_SUFFIX)
  message(STATUS "Using LAPACK with no symbol mangling.")
elseif (LAPACK_USE_UNDERSCORE)
  set(${UPPER_PROJECT_NAME}_LAPACK_SUFFIX "_")
  message(STATUS "Using LAPACK with trailing underscore.")
else ()
  message(STATUS
    "Could not find LAPACK's copy function. Disabling LAPACK support.")
  set(${UPPER_PROJECT_NAME}_HAVE_LAPACK)
endif ()

# Check a few MKL features
check_function_exists(mkl_dcsrmv ${UPPER_PROJECT_NAME}_HAVE_MKL)
if (${UPPER_PROJECT_NAME}_HAVE_MKL)
  check_function_exists(dgemmt  ${UPPER_PROJECT_NAME}_HAVE_MKL_GEMMT)
  check_function_exists(dgemmt_ ${UPPER_PROJECT_NAME}_HAVE_MKL_GEMMT)
endif ()
set(CMAKE_REQUIRED_LIBRARIES)
