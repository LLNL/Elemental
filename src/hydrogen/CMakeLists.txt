add_subdirectory(blas)
add_subdirectory(device)

if (HYDROGEN_HAVE_GPU)
  set(GPU_SOURCES "${GPU_SOURCES}" PARENT_SCOPE)
endif ()

set_full_path(THIS_DIR_CXX_SOURCES
  Error.cpp)

set(SOURCES "${SOURCES}" "${THIS_DIR_CXX_SOURCES}" PARENT_SCOPE)
