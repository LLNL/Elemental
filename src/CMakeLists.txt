# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  io.cpp
  )

# Add the subdirectories
add_subdirectory(blas_like)
#add_subdirectory(control)
add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(lapack_like)
add_subdirectory(matrices)
add_subdirectory(hydrogen)

# Propagate the files up the tree
set(HYDROGEN_SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
