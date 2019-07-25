# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  blas.cpp
  flame.cpp
  lapack.cpp
  mkl.cpp
  mpfr.cpp
  mpi.cpp
  openblas.cpp
  qd.cpp
  qt5.cpp
  scalapack.cpp
  )

if (HYDROGEN_HAVE_CUB)
  set_full_path(CUB_SOURCES cub.cpp)
  list(APPEND THIS_DIR_SOURCES ${CUB_SOURCES})
endif()

# Add the subdirectories
add_subdirectory(blas)
if (HYDROGEN_HAVE_LAPACK)
  add_subdirectory(lapack)
endif ()
add_subdirectory(mpfr)
add_subdirectory(scalapack)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
