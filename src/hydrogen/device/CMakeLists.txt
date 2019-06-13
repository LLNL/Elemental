if (HYDROGEN_HAVE_GPU)
  if (HYDROGEN_HAVE_CUDA)
    set_full_path(THIS_DIR_CXX_SOURCES
      CUDA.cpp
      cuBLAS.cpp
      cuBLAS_API.cpp)
  endif ()
  if (HYDROGEN_HAVE_HIP)
    set_full_path(THIS_DIR_CXX_SOURCES
      ROCm.cpp)
  endif ()

  set(SOURCES "${SOURCES}" "${THIS_DIR_CXX_SOURCES}" PARENT_SCOPE)
endif ()
