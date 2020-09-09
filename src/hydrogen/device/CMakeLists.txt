if (HYDROGEN_HAVE_GPU)
  if (HYDROGEN_HAVE_CUDA)
    set_full_path(THIS_DIR_CXX_SOURCES
      CUDA.cpp
      cuBLAS.cpp
      cuBLAS_API.cpp
      cuSOLVER.cpp
      cuSOLVER_API.cpp
      GPU.cpp)
  endif ()
  if (HYDROGEN_HAVE_ROCM)
    set_full_path(THIS_DIR_CXX_SOURCES
      GPU.cpp
      ROCm.cpp
      rocBLAS.cpp
      rocBLAS_API.cpp
      rocSOLVER.cpp
      rocSOLVER_API.cpp
      )
  endif ()

  set(SOURCES "${SOURCES}" "${THIS_DIR_CXX_SOURCES}" PARENT_SCOPE)
endif ()
