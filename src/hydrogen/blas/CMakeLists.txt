if (HYDROGEN_HAVE_GPU)
  add_subdirectory(gpu)

  set(GPU_SOURCES "${GPU_SOURCES}" PARENT_SCOPE)
endif ()
