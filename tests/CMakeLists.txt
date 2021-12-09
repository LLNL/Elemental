# Add the subdirectories
add_subdirectory(blas_like)
add_subdirectory(core)
add_subdirectory(lapack_like)

foreach (src_file ${SOURCES})

  get_filename_component(__test_name "${src_file}" NAME_WE)

  # Create the executable
  add_executable("${__test_name}" ${src_file})
  target_link_libraries("${__test_name}" PRIVATE ${HYDROGEN_LIBRARIES})

  # Create test
  add_test(NAME "${__test_name}.test"
    COMMAND "${__test_name}")

  add_test(NAME "${__test_name}_mpi_np4.test"
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPI_PREFLAGS}
    $<TARGET_FILE:${__test_name}> ${MPI_POSTFLAGS})
endforeach ()

set_target_properties(HermitianEig
  PROPERTIES
  CXX_STANDARD 17
  CXX_EXTENSIONS OFF
  CXX_STANDARD_REQUIRED TRUE)
