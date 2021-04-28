list(APPEND HYDROGEN_CATCH2_TEST_FILES
  matrix_test.cpp
  )
if (HYDROGEN_HAVE_GPU)
  list(APPEND HYDROGEN_CATCH2_TEST_FILES
    gpu_test.cpp
    gpu_combine_test.cu
    gpu_entrywise_map_test.cu
    copy_cpu_gpu_test.cpp
    )
  if (HYDROGEN_HAVE_HALF AND HYDROGEN_GPU_USE_FP16)
    list(APPEND HYDROGEN_CATCH2_TEST_FILES
      copy_cpu_gpu_half_test.cpp
      )
  endif ()
endif (HYDROGEN_HAVE_GPU)

# Add the sequential test main() function
if (HYDROGEN_HAVE_ROCM)
  hip_add_executable(seq-catch-tests
    SequentialCatchMain.cpp "${HYDROGEN_CATCH2_TEST_FILES}")
else ()
  add_executable(seq-catch-tests
    SequentialCatchMain.cpp "${HYDROGEN_CATCH2_TEST_FILES}")
endif (HYDROGEN_HAVE_ROCM)
target_link_libraries(seq-catch-tests
  PRIVATE
  ${HYDROGEN_LIBRARIES} Catch2::Catch2)

catch_discover_tests(seq-catch-tests)

# Add the parallel test main() function -- TODO
