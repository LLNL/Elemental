list(APPEND HYDROGEN_CATCH2_TEST_FILES
  matrix_test.cpp
  )
if (HYDROGEN_HAVE_GPU)
  list(APPEND HYDROGEN_CATCH2_TEST_FILES
    copy_cpu_gpu_test.cpp
    )
  if (HYDROGEN_HAVE_HALF AND HYDROGEN_GPU_USE_FP16)
    list(APPEND HYDROGEN_CATCH2_TEST_FILES
      copy_cpu_gpu_half_test.cpp
      )
  endif ()
endif (HYDROGEN_HAVE_GPU)

if (HYDROGEN_HAVE_GPU)
  list(APPEND HYDROGEN_CATCH2_TEST_FILES
    gpu_test.cpp)

  if (HYDROGEN_HAVE_ROCM)
    list(APPEND HYDROGEN_CATCH2_TEST_FILES
      hip_half_test.cpp
      )
  endif ()
endif ()
# Add the sequential test main() function
add_executable(seq-catch-tests
  SequentialCatchMain.cpp "${HYDROGEN_CATCH2_TEST_FILES}")
target_link_libraries(seq-catch-tests
  PRIVATE
  ${HYDROGEN_LIBRARIES} Catch2::Catch2)

catch_discover_tests(seq-catch-tests)

# Add the parallel test main() function -- TODO
