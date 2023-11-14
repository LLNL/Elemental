list(APPEND HYDROGEN_CATCH2_TEST_FILES
  matrix_test.cpp
  )
if (HYDROGEN_HAVE_GPU)
  list(APPEND HYDROGEN_CATCH2_TEST_FILES
    gpu_test.cpp
    gpu_combine_test.cu
    gpu_entrywise_map_test.cu
    copy_cpu_gpu_test.cpp
    pool_allocator_test.cpp
    )
  if (HYDROGEN_HAVE_HALF AND HYDROGEN_GPU_USE_FP16)
    list(APPEND HYDROGEN_CATCH2_TEST_FILES
      copy_cpu_gpu_half_test.cpp
      )
  endif ()
endif (HYDROGEN_HAVE_GPU)

# Add the sequential test main() function
add_executable(seq-catch-tests
  SequentialCatchMain.cpp "${HYDROGEN_CATCH2_TEST_FILES}")

target_link_libraries(seq-catch-tests
  PRIVATE
  ${HYDROGEN_LIBRARIES} Catch2::Catch2)

if (HYDROGEN_HAVE_CUDA)
  check_compiler_flag(CUDA "--expt-extended-lambda"
    CUDA_COMPILER_SUPPORTS_EXTENDED_LAMBDA)
  if (CUDA_COMPILER_SUPPORTS_EXTENDED_LAMBDA)
    target_compile_options(seq-catch-tests
      PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
  endif ()
endif ()

catch_discover_tests(seq-catch-tests)

# Add the parallel test main() function -- TODO
