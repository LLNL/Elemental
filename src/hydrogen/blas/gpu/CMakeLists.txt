set_full_path(THIS_DIR_GPU_SOURCES
  Axpy.cu
  Copy.cu
  Fill.cu
  Hadamard.cu
  Scale.cu
  Transpose.cu
  )

# Propagate the files up the tree
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_GPU_SOURCES}" PARENT_SCOPE)
