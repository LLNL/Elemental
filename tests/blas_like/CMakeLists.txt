# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  Axpy.cpp
  BasicGemm.cpp
  ColumnNorms.cpp
  Dot.cpp
  EntrywiseMap.cpp
  Gemm.cpp
  Gemm_Suite.cpp
  Gemv.cpp
  Hadamard.cpp
#  MaxAbs.cpp
#  MultiShiftQuasiTrsm.cpp
#  MultiShiftTrsm.cpp
#  QuasiTrsm.cpp
#  SafeMultiShiftTrsm.cpp
#  Symm.cpp
#  Symv.cpp
#  Syr2k.cpp
#  Syrk.cpp
#  Trmm.cpp
#  Trsm.cpp
#  Trsv.cpp
#  TwoSidedTrmm.cpp
#  TwoSidedTrsm.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
