# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  Gemm.cpp
#  Hemm.cpp
#  Her2k.cpp
#  Herk.cpp
#  HermitianFromEVD.cpp
#  MultiShiftQuasiTrsm.cpp
#  MultiShiftTrsm.cpp
#  NormalFromEVD.cpp
#  QuasiTrsm.cpp
#  SafeMultiShiftTrsm.cpp
#  Symm.cpp
#  Syr2k.cpp
#  Syrk.cpp
#  Trdtrmm.cpp
#  Trmm.cpp
#  Trr2k.cpp
#  Trrk.cpp
#  Trsm.cpp
#  Trstrm.cpp
#  Trtrmm.cpp
#  TwoSidedTrmm.cpp
#  TwoSidedTrsm.cpp
  )

# Add the subdirectories
add_subdirectory(Gemm)
#add_subdirectory(MultiShiftQuasiTrsm)
#add_subdirectory(MultiShiftTrsm)
#add_subdirectory(QuasiTrsm)
#add_subdirectory(SafeMultiShiftTrsm)
#add_subdirectory(Symm)
#add_subdirectory(Syr2k)
#add_subdirectory(Syrk)
#add_subdirectory(Trdtrmm)
#add_subdirectory(Trmm)
#add_subdirectory(Trr2k)
#add_subdirectory(Trrk)
#add_subdirectory(Trsm)
#add_subdirectory(Trstrm)
#add_subdirectory(Trtrmm)
#add_subdirectory(TwoSidedTrmm)
#add_subdirectory(TwoSidedTrsm)

if (HYDROGEN_HAVE_GPU AND HYDROGEN_HAVE_ALUMINUM)
  set_full_path(THIS_DIR_CATCH2_TESTS
    sync_info_pool_test.cpp
    )
endif ()

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(CATCH2_TESTS "${CATCH2_TESTS}" "${THIS_DIR_CATCH2_TESTS}" PARENT_SCOPE)
