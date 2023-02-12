# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  # ApplyPackedReflectors.cpp
  # Bidiag.cpp
  # BidiagDCSVD.cpp
  # Cholesky.cpp
  # CholeskyMod.cpp
  # CholeskyQR.cpp
  # Eig.cpp
  HermitianEig.cpp
  # HermitianGenDefEig.cpp
  # HermitianTridiag.cpp
  # HermitianTridiagEig.cpp
  # Hessenberg.cpp
  # HessenbergSchur.cpp
  # LDL.cpp
  # LQ.cpp
  # LU.cpp
  # LUMod.cpp
  # MultiShiftHessSolve.cpp
  # QR.cpp
  # RQ.cpp
  # SVD.cpp
  # SVDTwoByTwoUpper.cpp
  # Schur.cpp
  # SchurSwap.cpp
  # SecularEVD.cpp
  # SecularSVD.cpp
  # TSQR.cpp
  # TSSVD.cpp
  # TriangEig.cpp
  # TriangularInverse.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
