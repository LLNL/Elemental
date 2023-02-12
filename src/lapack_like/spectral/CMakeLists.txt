# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  # BidiagSVD.cpp
  CubicSecular.cpp
  # Eig.cpp
  HermitianEig.cpp
  # HermitianGenDefEig.cpp
  # HermitianSVD.cpp
  HermitianTridiagEig.cpp
  # HessenbergSchur.cpp
  # ImageAndKernel.cpp
  # Polar.cpp
  # Pseudospectra.cpp
  # SVD.cpp
  # Schur.cpp
  SecularEVD.cpp
  # SecularSVD.cpp
  # SkewHermitianEig.cpp
  # TriangEig.cpp
  )

# Add the subdirectories
# add_subdirectory(BidiagSVD)
add_subdirectory(HermitianEig)
add_subdirectory(HermitianTridiagEig)
# add_subdirectory(HessenbergSchur)
# add_subdirectory(Polar)
# add_subdirectory(Pseudospectra)
# add_subdirectory(SVD)
# add_subdirectory(Schur)
add_subdirectory(SecularEVD)
# add_subdirectory(SecularSVD)
# add_subdirectory(TriangEig)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
