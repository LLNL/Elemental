# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  # Demmel.cpp
  # DruinskyToledo.cpp
  # DynamicRegCounter.cpp
  # Ehrenfest.cpp
  # ExtendedKahan.cpp
  # GEPPGrowth.cpp
  # GKS.cpp
  # Gear.cpp
  # Hanowa.cpp
  # JordanCholesky.cpp
  # KMS.cpp
  # Kahan.cpp
  # Lauchli.cpp
  # Legendre.cpp
  # Lehmer.cpp
  # Lotkin.cpp
  # MinIJ.cpp
  # Parter.cpp
  # Pei.cpp
  # Redheffer.cpp
  # Riffle.cpp
  # Ris.cpp
  Wilkinson.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
