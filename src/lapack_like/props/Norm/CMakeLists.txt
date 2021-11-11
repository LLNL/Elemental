# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  Entrywise.cpp
  Frobenius.cpp
  Infinity.cpp
#  KyFan.cpp
#  KyFanSchatten.cpp
  Max.cpp
#  Nuclear.cpp
  One.cpp
#  Schatten.cpp
#  Two.cpp
#  TwoEstimate.cpp
#  Zero.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
