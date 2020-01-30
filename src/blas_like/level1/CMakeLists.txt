# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  ColumnMinAbs.cpp
  ColumnNorms.cpp
  Copy.cpp
  HilbertSchmidt.cpp
  Instantiate.cpp
  Max.cpp
  MaxAbs.cpp
  MaxAbsLoc.cpp
  MaxLoc.cpp
  Min.cpp
  MinAbsLoc.cpp
  MinLoc.cpp
  RowMinAbs.cpp
  RowNorms.cpp
  Swap.cpp
  Symmetric2x2Inv.cpp
  Transform2x2.cpp
  NormsFromScaledSquares.hpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
