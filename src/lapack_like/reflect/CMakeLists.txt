# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  ApplyPacked.cpp
  ExpandPacked.cpp
  Householder.cpp
  # Hyperbolic.cpp
  )

# Add the subdirectories
add_subdirectory(ApplyPacked)
# add_subdirectory(ExpandPacked)
# add_subdirectory(Householder)
# add_subdirectory(Hyperbolic)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
