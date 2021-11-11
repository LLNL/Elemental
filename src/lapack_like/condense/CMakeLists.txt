# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  # Bidiag.cpp
  HermitianTridiag.cpp
  # Hessenberg.cpp
  )

# Add the subdirectories
# add_subdirectory(Bidiag)
add_subdirectory(HermitianTridiag)
# add_subdirectory(Hessenberg)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
