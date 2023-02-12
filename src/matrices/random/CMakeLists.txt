# Add the subdirectories
add_subdirectory(independent)
add_subdirectory(lattice)
add_subdirectory(misc)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
