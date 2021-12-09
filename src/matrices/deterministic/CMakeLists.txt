# Add the subdirectories
add_subdirectory(classical)
#add_subdirectory(integral)
add_subdirectory(misc)
#add_subdirectory(pde)
#add_subdirectory(sparse_toeplitz)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
