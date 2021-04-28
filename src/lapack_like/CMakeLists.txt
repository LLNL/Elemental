# Add the subdirectories
#add_subdirectory(condense)
#add_subdirectory(equilibrate)
#add_subdirectory(euclidean_min)
add_subdirectory(factor)
#add_subdirectory(funcs)
#add_subdirectory(perm)
add_subdirectory(props)
#add_subdirectory(reflect)
#add_subdirectory(solve)
#add_subdirectory(spectral)
#add_subdirectory(util)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
