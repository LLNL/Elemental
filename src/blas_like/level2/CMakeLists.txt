# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
#  ApplyGivensSequence.cpp
  Gemv.cpp
#  Ger.cpp
#  Geru.cpp
#  Hemv.cpp
#  Her.cpp
#  Her2.cpp
#  QuasiTrsv.cpp
#  Symv.cpp
#  Syr.cpp
#  Syr2.cpp
#  Trmv.cpp
#  Trr.cpp
#  Trr2.cpp
  Trsv.cpp
  )

# Add the subdirectories
add_subdirectory(Gemv)
#add_subdirectory(QuasiTrsv)
#add_subdirectory(Symv)
#add_subdirectory(Trsv)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
