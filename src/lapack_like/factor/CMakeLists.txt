# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  Cholesky.cpp
#  GQR.cpp
#  GRQ.cpp
#  ID.cpp
#  LDL.cpp
#  LQ.cpp
#  LU.cpp
#  QR.cpp
#  RQ.cpp
#  Skeleton.cpp
  )

# Add the subdirectories
add_subdirectory(Cholesky)
#add_subdirectory(LDL)
#add_subdirectory(LQ)
#add_subdirectory(LU)
#add_subdirectory(QR)
#add_subdirectory(RQ)
#add_subdirectory(RegularizedLDL)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
