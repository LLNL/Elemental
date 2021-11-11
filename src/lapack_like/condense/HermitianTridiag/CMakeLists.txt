# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  # ApplyQ.hpp
  LowerBlocked.hpp
  # LowerBlockedSquare.hpp
  # LowerPanel.hpp
  # LowerPanelSquare.hpp
  UpperBlocked.hpp
  # UpperBlockedSquare.hpp
  # UpperPanel.hpp
  # UpperPanelSquare.hpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
