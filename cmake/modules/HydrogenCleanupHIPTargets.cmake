# This just finds some stuff correctly and cleans up the HIP targets.
macro(h_clean_hip_targets)
  set(HIP_CLANG_ROOT "$ENV{ROCM_PATH}/llvm")

  file(GLOB HIP_CLANG_INCLUDE_SEARCH_PATHS
    "${HIP_CLANG_ROOT}/lib/clang/*/include")
  find_path(HIP_CLANG_INCLUDE_PATH stddef.h
    HINTS "${HIP_CLANG_INCLUDE_SEARCH_PATHS}"
    NO_DEFAULT_PATH)

  if (HIP_CLANG_INCLUDE_PATH)
    message(STATUS "Found clang include path: ${HIP_CLANG_INCLUDE_PATH}")
  else ()
    message(WARNING
      "Could not find clang include path. "
      "Using whatever is in the hip IMPORTED targets")
  endif ()
  
  file(GLOB HIP_CLANGRT_LIB_SEARCH_PATHS
    "${HIP_CLANG_ROOT}/lib/clang/*/lib/*")
  find_library(ACTUAL_CLANGRT_BUILTINS clangrt-builtins
    NAMES
    clang_rt.builtins
    clang_rt.builtins-x86_64
    PATHS
    "${HIP_CLANGRT_LIB_SEARCH_PATHS}")

  if (ACTUAL_CLANGRT_BUILTINS)
    message(STATUS "Found clangrt builtins: ${ACTUAL_CLANGRT_BUILTINS}")
  else ()
    message(WARNING
      "Could not find clangrt builtins. "
      "Using whatever is in the hip IMPORTED targets")
  endif ()

  get_target_property(_HIP_HOST_LIBS hip::host INTERFACE_LINK_LIBRARIES)
  get_target_property(_HIP_DEVICE_LIBS hip::device INTERFACE_LINK_LIBRARIES)

  string(REPLACE
    "CLANGRT_BUILTINS-NOTFOUND"
    "${ACTUAL_CLANGRT_BUILTINS}"
    _NEW_HIP_HOST_LIBS
    "${_HIP_HOST_LIBS}")
  string(REPLACE
    "CLANGRT_BUILTINS-NOTFOUND"
    "${ACTUAL_CLANGRT_BUILTINS}"
    _NEW_HIP_DEVICE_LIBS
    "${_HIP_DEVICE_LIBS}")
  
  set_property(TARGET hip::host
    PROPERTY INTERFACE_LINK_LIBRARIES ${_NEW_HIP_HOST_LIBS})
  set_property(TARGET hip::device
    PROPERTY INTERFACE_LINK_LIBRARIES ${_NEW_HIP_DEVICE_LIBS})
endmacro()
