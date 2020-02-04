include(CheckCXXSourceCompiles)

# Attempt to use the built-in module
find_package(OpenMP COMPONENTS CXX)

if (NOT OpenMP_FOUND AND CMAKE_CXX_COMPILER_ID MATCHES "[Cc]lang")
  find_library(_OpenMP_LIBRARY
    NAMES omp gomp iomp5md
    HINTS ${OpenMP_DIR} $ENV{OpenMP_DIR}
    PATH_SUFFIXES lib64 lib
    NO_DEFAULT_PATH
    DOC "The libomp library.")
  find_library(_OpenMP_LIBRARY
    NAMES omp gomp iomp5md)
  mark_as_advanced(_OpenMP_LIBRARY)

  if (NOT _OpenMP_LIBRARY)
    message(FATAL_ERROR "No OpenMP library found.")
  else ()

    get_filename_component(_OpenMP_LIB_DIR "${_OpenMP_LIBRARY}" DIRECTORY)

    if (${_OpenMP_LIBRARY} MATCHES "libomp")
      set(OpenMP_libomp_LIBRARY ${_OpenMP_LIBRARY}
        CACHE PATH "The OpenMP omp library.")
      set(OpenMP_CXX_LIB_NAMES omp)
      set(OpenMP_CXX_FLAGS "-fopenmp=libomp")
      set(OpenMP_omp_LIBRARY "${_OpenMP_LIBRARY}")
    elseif (${_OpenMP_LIBRARY} MATCHES "libgomp")
      set(OpenMP_libgomp_LIBRARY ${_OpenMP_LIBRARY}
        CACHE PATH "The OpenMP gomp library.")
      set(OpenMP_CXX_LIB_NAMES gomp)
      set(OpenMP_CXX_FLAGS "-fopenmp")
      set(OpenMP_gomp_LIBRARY "${_OpenMP_LIBRARY}")
    elseif (${_OpenMP_LIBRARY} MATCHES "libiomp5md")
      set(OpenMP_libiomp5md_LIBRARY ${_OpenMP_LIBRARY}
        CACHE PATH "The OpenMP iomp5md library.")
      set(OpenMP_CXX_LIB_NAMES iomp5md)
      set(OpenMP_CXX_FLAGS "-fopenmp=libiomp5")
      set(OpenMP_iomp5md_LIBRARY "${_OpenMP_LIBRARY}")
    endif ()

    # Let's try this again
    find_package(OpenMP COMPONENTS CXX)
    if (OpenMP_CXX_FOUND)
      if (CMAKE_VERSION VERSION_GREATER 3.13.0)
        target_link_directories(
          OpenMP::OpenMP_CXX INTERFACE "${_OpenMP_LIB_DIR}")
      else ()
        # This isn't great, but it should work. The better solution is
        # to use a version of CMake that is at least 3.13.0.
        set_property(TARGET OpenMP::OpenMP_CXX APPEND
          PROPERTY INTERFACE_LINK_LIBRARIES "-L${_OpenMP_LIB_DIR}")
      endif ()
    endif ()
  endif (NOT _OpenMP_LIBRARY)
endif ()

set(_OPENMP_TEST_SOURCE
  "
#include <omp.h>
int main() {
#pragma omp parallel
{
  int x = omp_get_num_threads();
}
}")

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "${OpenMP_CXX_FLAGS}")
set(CMAKE_REQUIRED_LIBRARIES OpenMP::OpenMP_CXX)
check_cxx_source_compiles("${_OPENMP_TEST_SOURCE}" _OPENMP_TEST_COMPILES)

get_target_property(_OMP_FLAGS OpenMP::OpenMP_CXX INTERFACE_COMPILE_OPTIONS)
set_property(TARGET OpenMP::OpenMP_CXX PROPERTY
  INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:${_OMP_FLAGS}>)

set(OpenMP_FOUND ${_OPENMP_TEST_COMPILES})

if (OpenMP_FOUND)
  set(EL_HAVE_OPENMP TRUE)
else ()
  set(EL_HAVE_OPENMP FALSE)
endif ()

if (EL_HAVE_OPENMP)
  set(OMP_COLLAPSE_CODE
    "#include <omp.h>
     int main( int argc, char* argv[] )
     {
         int k[100];
     #pragma omp parallel for collapse(2)
         for( int i=0; i<10; ++i )
             for( int j=0; j<10; ++j )
                 k[i+j*10] = i+j;
         return 0;
     }")
  check_cxx_source_compiles("${OMP_COLLAPSE_CODE}" EL_HAVE_OMP_COLLAPSE)

  set(OMP_SIMD_CODE
      "#include <omp.h>
       int main( int argc, char* argv[] )
       {
           int k[10];
       #pragma omp simd
           for( int i=0; i<10; ++i )
               k[i] = i;
           return 0;
       }")
  check_cxx_source_compiles("${OMP_SIMD_CODE}" EL_HAVE_OMP_SIMD)

  # See if we have 'taskloop' support, which was introduced in OpenMP 4.0
  if (${PROJECT_NAME}_ENABLE_OMP_TASKLOOP)
    set(OMP_TASKLOOP_CODE
      "#include <omp.h>
       int main( int argc, char* argv[] )
       {
           int k[10];
       #pragma omp taskloop
           for( int i=0; i<10; ++i )
               k[i] = i;
           return 0;
       }")
    check_cxx_source_compiles("${OMP_TASKLOOP_CODE}" HYDROGEN_HAVE_OMP_TASKLOOP)
    set(${PROJECT_NAME}_ENABLE_OMP_TASKLOOP ${HYDROGEN_HAVE_OMP_TASKLOOP})
  else ()
    set(HYDROGEN_HAVE_OMP_TASKLOOP FALSE)
    set(${PROJECT_NAME}_ENABLE_OMP_TASKLOOP FALSE)
  endif ()
else ()
  set(HYDROGEN_HAVE_OMP_TASKLOOP FALSE)
  set(EL_HAVE_OMP_SIMD FALSE)
  set(EL_HAVE_OMP_COLLAPSE FALSE)
  set(Hydrogen_ENABLE_OMP_TASKLOOP FALSE)
endif (EL_HAVE_OPENMP)

set(CMAKE_REQUIRED_FLAGS)
set(CMAKE_REQUIRED_LIBRARIES)
set(CMAKE_REQUIRED_LINK_OPTIONS)
