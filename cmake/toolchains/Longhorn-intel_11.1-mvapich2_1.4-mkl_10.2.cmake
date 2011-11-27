# The serial Intel compilers
set(CMAKE_C_COMPILER       /opt/apps/intel/11.1/bin/intel64/icc)
set(CMAKE_CXX_COMPILER     /opt/apps/intel/11.1/bin/intel64/icpc)
set(CMAKE_Fortran_COMPILER /opt/apps/intel/11.1/bin/intel64/ifort)

# The MPI wrappers for the C and C++ compilers
set(MPI_C_COMPILER   /opt/apps/intel11_1/mvapich2/1.4/bin/mpicc)
set(MPI_CXX_COMPILER /opt/apps/intel11_1/mvapich2/1.4/bin/mpicxx)
set(MPI_C_COMPILE_FLAGS "-Wl,-rpath,/opt/apps/intel/11.1/lib/intel64 -Wl,-rpath,/opt/apps/intel/11.1/lib/intel64 -Wl,-rpath")
set(MPI_CXX_COMPILE_FLAGS "-Wl,-rpath,/opt/apps/intel/11.1/lib/intel64 -Wl,-rpath,/opt/apps/intel/11.1/lib/intel64 -Wl,-rpath")
set(MPI_C_INCLUDE_PATH   /opt/apps/intel11_1/mvapich2/1.4/include)
set(MPI_CXX_INCLUDE_PATH /opt/apps/intel11_1/mvapich2/1.4/include)
set(MPI_C_LINK_FLAGS "-L/opt/apps/intel11_1/mvapich2/1.4/lib -L/opt/ofed/lib64/")
set(MPI_CXX_LINK_FLAGS "-L/opt/apps/intel11_1/mvapich2/1.4/lib -L/opt/ofed/lib64/")
set(MPI_C_LIBRARIES "-limf -lmpich -lpthread -lrdmacm -libverbs -libumad -ldl -lrt")
set(MPI_CXX_LIBRARIES "-limf -lmpichcxx -lmpich -lpthread -lrdmacm -libverbs -libumad -ldl -lrt")

set(CXX_PURE_DEBUG_FLAGS "-g")
set(CXX_PURE_RELEASE_FLAGS "-O3")
set(CXX_HYBRID_DEBUG_FLAGS "-g")
set(CXX_HYBRID_RELEASE_FLAGS "-O3")

set(OpenMP_CXX_FLAGS "-openmp")

if(CMAKE_BUILD_TYPE MATCHES PureDebug OR
   CMAKE_BUILD_TYPE MATCHES PureRelease)
  set(MATH_LIBS "-L/opt/apps/intel/mkl/10.2.2.025/lib/em64t -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm")
else(CMAKE_BUILD_TYPE MATCHES PureDebug OR 
     CMAKE_BUILD_TYPE MATCHES PureRelease)
  set(MATH_LIBS "-L/opt/apps/intel/11.1/mkl/lib/em64t -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lguide -lpthread -lm")
endif(CMAKE_BUILD_TYPE MATCHES PureDebug OR 
      CMAKE_BUILD_TYPE MATCHES PureRelease)
