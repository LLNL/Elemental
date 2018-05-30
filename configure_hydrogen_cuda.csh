#!/bin/csh

# Sets up the stuff LC is nice enough to provide
use gcc-4.9.3p
module load cudatoolkit/8.0

if (`hostname | gawk '/surface/ {print "yes"}'` == "yes") then
	set NCCL_DIR="/usr/workspace/wsb/brain/nccl2/nccl_2.1.15-1+cuda9.1_x86_64/"
else if (`hostname | gawk '/pascal/ {print "yes"}'` == "yes") then
	set NCCL_DIR="/usr/workspace/wsb/brain/nccl2/nccl_2.1.15-1+cuda9.1_x86_64/"
else if (`hostname | gawk '/ray/ {print "yes"}'` == "yes") then
	set NCCL_DIR="/usr/workspace/wsb/brain/nccl2/nccl_2.0.5-3+cuda8.0_ppc64el/"
endif

#
#
# Adds modern CMake, ninja, and CUDA-aware MPI to the path
set path = ( /usr/workspace/wsb/brain/utils/toss2/cmake-3.9.6/bin /usr/workspace/wsb/brain/utils/toss2/ninja/bin /usr/global/tools/mpi/sideinstalls/chaos_5_x86_64_ib/mvapich2-2.2/install-gcc-cuda/bin $path )

# Adds the CUDA-aware MPI library to the LD_LIBRARY path to ensure preference
setenv LD_LIBRARY_PATH /usr/global/tools/mpi/sideinstalls/chaos_5_x86_64_ib/mvapich2-2.2/install-gcc-cuda/lib:/g/g0/ayoo/aluminum/Aluminum/:${LD_LIBRARY_PATH}

# Trick CMake into picking the right MPI
setenv CMAKE_PREFIX_PATH /usr/global/tools/mpi/sideinstalls/chaos_5_x86_64_ib/mvapich2-2.2/install-gcc-cuda

cmake -DCMAKE_C_COMPILER=`which gcc` \
    -DCMAKE_CXX_COMPILER=`which g++` \
    -DBUILD_SHARED_LIBS=YES \
    -DCMAKE_INSTALL_PREFIX=test_install_hydrogen \
    -DHydrogen_GENERAL_LAPACK_FALLBACK=ON \
    -DHydrogen_ENABLE_CUDA=ON \
    -DHydrogen_ENABLE_CUB=OFF  \
    -DCMAKE_CXX_FLAGS="-g -O0" \
    -DCMAKE_CUDA_FLAGS="-g -O0" \
    ..
#    -DUSE_ALUMINUM=YES \
#    -DUSE_NCCL=YES \
#    -DNCCL_HOME=${NCCL_DIR} \
