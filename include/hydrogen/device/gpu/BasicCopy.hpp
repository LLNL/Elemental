#ifndef HYDROGEN_DEVICE_GPU_BASICCOPY_HPP
#define HYDROGEN_DEVICE_GPU_BASICCOPY_HPP

#include <El/hydrogen_config.h>

#if defined(HYDROGEN_HAVE_CUDA)
#include "cuda/CUDACopy.hpp"
#elif defined(HYDROGEN_HAVE_ROCM)
#include "rocm/ROCmCopy.hpp"
#endif

#endif // HYDROGEN_DEVICE_GPU_BASICCOPY_HPP
