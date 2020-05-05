#ifndef HYDROGEN_DEVICE_GPU_SYNCINFO_HPP_
#define HYDROGEN_DEVICE_GPU_SYNCINFO_HPP_

#include <El/hydrogen_config.h>

#if defined HYDROGEN_HAVE_CUDA
#include "cuda/SyncInfo.hpp"
#elif defined HYDROGEN_HAVE_ROCM
#include "rocm/SyncInfo.hpp"
#endif

#endif // HYDROGEN_DEVICE_GPU_SYNCINFO_HPP_
