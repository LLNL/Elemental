#ifndef HYDROGEN_INCLUDE_HYDROGEN_DEVICE_GPU_GPURUNTIME_HPP_
#define HYDROGEN_INCLUDE_HYDROGEN_DEVICE_GPU_GPURUNTIME_HPP_

#include <El/hydrogen_config.h>

#include <hydrogen/meta/TypeTraits.hpp>
#ifdef HYDROGEN_HAVE_CUDA
#include "CUDA.hpp"
#include <cuda_runtime.h>
#elif defined(HYDROGEN_HAVE_ROCM)
#include "ROCm.hpp"
#include <hip/hip_runtime.h>
#endif

#endif // HYDROGEN_INCLUDE_HYDROGEN_DEVICE_GPU_GPURUNTIME_HPP_
