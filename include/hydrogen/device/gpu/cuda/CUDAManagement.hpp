#ifndef HYDROGEN_IMPORTS_CUDA_HPP_
#define HYDROGEN_IMPORTS_CUDA_HPP_

#include <El/hydrogen_config.h>

#include <hydrogen/Device.hpp>
#include <hydrogen/utils/HalfPrecision.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "CUDAError.hpp"

namespace hydrogen
{
namespace cuda
{
cudaEvent_t GetDefaultEvent() noexcept;
cudaStream_t GetDefaultStream() noexcept;
}// namespace cuda
}// namespace hydrogen

#endif // HYDROGEN_IMPORTS_CUDA_HPP_
