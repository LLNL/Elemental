#ifndef HYDROGEN_DEVICE_GPU_CUDAMANAGEMENT_HPP_
#define HYDROGEN_DEVICE_GPU_CUDAMANAGEMENT_HPP_

#include <cuda_runtime.h>

namespace hydrogen
{

using gpuEvent_t = cudaEvent_t;
using gpuStream_t = cudaStream_t;

namespace cuda
{
cudaEvent_t GetDefaultEvent() noexcept;
cudaStream_t GetDefaultStream() noexcept;
}// namespace cuda
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_CUDAMANAGEMENT_HPP_
