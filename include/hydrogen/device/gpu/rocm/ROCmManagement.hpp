#ifndef HYDROGEN_DEVICE_GPU_ROCMMANAGEMENT_HPP_
#define HYDROGEN_DEVICE_GPU_ROCMMANAGEMENT_HPP_

#include <hip/hip_runtime.h>

namespace hydrogen
{

using gpuEvent_t = hipEvent_t;
using gpuStream_t = hipStream_t;

namespace rocm
{
hipEvent_t GetDefaultEvent() noexcept;
hipStream_t GetDefaultStream() noexcept;
}// namespace rocm
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_ROCMMANAGEMENT_HPP_
