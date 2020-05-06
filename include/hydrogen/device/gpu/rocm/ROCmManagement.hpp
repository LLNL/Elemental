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
hipEvent_t GetNewEvent();
hipStream_t GetNewStream();
void FreeEvent(hipEvent_t& event);
void FreeStream(hipStream_t& stream);
}// namespace rocm
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_ROCMMANAGEMENT_HPP_
