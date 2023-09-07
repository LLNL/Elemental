#include <El/hydrogen_config.h>

#include <hydrogen/device/GPU.hpp>
#include <hydrogen/device/gpu/ROCm.hpp>

#include <El/core/MemoryPool.hpp>

namespace hydrogen
{
namespace gpu
{

size_t DeviceCount()
{
    int count;
    H_CHECK_HIP(hipGetDeviceCount(&count));
    return count;
}

int DefaultDevice()
{
    static int device_id = ComputeDeviceId(DeviceCount());
    return device_id;
}

int CurrentDevice()
{
    int device_id;
    H_CHECK_HIP(hipGetDevice(&device_id));
    return device_id;
}

void SetDevice(int device_id)
{
    H_CHECK_HIP(hipSetDevice(device_id));
}

void SynchronizeDevice()
{
    H_CHECK_HIP(hipDeviceSynchronize());
}

}// namespace gpu

namespace rocm
{

std::string BuildHipErrorMessage(std::string const& cmd, hipError_t error_code)
{
    std::ostringstream oss;
    oss << "ROCm error detected in command: \"" << cmd << "\"\n\n"
        << "    Error Code: " << error_code << "\n"
        << "    Error Name: " << hipGetErrorName(error_code) << "\n"
        << "    Error Mesg: " << hipGetErrorString(error_code);
    return oss.str();
}

hipEvent_t GetDefaultEvent() noexcept
{
    return gpu::DefaultSyncInfo().Event();
}

hipStream_t GetDefaultStream() noexcept
{
    return gpu::DefaultSyncInfo().Stream();
}

hipStream_t GetNewStream()
{
    hipStream_t stream;
    H_CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    return stream;
}

hipEvent_t GetNewEvent()
{
    hipEvent_t event;
#if HIP_VERSION < 50600000
    H_CHECK_HIP(hipEventCreateWithFlags(&event, hipEventDisableTiming));
#else
    H_CHECK_HIP(hipEventCreateWithFlags(
                    &event,
                    hipEventDisableTiming | hipEventDisableSystemFence));
#endif
    return event;
}

void FreeStream(hipStream_t& stream)
{
    if (stream)
    {
        H_CHECK_HIP(hipStreamDestroy(stream));
        stream = nullptr;
    }
}

void FreeEvent(hipEvent_t& event)
{
    if (event)
    {
        H_CHECK_HIP(hipEventDestroy(event));
        event = nullptr;
    }
}

}// namespace rocm
}// namespace hydrogen
