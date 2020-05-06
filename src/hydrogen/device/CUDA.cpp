#include "El/hydrogen_config.h"

#include "hydrogen/device/GPU.hpp"
#include "hydrogen/device/gpu/CUDA.hpp"

#include "hydrogen/Device.hpp"
#include "hydrogen/Error.hpp"
#include "hydrogen/SyncInfo.hpp"

#include "El/core/MemoryPool.hpp"

#include <nvml.h>

#include <iostream>
#include <sstream>

#define H_CHECK_NVML(cmd)                                               \
    {                                                                   \
        auto h_check_nvml_error_code = cmd;                             \
        H_ASSERT(h_check_nvml_error_code == NVML_SUCCESS,               \
                 NVMLError,                                             \
                 BuildNVMLErrorMessage(#cmd,                            \
                                       h_check_nvml_error_code));       \
    }

namespace hydrogen
{
namespace gpu
{
namespace
{

/** @class NVMLError
 *  @brief Exception class for errors detected in NVML
 */
H_ADD_BASIC_EXCEPTION_CLASS(NVMLError, GPUError);// struct NVMLError

/** @brief Write an error message describing what went wrong in NVML
 *  @param[in] cmd The expression that raised the error.
 *  @param[in] error_code The error code reported by NVML.
 *  @returns A string describing the error.
 */
std::string BuildNVMLErrorMessage(
    std::string const& cmd, nvmlReturn_t error_code)
{
    std::ostringstream oss;
    oss << "NVML error detected in command: \"" << cmd << "\"\n\n"
        << "    Error Code: " << error_code << "\n"
        << "    Error Mesg: " << nvmlErrorString(error_code) << "\n";
    return oss.str();
}

unsigned int PreCUDAInitDeviceCount()
{
    unsigned int count;
    H_CHECK_NVML(nvmlInit());
    H_CHECK_NVML(nvmlDeviceGetCount(&count));
    H_CHECK_NVML(nvmlShutdown());
    return count;
}

}// namespace hydrogen::gpu::<anon>

//
// GPU.hpp functions
//

int DefaultDevice()
{
    static int device_id =
        ComputeDeviceId(PreCUDAInitDeviceCount());
    return device_id;
}

size_t DeviceCount()
{
    int count;
    H_CHECK_CUDA(cudaGetDeviceCount(&count));
    return static_cast<size_t>(count);
}

int CurrentDevice()
{
    int device;
    H_CHECK_CUDA(cudaGetDevice(&device));
    return device;
}

void SetDevice(int device_id)
{
    H_CHECK_CUDA(cudaSetDevice(device_id));
    H_CHECK_CUDA(cudaGetLastError());
}

void SynchronizeDevice()
{
    H_CHECK_CUDA(cudaDeviceSynchronize());
}

}// namespace gpu

namespace cuda
{

std::string BuildCUDAErrorMessage(
    std::string const& cmd, cudaError_t error_code)
{
    std::ostringstream oss;
    oss << "CUDA error detected in command: \"" << cmd << "\"\n\n"
        << "    Error Code: " << error_code << "\n"
        << "    Error Name: " << cudaGetErrorName(error_code) << "\n"
        << "    Error Mesg: " << cudaGetErrorString(error_code);
    return oss.str();
}

cudaEvent_t GetDefaultEvent() noexcept
{
    return gpu::DefaultSyncInfo().Event();
}

cudaStream_t GetDefaultStream() noexcept
{
    return gpu::DefaultSyncInfo().Stream();
}

cudaStream_t GetNewStream()
{
    cudaStream_t stream;
    H_CHECK_CUDA(
        cudaStreamCreateWithFlags(
            &stream, cudaStreamNonBlocking));
    return stream;
}

cudaEvent_t GetNewEvent()
{
    cudaEvent_t event;
    H_CHECK_CUDA(
        cudaEventCreateWithFlags(
            &event, cudaEventDisableTiming));
    return event;
}

void FreeStream(cudaStream_t& stream)
{
    if (stream)
    {
        H_CHECK_CUDA(cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

void FreeEvent(cudaEvent_t& event)
{
    if (event)
    {
        H_CHECK_CUDA(cudaEventDestroy(event));
        event = nullptr;
    }
}

}// namespace cuda
}// namespace hydrogen
