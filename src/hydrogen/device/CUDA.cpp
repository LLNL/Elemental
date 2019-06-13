#include <El/hydrogen_config.h>

#include <hydrogen/device/GPU.hpp>
#include <hydrogen/device/gpu/CUDA.hpp>
#ifdef HYDROGEN_HAVE_CUB
#include "hydrogen/device/gpu/cuda/CUB.hpp"
#endif // HYDROGEN_HAVE_CUB

#include <hydrogen/Device.hpp>
#include <hydrogen/Error.hpp>
#include <hydrogen/SyncInfo.hpp>

#include <El/core/MemoryPool.hpp>

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

cudaStream_t GetNewStream()
{
    cudaStream_t stream;
    H_CHECK_CUDA(cudaStreamCreate(&stream));//, cudaStreamNonBlocking));
    return stream;
}

cudaEvent_t GetNewEvent()
{
    cudaEvent_t event;
    H_CHECK_CUDA(cudaEventCreate(&event));
    return event;
}

void FreeStream(cudaStream_t stream)
{
    if (stream)
        H_CHECK_CUDA(cudaStreamDestroy(stream));
}

void FreeEvent(cudaEvent_t event)
{
    if (event)
        H_CHECK_CUDA(cudaEventDestroy(event));
}

SyncInfo<Device::GPU> GetNewSyncInfo()
{
    return SyncInfo<Device::GPU>{GetNewStream(), GetNewEvent()};
}

void DestroySyncInfo(SyncInfo<Device::GPU> si)
{
    FreeStream(si.Stream());
    FreeEvent(si.Event());
}

int ComputeMyDeviceId(unsigned int device_count)
{
    if (device_count == 0U)
        return -1;
    if (device_count == 1U)
        return 0;

    // Get local rank (rank within compute node)
    //
    // TODO: Update to not rely on env vars
    // TODO: Use HWLOC or something to pick "closest GPU"
    int local_rank = 0;
    char* env = nullptr;
    if (!env) { env = std::getenv("SLURM_LOCALID"); }
    if (!env) { env = std::getenv("MV2_COMM_WORLD_LOCAL_RANK"); }
    if (!env) { env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"); }
    if (env) { local_rank = std::atoi(env); }

    // Try assigning GPUs to local ranks in round-robin fashion
    return local_rank % device_count;
}

//
// Global variables
//

bool cuda_initialized_ = false;
SyncInfo<Device::GPU> default_syncinfo_;

}// namespace hydrogen::gpu::<anon>

//
// GPU.hpp functions
//

void Initialize()
{
    if (cuda_initialized_)
        return; // Or should this throw??

    // This should fail if device < 0.
    SetDevice(ComputeMyDeviceId(PreCUDAInitDeviceCount()));

    // Setup a default stream and event.
    default_syncinfo_ = GetNewSyncInfo();

    // Set the global flag
    cuda_initialized_ = true;
}

void Finalize()
{
    // FIXME: This stuff should move.
#ifdef HYDROGEN_HAVE_CUB
    cub::DestroyMemoryPool();
#endif // HYDROGEN_HAVE_CUB
    El::DestroyPinnedHostMemoryPool();
    DestroySyncInfo(default_syncinfo_);
    cuda_initialized_ = false;
}

bool IsInitialized() noexcept
{
    return cuda_initialized_;
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

SyncInfo<Device::GPU> const& DefaultSyncInfo() noexcept
{
    return default_syncinfo_;
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

cudaEvent_t GetDefaultEvent() noexcept { return gpu::DefaultSyncInfo().Event(); }
cudaStream_t GetDefaultStream() noexcept { return gpu::DefaultSyncInfo().Stream(); }
}

}// namespace hydrogen
