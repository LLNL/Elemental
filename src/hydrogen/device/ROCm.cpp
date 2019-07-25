#include <El/hydrogen_config.h>

#include <hydrogen/device/GPU.hpp>
#include <hydrogen/device/gpu/ROCm.hpp>

namespace hydrogen
{
namespace gpu
{
namespace
{

hipStream_t GetNewStream()
{
    hipStream_t stream;
    H_CHECK_HIP(hipStreamCreate(&stream));//, hipStreamNonBlocking));
    return stream;
}

hipEvent_t GetNewEvent()
{
    hipEvent_t event;
    H_CHECK_HIP(hipEventCreate(&event));
    return event;
}

void FreeStream(hipStream_t stream)
{
    if (stream)
        H_CHECK_HIP(hipStreamDestroy(stream));
}

void FreeEvent(hipEvent_t event)
{
    if (event)
        H_CHECK_HIP(hipEventDestroy(event));
}

void DestroySyncInfo(SyncInfo<Device::GPU>& si)
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

bool rocm_initialized_ = false;
SyncInfo<Device::GPU> default_syncinfo_;
}// namespace <anon>

void Initialize()
{
    if (IsInitialized())
        return;
    SetDevice(ComputeMyDeviceId(DeviceCount()));

    default_syncinfo_ = SyncInfo<Device::GPU>(GetNewStream(), GetNewEvent());
    rocm_initialized_ = true;
}

void Finalize()
{
    //El::DestroyPinnedHostMemoryPool(); // FIXME
    DestroySyncInfo(default_syncinfo_);
    rocm_initialized_ = false;
}

bool IsInitialized() noexcept
{
    return rocm_initialized_;
}

size_t DeviceCount()
{
    int count;
    H_CHECK_HIP(hipGetDeviceCount(&count));
    return count;
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

SyncInfo<Device::GPU> const& DefaultSyncInfo() noexcept
{
    return default_syncinfo_;
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

}// namespace rocm
}// namespace hydrogen
