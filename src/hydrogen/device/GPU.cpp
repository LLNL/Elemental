#include "El/hydrogen_config.h"
#include "El/core/MemoryPool.hpp"
#include "hydrogen/device/GPU.hpp"

#if defined HYDROGEN_HAVE_CUDA
#include "hydrogen/device/gpu/CUDA.hpp"
namespace impl = ::hydrogen::cuda;
#elif defined HYDROGEN_HAVE_ROCM
#include "hydrogen/device/gpu/ROCm.hpp"
namespace impl = ::hydrogen::rocm;
#endif // HYDROGEN_HAVE_CUDA

#if defined HYDROGEN_HAVE_CUB
#include "hydrogen/device/gpu/CUB.hpp"
#endif

namespace hydrogen
{
namespace gpu
{
namespace
{

// Global variables
bool gpu_initialized_ = false;
SyncInfo<Device::GPU> default_syncinfo_;

}// namespace <anon>

int ComputeDeviceId(unsigned int device_count) noexcept
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
    if (!env) { env = std::getenv("FLUX_TASK_LOCAL_ID"); }
    if (!env) { env = std::getenv("SLURM_LOCALID"); }
    if (!env) { env = std::getenv("MV2_COMM_WORLD_LOCAL_RANK"); }
    if (!env) { env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"); }
    if (env) { local_rank = std::atoi(env); }

    // Try assigning GPUs to local ranks in round-robin fashion
    return local_rank % device_count;
}

void Initialize()
{
    if (gpu_initialized_)
        return; // Or should this throw??

    // This should fail if device < 0.
    SetDevice(DefaultDevice());

    // Setup a default stream and event.
    default_syncinfo_ = CreateNewSyncInfo<Device::GPU>();

    // Set the global flag
    gpu_initialized_ = true;
}

void Finalize()
{
    // FIXME: This stuff should move.
#ifdef HYDROGEN_HAVE_CUB
    cub::DestroyMemoryPool();
#endif // HYDROGEN_HAVE_CUB
    El::DestroyPinnedHostMemoryPool();
    DestroySyncInfo(default_syncinfo_);
    gpu_initialized_ = false;
}

bool IsInitialized() noexcept
{
    return gpu_initialized_;
}

SyncInfo<Device::GPU> const& DefaultSyncInfo() noexcept
{
    return default_syncinfo_;
}

}// namespace gpu

template <>
SyncInfo<Device::GPU> CreateNewSyncInfo<Device::GPU>()
{
    return SyncInfo<Device::GPU>{
        impl::GetNewStream(), impl::GetNewEvent()};
}

void DestroySyncInfo(SyncInfo<Device::GPU>& si)
{
    impl::FreeStream(si.stream_);
    impl::FreeEvent(si.event_);
    si.stream_ = nullptr;
    si.event_ = nullptr;
}

}// namespace hydrogen
