#ifndef HYDROGEN_DEVICE_GPU_CUDA_SYNCINFO_HPP_
#define HYDROGEN_DEVICE_GPU_CUDA_SYNCINFO_HPP_

#include <cuda_runtime_api.h>

#include <hydrogen/SyncInfo.hpp>
#include <hydrogen/meta/MetaUtilities.hpp>

#include "CUDAManagement.hpp"

namespace hydrogen
{

template <>
class SyncInfo<Device::GPU>
{
public:
    SyncInfo()
        : SyncInfo{cuda::GetDefaultStream(), cuda::GetDefaultEvent()}
    {}

    SyncInfo(cudaStream_t stream, cudaEvent_t event)
        : stream_{stream}, event_{event}
    {}

    void Merge(SyncInfo<Device::GPU> const& si) noexcept
    {
        if (si.stream_)
            stream_ = si.stream_;
        if (si.event_)
            event_ = si.event_;
    }

    cudaStream_t Stream() const noexcept { return stream_; }
    cudaEvent_t Event() const noexcept { return event_; }
private:
    cudaStream_t stream_;
    cudaEvent_t event_;
};// struct SyncInfo<Device::GPU>

inline void AddSynchronizationPoint(SyncInfo<Device::GPU> const& syncInfo)
{
    H_CHECK_CUDA(cudaEventRecord(syncInfo.Event(), syncInfo.Stream()));
}

inline void AddSynchronizationPoint(
    SyncInfo<Device::CPU> const& master,
    SyncInfo<Device::GPU> const& dependent)
{
    // The GPU must wait for the CPU.
    // This probably requires Aluminum, but it might also just be a logic error.
    throw std::logic_error(
        "Either this function doesn't make sense "
        "or it should use Al::GPUWait.");
}

inline void AddSynchronizationPoint(
    SyncInfo<Device::GPU> const& master,
    SyncInfo<Device::CPU> const& dependent)
{
    // The CPU must wait for the GPU to catch up.
    Synchronize(master); // wait for "master"
}

// This captures the work done on A and forces "others" to wait for
// completion.
template <typename... Ts>
inline
EnableWhen<AllMatch<SyncInfo<Device::GPU>,Ts...>>
AddSynchronizationPoint(
    SyncInfo<Device::GPU> const& master, Ts&&... others)
{
    AddSynchronizationPoint(master);

    auto sync_other = [&master](SyncInfo<Device::GPU> const& x)
        {
            if (master.Stream() != x.Stream())
                H_CHECK_CUDA(
                    cudaStreamWaitEvent(x.Stream(), master.Event(), 0));
            return 0;
        };

    int dummy[] = { sync_other(others)... };
    (void) dummy;
}

inline void Synchronize(SyncInfo<Device::GPU> const& syncInfo)
{
    H_CHECK_CUDA(cudaStreamSynchronize(syncInfo.Stream()));
}

}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_CUDA_SYNCINFO_HPP_
