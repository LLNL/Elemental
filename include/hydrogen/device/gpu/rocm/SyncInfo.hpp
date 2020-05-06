#ifndef HYDROGEN_DEVICE_GPU_ROCM_SYNCINFO_HPP_
#define HYDROGEN_DEVICE_GPU_ROCM_SYNCINFO_HPP_

#include <hip/hip_runtime.h>

#include <hydrogen/SyncInfo.hpp>
#include <hydrogen/meta/MetaUtilities.hpp>

#include "ROCmError.hpp"
#include "ROCmManagement.hpp"

namespace hydrogen
{

template <>
class SyncInfo<Device::GPU>
{
public:
    SyncInfo()
        : SyncInfo{rocm::GetDefaultStream(), rocm::GetDefaultEvent()}
    {}

    SyncInfo(hipStream_t stream, hipEvent_t event)
        : stream_{stream}, event_{event}
    {}

    void Merge(SyncInfo<Device::GPU> const& si) noexcept
    {
        if (si.stream_)
            stream_ = si.stream_;
        if (si.event_)
            event_ = si.event_;
    }

    hipStream_t Stream() const noexcept { return stream_; }
    hipEvent_t Event() const noexcept { return event_; }
private:
    friend void DestroySyncInfo(SyncInfo<Device::GPU>&);
    hipStream_t stream_;
    hipEvent_t event_;
};// struct SyncInfo<Device::GPU>

inline void AddSynchronizationPoint(SyncInfo<Device::GPU> const& syncInfo)
{
    H_CHECK_HIP(hipEventRecord(syncInfo.Event(), syncInfo.Stream()));
}

namespace details
{
inline void AddSyncPoint(
    SyncInfo<Device::CPU> const& master,
    SyncInfo<Device::GPU> const& dependent)
{
    // The GPU must wait for the CPU.
    // This probably requires Aluminum, but it might also just be a logic error.
    throw std::logic_error(
        "Either this function doesn't make sense "
        "or it should use Al::GPUWait.");
}

inline void AddSyncPoint(
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
void AddSyncPoint(
    SyncInfo<Device::GPU> const& master, SyncInfo<Device::GPU> const& other)
{
    if (master.Stream() != other.Stream())
        H_CHECK_HIP(
            hipStreamWaitEvent(other.Stream(), master.Event(), 0));
}
}// namespace details

inline void Synchronize(SyncInfo<Device::GPU> const& syncInfo)
{
    H_CHECK_HIP(hipStreamSynchronize(syncInfo.Stream()));
}

}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_ROCM_SYNCINFO_HPP_
