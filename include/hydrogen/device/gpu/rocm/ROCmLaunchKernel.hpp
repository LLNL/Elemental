#ifndef HYDROGEN_DEVICE_GPU_ROCMLAUNCHKERNEL_HPP_
#define HYDROGEN_DEVICE_GPU_ROCMLAUNCHKERNEL_HPP_

#include <hydrogen/Device.hpp>
#include <hydrogen/SyncInfo.hpp>

namespace hydrogen
{
namespace gpu
{

inline constexpr int Default2DTileSize() { return 64; }

template <typename F, typename... Args>
void LaunchKernel(
    F kernel, dim3 const& gridDim, dim3 const& blkDim,
    size_t sharedMem, SyncInfo<Device::GPU> const& si,
    Args&&... kernel_args)
{
    H_CHECK_HIP(hipGetLastError());
    // Note that this is (currently) implemented as a macro; not clear
    // if std::forward-ing the arguments is appropriate...
    hipLaunchKernelGGL(
        kernel, gridDim, blkDim,
        sharedMem, si.Stream(),
        std::forward<Args>(kernel_args)...);
    H_CHECK_HIP(hipGetLastError());
}

}// namespace gpu
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_ROCMLAUNCHKERNEL_HPP_
