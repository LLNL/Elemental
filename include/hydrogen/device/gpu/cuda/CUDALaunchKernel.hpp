#ifndef HYDROGEN_DEVICE_GPU_CUDALAUNCHKERNEL_HPP_
#define HYDROGEN_DEVICE_GPU_CUDALAUNCHKERNEL_HPP_

#include <cuda_runtime.h>

#include <hydrogen/Device.hpp>
#include <hydrogen/SyncInfo.hpp>

#include "CUDAError.hpp"

namespace hydrogen
{
namespace gpu
{

template <typename F, typename... Args>
void LaunchKernel(
    F kernel, dim3 const& gridDim, dim3 const& blkDim,
    size_t sharedMem, SyncInfo<Device::GPU> const& si,
    Args&&... kernel_args)
{
    void* args[] = { reinterpret_cast<void*>(&kernel_args)... };
    H_CHECK_GPU(
        cudaLaunchKernel(
            (void const*) kernel,
            gridDim, blkDim, args, sharedMem, si.Stream()));
}

}// namespace gpu
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_CUDALAUNCHKERNEL_HPP_
