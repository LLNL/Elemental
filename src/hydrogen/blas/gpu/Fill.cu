#include <hydrogen/blas/gpu/Fill.hpp>

#include <El/hydrogen_config.h>
#include <hydrogen/meta/TypeTraits.hpp>

#ifdef HYDROGEN_HAVE_CUDA
#include <hydrogen/device/gpu/CUDA.hpp>
#include <cuda_runtime.h>
#elif defined(HYDROGEN_HAVE_ROCM)
#include <hydrogen/device/gpu/ROCm.hpp>
#include <hip/hip_runtime.h>
#endif

namespace hydrogen
{
namespace
{

template <typename T>
__global__ void Fill1D_kernel(size_t size, T value, T* buffer)
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t numThreads = blockDim.x * gridDim.x;
    for (size_t pos = tid; pos < size; pos += numThreads)
    {
        buffer[pos] = value;
    }
}

template <typename T>
__global__ void Fill2D_kernel(size_t height, size_t width, T value,
                              T* buffer, size_t ldim)
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t numThreads = blockDim.x * gridDim.x;
    for (size_t pos = tid; pos < height * width; pos += numThreads)
    {
        const size_t i = pos % height;
        const size_t j = pos / height;
        buffer[i+j*ldim] = value;
    }
}

}// namespace <anon>

template <typename T, typename>
void Fill_GPU_impl(
    size_t height, size_t width, T const& value_in,
    T* buffer, size_t ldim,
    SyncInfo<Device::GPU> const& sync_info)
{
    using GPUValueType = NativeGPUType<T>;
    if (height <= 0 || width <= 0)
        return;

    size_t size = height * width;
    constexpr size_t blockDim = 256;
    const size_t gridDim = (size + blockDim - 1) / blockDim;

    GPUValueType value = *AsNativeGPUType(&value_in);
    if (width == 1 || ldim == height)
    {
        gpu::LaunchKernel(
            Fill1D_kernel<GPUValueType>,
            gridDim, blockDim, 0, sync_info,
            size, value, AsNativeGPUType(buffer));
    }
    else
    {
        gpu::LaunchKernel(
            Fill2D_kernel<GPUValueType>,
            gridDim, blockDim, 0, sync_info,
            height, width, value, AsNativeGPUType(buffer), ldim);
    }

}

#define ETI(T)                                 \
    template void Fill_GPU_impl(               \
        size_t, size_t, T const&, T*, size_t,  \
        SyncInfo<Device::GPU> const&)

#ifdef HYDROGEN_GPU_USE_FP16
ETI(gpu_half_type);
#endif
ETI(float);
ETI(double);
ETI(El::Complex<float>);
ETI(El::Complex<double>);

}// namespace hydrogen
