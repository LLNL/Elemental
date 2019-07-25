#include <hydrogen/blas/gpu/Hadamard.hpp>

#include <El/hydrogen_config.h>
#ifdef HYDROGEN_HAVE_CUDA
#include <hydrogen/device/gpu/CUDA.hpp>
#include <cuda_runtime.h>
using gpuStream_t = cudaStream_t;
#elif defined(HYDROGEN_HAVE_ROCM)
#include <hydrogen/device/gpu/ROCm.hpp>
#include <hip/hip_runtime.h>
using gpuStream_t = hipStream_t;
#endif

namespace
{

template <typename T>
__global__ void Hadamard1D_kernel(size_t size,
                                  T const* __restrict__ X,
                                  T const* __restrict__ Y,
                                  T* __restrict__ Z)
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t numThreads = blockDim.x * gridDim.x;
    for (size_t pos = tid; pos < size; pos += numThreads)
    {
        Z[pos] = X[pos] * Y[pos];
    }
}

template <typename T>
__global__ void MultAssign_kernel(size_t size, T const* X, T* Y)
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t numThreads = blockDim.x * gridDim.x;
    for (size_t pos = tid; pos < size; pos += numThreads)
    {
        Y[pos] *= X[pos];
    }
}

template <typename T>
__global__ void Hadamard2D_kernel(size_t height, size_t width,
                                  T const* X, size_t colStrideX, size_t rowStrideX,
                                  T const* Y, size_t colStrideY, size_t rowStrideY,
                                  T* Z, size_t colStrideZ, size_t rowStrideZ)
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t numThreads = blockDim.x * gridDim.x;
    for (size_t pos = tid; pos < height * width; pos += numThreads)
    {
        const size_t i = pos % height;
        const size_t j = pos / height;
        const auto& x_local = X[i*colStrideX+j*rowStrideX];
        const auto& y_local = Y[i*colStrideY+j*rowStrideY];
        Z[i*colStrideZ+j*rowStrideZ] = x_local * y_local;
    }
}

}// namespace <anon>

namespace hydrogen
{

template <typename T, typename>
void Hadamard_GPU_impl(
    size_t height, size_t width,
    T const* X, size_t colStrideX, size_t rowStrideX,
    T const* Y, size_t colStrideY, size_t rowStrideY,
    T* Z, size_t colStrideZ, size_t rowStrideZ,
    gpuStream_t stream)
{
    if (height <= 0 || width <= 0) { return; }
    size_t size = height * width;
    size_t const blockDim = 256;
    size_t const gridDim = (size + blockDim - 1) / blockDim;
    if (colStrideX == 1 && rowStrideX == height
        && colStrideY == 1 && rowStrideY == height
        && colStrideZ == 1 && rowStrideZ == height)
    {
        if (X == Z)
        {
            gpu::LaunchKernel(
                MultAssign_kernel<T>,
                gridDim, blockDim, 0, SyncInfo<Device::GPU>(stream, nullptr),
                size, Y, Z);
        }
        else if (Y == Z)
        {
            gpu::LaunchKernel(
                MultAssign_kernel<T>,
                gridDim, blockDim, 0, SyncInfo<Device::GPU>(stream, nullptr),
                size, X, Z);
        }
        else
        {
            gpu::LaunchKernel(
                Hadamard1D_kernel<T>,
                gridDim, blockDim, 0, SyncInfo<Device::GPU>(stream, nullptr),
                size, X, Y, Z);
        }
    }
    else
    {
        gpu::LaunchKernel(
            Hadamard2D_kernel<T>,
            gridDim, blockDim, 0, SyncInfo<Device::GPU>(stream, nullptr),
            height, width,
            X, colStrideX, rowStrideX,
            Y, colStrideY, rowStrideY,
            Z, colStrideZ, rowStrideZ);
    }

}

#define ETI(T)                                  \
    template void Hadamard_GPU_impl(            \
        size_t, size_t,                         \
        T const*, size_t, size_t,               \
        T const*, size_t, size_t,               \
        T*, size_t, size_t, gpuStream_t)

#ifdef HYDROGEN_GPU_USE_FP16
ETI(gpu_half_type);
#endif

ETI(float);
ETI(double);

}// namespace hydrogen
