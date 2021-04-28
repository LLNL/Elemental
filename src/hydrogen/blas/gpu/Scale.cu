#include <hydrogen/blas/gpu/Scale.hpp>

#include <El/hydrogen_config.h>
#include <hydrogen/meta/TypeTraits.hpp>
#ifdef HYDROGEN_HAVE_CUDA
#include <hydrogen/device/gpu/CUDA.hpp>
#include <cuda_runtime.h>
#elif defined(HYDROGEN_HAVE_ROCM)
#include <hydrogen/device/gpu/ROCm.hpp>
#include <hip/hip_runtime.h>
#endif

namespace
{

template <typename T, typename SizeT>
__global__ void scale_1d_kernel_naive(
    SizeT num_entries, T alpha, T* A, SizeT stride_A)
{
    SizeT const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_entries)
        A[idx*stride_A] *= alpha;
}

template <int TILE_DIM, int BLK_COLS, typename T, typename SizeT>
__global__ void scale_2d_kernel_naive(
    SizeT m, SizeT n, T alpha, T* A, SizeT lda)
{
    size_t const row_idx = blockIdx.x * TILE_DIM + threadIdx.x;
    size_t const col_idx = blockIdx.y * TILE_DIM + threadIdx.y;

    if (row_idx < m)
    {
        for (int ii = 0; ii < TILE_DIM && col_idx + ii < n; ii += BLK_COLS)
            A[row_idx + (col_idx+ii)*lda] *= alpha;
    }
}

}// namespace <anon>

namespace hydrogen
{

template <typename T, typename SizeT, typename>
void Scale_GPU_impl(
    SizeT num_entries,
    T const& alpha_in, T* A, SizeT lda,
    SyncInfo<Device::GPU> const& sync_info)
{
    if (!num_entries)
        return;

    constexpr size_t threads_per_block = 128;
    auto blocks = (num_entries + threads_per_block - 1)/ threads_per_block;
    NativeGPUType<T> alpha = *AsNativeGPUType(&alpha_in);
    gpu::LaunchKernel(
        scale_1d_kernel_naive<NativeGPUType<T>, SizeT>,
        blocks, threads_per_block, 0, sync_info,
        num_entries, alpha, AsNativeGPUType(A), lda);
}

template <typename T, typename SizeT, typename>
void Scale_GPU_impl(
    SizeT num_rows, SizeT num_cols,
    T const& alpha_in, T* A, SizeT lda,
    SyncInfo<Device::GPU> const& sync_info)
{
    if (num_rows == TypeTraits<SizeT>::Zero()
        || num_cols == TypeTraits<SizeT>::Zero())
    {
        return;
    }

    constexpr int TILE_DIM = 32;
    constexpr int BLK_COLS = 8;

    dim3 blks((num_rows + TILE_DIM - 1) / TILE_DIM,
              (num_cols + TILE_DIM - 1) / TILE_DIM,
              1);
    dim3 thds(TILE_DIM, BLK_COLS, 1);

    NativeGPUType<T> alpha = *AsNativeGPUType(&alpha_in);
    gpu::LaunchKernel(
        scale_2d_kernel_naive<TILE_DIM,BLK_COLS,NativeGPUType<T>,SizeT>,
        blks, thds, 0, sync_info,
        num_rows, num_cols, alpha, AsNativeGPUType(A), lda);
}

#define ETI(DataType, SizeType)                         \
    template void Scale_GPU_impl(                       \
        SizeType,                                       \
        DataType const&, DataType*, SizeType,           \
        SyncInfo<Device::GPU> const&);                  \
    template void Scale_GPU_impl(                       \
        SizeType, SizeType,                             \
        DataType const&, DataType*, SizeType,           \
        SyncInfo<Device::GPU> const&)

#define ETI_ALL_SIZE_TYPES(ScalarT)               \
    ETI(ScalarT, int);                            \
    ETI(ScalarT, long);                           \
    ETI(ScalarT, long long);                      \
    ETI(ScalarT, unsigned);                       \
    ETI(ScalarT, size_t)

#ifdef HYDROGEN_GPU_USE_FP16
ETI_ALL_SIZE_TYPES(gpu_half_type);
#endif

ETI_ALL_SIZE_TYPES(float);
ETI_ALL_SIZE_TYPES(double);
ETI_ALL_SIZE_TYPES(El::Complex<float>);
ETI_ALL_SIZE_TYPES(El::Complex<double>);

}// namespace hydrogen
