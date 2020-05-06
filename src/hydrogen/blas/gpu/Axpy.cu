#include <hydrogen/blas/gpu/Axpy.hpp>

#include <El/hydrogen_config.h>
#include <hydrogen/meta/TypeTraits.hpp>

#ifdef HYDROGEN_HAVE_CUDA
#include <hydrogen/device/gpu/CUDA.hpp>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#elif defined(HYDROGEN_HAVE_ROCM)
#include <hydrogen/device/gpu/ROCm.hpp>
#include <hip/hip_runtime.h>
#endif

namespace
{

// NOTE: B has dimension m x n.
template <int TILE_SIZE, int BLK_COLS, typename T, typename SizeT>
__global__ void axpy_2d_transpose_tiled_kernel(
    SizeT m, SizeT n, T alpha, T const* A, SizeT lda, T* B, SizeT ldb)
{

    // All the fun of a transpose meets the awesomeness of Axpy. :D
    //
    // remember: B is m x n, A is n x m
#ifdef HYDROGEN_HAVE_CUDA
    cg::thread_block cta = cg::this_thread_block();
#endif
    __shared__ T tile[TILE_SIZE][TILE_SIZE+1];

    auto const row_start_A = blockIdx.y * TILE_SIZE + threadIdx.x;
    auto const col_start_A = blockIdx.x * TILE_SIZE + threadIdx.y;

    A += row_start_A + col_start_A * lda;

    auto const row_start_B = blockIdx.x * TILE_SIZE + threadIdx.x;
    auto const col_start_B = blockIdx.y * TILE_SIZE + threadIdx.y;

    B += row_start_B + col_start_B * ldb;

    // If I am a valid row in A, I need to load things
    if (row_start_A < n)
    {
        if (col_start_A + TILE_SIZE <= m)
        {
            #pragma unroll
            for (int ii = 0; ii < TILE_SIZE; ii += BLK_COLS)
                tile[threadIdx.y+ii][threadIdx.x] = alpha * A[ii*lda];
        }
        else
        {
            for (int ii = 0; ii < TILE_SIZE && col_start_A + ii < m; ii += BLK_COLS)
                tile[threadIdx.y+ii][threadIdx.x] = alpha * A[ii*lda];
         }
    }

#ifdef HYDROGEN_HAVE_CUDA
    cg::sync(cta);
#else
    __syncthreads();
#endif

    // If I am a valid row in B, I need to store things
    if (row_start_B < m)
    {
        if (col_start_B + TILE_SIZE <= n)
        {
            #pragma unroll
            for (int ii = 0; ii < TILE_SIZE; ii += BLK_COLS)
                B[ii*ldb] += tile[threadIdx.x][threadIdx.y+ii];
        }
        else
        {
            for (int ii = 0; ii < TILE_SIZE && col_start_B + ii < n; ii += BLK_COLS)
                B[ii*ldb] += tile[threadIdx.x][threadIdx.y+ii];
        }
    }
}

template <int TILE_SIZE, int BLK_COLS, typename T, typename SizeT>
__global__ void axpy_2d_tiled_kernel(
    SizeT m, SizeT n, T alpha,
    T const* A, SizeT row_stride_A, SizeT col_stride_A,
    T* B, SizeT row_stride_B, SizeT col_stride_B)
{
    auto row_start = blockIdx.x * TILE_SIZE + threadIdx.x;
    auto col_start = blockIdx.y * TILE_SIZE + threadIdx.y;

    auto idx_in = row_start*row_stride_A + col_start*col_stride_A;
    auto idx_out = row_start*row_stride_B + col_start*col_stride_B;

    if (row_start < m)
    {
        A += idx_in;
        B += idx_out;
        if (col_start + TILE_SIZE <= n)
        {
            #pragma unroll
            for (int ii = 0; ii < TILE_SIZE; ii += BLK_COLS)
                B[ii*col_stride_B] += alpha * A[ii*col_stride_A];
        }
        else
        {
            for (int ii = 0; ii < TILE_SIZE && col_start + ii < n; ii += BLK_COLS)
                B[ii*col_stride_B] += alpha * A[ii*col_stride_A];
        }
    }
}

}// namespace <anon>

namespace hydrogen
{

template <typename T, typename SizeT, typename>
void Axpy_GPU_impl(
    SizeT height, SizeT width,
    T alpha,
    T const* X, SizeT colStrideX, SizeT rowStrideX,
    T* Y, SizeT colStrideY, SizeT rowStrideY,
    SyncInfo<Device::GPU> const& sync_info)
{
    if (height == TypeTraits<SizeT>::Zero()
        || width == TypeTraits<SizeT>::Zero())
    {
        return;
    }

    constexpr int TILE_SIZE = 32;
    constexpr int BLK_COLS = 8;

    // Short-circuit
    if (height <= 0 || width <= 0)
        return;

    dim3 blks((height + TILE_SIZE - 1) / TILE_SIZE,
              (width + TILE_SIZE - 1) / TILE_SIZE, 1);
    dim3 thds(TILE_SIZE, BLK_COLS, 1);

    gpu::LaunchKernel(
        axpy_2d_tiled_kernel<TILE_SIZE, BLK_COLS, T, SizeT>,
        blks, thds, 0, sync_info,
        height, width, alpha,
        X, colStrideX, rowStrideX,
        Y, colStrideY, rowStrideY);
}

template <typename T, typename SizeT, typename>
void Axpy_GPU_impl(
    TransposeMode transpA,
    SizeT height, SizeT width,
    T alpha,
    T const* A, SizeT lda,
    T* B, SizeT ldb,
    SyncInfo<Device::GPU> const& sync_info)
{
    // Short-circuit
    if (height <= TypeTraits<SizeT>::Zero()
        || width <= TypeTraits<SizeT>::Zero())
    {
        return;
    }

    if (transpA == TransposeMode::NORMAL)
        return Axpy_GPU_impl(
            height, width, alpha,
            A, TypeTraits<SizeT>::One(), lda,
            B, TypeTraits<SizeT>::One(), ldb, sync_info);

    constexpr int TILE_SIZE = 32;
    constexpr int BLK_COLS = 8;

    dim3 blks((height + TILE_SIZE - 1) / TILE_SIZE,
              (width + TILE_SIZE - 1) / TILE_SIZE, 1);
    dim3 thds(TILE_SIZE, BLK_COLS, 1);

    gpu::LaunchKernel(
        axpy_2d_transpose_tiled_kernel<TILE_SIZE, BLK_COLS, T, SizeT>,
        blks, thds, 0, sync_info,
        height, width, alpha, A, lda, B, ldb);
}

#define ETI(ScalarT, SizeT)                                    \
    template void Axpy_GPU_impl(                               \
        SizeT, SizeT, ScalarT,                                 \
        ScalarT const*, SizeT, SizeT,                          \
        ScalarT*, SizeT, SizeT, SyncInfo<Device::GPU> const&); \
    template void Axpy_GPU_impl(                               \
        TransposeMode, SizeT, SizeT, ScalarT,                  \
        ScalarT const*, SizeT,                                 \
        ScalarT*, SizeT, SyncInfo<Device::GPU> const&)


#ifdef HYDROGEN_GPU_USE_FP16
ETI(gpu_half_type, int);
ETI(gpu_half_type, long);
ETI(gpu_half_type, long long);
ETI(gpu_half_type, unsigned);
ETI(gpu_half_type, size_t);
#endif

ETI(float, int);
ETI(float, long);
ETI(float, long long);
ETI(float, unsigned);
ETI(float, size_t);

ETI(double, int);
ETI(double, long);
ETI(double, long long);
ETI(double, unsigned);
ETI(double, size_t);

}// namespace hydrogen
