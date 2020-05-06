#include <hydrogen/blas/gpu/Transpose.hpp>

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

// A is m x n, B is n x m. Memory access is coalesced.
template <int TILE_DIM, int BLK_COLS, typename T, typename SizeT>
__global__ void transpose_kernel(
    SizeT const m, SizeT const n,
    T const* __restrict__ A, SizeT const lda,
    T* __restrict__ B, SizeT const ldb)
{
#ifdef HYDROGEN_HAVE_CUDA
    cg::thread_block cta = cg::this_thread_block();
#endif
    __shared__ T tile[TILE_DIM][TILE_DIM+1];

    SizeT row_idx_A = blockIdx.x * TILE_DIM + threadIdx.x;
    SizeT col_idx_A = blockIdx.y * TILE_DIM + threadIdx.y;

    // Starting point is inside the matrix
    bool const do_anything = (row_idx_A < m && col_idx_A < n);

    // Ending point is inside the matrix
    bool const do_all = (do_anything) && (col_idx_A + TILE_DIM <= n);

    size_t const idx_in = row_idx_A + col_idx_A*lda;

    SizeT row_idx = blockIdx.y * TILE_DIM + threadIdx.x;
    SizeT col_idx = blockIdx.x * TILE_DIM + threadIdx.y;

    size_t const idx_out = row_idx + col_idx*ldb;
    if (do_all)
    {
        //
        // Bounds are fine; just do the thing
        //
        #pragma unroll
        for (int ii = 0; ii < TILE_DIM; ii += BLK_COLS)
        {
            tile[threadIdx.y+ii][threadIdx.x] = A[idx_in + ii*lda];
        }

#ifdef HYDROGEN_HAVE_CUDA
        cg::sync(cta);
#else
        __syncthreads();
#endif

        #pragma unroll
        for (int ii = 0; ii < TILE_DIM; ii += BLK_COLS)
        {
            B[idx_out + ii*ldb] = tile[threadIdx.x][threadIdx.y+ii];
        }
    }
    else
    {
        //
        // Some work doesn't get done. Be more careful
        //

        // Make sure we don't grab extra columns
        if (row_idx_A < m)
        {
            for (int ii = 0; ii < TILE_DIM && col_idx_A + ii < n; ii += BLK_COLS)
                tile[threadIdx.y+ii][threadIdx.x] = A[idx_in + ii*lda];
        }

        // Same warp-sync stuff -- I assume this still needs to happen.
#ifdef HYDROGEN_HAVE_CUDA
        cg::sync(cta);
#else
        __syncthreads();
#endif

        // Don't write rows of the new matrix that don't exist.
        if (row_idx < n)
        {
            for (int ii = 0; ii < TILE_DIM && col_idx + ii < m; ii += BLK_COLS)
                B[idx_out + ii*ldb] = tile[threadIdx.x][threadIdx.y+ii];
        }
    }
}

}// namespace <anon>


namespace hydrogen
{

template <typename T, typename SizeT, typename>
void Transpose_GPU_impl(
    SizeT m, SizeT n, T const* A, SizeT lda, T* B, SizeT ldb,
    SyncInfo<Device::GPU> const& sync_info)
{
    if (m == TypeTraits<SizeT>::Zero() || n == TypeTraits<SizeT>::Zero())
        return;

    constexpr int TILE_DIM = 32;
    constexpr int BLK_COLS = 8;

    dim3 blks((m + TILE_DIM - 1) / TILE_DIM,
              (n + TILE_DIM - 1) / TILE_DIM,
              1);
    dim3 thds(TILE_DIM, BLK_COLS, 1);

    gpu::LaunchKernel(
        transpose_kernel<TILE_DIM,BLK_COLS,T,SizeT>,
        blks, thds, 0, sync_info,
        m, n, A, lda, B, ldb);
}

#define ETI(DataType, SizeType)                            \
    template void Transpose_GPU_impl(                      \
        SizeType, SizeType,                                \
        DataType const*, SizeType,                         \
        DataType*, SizeType, SyncInfo<Device::GPU> const&)

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

#ifdef HYDROGEN_GPU_USE_FP16
ETI(gpu_half_type, int);
ETI(gpu_half_type, long);
ETI(gpu_half_type, long long);
ETI(gpu_half_type, unsigned);
ETI(gpu_half_type, size_t);
#endif // HYDROGEN_GPU_USE_FP16
}// namespace hydrogen
