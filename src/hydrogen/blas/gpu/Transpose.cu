#include <hydrogen/blas/gpu/Transpose.hpp>

#include <El/hydrogen_config.h>
#include <hydrogen/meta/TypeTraits.hpp>

#include <hydrogen/device/gpu/GPURuntime.hpp>
#ifdef HYDROGEN_HAVE_CUDA
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#endif

namespace
{

// A is m x n, B is n x m. Memory access is coalesced.
template <int TILE_SIZE, int BLK_COLS, typename T, typename SizeT>
__global__ void transpose_kernel(
    SizeT const m, SizeT const n,
    T const* __restrict__ A, SizeT const lda,
    T* __restrict__ B, SizeT const ldb)
{
#ifdef HYDROGEN_HAVE_CUDA
    cg::thread_block cta = cg::this_thread_block();
#endif
    using StorageType = hydrogen::GPUStaticStorageType<T>;
    __shared__ StorageType tile_shared[TILE_SIZE][TILE_SIZE+1];
    auto tile = reinterpret_cast<T(*)[TILE_SIZE+1]>(tile_shared);

    SizeT row_idx_A = blockIdx.x * TILE_SIZE + threadIdx.x;
    SizeT col_idx_A = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Starting point is inside the matrix
    bool const do_anything = (row_idx_A < m && col_idx_A < n);

    // Ending point is inside the matrix
    bool const do_all = (do_anything) && (col_idx_A + TILE_SIZE <= n);

    size_t const idx_in = row_idx_A + col_idx_A*lda;

    SizeT row_idx = blockIdx.y * TILE_SIZE + threadIdx.x;
    SizeT col_idx = blockIdx.x * TILE_SIZE + threadIdx.y;

    size_t const idx_out = row_idx + col_idx*ldb;
    if (do_all)
    {
        //
        // Bounds are fine; just do the thing
        //
        #pragma unroll
        for (int ii = 0; ii < TILE_SIZE; ii += BLK_COLS)
        {
            tile[threadIdx.y+ii][threadIdx.x] = A[idx_in + ii*lda];
        }

#ifdef HYDROGEN_HAVE_CUDA
        cg::sync(cta);
#else
        __syncthreads();
#endif

        #pragma unroll
        for (int ii = 0; ii < TILE_SIZE; ii += BLK_COLS)
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
            for (int ii = 0; ii < TILE_SIZE && col_idx_A + ii < n; ii += BLK_COLS)
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
            for (int ii = 0; ii < TILE_SIZE && col_idx + ii < m; ii += BLK_COLS)
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

    constexpr int TILE_SIZE = 32;
    constexpr int BLK_COLS = 8;
    dim3 blks((m + TILE_SIZE - 1) / TILE_SIZE,
              (n + TILE_SIZE - 1) / TILE_SIZE,
              1);
    dim3 thds(TILE_SIZE, BLK_COLS, 1);

    gpu::LaunchKernel(
        transpose_kernel<TILE_SIZE,BLK_COLS,NativeGPUType<T>,SizeT>,
        blks, thds, 0, sync_info,
        m, n, AsNativeGPUType(A), lda, AsNativeGPUType(B), ldb);
}

#define ETI(DataType, SizeType)                            \
    template void Transpose_GPU_impl(                      \
        SizeType, SizeType,                                \
        DataType const*, SizeType,                         \
        DataType*, SizeType, SyncInfo<Device::GPU> const&)

#define ETI_ALL_SIZE_TYPES(ScalarT) \
  ETI(ScalarT, int);                \
  ETI(ScalarT, long);               \
  ETI(ScalarT, long long);          \
  ETI(ScalarT, unsigned);           \
  ETI(ScalarT, size_t)

#ifdef HYDROGEN_GPU_USE_FP16
ETI_ALL_SIZE_TYPES(gpu_half_type);
#endif // HYDROGEN_GPU_USE_FP16

ETI_ALL_SIZE_TYPES(float);
ETI_ALL_SIZE_TYPES(double);
ETI_ALL_SIZE_TYPES(El::Complex<float>);
ETI_ALL_SIZE_TYPES(El::Complex<double>);

}// namespace hydrogen
