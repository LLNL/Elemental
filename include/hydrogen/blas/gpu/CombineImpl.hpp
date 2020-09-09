#ifndef HYDROGEN_SRC_HYDROGEN_BLAS_GPU_COMBINEIMPL_HPP_
#define HYDROGEN_SRC_HYDROGEN_BLAS_GPU_COMBINEIMPL_HPP_

/**
 * @file
 *
 * This file provides general-purpose 1D or 2D "combine" capability
 * for GPU-based matrices. "Combine" is basically entrywise binary
 * operations: B(i,j) <- F(A(i,j), B(i,j)). This file contains device
 * code. It will appear empty if `#include`-d somewhere device code is
 * not valid.
 */

#include <El/hydrogen_config.h>

#include <hydrogen/meta/TypeTraits.hpp>
#include <hydrogen/device/gpu/GPURuntime.hpp>

#if defined __CUDACC__ || defined __HIPCC__

namespace hydrogen
{
namespace device
{
namespace kernel
{

/** @brief Apply a functor to 1-D buffers.
 *
 *  @tparam S (Inferred) Type of source buffer.
 *  @tparam T (Inferred) Type of target buffer.
 *  @tparam SizeT (Inferred) Type of integer used to express sizes.
 *  @tparam FunctorT (Inferred) Type of functor. Must be equivalent to
 *                              `T(S const&, T const&)`.
 *
 *  @param num_entries The number of entries to which the functor is
 *                     applied.
 *  @param A The source data buffer.
 *  @param stride_A The stride between elements of A in terms of
 *                  elements of type S.
 *  @param B The target data buffer.
 *  @param stride_B The stride between elements of B in terms of
 *                  elements of type T.
 *  @param func The functor to apply. Must be device-invocable.
 */
template <typename S, typename T, typename SizeT, typename FunctorT>
__global__ void combine_1d_kernel_naive(
    SizeT num_entries,
    S const* A, SizeT stride_A,
    T * B, SizeT stride_B,
    FunctorT func)
{
    SizeT const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_entries)
        B[idx*stride_B] = func(A[idx*stride_A], B[idx*stride_B]);
}

/** @brief Apply a functor to a 2-D column-major matrix buffer.
 *
 *  This can be applied to a row-major matrix by logically transposing
 *  the matrix.
 *
 *  @tparam TILE_DIM The number of rows/columns being processed by a
 *                   thread block.
 *  @tparam BLK_COLS The number of columns handled at one time in the
 *                   block.
 *
 *  @tparam S (Inferred) Type of source buffer.
 *  @tparam T (Inferred) Type of target buffer.
 *  @tparam SizeT (Inferred) Type of integer used to express sizes.
 *  @tparam FunctorT (Inferred) Type of functor. Must be equivalent to
 *                              `T(S const&)`.
 *
 *  @param m The number of rows in A/B.
 *  @param n The number of columns in A/B. Columns must be contiguous
 *           in memory.
 *  @param A The source matrix buffer.
 *  @param lda The stride between columns of A in terms of elements of
 *             type S.
 *  @param B The target matrix buffer.
 *  @param ldb The stride between columns of B in terms of elements of
 *             type T.
 *  @param func The functor to apply. Must be device-invocable.
 */
template <int TILE_DIM, int BLK_COLS,
          typename S, typename T, typename SizeT, typename FunctorT>
__global__ void combine_2d_kernel_naive(
    SizeT m, SizeT n,
    S const* A, SizeT lda,
    T * B, SizeT ldb,
    FunctorT func)
{
    size_t const row_idx = blockIdx.x * TILE_DIM + threadIdx.x;
    size_t const col_idx = blockIdx.y * TILE_DIM + threadIdx.y;

    if (row_idx < m)
    {
        for (int ii = 0; ii < TILE_DIM && col_idx + ii < n; ii += BLK_COLS)
            B[row_idx + (col_idx+ii)*ldb] =
                func(A[row_idx + (col_idx+ii)*lda],
                     B[row_idx + (col_idx+ii)*ldb]);
    }
}
}// namespace kernel

/** @brief Apply a functor to a 1-D buffer.
 *
 *  @warning Calling this function is only valid in device code.
 *
 *  @tparam S (Inferred) Type of source buffer.
 *  @tparam T (Inferred) Type of target buffer.
 *  @tparam SizeT (Inferred) Type of integer used to express sizes.
 *  @tparam FunctorT (Inferred) Type of functor. Must be equivalent to
 *                              `T(S const&, T const&)`.
 *
 *  @param num_entries The number of entries to which the functor is
 *                     applied.
 *  @param A The source data buffer.
 *  @param stride_A The stride between elements of A in terms of
 *                  elements of type S.
 *  @param B The target data buffer.
 *  @param stride_B The stride between elements of B in terms of
 *                  elements of type T.
 *  @param func The functor to apply. Must be device-invocable.
 */
template <typename S, typename T, typename SizeT, typename FunctorT>
void CombineImpl(
    SizeT size,
    S const* A, SizeT stride_A,
    T * B, SizeT stride_B,
    FunctorT func,
    SyncInfo<Device::GPU> const& sync_info)
{
    if (size == 0)
        return;

    constexpr size_t threads_per_block = 256ULL;
    auto blocks = (size + threads_per_block - 1) / threads_per_block;
    gpu::LaunchKernel(
        kernel::combine_1d_kernel_naive<
            NativeGPUType<S>, NativeGPUType<T>, SizeT, FunctorT>,
        blocks, threads_per_block, 0, sync_info,
        size,
        AsNativeGPUType(A), stride_A,
        AsNativeGPUType(B), stride_B,
        func);
}

/** @brief Apply a functor to 2-D column-major matrix buffers.
 *
 *  This can be applied to a row-major matrix by logically transposing
 *  the matrix.
 *
 *  @warning Calling this function is only valid in device code.
 *
 *  @tparam S (Inferred) Type of source buffer.
 *  @tparam T (Inferred) Type of target buffer.
 *  @tparam SizeT (Inferred) Type of integer used to express sizes.
 *  @tparam FunctorT (Inferred) Type of functor. Must be equivalent to
 *                              `T(S const&, T const&)`.
 *
 *  @param m The number of rows in A/B.
 *  @param n The number of columns in A/B. Columns must be contiguous
 *           in memory.
 *  @param A The source matrix buffer.
 *  @param lda The stride between columns of A in terms of elements of
 *             type S.
 *  @param B The target matrix buffer.
 *  @param ldb The stride between columns of B in terms of elements of
 *             type T.
 *  @param func The functor to apply. Must be device-invocable.
 */
template <typename S, typename T, typename SizeT, typename FunctorT>
void CombineImpl(
    SizeT m, SizeT n,
    S const* A, SizeT lda,
    T * B, SizeT ldb,
    FunctorT func,
    SyncInfo<Device::GPU> const& sync_info)
{

    if (m == TypeTraits<SizeT>::Zero() || n == TypeTraits<SizeT>::Zero())
    {
        // Nothing to do
        return;
    }

    constexpr int TILE_DIM = gpu::Default2DTileSize();
    constexpr int BLK_COLS = 8;

    static_assert(TILE_DIM % BLK_COLS == 0,
                  "Incompatible TILE_DIM, BLK_COLS.");

    dim3 blks((m + TILE_DIM - 1) / TILE_DIM,
              (n + TILE_DIM - 1) / TILE_DIM,
              1);
    dim3 thds(TILE_DIM, BLK_COLS, 1);

    gpu::LaunchKernel(
        kernel::combine_2d_kernel_naive<
            TILE_DIM, BLK_COLS,
            NativeGPUType<S>, NativeGPUType<T>, SizeT, FunctorT>,
        blks, thds, 0, sync_info,
        m, n,
        AsNativeGPUType(A), lda,
        AsNativeGPUType(B), ldb,
        func);
}

}// namespace device
}// namespace hydrogen
#endif // defined __CUDACC__ || defined __HIPCC__
#endif // HYDROGEN_SRC_HYDROGEN_BLAS_GPU_COMBINEIMPL_HPP_
