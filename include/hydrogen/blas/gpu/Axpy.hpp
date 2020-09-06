#ifndef HYDROGEN_BLAS_GPU_AXPY_HPP_
#define HYDROGEN_BLAS_GPU_AXPY_HPP_

#include <hydrogen/Device.hpp>
#include <hydrogen/blas/BLAS_Common.hpp>
#include <hydrogen/meta/MetaUtilities.hpp>

#ifdef HYDROGEN_HAVE_CUDA
#include <hydrogen/device/gpu/CUDA.hpp>
#elif defined(HYDROGEN_HAVE_ROCM)
#include <hydrogen/device/gpu/ROCm.hpp>
#endif

#include <stdexcept>

namespace hydrogen
{

/** @brief Execute a 2-D AXPY operation on the GPU
 *
 *  Writes dest = alpha*src + dest, taking into account 2D stride
 *  information.
 *
 *  @tparam T (Inferred) The type of data. Must be the same for source
 *      and destination matrices.
 *
 *  @param[in] num_rows The number of rows in the matrix.
 *  @param[in] num_cols The number of columns in the matrix.
 *  @param[in] alpha The scaling factor.
 *  @param[in] src The source matrix, in column-major ordering. Must
 *      not overlap with the destination matrix.
 *  @param[in] src_row_stride The number of `T`s between rows in a
 *      column of the source matrix. For "traditional" packed
 *      matrices, this will be "1".
 *  @param[in] src_col_stride The number of `T`s between columns in a
 *      row of the source matrix. For "traditional" packed matrices,
 *      this will be the leading dimension.
 *  @param[out] dest The destination matrix, in column-major
 *      ordering. Must not overlap with the source matrix.
 *  @param[in] dest_row_stride The number of `T`s between rows in a
 *      column of the destination matrix. For "traditional" packed
 *      matrices, this will be "1".
 *  @param[in] dest_col_stride The number of `T`s between columns in a
 *      row of the destination matrix. For "traditional" packed
 *      matrices, this will be the leading dimension.
 *  @param[in] sync_info The sync info wrapping the stream on which
 *      the kernel should be launched.
 */
template <typename T, typename SizeT,
          typename=EnableWhen<IsComputeType<T,Device::GPU>>>
void Axpy_GPU_impl(
    SizeT num_rows, SizeT num_cols, T const& alpha,
    T const* src, SizeT src_row_stride, SizeT src_col_stride,
    T* dest, SizeT dest_row_stride, SizeT dest_col_stride,
    SyncInfo<Device::GPU> const& sync_info);

template <typename T, typename SizeT,
          typename=EnableUnless<IsComputeType<T,Device::GPU>>,
          typename=void>
void Axpy_GPU_impl(
    SizeT, SizeT, T,
    T const*, SizeT, SizeT, T*, SizeT, SizeT,
    SyncInfo<Device::GPU> const&)
{
    throw std::logic_error("Axpy: Type not valid on GPU.");
}

/** @brief Execute a 2-D AXPY operation on the GPU
 *
 *  Writes B = alpha*op(A) + B, taking into account leading dimension
 *  information.
 *
 *  @tparam T (Inferred) The type of data. Must be the same for source
 *      and destination matrices.
 *  @tparam SizeT (Inferred) The type of size information.
 *
 *  @param[in] transpA The transpose mode of A.
 *  @param[in] num_rows The number of rows in the matrix.
 *  @param[in] num_cols The number of columns in the matrix.
 *  @param[in] alpha The scaling factor.
 *  @param[in] A The source matrix, in column-major ordering. Must not
 *      overlap with the destination matrix.
 *  @param[in] lda The leading dimension of A.
 *  @param[in,out] B The destination matrix, in column-major
 *      ordering. Must not overlap with the source matrix.
 *  @param[in] ldb The leading dimension of B.
 *  @param[in] sync_info The sync info wrapping the stream on which
 *      the kernel should be launched.
 */
template <typename T, typename SizeT,
          typename=EnableWhen<IsComputeType<T,Device::GPU>>>
void Axpy_GPU_impl(
    TransposeMode transpA,
    SizeT num_rows, SizeT num_cols,
    T const& alpha, T const* A, SizeT lda, T* B, SizeT ldb,
    SyncInfo<Device::GPU> const& sync_info);

}// namespace hydrogen
#endif // HYDROGEN_BLAS_GPU_AXPY_HPP_
