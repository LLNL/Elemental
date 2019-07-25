#ifndef HYDROGEN_BLAS_GPU_TRANSPOSE_HPP_
#define HYDROGEN_BLAS_GPU_TRANSPOSE_HPP_

#include <hydrogen/Device.hpp>
#include <hydrogen/meta/MetaUtilities.hpp>

#if defined(HYDROGEN_HAVE_CUDA)
#include <hydrogen/device/gpu/CUDA.hpp>
#elif defined(HYDROGEN_HAVE_ROCM)
#include <hydrogen/device/gpu/ROCm.hpp>
#endif

#include <stdexcept>

namespace hydrogen
{

/** @brief Execute a 2-D COPY operation on the GPU
 *
 *  Writes B = A^T, taking into account 2D stride
 *  information.
 *
 *  @tparam T (Inferred) The type of data. Must be the same for source
 *      and destination matrices.
 *  @tparam SizeT (Inferred) The type of size information.
 *
 *  @param[in] num_rows The number of rows in the matrix A.
 *  @param[in] num_cols The number of columns in the matrix A.
 *  @param[in] A The source matrix, in column-major ordering. Must not
 *      overlap with the destination matrix.
 *  @param[in] lda The leading dimension of A.
 *  @param[out] dest The destination matrix, in column-major ordering. Must not
 *      overlap with the source matrix. Contents will be overwritten.
 *  @param[in] ldb The leading dimension of B.
 *  @param stream The CUDA stream on which the kernel should be
 *      launched.
 */
template <typename T, typename SizeT,
          typename=EnableWhen<IsStorageType<T,Device::GPU>>>
void Transpose_GPU_impl(
    SizeT num_rows, SizeT num_cols,
    T const* A, SizeT lda,
    T* B, SizeT ldb,
    gpuStream_t stream);

template <typename T, typename SizeT,
          typename=EnableUnless<IsStorageType<T,Device::GPU>>,
          typename=void>
void Transpose_GPU_impl(
    SizeT const&, SizeT const&,
    T const* const&, SizeT const&,
    T* const&, SizeT const&,
    gpuStream_t const&)
{
    throw std::logic_error("Copy: Type not valid on GPU.");
}

}// namespace hydrogen
#endif // HYDROGEN_BLAS_GPU_TRANSPOSE_HPP_
