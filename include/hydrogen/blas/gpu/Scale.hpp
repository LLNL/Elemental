#ifndef HYDROGEN_BLAS_GPU_SCALE_HPP_
#define HYDROGEN_BLAS_GPU_SCALE_HPP_

#include <hydrogen/Device.hpp>
#include <hydrogen/meta/MetaUtilities.hpp>

#ifdef HYDROGEN_HAVE_CUDA
#include <hydrogen/device/gpu/CUDA.hpp>
#elif defined(HYDROGEN_HAVE_ROCM)
#include <hydrogen/device/gpu/ROCm.hpp>
#endif

#include <stdexcept>

namespace hydrogen
{

/** @brief Execute a 1D SCALE operation on the GPU
 *
 *  Writes `buffer *= alpha`, taking into account 1D stride
 *  information. This substitutes cublas<T>scal for types that cuBLAS
 *  does not support in that operation.
 *
 *  @tparam T (Inferred) The type of data.
 *  @tparam SizeT (Inferred) The type of size information.
 *
 *  @param[in] num_entries The number of entries in the array.
 *  @param[in] alpha The scaling parameter
 *  @param[in,out] buffer The array.
 *  @param[in] stride The number of `T`s between entries in the input
 *      array.
 *  @param[in] sync_info The sync info wrapping the stream on which
 *      the kernel should be launched.
 *
 *  @throws std::logic_error If the type is not supported on GPU.
 */
template <typename T, typename SizeT,
          typename=EnableWhen<IsComputeType<T,Device::GPU>>>
void Scale_GPU_impl(
    SizeT num_entries,
    T const& alpha,
    T* buffer, SizeT stride,
    SyncInfo<Device::GPU> const& sync_info);

template <typename T, typename SizeT,
          typename=EnableUnless<IsStorageType<T,Device::GPU>>,
          typename=void>
void Scale_GPU_impl(
    SizeT const&,
    T const&,
    T const* const&, SizeT const&,
    SyncInfo<Device::GPU> const&)
{
    throw std::logic_error("Scale: Type not valid on GPU");
}

/** @brief Execute a 2-D SCALE operation on the GPU
 *
 *  Writes `buffer *= alpha`, taking into account leading dimension
 *  information.
 *
 *  @tparam T (Inferred) The type of data.
 *  @tparam SizeT (Inferred) Tye type of size information.
 *
 *  @param[in] num_rows The number of rows in the matrix.
 *  @param[in] num_cols The number of columns in the matrix.
 *  @param[in] alpha The scaling parameter.
 *  @param[in,out] buffer The matrix, in column-major ordering.
 *  @param[in] ldim The leading dimension of the data in buffer.
 *  @param[in] sync_info The sync info wrapping the stream on which
 *      the kernel should be launched.
 *
 *  @todo See if we can statically assert that the operator*= will
 *        succeed on the device.
 */
template <typename T, typename SizeT,
          typename=EnableWhen<IsStorageType<T,Device::GPU>>>
void Scale_GPU_impl(
    SizeT num_rows, SizeT num_cols,
    T const& alpha,
    T* buffer, SizeT ldim,
    SyncInfo<Device::GPU> const& sync_info);

template <typename T, typename SizeT,
          typename=EnableUnless<IsStorageType<T,Device::GPU>>,
          typename=void>
void Scale_GPU_impl(SizeT const&, SizeT const&,
                    T const&, T const* const&, SizeT const&,
                    SyncInfo<Device::GPU> const&)
{
    throw std::logic_error("Scale: Type not valid on GPU.");
}

}// namespace hydrogen
#endif // HYDROGEN_BLAS_GPU_SCALE_HPP_
