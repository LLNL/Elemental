#ifndef HYDROGEN_BLAS_GPU_COPY_HPP_
#define HYDROGEN_BLAS_GPU_COPY_HPP_

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

/** @brief Execute a 1D COPY operation on the GPU
 *
 *  Writes dest = src, taking into account 1D stride information. This
 *  substitutes cublas<T>copy for types that cublas does not support
 *  in that operation (e.g., "__half").
 *
 *  @tparam SrcT (Inferred) The type of source data.
 *  @tparam DestT (Inferred) The type of destination data. Must be
 *          copy-assignable from SrcT.
 *
 *  @param[in] num_entries The number of entries in the array
 *  @param[in] src The source array. Must not overlap with the
 *      destination array.
 *  @param[in] src_stride The number of `T`s between entries in the
 *      source array.
 *  @param[out] dest The destination array. Must not overlap with the
 *      source array.
 *  @param[in] dest_stride The number of `T`s between entires in the
 *      destination array.
 *  @param[in] sync_info The sync info wrapping the stream on which
 *      the kernel should be launched.
 *
 *  @throws std::logic_error If the type is not supported on GPU or if
 *      the arrays overlap.
 */
template <typename SrcT, typename DestT, typename SizeT,
          typename=EnableWhen<And<IsStorageType<SrcT,Device::GPU>,
                                  IsStorageType<DestT,Device::GPU>>>>
void Copy_GPU_impl(
    SizeT num_entries,
    SrcT const* src, SizeT src_stride,
    DestT* dest, SizeT dest_stride,
    SyncInfo<Device::GPU> const& sync_info);

template <typename SrcT, typename DestT, typename SizeT,
          typename=EnableUnless<And<IsStorageType<SrcT,Device::GPU>,
                                    IsStorageType<DestT,Device::GPU>>>,
          typename=void>
void Copy_GPU_impl(
    SizeT const&,
    SrcT const* const&, SizeT const&,
    DestT* const&, SizeT const&,
    SyncInfo<Device::GPU> const&)
{
    throw std::logic_error("Type not valid on GPU");
}

/** @brief Execute a 2-D COPY operation on the GPU
 *
 *  Writes dest = src, taking into account 2D stride
 *  information.
 *
 *  @tparam T (Inferred) The type of data. Must be the same for source
 *      and destination matrices.
 *
 *  @param[in] num_rows The number of rows in the matrix.
 *  @param[in] num_cols The number of columns in the matrix.
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
 *
 *  @todo See if we can statically assert that the operator= between
 *        SrcT and DestT will succeed on the device.
 */
template <typename SrcT, typename DestT, typename SizeT,
          typename=EnableWhen<And<IsStorageType<SrcT,Device::GPU>,
                                  IsStorageType<DestT,Device::GPU>>>>
void Copy_GPU_impl(
    SizeT num_rows, SizeT num_cols,
    SrcT const* src, SizeT src_row_stride, SizeT src_col_stride,
    DestT* dest, SizeT dest_row_stride, SizeT dest_col_stride,
    SyncInfo<Device::GPU> const& sync_info);

template <typename SrcT, typename DestT, typename SizeT,
          typename=EnableUnless<And<IsStorageType<SrcT,Device::GPU>,
                                    IsStorageType<DestT,Device::GPU>>>,
          typename=void>
void Copy_GPU_impl(SizeT const&, SizeT const&,
                   SrcT const* const&, SizeT const&, SizeT const&,
                   DestT* const&, SizeT const&, SizeT const&,
                   SyncInfo<Device::GPU> const&)
{
    throw std::logic_error("Copy: Type not valid on GPU.");
}

}// namespace hydrogen
#endif // HYDROGEN_BLAS_GPU_COPY_HPP_
