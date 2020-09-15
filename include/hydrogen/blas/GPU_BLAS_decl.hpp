#ifndef HYDROGEN_GPU_BLAS_DECL_HPP_
#define HYDROGEN_GPU_BLAS_DECL_HPP_

#include <hydrogen/Device.hpp>
#include <hydrogen/SyncInfo.hpp>
#include <hydrogen/blas/BLAS_Common.hpp>

/** @file
 *
 *  This file describes the BLAS-like interface as exposed to the GPU
 *  device.
 */

/** @defgroup device_blas Device-Specific BLAS-like APIs
 *
 *  This API abstracts the node-local BLAS API to run on arbitrary
 *  accelerators, or `El::Device`s. Note that we do *not* expose the
 *  entire BLAS interface on every device. Additional functionality
 *  may be added as needed.
 */

namespace hydrogen
{
namespace details
{
template <typename T>
struct BaseT
{
    using type = T;
};

// Ehhhhh this is certainly not fool-proof... But I don't want to
// include all the El::Complex stuff here.
template <template <typename> class AugmentedT, typename T>
struct BaseT<AugmentedT<T>>
{
    using type = T;
};
}// namespace details

template <typename T>
using TmpBase = typename details::BaseT<T>::type;

/** @namespace gpu_blas
 *  @brief A collection of BLAS routines that are exposed for GPUs
 *
 *  This namespace contains facilities and interfaces for performing
 *  BLAS operations on device memory. Generally, the functions will
 *  dispatch to a programming-model-specific optimized BLAS library
 *  (cuBLAS, rocBLAS) if the data are of suitable type for interaction
 *  with those libraries. Fallback kernels are implemented for some
 *  additional types (e.g., the FP16 types). Additional support can be
 *  added as needed.
 */
namespace gpu_blas
{

/** @name BLAS-1 Routines */
///@{

/** @brief 1-D Strided Axpy operation in GPU memory.
 *
 *  The Axpy operation is a scaled vector add:
 *
 *  @f[ Y = \alpha X + Y\qquad X,Y\in\mathbb{F}^{\text{size}}. @f]
 *
 *  The vectors @f$X@f$, @f$Y@f$ may be stored with different stride
 *  but must have the same size.
 *
 *  @tparam T (Inferred) The type of data.
 *  @tparam SizeT (Inferred) The type used to express size information.
 *
 *  @param[in] size The number of entries in X and Y
 *  @param[in] alpha The scaling parameter on X
 *  @param[in] X The source vector
 *  @param[in] incx The stride between entries of X
 *  @param[in,out] Y The target vector
 *  @param[in] incy The stride between entries of Y
 *  @param[in] syncinfo The SyncInfo to use for this operation
 *
 *  @ingroup device_blas
 */
template <typename T, typename SizeT>
void Axpy(SizeT size, T const& alpha,
          T const* X, SizeT incx,
          T* Y, SizeT incy,
          SyncInfo<Device::GPU> const& syncinfo);

/** @brief 2-D Axpy operation in GPU memory.
 *
 *  The 2-D Axpy operation is a scaled matrix add:
 *
 *  @f[ B = \alpha A + B\qquad A,B\in\mathbb{F}^{\text{rows}\times\text{cols}}. @f]
 *
 *  The matrices @f$A@f$, @f$B@f$ may have different leading
 *  dimensions but must have the same size.
 *
 *  @tparam T (Inferred) The type of data
 *  @tparam SizeT (Inferred) The type used to express size information.
 *
 *  @param[in] num_rows The number of rows in A and B
 *  @param[in] num_cols The number of columns in A and B
 *  @param[in] alpha The scaling parameter on A
 *  @param[in] A The source matrix, in column-major storage
 *  @param[in] lda The stride between columns of A
 *  @param[in,out] B The target matrix, in column-major storage
 *  @param[in] ldb The stride between columns of B
 *  @param[in] syncinfo Synchronization information for this operation
 *
 *  @ingroup device_blas
 */
template <typename T, typename SizeT>
void Axpy(SizeT num_rows, SizeT num_cols,
          T const& alpha,
          T const* A, SizeT lda,
          T* B, SizeT ldb,
          SyncInfo<Device::GPU> const& syncinfo);

/** @brief 2-D Axpy operation in GPU memory with optional transpose.
 *
 *  The 2-D Axpy operation is a scaled matrix add:
 *
 *  @f[ B = \alpha op(A) + B\qquad A,B\in\mathbb{F}^{\text{rows}\times\text{cols}}. @f]
 *
 *  The matrices @f$A@f$, @f$B@f$ may have different leading
 *  dimensions but must have the same size.
 *
 *  @tparam T (Inferred) The type of data
 *  @tparam SizeT (Inferred) The type used to express size information.
 *
 *  @param[in] transpA The transpose mode for A.
 *  @param[in] num_rows The number of rows in B.
 *  @param[in] num_cols The number of columns in B.
 *  @param[in] alpha The scaling parameter on A.
 *  @param[in] A The source matrix, in column-major storage.
 *  @param[in] lda The stride between columns of A.
 *  @param[in,out] B The target matrix, in column-major storage.
 *  @param[in] ldb The stride between columns of B.
 *  @param[in] syncinfo Synchronization information for this operation.
 *
 *  @ingroup device_blas
 */
template <typename T, typename SizeT>
void Axpy(
    TransposeMode transpA,
    SizeT num_rows, SizeT num_cols,
    T const& alpha,
    T const* A, SizeT lda,
    T* B, SizeT ldb,
    SyncInfo<Device::GPU> const& syncinfo);

/** @brief 2-D Axpy operation in GPU memory with 2 strides.
 *
 *  The 2-D Axpy operation is a scaled matrix add:
 *
 *  @f[ B = \alpha A + B\qquad A,B\in\mathbb{F}^{\text{rows}\times\text{cols}}. @f]
 *
 *  The matrices @f$A@f$, @f$B@f$ may have different row strides and
 *  leading dimensions but must have the same size.
 *
 *  @tparam T (Inferred) The type of data
 *  @tparam SizeT (Inferred) The type used to express size information.
 *
 *  @param[in] num_rows The number of rows in A and B.
 *  @param[in] num_cols The number of columns in A and B.
 *  @param[in] alpha The scaling parameter on A.
 *  @param[in] A The source matrix, in column-major storage.
 *  @param[in] row_stride_A The stride between rows of A.
 *  @param[in] lda The stride between columns of A.
 *  @param[in,out] B The target matrix, in column-major storage.
 *  @param[in] row_stride_B The stride between rows of B.
 *  @param[in] ldb The stride between columns of B.
 *  @param[in] syncinfo Synchronization information for this
 *                      operation.
 *
 *  @ingroup device_blas
 */
template <typename T, typename SizeT>
void Axpy(SizeT num_rows, SizeT num_cols,
          T const& alpha,
          T const* A, SizeT row_stride_A, SizeT lda,
          T* B, SizeT row_stride_B, SizeT ldb,
          SyncInfo<Device::GPU> const& syncinfo);

/** @brief Strided 1-D Copy operation in GPU memory.
 *
 *  This is simple copy assignment:
 *
 *  @f[ Y = X\qquad X,Y\in\mathbb{F}^{\text{size}}. @f]
 *
 *  The vectors @f$X@f$, @f$Y@f$ may be stored with different stride
 *  but must have the same size.
 *
 *  @tparam T (Inferred) The type of data.
 *  @tparam SizeT (Inferred) The type used to express size information.
 *
 *  @param[in] size The number of entries in X and Y.
 *  @param[in] X The source vector.
 *  @param[in] incx The stride between entries of X.
 *  @param[out] Y The target vector.
 *  @param[in] incy The stride between entries of Y.
 *  @param[in] syncinfo Synchronization information for this
 *                      operation.
 *
 *  @ingroup device_blas
 */
template <typename T, typename SizeT>
void Copy(SizeT size,
          T const* X, SizeT incx,
          T* Y, SizeT incy,
          SyncInfo<Device::GPU> const& syncinfo);

/** @brief 2-D Copy operation in GPU memory.
 *
 *  This is simple copy assignment:
 *
 *  @f[ B = op(A) @f]
 *
 *  The matrices @f$A@f$, @f$B@f$ may have different leading
 *  dimensions but must have the same size.
 *
 *  @tparam T (Inferred) The type of data
 *  @tparam SizeT (Inferred) The type used to express size information.
 *
 *  @param[in] transpA The transpose operation to apply to A (`NORMAL`,
 *                     `TRANSPOSE`, or `CONJ_TRANSPOSE`).
 *  @param[in] num_rows The number of rows in op(A) and B.
 *  @param[in] num_cols The number of columns in op(A) and B.
 *  @param[in] A The source matrix, in column-major storage.
 *  @param[in] lda The stride between columns of A.
 *  @param[out] B The target matrix, in column-major storage.
 *  @param[in] ldb The stride between columns of B.
 *  @param[in] syncinfo Synchronization information for this
 *                      operation.
 *
 *  @ingroup device_blas
 */
template <typename T, typename U, typename SizeT>
void Copy(TransposeMode transpA,
          SizeT num_rows, SizeT num_cols,
          T const* A, SizeT lda,
          U* B, SizeT ldb,
          SyncInfo<Device::GPU> const& syncinfo);

/** @brief 2-D Copy operation in GPU memory with 2 strides.
 *
 *  The 2-D Axpy operation is a scaled matrix add:
 *
 *  This is simple copy assignment:
 *
 *  @f[ B = A\qquad A,B\in\mathbb{F}^{\text{rows}\times\text{cols}}. @f]
 *
 *  The matrices @f$A@f$, @f$B@f$ may have different row strides and
 *  leading dimensions but must have the same size.
 *
 *  @tparam T (Inferred) The type of data
 *  @tparam SizeT (Inferred) The type used to express size information.
 *
 *  @param[in] num_rows The number of rows in A and B.
 *  @param[in] num_cols The number of columns in A and B.
 *  @param[in] A The source matrix, in column-major storage.
 *  @param[in] row_stride_A The stride between rows of A.
 *  @param[in] lda The stride between columns of A.
 *  @param[out] B The target matrix, in column-major storage.
 *  @param[in] row_stride_B The stride between rows of B.
 *  @param[in] ldb The stride between columns of B.
 *  @param[in] syncinfo Synchronization information for this
 *                      operation.
 *
 *  @ingroup device_blas
 */
template <typename T, typename SizeT>
void Copy(SizeT num_rows, SizeT num_cols,
          T const* A, SizeT row_stride_A, SizeT lda,
          T* B, SizeT row_stride_B, SizeT ldb,
          SyncInfo<Device::GPU> const& syncinfo);

/** @brief A dot-product operation for 1-D memory.
 *
 *  @tparam T (Inferred) The type of data.
 *  @tparam SizeT (Inferred) The type used to express size information.
 *
 *  @param num_entries The number of entries in X and Y.
 *  @param X The first vector (device memory).
 *  @param stride_X The stride of X.
 *  @param Y The second vector (device memory).
 *  @param stride_Y The stride of Y.
 *  @param result The result of the dot product (host or device memory).
 *  @param[in] syncinfo The synchronization information for this
 *                      operation.
 *
 *  @ingroup device_blas
 */
template <typename T, typename SizeT>
void Dot(SizeT num_entries,
         T const* X, SizeT stride_X,
         T const* Y, SizeT stride_Y,
         T* result,
         SyncInfo<Device::GPU> const& syncinfo);

/** @brief Computes the 2-norm of 1-D memory.
 *
 *  @tparam T (Inferred) The type of data.
 *  @tparam SizeT (Inferred) The type used to express size information.
 *
 *  @param num_entries The number of entries in X.
 *  @param X The vector (device memory).
 *  @param stride_X The stride of X.
 *  @param result The result of the norm (host or device memory).
 *  @param[in] syncinfo The synchronization information for this
 *                      operation.
 *
 *  @ingroup device_blas
 */
template <typename T, typename SizeT>
void Nrm2(SizeT num_entries,
          T const* X, SizeT stride_X,
          T* result,
          SyncInfo<Device::GPU> const& syncinfo);

/** @brief 1-D Scale operation in GPU memory.
 *
 *  This is in-place scaling:
 *
 *  @f[ A *= \alpha. @f]
 *
 *  @tparam T (Inferred) The type of data
 *  @tparam SizeT (Inferred) The type used to express size information.
 *
 *  @param[in] num_entries The number of entries in the array A.
 *  @param[in] alpha The scaling parameter
 *  @param[in,out] A The input array.
 *  @param[in] stride The stride between entries of A.
 *  @param[in] syncinfo Synchronization information for this
 *                      operation.
 *
 *  @ingroup device_blas
 */
template <typename T, typename SizeT>
void Scale(SizeT num_entries,
           T const& alpha, T* A, SizeT stride,
           SyncInfo<Device::GPU> const& syncinfo);

/** @brief 2-D Scale operation in GPU memory.
 *
 *  This is in-place scaling:
 *
 *  @f[ A *= \alpha. @f]
 *
 *  @tparam T (Inferred) The type of data
 *  @tparam SizeT (Inferred) The type used to express size information.
 *
 *  @param[in] num_rows The number of rows in A.
 *  @param[in] num_cols The number of columns in A.
 *  @param[in] alpha The scaling parameter
 *  @param[in,out] A The matrix, in column-major storage.
 *  @param[in] lda The stride between columns of A.
 *  @param[in] syncinfo Synchronization information for this
 *                      operation.
 *
 *  @ingroup device_blas
 */
template <typename T, typename SizeT>
void Scale(SizeT num_rows, SizeT num_cols,
           T const& alpha, T* A, SizeT lda,
           SyncInfo<Device::GPU> const& syncinfo);

///@}
/** @name BLAS-2 Routines */
///@{

/** @brief Matrix-vector product in GPU memory.
 *
 *  Perform a scaled matrix-vector product:
 *
 *  @f[ y = \alpha\text{op}(A)x + \beta y. @f]
 *
 *  @tparam T (Inferred) The type of the data. Should be a field.
 *  @tparam SizeT (Inferred) The type used to express size information.
 *
 *  @param[in] transpA The operation flag for A indicating `NORMAL`,
 *                     `TRANSPOSE`, or `CONJ_TRANSPOSE`.
 *  @param[in] m The number of rows in `op(A)` and y.
 *  @param[in] n The number of columns in `op(A)` and rows in x.
 *  @param[in] alpha The scaling term on the matvec term.
 *  @param[in] A The matrix in column-major format.
 *  @param[in] lda The leading dimension of A.
 *  @param[in] x The source vector.
 *  @param[in] incx The stride between elements of x.
 *  @param[in] beta The scaling applied to the input value of the
 *                  target vector.
 *  @param[in,out] y The target vector. Inital values are scaled by
 *                   beta and updated with the result of the matvec.
 *  @param[in] incy The stride between elements of y.
 *  @param[in] syncinfo The synchronization information for this
 *                      operation.
 *
 *  @ingroup device_blas
 */
template <typename T, typename SizeT>
void Gemv(
    TransposeMode transpA, SizeT m, SizeT n,
    T const& alpha,
    T const* A, SizeT lda,
    T const* x, SizeT incx,
    T const& beta,
    T* y, SizeT incy,
    SyncInfo<Device::GPU> const& syncinfo);

///@}
/** @name BLAS-3 Routines */
///@{

/** @brief Hermitian rank-K update of matrices in GPU memory.
 *  @todo Finish documentation.
 */
template <typename T, typename SizeT>
void Herk(
    FillMode uplo, TransposeMode trans,
    SizeT n, SizeT k,
    TmpBase<T> const& alpha,
    T const* A, SizeT lda,
    TmpBase<T> const& beta,
    T* C, SizeT ldc,
    SyncInfo<Device::GPU> const& syncinfo);

/** @brief Symmetric rank-K update of matrices in GPU memory.
 *  @todo Finish documentation.
 */
template <typename T, typename SizeT>
void Syrk(
    FillMode uplo, TransposeMode trans,
    SizeT n, SizeT k,
    T const& alpha,
    T const* A, SizeT lda,
    T const& beta,
    T* C, SizeT ldc,
    SyncInfo<Device::GPU> const& syncinfo);

/** @brief Triangular matrix solve.
 *  @todo Finish documentation.
 */
template <typename T, typename SizeT>
void Trsm(
    SideMode side, FillMode uplo,
    TransposeMode trans, DiagType diag,
    SizeT m, SizeT n,
    T const& alpha,
    T const* A, SizeT lda,
    T* B, SizeT ldb,
    SyncInfo<Device::GPU> const& syncinfo);

/** @brief Matrix-matrix product in GPU memory.
 *
 *  Perform a scaled matrix-matrix product:
 *
 *  @f[ C = \alpha\text{op}(A)\text{op}(B) + \beta C. @f]
 *
 *  @tparam T (Inferred) The type of the data. Should be a field.
 *  @tparam SizeT (Inferred) The type used to express size information.
 *
 *  @param[in] transpA The operation flag for `A` indicating `NORMAL`,
 *                     `TRANSPOSE`, or `CONJ_TRANSPOSE`.
 *  @param[in] transpB The operation flag for `B` indicating `NORMAL`,
 *                     `TRANSPOSE`, or `CONJ_TRANSPOSE`.
 *  @param[in] m The number of rows in `op(A)` and C.
 *  @param[in] n The number of columns in `op(B)` and C.
 *  @param[in] k The number of columns in `op(A)` and rows in `op(B)`.
 *  @param[in] alpha The scaling term on the multiplicative term.
 *  @param[in] A A matrix in column-major format.
 *  @param[in] lda The leading dimension of A.
 *  @param[in] B A matrix in column-major format.
 *  @param[in] ldb The leading dimension of B
 *  @param[in] beta The scaling applied to the input value of the
 *                  target matrix.
 *  @param[in,out] C The target matrix. Inital values are scaled by
 *                   beta and updated with the result of the product.
 *  @param[in] ldc The leading dimension of C.
 *  @param[in] syncinfo The synchronization information for this
 *                      operation.
 *
 *  @ingroup device_blas
 */
template <typename T, typename SizeT>
void Gemm(
    TransposeMode transpA, TransposeMode transpB,
    SizeT m, SizeT n, SizeT k,
    T const& alpha,
    T const* A, SizeT lda,
    T const* B, SizeT ldb,
    T const& beta,
    T* C, SizeT ldc,
    SyncInfo<Device::GPU> const& syncinfo);

/** @brief Batched, strided matrix-matrix product in GPU memory.
 *
 *  @todo Write documentation.
 *
 *  @tparam T (Inferred) The type of the data. Should be a field.
 *  @tparam SizeT (Inferred) The type used to express size information.
 *  @tparam StrideT (Inferred) The type used to express stride information.
 *
 *  @param[in] transpA The operation flag for `A` indicating `NORMAL`,
 *                     `TRANSPOSE`, or `CONJ_TRANSPOSE`.
 *  @param[in] transpB The operation flag for `B` indicating `NORMAL`,
 *                     `TRANSPOSE`, or `CONJ_TRANSPOSE`.
 *  @param[in] m The number of rows in `op(A)` and C.
 *  @param[in] n The number of columns in `op(B)` and C.
 *  @param[in] k The number of columns in `op(A)` and rows in `op(B)`.
 *  @param[in] alpha The scaling term on the multiplicative term.
 *  @param[in] A A matrix in column-major format.
 *  @param[in] lda The leading dimension of A.
 *  @param[in] strideA The between A matrices.
 *  @param[in] B A matrix in column-major format.
 *  @param[in] ldb The leading dimension of B
 *  @param[in] strideB The between B matrices.
 *  @param[in] beta The scaling applied to the input value of the
 *                  target matrix.
 *  @param[in,out] C The target matrix. Inital values are scaled by
 *                   beta and updated with the result of the product.
 *  @param[in] ldc The leading dimension of C.
 *  @param[in] strideC The between C matrices.
 *  @param[in] batchCount The number of GEMMs in the batch.
 *  @param[in] syncinfo The synchronization information for this
 *                      operation.
 *
 *  @ingroup device_blas
 */
template <typename T, typename SizeT, typename StrideT>
void GemmStridedBatched(
    TransposeMode transpA, TransposeMode transpB,
    SizeT m, SizeT n, SizeT k,
    T const& alpha,
    T const* A, SizeT lda, StrideT strideA,
    T const* B, SizeT ldb, StrideT strideB,
    T const& beta,
    T* C, SizeT ldc, StrideT strideC,
    SizeT batchCount,
    SyncInfo<Device::GPU> const& syncinfo);

///@}
/** @name BLAS-like Extension Routines */
///@{

/** @brief Matrix-matrix product with a diagonal matrix in GPU memory.
 *
 *  Perform that product of a matrix with a diagonal matrix. If
 *  `side == LEFT`,
 *
 *  @f[ C = \text{diag}(X) \times A. @f]
 *
 *  If `side == RIGHT`,
 *
 *  @f[ C = A \times \text{diag}(X). @f]
 *
 *  @tparam T (Inferred) The type of the data. Should be a field.
 *  @tparam SizeT (Inferred) The type used to express size information.
 *
 *  @param[in] side The side of the source matrix on which the
 *                  diagonal matrix appears.
 *  @param[in] m The number of rows in A and C.
 *  @param[in] n The number of columns in A and C.
 *  @param[in] A A matrix in column-major format.
 *  @param[in] lda The leading dimension of A.
 *  @param[in] X The diagonal matrix stored as a single vector of size
 *               `m` if `side == LEFT` and size `n` if `side == RIGHT`.
 *  @param[in] incx The stride of X.
 *  @param[out] C The target matrix.
 *  @param[in] ldc The leading dimension of C.
 *  @param[in] syncinfo The synchronization information for this
 *                      operation.
 *
 *  @ingroup device_blas
 */
template <typename T, typename SizeT>
void Dgmm(SideMode side,
          SizeT m, SizeT n,
          T const* A, SizeT lda,
          T const* X, SizeT incx,
          T* C, SizeT ldc,
          SyncInfo<Device::GPU> const& syncinfo);

///@}

}// namespace gpu_blas

namespace gpu_lapack
{

/** @brief Compute the Cholesky factorization of A.
 *
 *  This routine will allocate the properly-sized workspace and info
 *  parameter, using the caching allocators if available.
 *
 *  @note The `info` parameter to the underlying library will only be
 *        checked in Debug builds.
 */
template <typename T, typename SizeT>
void CholeskyFactorize(FillMode uplo,
                       SizeT n,
                       T* A, SizeT lda,
                       SyncInfo<Device::GPU> const& si);

/** @brief Compute the Cholesky factorization of A with a user-provided
 *         workspace.
 *
 *  This routine will allocate the info argument.
 *
 *  @note The `info` parameter to the underlying library will only be
 *        checked in Debug builds.
 *  @todo Improve the error reporting.
 */
template <typename T, typename SizeT>
void CholeskyFactorize(FillMode uplo,
                       SizeT n,
                       T* A, SizeT lda,
                       T* workspace, SizeT workspace_size,
                       SyncInfo<Device::GPU> const& si);

/** @brief Compute the Cholesky factorization of A with a user-provided
 *         workspace and info parameter.
 *
 *  @note The `info` parameter will not be checked by this routine.
 */
template <typename T, typename SizeT, typename InfoT>
void CholeskyFactorize(FillMode uplo,
                       SizeT n,
                       T* A, SizeT lda,
                       T* workspace, SizeT workspace_size,
                       InfoT* info,
                       SyncInfo<Device::GPU> const& si);

}// namespace gpu_lapack
}// namespace hydrogen
#endif // HYDROGEN_GPU_BLAS_DECL_HPP_
