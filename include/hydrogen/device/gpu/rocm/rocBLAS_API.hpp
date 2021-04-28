#ifndef HYDROGEN_DEVICE_GPU_ROCM_ROCBLAS_API_HPP_
#define HYDROGEN_DEVICE_GPU_ROCM_ROCBLAS_API_HPP_

#include <El/hydrogen_config.h>

#include <rocblas.h>

namespace hydrogen
{
namespace rocblas
{

/** @name BLAS-1 Routines */
///@{

#define ADD_AXPY_DECL(ScalarType)                       \
    void Axpy(rocblas_handle handle,                    \
              rocblas_int n, ScalarType const& alpha,   \
              ScalarType const* X, rocblas_int incx,    \
              ScalarType* Y, rocblas_int incy)

#define ADD_COPY_DECL(ScalarType)                                       \
    void Copy(rocblas_handle handle,                                    \
              rocblas_int n, ScalarType const* X, rocblas_int incx,     \
              ScalarType* Y, rocblas_int incy)

#define ADD_DOT_DECL(ScalarType)                          \
    void Dot(rocblas_handle handle,                       \
             rocblas_int n,                               \
             ScalarType const* X, rocblas_int incx,       \
             ScalarType const* Y, rocblas_int incy,       \
             ScalarType* output)

#define ADD_NRM2_DECL(ScalarType)                       \
    void Nrm2(rocblas_handle handle,                    \
              rocblas_int n,                            \
              ScalarType const* X, rocblas_int incx,    \
              ScalarType* output)

#define ADD_SCALE_DECL(ScalarType)                      \
    void Scale(rocblas_handle handle,                   \
               rocblas_int n, ScalarType const& alpha,  \
               ScalarType* X, rocblas_int incx)

#ifdef HYDROGEN_GPU_USE_FP16
ADD_AXPY_DECL(rocblas_half);
#endif // HYDROGEN_GPU_USE_FP16
ADD_AXPY_DECL(float);
ADD_AXPY_DECL(double);

ADD_COPY_DECL(float);
ADD_COPY_DECL(double);

ADD_DOT_DECL(float);
ADD_DOT_DECL(double);

ADD_NRM2_DECL(float);
ADD_NRM2_DECL(double);

#ifdef HYDROGEN_GPU_USE_FP16
ADD_SCALE_DECL(rocblas_half);
#endif // HYDROGEN_GPU_USE_FP16
ADD_SCALE_DECL(float);
ADD_SCALE_DECL(double);

///@}
/** @name BLAS-2 Routines */
///@{

#define ADD_GEMV_DECL(ScalarType)                                       \
    void Gemv(                                                          \
        rocblas_handle handle,                                          \
        rocblas_operation transpA, rocblas_int m, rocblas_int n,        \
        ScalarType const& alpha,                                        \
        ScalarType const* A, rocblas_int lda,                           \
        ScalarType const* x, rocblas_int incx,                          \
        ScalarType const& beta,                                         \
        ScalarType* y, rocblas_int incy)

ADD_GEMV_DECL(float);
ADD_GEMV_DECL(double);

///@}
/** @name BLAS-3 Routines */
///@{

#define ADD_HERK_DECL(ScalarType, BaseScalarType)               \
    void Herk(                                                  \
        rocblas_handle handle,                                  \
        rocblas_fill uplo, rocblas_operation trans,             \
        rocblas_int n, rocblas_int k,                           \
        BaseScalarType const& alpha,                            \
        ScalarType const* A, rocblas_int lda,                   \
        BaseScalarType const& beta,                             \
        ScalarType * C, rocblas_int ldc)

#define ADD_SYRK_DECL(ScalarType)                               \
    void Syrk(                                                  \
        rocblas_handle handle,                                  \
        rocblas_fill uplo, rocblas_operation trans,             \
        rocblas_int n, rocblas_int k,                           \
        ScalarType const& alpha,                                \
        ScalarType const* A, rocblas_int lda,                   \
        ScalarType const& beta,                                 \
        ScalarType* C, rocblas_int ldc)

#define ADD_TRSM_DECL(ScalarType)                             \
    void Trsm(                                                \
        rocblas_handle handle,                                \
        rocblas_side side, rocblas_fill uplo,                 \
        rocblas_operation trans, rocblas_diagonal diag,       \
        rocblas_int m, rocblas_int n,                         \
        ScalarType const& alpha,                              \
        ScalarType const* A, int lda,                         \
        ScalarType* B, int ldb)

#define ADD_GEMM_DECL(ScalarType)                       \
    void Gemm(                                          \
        rocblas_handle handle,                          \
        rocblas_operation transpA,                      \
        rocblas_operation transpB,                      \
        rocblas_int m, rocblas_int n, rocblas_int k,    \
        ScalarType const& alpha,                        \
        ScalarType const* A, rocblas_int lda,           \
        ScalarType const* B, rocblas_int ldb,           \
        ScalarType const& beta,                         \
        ScalarType* C, rocblas_int ldc)

#define ADD_GEMM_STRIDED_BATCHED_DECL(ScalarType)         \
    void GemmStridedBatched(                              \
        rocblas_handle handle,                            \
        rocblas_operation transpA,                        \
        rocblas_operation transpB,                        \
        rocblas_int m, rocblas_int n, rocblas_int k,      \
        ScalarType const* alpha,                          \
        ScalarType const* A, rocblas_int lda,             \
        rocblas_stride strideA,                           \
        ScalarType const* B, rocblas_int ldb,             \
        rocblas_stride strideB,                           \
        ScalarType const* beta,                           \
        ScalarType* C, rocblas_int ldc,                   \
        rocblas_stride strideC,                           \
        rocblas_int batchCount)

ADD_HERK_DECL(rocblas_float_complex, float);
ADD_HERK_DECL(rocblas_double_complex, double);

ADD_SYRK_DECL(float);
ADD_SYRK_DECL(double);
ADD_SYRK_DECL(rocblas_float_complex);
ADD_SYRK_DECL(rocblas_double_complex);

ADD_TRSM_DECL(float);
ADD_TRSM_DECL(double);
ADD_TRSM_DECL(rocblas_float_complex);
ADD_TRSM_DECL(rocblas_double_complex);

#ifdef HYDROGEN_GPU_USE_FP16
ADD_GEMM_DECL(rocblas_half);
#endif // HYDROGEN_GPU_USE_FP16
ADD_GEMM_DECL(float);
ADD_GEMM_DECL(double);

#ifdef HYDROGEN_GPU_USE_FP16
ADD_GEMM_STRIDED_BATCHED_DECL(rocblas_half);
#endif // HYDROGEN_GPU_USE_FP16
ADD_GEMM_STRIDED_BATCHED_DECL(float);
ADD_GEMM_STRIDED_BATCHED_DECL(double);

///@}
/** @name BLAS-like Extension Routines */
///@{

// We use this for Axpy2D, Copy2D, and Transpose
#define ADD_GEAM_DECL(ScalarType)                       \
    void Geam(rocblas_handle handle,                    \
              rocblas_operation transpA,                \
              rocblas_operation transpB,                \
              rocblas_int m, rocblas_int n,             \
              ScalarType const& alpha,                  \
              ScalarType const* A, rocblas_int lda,     \
              ScalarType const& beta,                   \
              ScalarType const* B, rocblas_int ldb,     \
              ScalarType* C, rocblas_int ldc)

ADD_GEAM_DECL(float);
ADD_GEAM_DECL(double);

#define ADD_DGMM_DECL(ScalarType)                       \
    void Dgmm(rocblas_handle handle,                    \
              rocblas_side side,                        \
              rocblas_int m, rocblas_int n,             \
              ScalarType const* A, rocblas_int lda,     \
              ScalarType const* X, rocblas_int incx,    \
              ScalarType* C, rocblas_int ldc)
ADD_DGMM_DECL(float);
ADD_DGMM_DECL(double);

///@}
}// namespace rocblas
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_ROCM_ROCBLAS_API_HPP_
