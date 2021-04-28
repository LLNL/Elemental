#ifndef HYDROGEN_DEVICE_GPU_CUDA_CUBLAS_API_HPP_
#define HYDROGEN_DEVICE_GPU_CUDA_CUBLAS_API_HPP_

#include <El/hydrogen_config.h>

#include <cublas_v2.h>

namespace hydrogen
{
namespace cublas
{

/** @name BLAS-1 Routines */
///@{

#define ADD_AXPY_DECL(ScalarType)               \
    void Axpy(cublasHandle_t handle,            \
              int n, ScalarType const& alpha,   \
              ScalarType const* X, int incx,    \
              ScalarType* Y, int incy)

#define ADD_COPY_DECL(ScalarType)                       \
    void Copy(cublasHandle_t handle,                    \
              int n, ScalarType const* X, int incx,     \
              ScalarType* Y, int incy)

#define ADD_DOT_DECL(ScalarType)                \
    void Dot(cublasHandle_t handle,             \
             int n,                             \
             ScalarType const* X, int incx,     \
             ScalarType const* Y, int incy,     \
             ScalarType* output)

#define ADD_NRM2_DECL(ScalarType)               \
    void Nrm2(cublasHandle_t handle,            \
              int n,                            \
              ScalarType const* X, int incx,    \
              ScalarType* output)

#define ADD_SCALE_DECL(ScalarType)                       \
    void Scale(cublasHandle_t handle,                    \
               int n, ScalarType const& alpha,           \
               ScalarType* X, int incx)

#ifdef HYDROGEN_GPU_USE_FP16
ADD_AXPY_DECL(__half);
#endif // HYDROGEN_GPU_USE_FP16
ADD_AXPY_DECL(float);
ADD_AXPY_DECL(double);
ADD_AXPY_DECL(cuComplex);
ADD_AXPY_DECL(cuDoubleComplex);

ADD_COPY_DECL(float);
ADD_COPY_DECL(double);
ADD_COPY_DECL(cuComplex);
ADD_COPY_DECL(cuDoubleComplex);

#ifdef HYDROGEN_GPU_USE_FP16
ADD_DOT_DECL(__half);
#endif // HYDROGEN_GPU_USE_FP16
ADD_DOT_DECL(float);
ADD_DOT_DECL(double);
ADD_DOT_DECL(cuComplex);
ADD_DOT_DECL(cuDoubleComplex);

#ifdef HYDROGEN_GPU_USE_FP16
ADD_NRM2_DECL(__half);
#endif // HYDROGEN_GPU_USE_FP16
ADD_NRM2_DECL(float);
ADD_NRM2_DECL(double);
ADD_NRM2_DECL(cuComplex);
ADD_NRM2_DECL(cuDoubleComplex);

#ifdef HYDROGEN_GPU_USE_FP16
ADD_SCALE_DECL(__half);
#endif // HYDROGEN_GPU_USE_FP16
ADD_SCALE_DECL(float);
ADD_SCALE_DECL(double);
ADD_SCALE_DECL(cuComplex);
ADD_SCALE_DECL(cuDoubleComplex);

///@}
/** @name BLAS-2 Routines */
///@{

#define ADD_GEMV_DECL(ScalarType)                       \
    void Gemv(                                          \
        cublasHandle_t handle,                          \
        cublasOperation_t transpA, int m, int n,        \
        ScalarType const& alpha,                        \
        ScalarType const* A, int lda,                   \
        ScalarType const* x, int incx,                  \
        ScalarType const& beta,                         \
        ScalarType* y, int incy)

ADD_GEMV_DECL(float);
ADD_GEMV_DECL(double);
ADD_GEMV_DECL(cuComplex);
ADD_GEMV_DECL(cuDoubleComplex);

///@}
/** @name BLAS-3 Routines */
///@{

#define ADD_HERK_DECL(ScalarType, BaseScalarType)               \
    void Herk(                                                  \
        cublasHandle_t handle,                                  \
        cublasFillMode_t uplo, cublasOperation_t trans,         \
        int n, int k,                                           \
        BaseScalarType const& alpha,                            \
        ScalarType const* A, int lda,                           \
        BaseScalarType const& beta,                             \
        ScalarType * C, int ldc)

#define ADD_SYRK_DECL(ScalarType)                       \
    void Syrk(                                          \
        cublasHandle_t handle,                          \
        cublasFillMode_t uplo, cublasOperation_t trans, \
        int n, int k,                                   \
        ScalarType const& alpha,                        \
        ScalarType const* A, int lda,                   \
        ScalarType const& beta,                         \
        ScalarType* C, int ldc)

#define ADD_TRSM_DECL(ScalarType)                             \
    void Trsm(                                                \
        cublasHandle_t handle,                                \
        cublasSideMode_t side, cublasFillMode_t uplo,         \
        cublasOperation_t trans, cublasDiagType_t diag,       \
        int m, int n,                                         \
        ScalarType const& alpha,                              \
        ScalarType const* A, int lda,                         \
        ScalarType* B, int ldb)

#define ADD_GEMM_DECL(ScalarType)               \
    void Gemm(                                  \
        cublasHandle_t handle,                  \
        cublasOperation_t transpA,              \
        cublasOperation_t transpB,              \
        int m, int n, int k,                    \
        ScalarType const& alpha,                \
        ScalarType const* A, int lda,           \
        ScalarType const* B, int ldb,           \
        ScalarType const& beta,                 \
        ScalarType* C, int ldc)

#define ADD_GEMM_STRIDED_BATCHED_DECL(ScalarType)       \
    void GemmStridedBatched(                            \
        cublasHandle_t handle,                          \
        cublasOperation_t transpA,                      \
        cublasOperation_t transpB,                      \
        int m, int n, int k,                            \
        ScalarType const* alpha,                        \
        ScalarType const* A, int lda,                   \
        long long int strideA,                          \
        ScalarType const* B, int ldb,                   \
        long long int strideB,                          \
        ScalarType const* beta,                         \
        ScalarType* C, int ldc,                         \
        long long int strideC,                          \
        int batchCount)

ADD_HERK_DECL(float, float);
ADD_HERK_DECL(double, double);
ADD_HERK_DECL(cuComplex, float);
ADD_HERK_DECL(cuDoubleComplex, double);

ADD_SYRK_DECL(float);
ADD_SYRK_DECL(double);
ADD_SYRK_DECL(cuComplex);
ADD_SYRK_DECL(cuDoubleComplex);

ADD_TRSM_DECL(float);
ADD_TRSM_DECL(double);
ADD_TRSM_DECL(cuComplex);
ADD_TRSM_DECL(cuDoubleComplex);

#ifdef HYDROGEN_GPU_USE_FP16
ADD_GEMM_DECL(__half);
#endif // HYDROGEN_GPU_USE_FP16
ADD_GEMM_DECL(float);
ADD_GEMM_DECL(double);
ADD_GEMM_DECL(cuComplex);
ADD_GEMM_DECL(cuDoubleComplex);

#ifdef HYDROGEN_GPU_USE_FP16
ADD_GEMM_STRIDED_BATCHED_DECL(__half);
#endif // HYDROGEN_GPU_USE_FP16
ADD_GEMM_STRIDED_BATCHED_DECL(float);
ADD_GEMM_STRIDED_BATCHED_DECL(double);
ADD_GEMM_STRIDED_BATCHED_DECL(cuComplex);
ADD_GEMM_STRIDED_BATCHED_DECL(cuDoubleComplex);

///@}
/** @name BLAS-like Extension Routines */
///@{

// We use this for Axpy2D, Copy2D, and Transpose
#define ADD_GEAM_DECL(ScalarType)               \
    void Geam(cublasHandle_t handle,            \
              cublasOperation_t transpA,        \
              cublasOperation_t transpB,        \
              int m, int n,                     \
              ScalarType const& alpha,          \
              ScalarType const* A, int lda,     \
              ScalarType const& beta,           \
              ScalarType const* B, int ldb,     \
              ScalarType* C, int ldc)

#define ADD_DGMM_DECL(ScalarType)               \
    void Dgmm(cublasHandle_t handle,            \
              cublasSideMode_t side,            \
              int m, int n,                     \
              ScalarType const* A, int lda,     \
              ScalarType const* X, int incx,    \
              ScalarType* C, int ldc)

ADD_GEAM_DECL(float);
ADD_GEAM_DECL(double);
ADD_GEAM_DECL(cuComplex);
ADD_GEAM_DECL(cuDoubleComplex);

ADD_DGMM_DECL(float);
ADD_DGMM_DECL(double);
ADD_DGMM_DECL(cuComplex);
ADD_DGMM_DECL(cuDoubleComplex);

///@}
}// namespace cublas
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_CUDA_CUBLAS_API_HPP_
