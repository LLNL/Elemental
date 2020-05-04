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

#define ADD_AXPY_DECL(ScalarType)               \
    void Axpy(rocblas_handle handle,            \
              int n, ScalarType const& alpha,   \
              ScalarType const* X, int incx,    \
              ScalarType* Y, int incy)

#define ADD_COPY_DECL(ScalarType)                       \
    void Copy(rocblas_handle handle,                    \
              int n, ScalarType const* X, int incx,     \
              ScalarType* Y, int incy)

#define ADD_DOT_DECL(ScalarType)                \
    void Dot(rocblasHandle_t handle,            \
             int n,                             \
             ScalarType const* X, int incx,     \
             ScalarType const* Y, int incy,     \
             ScalarType* output)

#define ADD_NRM2_DECL(ScalarType)               \
    void Nrm2(rocblasHandle_t handle,           \
              int n,                            \
              ScalarType const* X, int incx,    \
              ScalarType* output)

#define ADD_SCALE_DECL(ScalarType)                       \
    void Scale(rocblas_handle handle,                    \
               int n, ScalarType const& alpha,           \
               ScalarType* X, int incx)

#ifdef HYDROGEN_GPU_USE_FP16
ADD_AXPY_DECL(rocblas_half);
#endif // HYDROGEN_GPU_USE_FP16
ADD_AXPY_DECL(float);
ADD_AXPY_DECL(double);

ADD_COPY_DECL(float);
ADD_COPY_DECL(double);

#ifdef HYDROGEN_GPU_USE_FP16
ADD_SCALE_DECL(rocblas_half);
#endif // HYDROGEN_GPU_USE_FP16
ADD_SCALE_DECL(float);
ADD_SCALE_DECL(double);

///@}
/** @name BLAS-2 Routines */
///@{

#define ADD_GEMV_DECL(ScalarType)                       \
    void Gemv(                                          \
        rocblas_handle handle,                          \
        rocblas_operation transpA, int m, int n,        \
        ScalarType const& alpha,                        \
        ScalarType const* A, int lda,                   \
        ScalarType const* x, int incx,                  \
        ScalarType const& beta,                         \
        ScalarType* y, int incy)

ADD_GEMV_DECL(float);
ADD_GEMV_DECL(double);

///@}
/** @name BLAS-3 Routines */
///@{

#define ADD_GEMM_DECL(ScalarType)               \
    void Gemm(                                  \
        rocblas_handle handle,                  \
        rocblas_operation transpA,              \
        rocblas_operation transpB,              \
        int m, int n, int k,                    \
        ScalarType const& alpha,                \
        ScalarType const* A, int lda,           \
        ScalarType const* B, int ldb,           \
        ScalarType const& beta,                 \
        ScalarType* C, int ldc)

#ifdef HYDROGEN_GPU_USE_FP16
ADD_GEMM_DECL(rocblas_half);
#endif // HYDROGEN_GPU_USE_FP16
ADD_GEMM_DECL(float);
ADD_GEMM_DECL(double);

///@}
/** @name BLAS-like Extension Routines */
///@{

// We use this for Axpy2D, Copy2D, and Transpose
#define ADD_GEAM_DECL(ScalarType)               \
    void Geam(rocblas_handle handle,            \
              rocblas_operation transpA,        \
              rocblas_operation transpB,        \
              int m, int n,                     \
              ScalarType const& alpha,          \
              ScalarType const* A, int lda,     \
              ScalarType const& beta,           \
              ScalarType const* B, int ldb,     \
              ScalarType* C, int ldc)

ADD_GEAM_DECL(float);
ADD_GEAM_DECL(double);

#define ADD_DGMM_DECL(ScalarType)               \
    void Dgmm(rocblas_handle handle,            \
              rocblas_side side,                \
              int m, int n,                     \
              ScalarType const* A, int lda,     \
              ScalarType const* X, int incx,    \
              ScalarType* C, int ldc)
ADD_DGMM_DECL(float);
ADD_DGMM_DECL(double);

///@}
}// namespace rocblas
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_ROCM_ROCBLAS_API_HPP_
