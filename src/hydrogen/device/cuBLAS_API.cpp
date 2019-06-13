#include <hydrogen/device/gpu/CUDA.hpp>
#include <hydrogen/device/gpu/cuda/cuBLAS.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef HYDROGEN_GPU_USE_FP16
#include <cuda_fp16.h>
#endif // HYDROGEN_GPU_USE_FP16

#include <cublas_v2.h>

namespace hydrogen
{
namespace cublas
{

//
// BLAS 1
//

#ifdef HYDROGEN_GPU_USE_FP16
void Axpy(cublasHandle_t handle,
          int n, __half const& alpha,
          __half const* X, int incx,
          __half* Y, int incy)
{
    H_CHECK_CUBLAS(
        cublasAxpyEx(
            handle,
            n, &alpha, CUDA_R_16F,
            X, CUDA_R_16F, incx,
            Y, CUDA_R_16F, incy,
            CUDA_R_32F));
}

void Scale(cublasHandle_t handle,
            int n, __half const& alpha,
            __half* X, int incx)
{
    H_CHECK_CUBLAS(
        cublasScalEx(
            handle, n, &alpha, CUDA_R_16F, X, CUDA_R_16F, incx, CUDA_R_32F));
}
#endif // HYDROGEN_GPU_USE_FP16

#define ADD_AXPY_IMPL(ScalarType, TypeChar)                     \
    void Axpy(cublasHandle_t handle,                            \
              int n, ScalarType const& alpha,                   \
              ScalarType const* X, int incx,                    \
              ScalarType* Y, int incy)                          \
    {                                                           \
        H_CHECK_CUBLAS(                                        \
            cublas ## TypeChar ## axpy(                         \
                handle,                                         \
                n, &alpha, X, incx, Y, incy));                  \
    }

#define ADD_COPY_IMPL(ScalarType, TypeChar)             \
    void Copy(cublasHandle_t handle,                    \
              int n, ScalarType const* X, int incx,     \
              ScalarType* Y, int incy)                  \
    {                                                   \
        H_CHECK_CUBLAS(                                \
            cublas ## TypeChar ## copy(                 \
                handle,                                 \
                n, X, incx, Y, incy));                  \
    }

#define ADD_SCALE_IMPL(ScalarType, TypeChar)               \
    void Scale(cublasHandle_t handle,                      \
               int n, ScalarType const& alpha,             \
               ScalarType* X, int incx)                    \
    {                                                      \
        H_CHECK_CUBLAS(                                   \
            cublas ## TypeChar ## scal(                    \
                handle, n, &alpha, X, incx));              \
    }

//
// BLAS 2
//
#define ADD_GEMV_IMPL(ScalarType, TypeChar)              \
    void Gemv(                                           \
        cublasHandle_t handle,                           \
        cublasOperation_t transpA, int m, int n,         \
        ScalarType const& alpha,                         \
        ScalarType const* A, int lda,                    \
        ScalarType const* B, int ldb,                    \
        ScalarType const& beta,                          \
        ScalarType* C, int ldc)                          \
    {                                                    \
        H_CHECK_CUBLAS(cublas ## TypeChar ## gemv(      \
                            handle,                      \
                            transpA,                     \
                            m, n,                        \
                            &alpha, A, lda, B, ldb,      \
                            &beta, C, ldc));             \
    }

//
// BLAS 3
//
#define ADD_GEMM_IMPL(ScalarType, TypeChar)             \
    void Gemm(                                          \
        cublasHandle_t handle,                          \
        cublasOperation_t transpA,                      \
        cublasOperation_t transpB,                      \
        int m, int n, int k,                            \
        ScalarType const& alpha,                        \
        ScalarType const* A, int lda,                   \
        ScalarType const* B, int ldb,                   \
        ScalarType const& beta,                         \
        ScalarType* C, int ldc)                         \
    {                                                   \
        H_CHECK_CUBLAS(                                \
            cublas ## TypeChar ## gemm(                 \
                handle,                                 \
                transpA, transpB,                       \
                m, n, k, &alpha, A, lda, B, ldb,        \
                &beta, C, ldc));                        \
    }

//
// BLAS-like Extension
//
#define ADD_GEAM_IMPL(ScalarType, TypeChar)     \
    void Geam(                                  \
        cublasHandle_t handle,                  \
        cublasOperation_t transpA,              \
        cublasOperation_t transpB,              \
        int m, int n,                           \
        ScalarType const& alpha,                \
        ScalarType const* A, int lda,           \
        ScalarType const& beta,                 \
        ScalarType const* B, int ldb,           \
        ScalarType* C, int ldc)                 \
    {                                           \
        H_CHECK_CUBLAS(                        \
            cublas ## TypeChar ## geam(         \
                handle,                         \
                transpA, transpB,               \
                m, n,                           \
                &alpha, A, lda,                 \
                &beta, B, ldb,                  \
                C, ldc));                       \
    }

#define ADD_DGMM_IMPL(ScalarType, TypeChar)             \
    void Dgmm(                                          \
        cublasHandle_t handle,                          \
        cublasSideMode_t side,                          \
        int m, int n,                                   \
        ScalarType const* A, int lda,                   \
        ScalarType const* X, int incx,                  \
        ScalarType* C, int ldc)                         \
    {                                                   \
        H_CHECK_CUBLAS(                                \
            cublas ## TypeChar ## dgmm(                 \
                handle,                                 \
                side, m, n, A, lda, X, incx, C, ldc));  \
    }

// BLAS 1
ADD_AXPY_IMPL(float, S)
ADD_AXPY_IMPL(double, D)
ADD_AXPY_IMPL(cuComplex, C)
ADD_AXPY_IMPL(cuDoubleComplex, Z)

ADD_COPY_IMPL(float, S)
ADD_COPY_IMPL(double, D)
ADD_COPY_IMPL(cuComplex, C)
ADD_COPY_IMPL(cuDoubleComplex, Z)

ADD_SCALE_IMPL(float, S)
ADD_SCALE_IMPL(double, D)
ADD_SCALE_IMPL(cuComplex, C)
ADD_SCALE_IMPL(cuDoubleComplex, Z)

// BLAS 2
ADD_GEMV_IMPL(float, S)
ADD_GEMV_IMPL(double, D)
ADD_GEMV_IMPL(cuComplex, C)
ADD_GEMV_IMPL(cuDoubleComplex, Z)

// BLAS 3
ADD_GEMM_IMPL(__half, H)
ADD_GEMM_IMPL(float, S)
ADD_GEMM_IMPL(double, D)
ADD_GEMM_IMPL(cuComplex, C)
ADD_GEMM_IMPL(cuDoubleComplex, Z)

// BLAS-like extension
ADD_GEAM_IMPL(float, S)
ADD_GEAM_IMPL(double, D)
ADD_GEAM_IMPL(cuComplex, C)
ADD_GEAM_IMPL(cuDoubleComplex, Z)

ADD_DGMM_IMPL(float, S)
ADD_DGMM_IMPL(double, D)
ADD_DGMM_IMPL(cuComplex, C)
ADD_DGMM_IMPL(cuDoubleComplex, Z)

//
// "STATIC" UNIT TEST
//

#define ASSERT_SUPPORT(type, op)                        \
    static_assert(IsSupportedType<type, op>::value, "")

#define ASSERT_NO_SUPPORT(type, op)                             \
    static_assert(!IsSupportedType<type, op>::value, "")

ASSERT_SUPPORT(float, BLAS_Op::AXPY);
ASSERT_SUPPORT(float, BLAS_Op::COPY);
ASSERT_SUPPORT(float, BLAS_Op::DGMM);
ASSERT_SUPPORT(float, BLAS_Op::GEAM);
ASSERT_SUPPORT(float, BLAS_Op::GEMM);
ASSERT_SUPPORT(float, BLAS_Op::GEMV);
ASSERT_SUPPORT(float, BLAS_Op::SCAL);

ASSERT_SUPPORT(double, BLAS_Op::AXPY);
ASSERT_SUPPORT(double, BLAS_Op::COPY);
ASSERT_SUPPORT(double, BLAS_Op::DGMM);
ASSERT_SUPPORT(double, BLAS_Op::GEAM);
ASSERT_SUPPORT(double, BLAS_Op::GEMM);
ASSERT_SUPPORT(double, BLAS_Op::GEMV);
ASSERT_SUPPORT(double, BLAS_Op::SCAL);

ASSERT_SUPPORT(std::complex<float>, BLAS_Op::AXPY);
ASSERT_SUPPORT(std::complex<float>, BLAS_Op::COPY);
ASSERT_SUPPORT(std::complex<float>, BLAS_Op::DGMM);
ASSERT_SUPPORT(std::complex<float>, BLAS_Op::GEAM);
ASSERT_SUPPORT(std::complex<float>, BLAS_Op::GEMM);
ASSERT_SUPPORT(std::complex<float>, BLAS_Op::GEMV);
ASSERT_SUPPORT(std::complex<float>, BLAS_Op::SCAL);

ASSERT_SUPPORT(std::complex<double>, BLAS_Op::AXPY);
ASSERT_SUPPORT(std::complex<double>, BLAS_Op::COPY);
ASSERT_SUPPORT(std::complex<double>, BLAS_Op::DGMM);
ASSERT_SUPPORT(std::complex<double>, BLAS_Op::GEAM);
ASSERT_SUPPORT(std::complex<double>, BLAS_Op::GEMM);
ASSERT_SUPPORT(std::complex<double>, BLAS_Op::GEMV);
ASSERT_SUPPORT(std::complex<double>, BLAS_Op::SCAL);

#ifdef HYDROGEN_GPU_USE_FP16
ASSERT_SUPPORT(__half, BLAS_Op::AXPY);
ASSERT_SUPPORT(__half, BLAS_Op::GEMM);
ASSERT_SUPPORT(__half, BLAS_Op::SCAL);
ASSERT_NO_SUPPORT(__half, BLAS_Op::COPY);
ASSERT_NO_SUPPORT(__half, BLAS_Op::DGMM);
ASSERT_NO_SUPPORT(__half, BLAS_Op::GEAM);
ASSERT_NO_SUPPORT(__half, BLAS_Op::GEMV);

#ifdef HYDROGEN_HAVE_HALF
ASSERT_SUPPORT(cpu_half_type, BLAS_Op::AXPY);
ASSERT_SUPPORT(cpu_half_type, BLAS_Op::GEMM);
ASSERT_SUPPORT(cpu_half_type, BLAS_Op::SCAL);
ASSERT_NO_SUPPORT(cpu_half_type, BLAS_Op::COPY);
ASSERT_NO_SUPPORT(cpu_half_type, BLAS_Op::DGMM);
ASSERT_NO_SUPPORT(cpu_half_type, BLAS_Op::GEAM);
ASSERT_NO_SUPPORT(cpu_half_type, BLAS_Op::GEMV);
#endif // HYDROGEN_HAVE_HALF
#endif // HYDROGEN_GPU_USE_FP16

// One type that should be entirely unsupported, just for sanity.
ASSERT_NO_SUPPORT(int, BLAS_Op::AXPY);
ASSERT_NO_SUPPORT(int, BLAS_Op::COPY);
ASSERT_NO_SUPPORT(int, BLAS_Op::DGMM);
ASSERT_NO_SUPPORT(int, BLAS_Op::GEAM);
ASSERT_NO_SUPPORT(int, BLAS_Op::GEMM);
ASSERT_NO_SUPPORT(int, BLAS_Op::GEMV);

} // namespace cublas

} // namespace hydrogen
