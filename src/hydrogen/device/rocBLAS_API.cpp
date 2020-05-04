#include <hydrogen/device/gpu/rocm/rocBLAS_API.hpp>

#include <hydrogen/device/gpu/ROCm.hpp>
#include <hydrogen/device/gpu/rocm/rocBLAS.hpp>

#include <rocblas.h>

namespace hydrogen
{
namespace rocblas
{

//
// BLAS 1
//

#define ADD_AXPY_IMPL(ScalarType, TypeChar)                     \
    void Axpy(rocblas_handle handle,                            \
              int n, ScalarType const& alpha,                   \
              ScalarType const* X, int incx,                    \
              ScalarType* Y, int incy)                          \
    {                                                           \
        H_CHECK_ROCBLAS(                                        \
            rocblas_ ## TypeChar ## axpy(                         \
                handle,                                         \
                n, &alpha, X, incx, Y, incy));                  \
    }

#define ADD_COPY_IMPL(ScalarType, TypeChar)             \
    void Copy(rocblas_handle handle,                    \
              int n, ScalarType const* X, int incx,     \
              ScalarType* Y, int incy)                  \
    {                                                   \
        H_CHECK_ROCBLAS(                                \
            rocblas_ ## TypeChar ## copy(                 \
                handle,                                 \
                n, X, incx, Y, incy));                  \
    }

#define ADD_DOT_IMPL(ScalarType, TypeChar)      \
    void Dot(rocblas_handle handle,             \
             int n,                             \
             ScalarType const* X, int incx,     \
             ScalarType const* Y, int incy,     \
             ScalarType* result)                \
    {                                           \
        H_CHECK_ROCBLAS(                        \
            rocblas_ ## TypeChar ## dot(        \
                handle,                         \
                n, X, incx, Y, incy, result));  \
    }

#define ADD_NRM2_IMPL(ScalarType, TypeChar)             \
    void Nrm2(rocblas_handle handle,                    \
              int n, ScalarType const* X, int incx,     \
              ScalarType* result)                       \
    {                                                   \
        H_CHECK_ROCBLAS(                                \
            rocblas_ ## TypeChar ## nrm2(               \
                handle,                                 \
                n, X, incx, result));                   \
    }

#define ADD_SCALE_IMPL(ScalarType, TypeChar)               \
    void Scale(rocblas_handle handle,                      \
               int n, ScalarType const& alpha,             \
               ScalarType* X, int incx)                    \
    {                                                      \
        H_CHECK_ROCBLAS(                                   \
            rocblas_ ## TypeChar ## scal(                    \
                handle, n, &alpha, X, incx));              \
    }

//
// BLAS 2
//
#define ADD_GEMV_IMPL(ScalarType, TypeChar)              \
    void Gemv(                                           \
        rocblas_handle handle,                           \
        rocblas_operation transpA, int m, int n,         \
        ScalarType const& alpha,                         \
        ScalarType const* A, int lda,                    \
        ScalarType const* B, int ldb,                    \
        ScalarType const& beta,                          \
        ScalarType* C, int ldc)                          \
    {                                                    \
        H_CHECK_ROCBLAS(rocblas_ ## TypeChar ## gemv(      \
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
        rocblas_handle handle,                          \
        rocblas_operation transpA,                      \
        rocblas_operation transpB,                      \
        rocblas_int m, rocblas_int n, rocblas_int k,                            \
        ScalarType const& alpha,                        \
        ScalarType const* A, rocblas_int lda,                   \
        ScalarType const* B, rocblas_int ldb,                   \
        ScalarType const& beta,                         \
        ScalarType* C, rocblas_int ldc)                         \
    {                                                   \
        H_CHECK_ROCBLAS(                                \
            rocblas_ ## TypeChar ## gemm(                 \
                handle,                                 \
                transpA, transpB,                       \
                m, n, k, &alpha, A, lda, B, ldb,        \
                &beta, C, ldc));                        \
    }

#define ADD_GEMM_STRIDED_BATCHED_IMPL(ScalarType, TypeChar)             \
    void GemmStridedBatched(                                            \
        rocblas_handle handle,                                          \
        rocblas_operation transpA,                                      \
        rocblas_operation transpB,                                      \
        rocblas_int m, rocblas_int n, rocblas_int k,                    \
        ScalarType const& alpha,                                        \
        ScalarType const* A, rocblas_int lda, rocblas_stride strideA,   \
        ScalarType const* B, rocblas_int ldb, rocblas_stride strideB,   \
        ScalarType const& beta,                                         \
        ScalarType* C, rocblas_int ldc, rocblas_stride strideC,         \
        rocblas_int batchCount)                                         \
    {                                                                   \
        H_CHECK_ROCBLAS(                                                \
            rocblas_ ## TypeChar ## gemm_strided_batched(               \
                handle,                                                 \
                transpA, transpB,                                       \
                m, n, k, &alpha,                                        \
                A, lda, strideA,                                        \
                B, ldb, strideB,                                        \
                &beta, C, ldc, strideC, batchCount));                   \
    }

//
// BLAS-like Extension
//
#define ADD_GEAM_IMPL(ScalarType, TypeChar)     \
    void Geam(                                  \
        rocblas_handle handle,                  \
        rocblas_operation transpA,              \
        rocblas_operation transpB,              \
        int m, int n,                           \
        ScalarType const& alpha,                \
        ScalarType const* A, int lda,           \
        ScalarType const& beta,                 \
        ScalarType const* B, int ldb,           \
        ScalarType* C, int ldc)                 \
    {                                           \
        H_CHECK_ROCBLAS(                        \
            rocblas_ ## TypeChar ## geam(       \
                handle,                         \
                transpA, transpB,               \
                m, n,                           \
                &alpha, A, lda,                 \
                &beta, B, ldb,                  \
                C, ldc));                       \
    }

#define ADD_DGMM_IMPL(ScalarType, TypeChar)                     \
    void Dgmm(                                                  \
        rocblas_handle handle,                                  \
        rocblas_side side,                                      \
        int m, int n,                                           \
        ScalarType const* A, int lda,                           \
        ScalarType const* X, int incx,                          \
        ScalarType* C, int ldc)                                 \
    {                                                           \
        H_CHECK_ROCBLAS(rocblas_status_not_implemented);        \
    }

// BLAS 1
ADD_AXPY_IMPL(rocblas_half, h)
ADD_AXPY_IMPL(float, s)
ADD_AXPY_IMPL(double, d)

ADD_COPY_IMPL(float, s)
ADD_COPY_IMPL(double, d)

//ADD_DOT_IMPL(rocblas_half, h)
ADD_DOT_IMPL(float, s)
ADD_DOT_IMPL(double, d)

//ADD_NRM2_IMPL(rocblas_half, h)
ADD_NRM2_IMPL(float, s)
ADD_NRM2_IMPL(double, d)

ADD_SCALE_IMPL(float, s)
ADD_SCALE_IMPL(double, d)

// BLAS 2
ADD_GEMV_IMPL(float, s)
ADD_GEMV_IMPL(double, d)

// BLAS 3
ADD_GEMM_IMPL(rocblas_half, h)
ADD_GEMM_IMPL(float, s)
ADD_GEMM_IMPL(double, d)

ADD_GEMM_STRIDED_BATCHED_IMPL(rocblas_half, h)
ADD_GEMM_STRIDED_BATCHED_IMPL(float, s)
ADD_GEMM_STRIDED_BATCHED_IMPL(double, d)

// BLAS-like extension
ADD_GEAM_IMPL(float, s)
ADD_GEAM_IMPL(double, d)

ADD_DGMM_IMPL(float, s)
ADD_DGMM_IMPL(double, d)

//
// "STATIC" UNIT TEST
//

#define ASSERT_SUPPORT(type, op)                        \
    static_assert(IsSupportedType<type, op>::value, "")

#define ASSERT_NO_SUPPORT(type, op)                             \
    static_assert(!IsSupportedType<type, op>::value, "")

ASSERT_SUPPORT(float, BLAS_Op::AXPY);
ASSERT_SUPPORT(float, BLAS_Op::COPY);
ASSERT_SUPPORT(float, BLAS_Op::GEAM);
ASSERT_SUPPORT(float, BLAS_Op::GEMM);
ASSERT_SUPPORT(float, BLAS_Op::GEMV);
ASSERT_SUPPORT(float, BLAS_Op::SCAL);
ASSERT_NO_SUPPORT(float, BLAS_Op::DGMM);
ASSERT_SUPPORT(float, BLAS_Op::DOT);
ASSERT_SUPPORT(float, BLAS_Op::NRM2);
ASSERT_SUPPORT(float, BLAS_Op::GEMMSTRIDEDBATCHED);

ASSERT_SUPPORT(double, BLAS_Op::AXPY);
ASSERT_SUPPORT(double, BLAS_Op::COPY);
ASSERT_SUPPORT(double, BLAS_Op::GEAM);
ASSERT_SUPPORT(double, BLAS_Op::GEMM);
ASSERT_SUPPORT(double, BLAS_Op::GEMV);
ASSERT_SUPPORT(double, BLAS_Op::SCAL);
ASSERT_NO_SUPPORT(double, BLAS_Op::DGMM);
ASSERT_SUPPORT(double, BLAS_Op::DOT);
ASSERT_SUPPORT(double, BLAS_Op::NRM2);
ASSERT_SUPPORT(double, BLAS_Op::GEMMSTRIDEDBATCHED);

#ifdef HYDROGEN_GPU_USE_FP16
ASSERT_SUPPORT(rocblas_half, BLAS_Op::AXPY);
ASSERT_SUPPORT(rocblas_half, BLAS_Op::GEMM);
ASSERT_NO_SUPPORT(rocblas_half, BLAS_Op::SCAL);
ASSERT_NO_SUPPORT(rocblas_half, BLAS_Op::COPY);
ASSERT_NO_SUPPORT(rocblas_half, BLAS_Op::DGMM);
ASSERT_NO_SUPPORT(rocblas_half, BLAS_Op::GEAM);
ASSERT_NO_SUPPORT(rocblas_half, BLAS_Op::GEMV);
ASSERT_SUPPORT(rocblas_half, BLAS_Op::DOT);
ASSERT_SUPPORT(rocblas_half, BLAS_Op::NRM2);
ASSERT_SUPPORT(rocblas_half, BLAS_Op::GEMMSTRIDEDBATCHED);

#ifdef HYDROGEN_HAVE_HALF
ASSERT_SUPPORT(cpu_half_type, BLAS_Op::AXPY);
ASSERT_SUPPORT(cpu_half_type, BLAS_Op::GEMM);
ASSERT_NO_SUPPORT(cpu_half_type, BLAS_Op::SCAL);
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
ASSERT_NO_SUPPORT(int, BLAS_Op::SCAL);
ASSERT_NO_SUPPORT(int, BLAS_Op::DOT);
ASSERT_NO_SUPPORT(int, BLAS_Op::NRM2);
ASSERT_NO_SUPPORT(int, BLAS_Op::GEMMSTRIDEDBATCHED);

} // namespace rocblas
} // namespace hydrogen
