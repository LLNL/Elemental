#include <hydrogen/device/gpu/ROCm.hpp>
#include <hydrogen/device/gpu/rocm/rocSOLVER.hpp>

#include <rocsolver/rocsolver.h>

namespace hydrogen {
namespace rocsolver {

#define ADD_POTRF_IMPL(ScalarType, TypeChar)                                   \
    rocblas_int GetPotrfWorkspaceSize(rocblas_handle /*handle*/,               \
                                      rocblas_fill /*uplo*/,                   \
                                      rocblas_int /*n*/,                       \
                                      DeviceArray<ScalarType> /*A*/,           \
                                      rocblas_int /*lda*/)                     \
    {                                                                          \
        return rocblas_int(0);                                                 \
    }                                                                          \
    void Potrf(rocblas_handle handle,                                          \
               rocblas_fill uplo,                                              \
               rocblas_int n,                                                  \
               DeviceArray<ScalarType> A,                                      \
               rocblas_int lda,                                                \
               DeviceArray<ScalarType> /*workspace*/,                          \
               rocblas_int /*workspace_size*/,                                 \
               DevicePtr<rocblas_int> devinfo)                                 \
    {                                                                          \
        H_CHECK_ROCSOLVER(                                                     \
            rocsolver_##TypeChar##potrf(handle, uplo, n, A, lda, devinfo));    \
    }

#define ADD_HEEV_IMPL(ScalarType, TypePrefix)                                  \
    rocsolver_int GetHeevWorkspaceSize(                                        \
        rocblas_handle /*handle*/,                                             \
        rocblas_fill /*uplo*/,                                                 \
        rocblas_int n,                                                         \
        DeviceArray<ScalarType> /*A*/,                                         \
        rocblas_int /*lda*/,                                                   \
        DeviceArray<RealType<ScalarType>> /*W*/)                               \
    {                                                                          \
        return n;                                                              \
    }                                                                          \
    void Heev(rocblas_handle handle,                                           \
              rocblas_fill uplo,                                               \
              rocblas_int n,                                                   \
              DeviceArray<ScalarType> A,                                       \
              rocblas_int lda,                                                 \
              DeviceArray<RealType<ScalarType>> W,                             \
              DeviceArray<ScalarType> workspace_in,                            \
              rocblas_int /*workspace_size*/,                                  \
              DevicePtr<rocblas_int> devinfo)                                  \
    {                                                                          \
        auto* workspace =                                                      \
            reinterpret_cast<RealType<ScalarType>*>(workspace_in);             \
        H_CHECK_ROCSOLVER(rocsolver_##TypePrefix##ev(handle,                   \
                                                     rocblas_evect_original,   \
                                                     uplo,                     \
                                                     n,                        \
                                                     A,                        \
                                                     lda,                      \
                                                     W,                        \
                                                     workspace,                \
                                                     devinfo));                \
    }

ADD_POTRF_IMPL(float, s)
ADD_POTRF_IMPL(double, d)
ADD_POTRF_IMPL(rocblas_float_complex, c)
ADD_POTRF_IMPL(rocblas_double_complex, z)

ADD_HEEV_IMPL(float, ssy)
ADD_HEEV_IMPL(double, dsy)
ADD_HEEV_IMPL(rocblas_float_complex, che)
ADD_HEEV_IMPL(rocblas_double_complex, zhe)

} // namespace rocsolver
} // namespace hydrogen
