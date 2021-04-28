#include <hydrogen/device/gpu/ROCm.hpp>
#include <hydrogen/device/gpu/rocm/rocSOLVER.hpp>

#include <rocsolver.h>

namespace hydrogen
{
namespace rocsolver
{

#define ADD_POTRF_IMPL(ScalarType, TypeChar)                    \
    rocblas_int GetPotrfWorkspaceSize(                          \
        rocblas_handle /*handle*/,                              \
        rocblas_fill /*uplo*/,                                  \
        rocblas_int /*n*/,                                      \
        DeviceArray<ScalarType> /*A*/, rocblas_int /*lda*/)     \
    {                                                           \
        return rocblas_int(0);                                  \
    }                                                           \
    void Potrf(                                                 \
        rocblas_handle handle,                                  \
        rocblas_fill uplo,                                      \
        rocblas_int n,                                          \
        DeviceArray<ScalarType> A, rocblas_int lda,             \
        DeviceArray<ScalarType> /*workspace*/,                  \
        rocblas_int /*workspace_size*/,                         \
        DevicePtr<rocblas_int> devinfo)                         \
    {                                                           \
        H_CHECK_ROCSOLVER(                                       \
            rocsolver_ ## TypeChar ## potrf(                    \
                handle, uplo, n, A, lda,                        \
                devinfo));                                      \
    }

ADD_POTRF_IMPL(float, s)
ADD_POTRF_IMPL(double, d)
ADD_POTRF_IMPL(rocblas_float_complex, c)
ADD_POTRF_IMPL(rocblas_double_complex, z)

} // namespace rocsolver
} // namespace hydrogen
