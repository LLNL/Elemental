#include <hydrogen/device/gpu/CUDA.hpp>
#include <hydrogen/device/gpu/cuda/cuSOLVER.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cusolverDn.h>

namespace hydrogen
{
namespace cusolver
{

#define ADD_POTRF_IMPL(ScalarType, TypeChar)                    \
    cusolver_int GetPotrfWorkspaceSize(                         \
        cusolverDnHandle_t handle,                              \
        cublasFillMode_t uplo,                                  \
        cusolver_int n,                                         \
        DeviceArray<ScalarType> A, cusolver_int lda)            \
    {                                                           \
        cusolver_int workspace_size;                            \
        H_CHECK_CUSOLVER(                                       \
            cusolverDn ## TypeChar ## potrf_bufferSize(         \
                handle, uplo, n, A, lda, &workspace_size));     \
        return workspace_size;                                  \
    }                                                           \
    void Potrf(                                                 \
        cusolverDnHandle_t handle,                              \
        cublasFillMode_t uplo,                                  \
        cusolver_int n,                                         \
        DeviceArray<ScalarType> A, cusolver_int lda,            \
        DeviceArray<ScalarType> workspace,                      \
        cusolver_int workspace_size,                            \
        DevicePtr<cusolver_int> devinfo)                        \
    {                                                           \
        H_CHECK_CUSOLVER(                                       \
            cusolverDn ## TypeChar ## potrf(                    \
                handle, uplo, n, A, lda,                        \
                workspace, workspace_size, devinfo));           \
    }

ADD_POTRF_IMPL(float, S)
ADD_POTRF_IMPL(double, D)
ADD_POTRF_IMPL(cuComplex, C)
ADD_POTRF_IMPL(cuDoubleComplex, Z)

} // namespace cusolver
} // namespace hydrogen
