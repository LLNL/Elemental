#include "hydrogen/device/gpu/cuda/cuSOLVERError.hpp"
#include <hydrogen/device/gpu/CUDA.hpp>
#include <hydrogen/device/gpu/cuda/cuSOLVER.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cusolverDn.h>

// There are some (old, unverified) posts on the interwebs about where
// exactly performance should fall off. The asymptotics of the Jacobi
// method aren't as favorable as QR, but for small matrices, it should
// have some benefit.
#define CUSOLVER_HEEV_JACOBI_THRESHOLD 1024

namespace hydrogen
{
namespace cusolver
{

#define ADD_POTRF_IMPL(ScalarType, TypeChar)                                   \
    cusolver_int GetPotrfWorkspaceSize(cusolverDnHandle_t handle,              \
                                       cublasFillMode_t uplo,                  \
                                       cusolver_int n,                         \
                                       DeviceArray<ScalarType> A,              \
                                       cusolver_int lda)                       \
    {                                                                          \
        cusolver_int workspace_size;                                           \
        H_CHECK_CUSOLVER(                                                      \
            cusolverDn##TypeChar##potrf_bufferSize(handle,                     \
                                                   uplo,                       \
                                                   n,                          \
                                                   A,                          \
                                                   lda,                        \
                                                   &workspace_size));          \
        return workspace_size;                                                 \
    }                                                                          \
    void Potrf(cusolverDnHandle_t handle,                                      \
               cublasFillMode_t uplo,                                          \
               cusolver_int n,                                                 \
               DeviceArray<ScalarType> A,                                      \
               cusolver_int lda,                                               \
               DeviceArray<ScalarType> workspace,                              \
               cusolver_int workspace_size,                                    \
               DevicePtr<cusolver_int> devinfo)                                \
    {                                                                          \
        H_CHECK_CUSOLVER(cusolverDn##TypeChar##potrf(handle,                   \
                                                     uplo,                     \
                                                     n,                        \
                                                     A,                        \
                                                     lda,                      \
                                                     workspace,                \
                                                     workspace_size,           \
                                                     devinfo));                \
    }

syevjInfo_t syevj_params;

#define ADD_HEEV_IMPL(ScalarType, TypePrefix)                                  \
    cusolver_int GetHeevWorkspaceSize(cusolverDnHandle_t handle,               \
                                      cublasFillMode_t uplo,                   \
                                      cusolver_int n,                          \
                                      DeviceArray<ScalarType> A,               \
                                      cusolver_int lda,                        \
                                      DeviceArray<RealType<ScalarType>> W)     \
    {                                                                          \
        cusolver_int workspace_size;                                           \
        if (n < CUSOLVER_HEEV_JACOBI_THRESHOLD)                                \
        {                                                                      \
            H_CHECK_CUSOLVER(cusolverDnCreateSyevjInfo(&syevj_params));        \
            H_CHECK_CUSOLVER(cusolverDn##TypePrefix##evj_bufferSize(           \
                handle,                                                        \
                CUSOLVER_EIG_MODE_VECTOR,                                      \
                uplo,                                                          \
                n,                                                             \
                A,                                                             \
                lda,                                                           \
                W,                                                             \
                &workspace_size,                                               \
                syevj_params));                                                \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            H_CHECK_CUSOLVER(cusolverDn##TypePrefix##evd_bufferSize(           \
                handle,                                                        \
                CUSOLVER_EIG_MODE_VECTOR,                                      \
                uplo,                                                          \
                n,                                                             \
                A,                                                             \
                lda,                                                           \
                W,                                                             \
                &workspace_size));                                             \
        }                                                                      \
        return workspace_size;                                                 \
    }                                                                          \
    void Heev(cusolverDnHandle_t handle,                                       \
              cublasFillMode_t uplo,                                           \
              cusolver_int n,                                                  \
              DeviceArray<ScalarType> A,                                       \
              cusolver_int lda,                                                \
              DeviceArray<RealType<ScalarType>> W,                             \
              DeviceArray<ScalarType> workspace,                               \
              cusolver_int workspace_size,                                     \
              DevicePtr<cusolver_int> devinfo)                                 \
    {                                                                          \
        if (n < CUSOLVER_HEEV_JACOBI_THRESHOLD)                                \
        {                                                                      \
            H_CHECK_CUSOLVER(                                                  \
                cusolverDn##TypePrefix##evj(handle,                            \
                                            CUSOLVER_EIG_MODE_VECTOR,          \
                                            uplo,                              \
                                            n,                                 \
                                            A,                                 \
                                            lda,                               \
                                            W,                                 \
                                            workspace,                         \
                                            workspace_size,                    \
                                            devinfo,                           \
                                            syevj_params));                    \
                                                                               \
            H_CHECK_CUSOLVER(                                                  \
                cusolverDnDestroySyevjInfo(syevj_params));                     \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            H_CHECK_CUSOLVER(                                                  \
                cusolverDn##TypePrefix##evd(handle,                            \
                                            CUSOLVER_EIG_MODE_VECTOR,          \
                                            uplo,                              \
                                            n,                                 \
                                            A,                                 \
                                            lda,                               \
                                            W,                                 \
                                            workspace,                         \
                                            workspace_size,                    \
                                            devinfo));                         \
        }                                                                      \
    }

ADD_POTRF_IMPL(float, S)
ADD_POTRF_IMPL(double, D)
ADD_POTRF_IMPL(cuComplex, C)
ADD_POTRF_IMPL(cuDoubleComplex, Z)

ADD_HEEV_IMPL(float, Ssy)
ADD_HEEV_IMPL(double, Dsy)
ADD_HEEV_IMPL(cuComplex, Che)
ADD_HEEV_IMPL(cuDoubleComplex, Zhe)

} // namespace cusolver
} // namespace hydrogen
