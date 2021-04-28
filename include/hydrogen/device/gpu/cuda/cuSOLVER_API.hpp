#ifndef HYDROGEN_DEVICE_GPU_CUDA_CUSOLVER_API_HPP_
#define HYDROGEN_DEVICE_GPU_CUDA_CUSOLVER_API_HPP_

#include <El/hydrogen_config.h>

#include <cusolverDn.h>

namespace hydrogen
{
namespace cusolver
{

// These are just for my own readability.
template <typename T>
using DeviceArray = T*;

template <typename T>
using HostPtr = T*;

template <typename T>
using DevicePtr = T*;

using cusolver_int = int;

// Cholesky factorization
#define ADD_POTRF_DECL(ScalarType)                      \
    cusolver_int GetPotrfWorkspaceSize(                 \
        cusolverDnHandle_t handle,                      \
        cublasFillMode_t uplo,                          \
        cusolver_int n,                                 \
        DeviceArray<ScalarType> A, cusolver_int lda);   \
    void Potrf(                                         \
        cusolverDnHandle_t handle,                      \
        cublasFillMode_t uplo,                          \
        cusolver_int n,                                 \
        DeviceArray<ScalarType> A, cusolver_int lda,    \
        DeviceArray<ScalarType> workspace,              \
        cusolver_int workspace_size,                    \
        DevicePtr<cusolver_int> devinfo)

ADD_POTRF_DECL(float);
ADD_POTRF_DECL(double);
ADD_POTRF_DECL(cuComplex);
ADD_POTRF_DECL(cuDoubleComplex);

}// namespace cusolver
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_CUDA_CUSOLVER_API_HPP_
