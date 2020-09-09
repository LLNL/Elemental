#ifndef HYDROGEN_DEVICE_GPU_ROCM_ROCSOLVER_API_HPP_
#define HYDROGEN_DEVICE_GPU_ROCM_ROCSOLVER_API_HPP_

#include <El/hydrogen_config.h>

#include <rocsolver.h>

namespace hydrogen
{
namespace rocsolver
{

// These are just for my own readability.
template <typename T>
using DeviceArray = T*;

template <typename T>
using HostPtr = T*;

template <typename T>
using DevicePtr = T*;

using rocsolver_int = rocblas_int;

// Cholesky factorization
//
// Due to shortcomings in the rocSOLVER API, the workspace size will
// always return zero, and the workspace pointer is unused.
#define ADD_POTRF_DECL(ScalarType)                      \
    rocblas_int GetPotrfWorkspaceSize(                  \
        rocblas_handle handle,                          \
        rocblas_fill uplo,                              \
        rocblas_int n,                                  \
        DeviceArray<ScalarType> A, rocblas_int lda);    \
    void Potrf(                                         \
        rocblas_handle handle,                          \
        rocblas_fill uplo,                              \
        rocblas_int n,                                  \
        DeviceArray<ScalarType> A, rocblas_int lda,     \
        DeviceArray<ScalarType> workspace,              \
        rocblas_int workspace_size,                     \
        DevicePtr<rocblas_int> devinfo)

ADD_POTRF_DECL(float);
ADD_POTRF_DECL(double);
ADD_POTRF_DECL(rocblas_float_complex);
ADD_POTRF_DECL(rocblas_double_complex);

}// namespace rocsolver
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_ROCM_ROCSOLVER_API_HPP_
