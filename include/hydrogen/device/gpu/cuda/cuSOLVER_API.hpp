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

template <typename T>
struct RealTypeT
{
    using type = T;
};

template <>
struct RealTypeT<cuComplex>
    : RealTypeT<float>
{
};

template <>
struct RealTypeT<cuDoubleComplex>
    : RealTypeT<double>
{
};

template <typename T> using RealType = typename RealTypeT<T>::type;

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

#define ADD_HEEV_DECL(ScalarType)                       \
    cusolver_int GetHeevWorkspaceSize(                  \
        cusolverDnHandle_t handle,                      \
        cublasFillMode_t uplo,                          \
        cusolver_int n,                                 \
        DeviceArray<ScalarType> A,                      \
        cusolver_int lda,                               \
        DeviceArray<RealType<ScalarType>> W);           \
    void Heev(                                          \
        cusolverDnHandle_t handle,                      \
        cublasFillMode_t uplo,                          \
        cusolver_int n,                                 \
        DeviceArray<ScalarType> A,                      \
        cusolver_int lda,                               \
        DeviceArray<RealType<ScalarType>> W,            \
        DeviceArray<ScalarType> workspace,              \
        cusolver_int workspace_size,                    \
        DevicePtr<cusolver_int> devinfo)

#define ADD_CUSOLVER_DECLS(ScalarType)          \
    ADD_POTRF_DECL(ScalarType);                 \
    ADD_HEEV_DECL(ScalarType)

ADD_CUSOLVER_DECLS(float);
ADD_CUSOLVER_DECLS(double);
ADD_CUSOLVER_DECLS(cuComplex);
ADD_CUSOLVER_DECLS(cuDoubleComplex);

}// namespace cusolver
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_CUDA_CUSOLVER_API_HPP_
