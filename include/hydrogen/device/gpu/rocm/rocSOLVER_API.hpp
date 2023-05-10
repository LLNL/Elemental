#ifndef HYDROGEN_DEVICE_GPU_ROCM_ROCSOLVER_API_HPP_
#define HYDROGEN_DEVICE_GPU_ROCM_ROCSOLVER_API_HPP_

#include <El/hydrogen_config.h>

#include <rocsolver/rocsolver.h>

namespace hydrogen {
namespace rocsolver {

// These are just for my own readability.
template <typename T> using DeviceArray = T*;

template <typename T> using HostPtr = T*;

template <typename T> using DevicePtr = T*;

using rocsolver_int = rocblas_int;

template <typename T>
struct RealTypeT
{
    using type = T;
};

template <>
struct RealTypeT<rocblas_float_complex>
    : RealTypeT<float>
{
};

template <>
struct RealTypeT<rocblas_double_complex>
    : RealTypeT<double>
{
};

template <typename T> using RealType = typename RealTypeT<T>::type;

// Cholesky factorization
//
// Due to shortcomings in the rocSOLVER API, the workspace size will
// always return zero, and the workspace pointer is unused.
#define ADD_POTRF_DECL(ScalarType)                                             \
    rocblas_int GetPotrfWorkspaceSize(rocblas_handle handle,                   \
                                      rocblas_fill uplo,                       \
                                      rocblas_int n,                           \
                                      DeviceArray<ScalarType> A,               \
                                      rocblas_int lda);                        \
    void Potrf(rocblas_handle handle,                                          \
               rocblas_fill uplo,                                              \
               rocblas_int n,                                                  \
               DeviceArray<ScalarType> A,                                      \
               rocblas_int lda,                                                \
               DeviceArray<ScalarType> workspace,                              \
               rocblas_int workspace_size,                                     \
               DevicePtr<rocblas_int> devinfo)

#define ADD_HEEV_DECL(ScalarType)                                              \
    rocblas_int GetHeevWorkspaceSize(rocblas_handle handle,                    \
                                     rocblas_fill uplo,                        \
                                     rocblas_int n,                            \
                                     DeviceArray<ScalarType> A,                \
                                     rocblas_int lda,                          \
                                     DeviceArray<RealType<ScalarType>> W);     \
    void Heev(rocblas_handle handle,                                           \
              rocblas_fill uplo,                                               \
              rocblas_int n,                                                   \
              DeviceArray<ScalarType> A,                                       \
              rocblas_int lda,                                                 \
              DeviceArray<RealType<ScalarType>> W,                             \
              DeviceArray<ScalarType> workspace,                               \
              rocblas_int workspace_size,                                      \
              DevicePtr<rocblas_int> devinfo)

#define ADD_ROCSOLVER_DECLS(ScalarType)                                        \
    ADD_POTRF_DECL(ScalarType);                                                \
    ADD_HEEV_DECL(ScalarType)

ADD_ROCSOLVER_DECLS(float);
ADD_ROCSOLVER_DECLS(double);
ADD_ROCSOLVER_DECLS(rocblas_float_complex);
ADD_ROCSOLVER_DECLS(rocblas_double_complex);

} // namespace rocsolver
} // namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_ROCM_ROCSOLVER_API_HPP_
