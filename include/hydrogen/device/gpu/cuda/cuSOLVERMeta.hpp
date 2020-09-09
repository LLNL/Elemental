#ifndef HYDROGEN_DEVICE_GPU_CUDA_CUSOLVERMETA_HPP_
#define HYDROGEN_DEVICE_GPU_CUDA_CUSOLVERMETA_HPP_

#include <El/hydrogen_config.h>

#include <hydrogen/blas/BLAS_Common.hpp>
#include <hydrogen/meta/MetaUtilities.hpp>
#include <hydrogen/utils/HalfPrecision.hpp>

#include <cusolverDn.h>

#include "cuBLASMeta.hpp"

namespace hydrogen
{
namespace cusolver
{
using cublas::NativeType;
using cublas::HasNativeType;

/** @class IsSupportedType_Base
 *  @brief Predicate indicating that a type is supported within cuBLAS
 *         for the given operation.
 *
 *  This is used to map internal cuSOLVER types to the operations that
 *  are supported.
 */
template <typename T, LAPACK_Op op>
struct IsSupportedType_Base : std::false_type {};

template <LAPACK_Op op>
struct IsSupportedType_Base<float, op> : std::true_type {};
template <LAPACK_Op op>
struct IsSupportedType_Base<double, op> : std::true_type {};
template <LAPACK_Op op>
struct IsSupportedType_Base<cuComplex, op> : std::true_type {};
template <LAPACK_Op op>
struct IsSupportedType_Base<cuDoubleComplex, op> : std::true_type {};

/** @class IsSupportedType
 *  @brief Predicate indicating that the given type is compatible with
 *         cuSOLVER.
 *
 *  This is true when either the type is a compatible cuSOLVER type
 *  (e.g., float) or when it is binarily equivalent to one (e.g.,
 *  std::complex<float>)..
 */
template <typename T, LAPACK_Op op, bool=HasNativeType<T>::value>
struct IsSupportedType
    : IsSupportedType_Base<NativeType<T>, op>
{};

template <typename T, LAPACK_Op op>
struct IsSupportedType<T,op,false> : std::false_type {};

}// namespace cusolver
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_CUDA_CUSOLVERMETA_HPP_
