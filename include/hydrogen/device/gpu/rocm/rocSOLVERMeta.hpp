#ifndef HYDROGEN_DEVICE_GPU_ROCM_ROCSOLVERMETA_HPP_
#define HYDROGEN_DEVICE_GPU_ROCM_ROCSOLVERMETA_HPP_

#include <El/hydrogen_config.h>

#include <hydrogen/blas/BLAS_Common.hpp>
#include <hydrogen/meta/MetaUtilities.hpp>
#include <hydrogen/utils/HalfPrecision.hpp>

#include <rocsolver/rocsolver.h>

#include "rocBLASMeta.hpp"

namespace hydrogen
{
namespace rocsolver
{
using rocblas::NativeType;
using rocblas::HasNativeType;

/** @class IsSupportedType_Base
 *  @brief Predicate indicating that a type is supported within
 *         rocSOLVER for the given operation.
 *
 *  This is used to map internal rocSOLVER types to the operations that
 *  are supported.
 */
template <typename T, LAPACK_Op op>
struct IsSupportedType_Base : std::false_type {};

template <LAPACK_Op op>
struct IsSupportedType_Base<float, op> : std::true_type {};
template <LAPACK_Op op>
struct IsSupportedType_Base<double, op> : std::true_type {};
template <LAPACK_Op op>
struct IsSupportedType_Base<rocblas_float_complex, op> : std::true_type {};
template <LAPACK_Op op>
struct IsSupportedType_Base<rocblas_double_complex, op> : std::true_type {};

/** @class IsSupportedType
 *  @brief Predicate indicating that the given type is compatible with
 *         rocSOLVER.
 *
 *  This is true when either the type is a compatible rocSOLVER type
 *  (e.g., float) or when it is binarily equivalent to one (e.g.,
 *  std::complex<float>)..
 */
template <typename T, LAPACK_Op op, bool=HasNativeType<T>::value>
struct IsSupportedType
    : IsSupportedType_Base<NativeType<T>, op>
{};

template <typename T, LAPACK_Op op>
struct IsSupportedType<T, op, false> : std::false_type {};

}// namespace rocsolver
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_ROCM_ROCSOLVERMETA_HPP_
