#ifndef HYDROGEN_DEVICE_GPU_CUDA_CUBLASMETA_HPP_
#define HYDROGEN_DEVICE_GPU_CUDA_CUBLASMETA_HPP_

#include <El/hydrogen_config.h>

#include <hydrogen/blas/BLAS_Common.hpp>
#include <hydrogen/meta/MetaUtilities.hpp>
#include <hydrogen/utils/HalfPrecision.hpp>

#include <cublas_v2.h>

namespace hydrogen
{
namespace cublas
{

/** @class NativeTypeT
 *  @brief Metafunction mapping type names to CUDA/cuBLAS equivalents.
 *
 *  The mapping should provide bitwise equivalence.
 *
 *  @note This belongs at this level because rocBLAS defines types (or
 *        names of types) that are local to the BLAS
 *        implementation. Additionally, it's feasible to conceive of
 *        custom types on the GPU that would, likewise, need to be
 *        mapped to the types that cuBLAS knows about.
 *
 *  @todo Add static assertions to ensure only valid types get mapped.
 */
template <typename T>
struct NativeTypeT;

// Built-in types are their own native types
template <> struct NativeTypeT<float> { using type = float; };
template <> struct NativeTypeT<double> { using type = double; };
template <>
struct NativeTypeT<cuComplex> { using type = cuComplex; };
template <>
struct NativeTypeT<cuDoubleComplex> { using type = cuDoubleComplex; };

// Complex and Double-Complex types require conversion
template <>
struct NativeTypeT<std::complex<float>> { using type = cuComplex; };
template <>
struct NativeTypeT<std::complex<double>> { using type = cuDoubleComplex; };

// Half precision requires conversion as well
#ifdef HYDROGEN_GPU_USE_FP16
template <> struct NativeTypeT<__half> { using type = __half; };
#ifdef HYDROGEN_HAVE_HALF
template <> struct NativeTypeT<cpu_half_type> { using type = __half; };
#endif // HYDROGEN_HAVE_HALF
#endif // HYDROGEN_GPU_USE_FP16

/** @brief Convenience wrapper for NativeTypeT */
template <typename T>
using NativeType = typename NativeTypeT<T>::type;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace meta_details
{
template <typename T>
auto Try_HasNativeType(int) -> SubstitutionSuccess<NativeType<T>>;
template <typename T>
auto Try_HasNativeType(...) -> std::false_type;
}// namespace meta_details
#endif // DOXYGEN_SHOULD_SKIP_THIS

/** @struct HasNativeType
 *  @brief Predicate that determines if a type is mappable to a
 *         library-native type.
 */
template <typename T>
struct HasNativeType : decltype(meta_details::Try_HasNativeType<T>(0)) {};

/** @class IsSupportedType_Base
 *  @brief Predicate indicating that a type is supported within cuBLAS
 *         for the given operation.
 *
 *  This is used to map internal cuBLAS types to the operations that
 *  are supported. For example, `float` is always supported but
 *  `__half` only has support in a few functions.
 */
template <typename T, BLAS_Op op>
struct IsSupportedType_Base : std::false_type {};

template <BLAS_Op op>
struct IsSupportedType_Base<float, op> : std::true_type {};
template <BLAS_Op op>
struct IsSupportedType_Base<double, op> : std::true_type {};
template <BLAS_Op op>
struct IsSupportedType_Base<cuComplex, op> : std::true_type {};
template <BLAS_Op op>
struct IsSupportedType_Base<cuDoubleComplex, op> : std::true_type {};

// No need to further test CUDA because this file isn't included if
// either we don't have GPUs at all or we don't have CUDA support.
#ifdef HYDROGEN_GPU_USE_FP16
template <>
struct IsSupportedType_Base<__half, BLAS_Op::AXPY> : std::true_type {};
template <>
struct IsSupportedType_Base<__half, BLAS_Op::GEMM> : std::true_type {};
template <>
struct IsSupportedType_Base<__half, BLAS_Op::SCAL> : std::true_type {};
#endif // HYDROGEN_GPU_USE_FP16

/** @class IsSupportedType
 *  @brief Predicate indicating that the given type is compatible with
 *         cuBLAS.
 *
 *  This is true when either the type is a compatible cuBLAS type
 *  (e.g., float) or when it is binarily equivalent to one (e.g.,
 *  std::complex<float>)..
 */
template <typename T, BLAS_Op op, bool=HasNativeType<T>::value>
struct IsSupportedType
    : IsSupportedType_Base<NativeType<T>, op>
{};

template <typename T, BLAS_Op op>
struct IsSupportedType<T,op,false> : std::false_type {};

}// namespace cublas
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_CUDA_CUBLASMETA_HPP_
