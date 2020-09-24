#ifndef HYDROGEN_DEVICE_GPU_ROCM_ROCBLASMETA_HPP_
#define HYDROGEN_DEVICE_GPU_ROCM_ROCBLASMETA_HPP_

#include <El/hydrogen_config.h>

#include <hydrogen/blas/BLAS_Common.hpp>
#include <hydrogen/meta/MetaUtilities.hpp>
#include <hydrogen/utils/HalfPrecision.hpp>

#include <rocblas.h>

namespace hydrogen
{
namespace rocblas
{

/** @class NativeTypeT
 *  @brief Metafunction mapping type names to HIP/rocBLAS equivalents.
 *
 *  The mapping should provide bitwise equivalence.
 *
 *  @note This belongs at this level because rocBLAS defines types (or
 *        names of types) that are local to the BLAS
 *        implementation. Additionally, it's feasible to conceive of
 *        custom types on the GPU that would, likewise, need to be
 *        mapped to the types that rocBLAS knows about.
 *
 *  @todo Add static assertions to ensure only valid types get mapped.
 */
template <typename T>
struct NativeTypeT
{
    using type = T;
};

// Complex and Double-Complex types require conversion
template <>
struct NativeTypeT<std::complex<float>> { using type = rocblas_float_complex; };
template <>
struct NativeTypeT<std::complex<double>> { using type = rocblas_double_complex; };

// Half precision requires conversion as well
#ifdef HYDROGEN_GPU_USE_FP16
template <> struct NativeTypeT<gpu_half_type> { using type = rocblas_half; };
#ifdef HYDROGEN_HAVE_HALF
template <> struct NativeTypeT<cpu_half_type> { using type = rocblas_half; };
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
 *  @brief Predicate indicating that a type is supported within rocBLAS
 *         for the given operation.
 *
 *  This is used to map internal rocBLAS types to the operations that
 *  are supported. For example, `float` is always supported but
 *  `rocblas_half` only has support in a few functions.
 */
template <typename T, BLAS_Op op>
struct IsSupportedType_Base : std::false_type {};

template <BLAS_Op op>
struct IsSupportedType_Base<float, op> : std::true_type {};
template <BLAS_Op op>
struct IsSupportedType_Base<double, op> : std::true_type {};

template <>
struct IsSupportedType_Base<float, BLAS_Op::HERK>
    : std::false_type {};
template <>
struct IsSupportedType_Base<double, BLAS_Op::HERK>
    : std::false_type {};
template <>
struct IsSupportedType_Base<rocblas_float_complex, BLAS_Op::HERK>
    : std::true_type {};
template <>
struct IsSupportedType_Base<rocblas_double_complex, BLAS_Op::HERK>
    : std::true_type {};
template <>
struct IsSupportedType_Base<rocblas_float_complex, BLAS_Op::SYRK>
    : std::true_type {};
template <>
struct IsSupportedType_Base<rocblas_double_complex, BLAS_Op::SYRK>
    : std::true_type {};
template <>
struct IsSupportedType_Base<rocblas_float_complex, BLAS_Op::TRSM>
    : std::true_type {};
template <>
struct IsSupportedType_Base<rocblas_double_complex, BLAS_Op::TRSM>
    : std::true_type {};

#ifdef HYDROGEN_GPU_USE_FP16
template <>
struct IsSupportedType_Base<rocblas_half, BLAS_Op::AXPY> : std::true_type {};
template <>
struct IsSupportedType_Base<rocblas_half, BLAS_Op::GEMM> : std::true_type {};
#endif // HYDROGEN_GPU_USE_FP16

/** @class IsSupportedType
 *  @brief Predicate indicating that the given type is compatible with
 *         rocBLAS.
 *
 *  This is true when either the type is a compatible rocBLAS type
 *  (e.g., float) or when it is binarily equivalent to one (e.g.,
 *  std::complex<float>)..
 */
template <typename T, BLAS_Op op, bool=HasNativeType<T>::value>
struct IsSupportedType
    : IsSupportedType_Base<NativeType<T>, op>
{};

template <typename T, BLAS_Op op>
struct IsSupportedType<T,op,false> : std::false_type {};

}// namespace rocblas
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_ROCM_ROCBLASMETA_HPP_
