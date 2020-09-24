#ifndef HYDROGEN_UTILS_HALFPRECISION_HPP_
#define HYDROGEN_UTILS_HALFPRECISION_HPP_

/** @file
 *
 *  Here we handle a few high-level metaprogramming tasks for 16-bit
 *  floating point support. This currently just includes IEEE-754
 *  support but will likely soon expand to include "bfloat16" as well.
 *
 *  When compiling using v2.1.0 of the Half library on OSX (10.14 with
 *  Apple LLVM version 10.0.1 (clang-1001.0.46.4)), there were many
 *  interesting issues that showed up. Hence the strange layout of
 *  this file.
 */

#include <El/hydrogen_config.h>
#include <hydrogen/meta/TypeTraits.hpp>

#include <iostream>

#ifdef HYDROGEN_HAVE_HALF

// Forward-declare things so I can start specializing templates.
namespace half_float
{
class half;
}// namespace half_float

// Specialize some STL-y things
#include <type_traits>
namespace std
{
template <>
struct is_floating_point<half_float::half> : true_type {};

template <>
struct is_integral<half_float::half> : false_type {};

template <>
struct is_arithmetic<half_float::half> : true_type {};
}// namespace std

// Fix an issue inside the Half library
#ifndef HALF_ENABLE_F16C_INTRINSICS
#define HALF_ENABLE_F16C_INTRINSICS __F16C__
#endif

// Now include the actual Half library.
#include <half.hpp>

// Declare the hydrogen typedef
namespace hydrogen
{
using namespace half_float::literal;

using cpu_half_type = half_float::half;

template <>
struct TypeTraits<cpu_half_type>
{
    static cpu_half_type One() noexcept { return 1._h; }
    static cpu_half_type Zero() noexcept { return 0._h; }
    static std::string Name() { return typeid(cpu_half_type).name(); }
};// struct TypeTraits<cpu_half_type>

}// namespace hydrogen

/** @name Bitwise operators, for MPI reduction. */
///@{
inline hydrogen::cpu_half_type operator~(hydrogen::cpu_half_type const&)
{
    throw std::logic_error(
        "Bitwise operations not supported for floating-point types.");
}
inline hydrogen::cpu_half_type operator|(hydrogen::cpu_half_type const&,
                                         hydrogen::cpu_half_type const&)
{
    throw std::logic_error(
        "Bitwise operations not supported for floating-point types.");
}
inline hydrogen::cpu_half_type operator&(hydrogen::cpu_half_type const&,
                                         hydrogen::cpu_half_type const&)
{
    throw std::logic_error(
        "Bitwise operations not supported for floating-point types.");
}
inline hydrogen::cpu_half_type operator^(hydrogen::cpu_half_type const&,
                                         hydrogen::cpu_half_type const&)
{
    throw std::logic_error(
        "Bitwise operations not supported for floating-point types.");
}
///@}
#endif // HYDROGEN_HAVE_HALF

// Finally, do the GPU stuff
#if defined HYDROGEN_HAVE_GPU && defined HYDROGEN_GPU_USE_FP16

// Grab the right header
#if defined(HYDROGEN_HAVE_CUDA)
#include <cuda_fp16.h>
#elif defined(HYDROGEN_HAVE_ROCM)
#include <hip/hip_fp16.h>
#endif // HYDROGEN_HAVE_CUDA

namespace hydrogen
{

#if defined(__HIP_NO_HALF_CONVERSIONS__)
static_assert(false, "__HIP_NO_HALF_CONVERSIONS__ is defined.");
#endif

/** @brief Unified name for the FP16 type on GPU */
using gpu_half_type = __half;

template <>
struct TypeTraits<gpu_half_type>
{
    static gpu_half_type One() noexcept { return 1.f; }
    static gpu_half_type Zero() noexcept { return 0.f; }
    static std::string Name() { return typeid(gpu_half_type).name(); }
};// struct TypeTraits<gpu_half_type>

}// namespace hydrogen

#if !(defined(__CUDACC__) || defined(__HIPCC__))

/** @brief Enable "update" functionality for __half. */
template <typename T>
hydrogen::gpu_half_type& operator+=(
    hydrogen::gpu_half_type& val, T const& rhs)
{
    val = float(val) + float(rhs);
    return val;
}

/** @brief Enable subtract-equal functionality for __half. */
template <typename T>
hydrogen::gpu_half_type& operator-=(
    hydrogen::gpu_half_type& val, T const& rhs)
{
    val = float(val) - float(rhs);
    return val;
}

/** @brief Enable "scale" functionality for __half. */
template <typename T>
hydrogen::gpu_half_type& operator*=(
    hydrogen::gpu_half_type& val, T const& rhs)
{
    val = float(val) * float(rhs);
    return val;
}

/** @brief Enable divide-equal functionality for __half. */
template <typename T>
hydrogen::gpu_half_type& operator/=(
    hydrogen::gpu_half_type& val, T const& rhs)
{
    val = float(val) / float(rhs);
    return val;
}

/** @brief Enable add functionality for __half. */
inline hydrogen::gpu_half_type operator+(
    hydrogen::gpu_half_type const& val, hydrogen::gpu_half_type const& rhs)
{
    return float(val) + float(rhs);
}

/** @brief Enable subtract functionality for __half. */
inline hydrogen::gpu_half_type operator-(
    hydrogen::gpu_half_type const& val, hydrogen::gpu_half_type const& rhs)
{
    return float(val) - float(rhs);
}

/** @brief Enable multiply functionality for __half. */
inline hydrogen::gpu_half_type operator*(
    hydrogen::gpu_half_type const& val, hydrogen::gpu_half_type const& rhs)
{
    return float(val) * float(rhs);
}

/** @brief Enable divide functionality for __half. */
inline hydrogen::gpu_half_type operator/(
    hydrogen::gpu_half_type const& val, hydrogen::gpu_half_type const& rhs)
{
    return float(val) / float(rhs);
}

/** @brief Enable unary minus functionality for __half. */
inline hydrogen::gpu_half_type operator-(
    hydrogen::gpu_half_type const& val)
{
    return -float(val);
}

#endif // !(defined(__CUDACC__) || defined(__HIPCC__))

inline std::ostream& operator<<(std::ostream& os, hydrogen::gpu_half_type const& x)
{
    return os << float(x) << "_h";
}

#endif // defined HYDROGEN_HAVE_GPU && defined HYDROGEN_GPU_USE_FP16
#endif // HYDROGEN_UTILS_HALFPRECISION_HPP_
