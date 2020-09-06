#ifndef EL_CORE_DEVICE_HPP_
#define EL_CORE_DEVICE_HPP_

#include <El/hydrogen_config.h>
#include <hydrogen/utils/HalfPrecision.hpp>
#include <hydrogen/meta/MetaUtilities.hpp>

#include <complex>
#include <string>
#include <type_traits>

// Forward declaration
namespace El
{
template <typename T>
class Complex;
}

namespace hydrogen
{

/** @defgroup device_meta Metaprogramming for Multiple Devices
 *
 *  @brief Simple metafunctions to help with metaprogramming based on
 *  the device type.
 *
 *  Hydrogen makes strong use of knowledge about the memory residence
 *  of its memory -- CPU, GPU, other. The general API will usually
 *  have a few layers of potential indirection before this information
 *  is needed, but eventually, nearly every call will need to be
 *  statically dispatched to a device-specific function. These
 *  functions and metafunctions help ensure that this level of static
 *  dispatch is valid and correct.
 */

/** @brief An enumeration of every known device type. */
enum class Device : unsigned char
{
    CPU
#ifdef HYDROGEN_HAVE_GPU
    , GPU
#endif
};

/** @brief Get a string representation of the device type. */
template <Device D>
std::string DeviceName();

template <> inline std::string DeviceName<Device::CPU>()
{ return "CPU"; }

#ifdef HYDROGEN_HAVE_GPU
template <> inline std::string DeviceName<Device::GPU>()
{ return "GPU"; }
#endif

/** @class IsComputeType
 *  @brief Test if one can compute with this type on the device.
 *
 *  This is true for every type that supports "normal" computation on
 *  the given device. E.g., on CPU, "normal" probably means it supports
 *  `operator{=,+,-,*,/,<,>,==}`.
 *
 *  @warning This predicate evaluating to `true` does not imply that
 *      its use for computation is recommended (e.g., half_float::half
 *      on the CPU), merely that it is possible.
 *
 *  @tparam T The type
 *  @tparam D The device type
 *
 *  @ingroup device_meta
 */
template <typename T, Device D>
struct IsComputeType : std::false_type {};

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <typename T>
struct IsComputeType<T,Device::CPU> : std::true_type {};

#ifdef HYDROGEN_HAVE_GPU
#ifdef HYDROGEN_GPU_USE_FP16
template <>
struct IsComputeType<gpu_half_type, Device::GPU> : std::true_type {};
#endif
template <> struct IsComputeType<float, Device::GPU> : std::true_type {};
template <> struct IsComputeType<double, Device::GPU> : std::true_type {};
template <>
struct IsComputeType<std::complex<float>, Device::GPU> : std::true_type {};
template <>
struct IsComputeType<std::complex<double>, Device::GPU> : std::true_type {};
template <>
struct IsComputeType<El::Complex<float>, Device::GPU> : std::true_type {};
template <>
struct IsComputeType<El::Complex<double>, Device::GPU> : std::true_type {};
#endif

#endif // DOXYGEN_SHOULD_SKIP_THIS

/** @brief Convenience function wrapping IsComputeType.
 *  @ingroup device_meta
 */
template <typename T, Device D>
constexpr bool IsComputeType_v() { return IsComputeType<T,D>::value; }

/** @class IsStorageType
 *  @brief Test if one can store values of this type on the device.
 *
 *  This is true for every type since every type is just a bunch of
 *  bits until a member function is called.
 *
 *  @tparam T The type
 *  @tparam D The device type
 *
 *  @ingroup device_meta
 */
template <typename T, Device D>
struct IsStorageType : std::false_type {};

template <typename T>
struct IsStorageType<T,Device::CPU> : std::true_type {};

#ifdef HYDROGEN_HAVE_GPU
#ifdef HYDROGEN_GPU_USE_FP16
template <>
struct IsStorageType<gpu_half_type, Device::GPU> : std::true_type {};
#endif
template <> struct IsStorageType<float, Device::GPU> : std::true_type {};
template <> struct IsStorageType<double, Device::GPU> : std::true_type {};
template <>
struct IsStorageType<std::complex<float>, Device::GPU> : std::true_type {};
template <>
struct IsStorageType<std::complex<double>, Device::GPU> : std::true_type {};
template <>
struct IsStorageType<El::Complex<float>, Device::GPU> : std::true_type {};
template <>
struct IsStorageType<El::Complex<double>, Device::GPU> : std::true_type {};
#endif

/** @brief Convenience function wrapping IsStorageType.
 *  @ingroup device_meta
 */
template <typename T, Device D>
constexpr bool IsStorageType_v() { return IsStorageType<T,D>::value; }

/** @class IsDeviceValidType
 *  @brief Backwards compatibility while things are cleaned up.
 *  @deprecated This predicate is deprecated. Prefer IsComputetype and
 *      IsStoragetype.
 *  @ingroup device_meta
 */
template <typename T, Device D>
struct IsDeviceValidType : IsComputeType<T,D> {};

/** @brief Convenience predicate to test if two devices are the same
 *  @ingroup device_meta
 */
template <Device D1, Device D2>
using SameDevice = EnumSame<Device,D1,D2>;

/** @brief Support for basic inter-device memory operations.
 *  @todo Figure out where InterDeviceCopy should live.
 */
template <Device SrcD, Device DestD> struct InterDeviceCopy;

// These should replace the InterDeviceCopy struct.
#if 0
template <typename T, Device SrcD, Device TgtD>
void MemCopy1DAsync(
    T const* __restrict__ const src,
    T * __restrict__ const dest,
    size_t const size,
    SyncInfo<SrcD> const& srcSyncInfo,
    SyncInfo<TgtD> const& destSyncInfo);

template <typename T, Device SrcD, Device TgtD>
void MemCopy2DAsync(
    T const* __restrict__ const src, size_t const src_ldim,
    T * __restrict__ const dest, size_t const dest_ldim,
    size_t const height, size_t const width,
    SyncInfo<SrcD> const& srcSyncInfo,
    SyncInfo<TgtD> const& destSyncInfo);
#endif // 0
}// namespace hydrogen
#endif // EL_CORE_DEVICE_HPP_
