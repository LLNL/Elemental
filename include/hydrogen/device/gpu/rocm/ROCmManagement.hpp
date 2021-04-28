#ifndef HYDROGEN_DEVICE_GPU_ROCMMANAGEMENT_HPP_
#define HYDROGEN_DEVICE_GPU_ROCMMANAGEMENT_HPP_

#include <hip/hip_runtime.h>

#include <hip/hip_complex.h>
#include <thrust/complex.h>

namespace hydrogen
{

using gpuEvent_t = hipEvent_t;
using gpuStream_t = hipStream_t;

template <typename T>
struct NativeGPUTypeT
{
    using type = T;
};

template <>
struct NativeGPUTypeT<El::Complex<float>>
{
    using type = thrust::complex<float>;
};

template <>
struct NativeGPUTypeT<El::Complex<double>>
{
    using type = thrust::complex<double>;
};

template <typename T>
using NativeGPUType = typename NativeGPUTypeT<T>::type;

template <typename T>
struct GPUStaticStorageTypeT
{
    using type = T;
};

template <>
struct GPUStaticStorageTypeT<thrust::complex<float>>
{
    using type = hipFloatComplex;
};

template <>
struct GPUStaticStorageTypeT<thrust::complex<double>>
{
    using type = hipDoubleComplex;
};

template <typename T>
struct GPUStaticStorageTypeT<El::Complex<T>>
    : GPUStaticStorageTypeT<NativeGPUType<T>>
{};

template <typename T>
using GPUStaticStorageType = typename GPUStaticStorageTypeT<T>::type;

template <typename T>
auto AsNativeGPUType(T* ptr) -> NativeGPUType<T>*
{
    return reinterpret_cast<NativeGPUType<T>*>(ptr);
}

template <typename T>
auto AsNativeGPUType(T const* ptr) -> NativeGPUType<T> const*
{
    return reinterpret_cast<NativeGPUType<T> const*>(ptr);
}

namespace rocm
{
hipEvent_t GetDefaultEvent() noexcept;
hipStream_t GetDefaultStream() noexcept;
hipEvent_t GetNewEvent();
hipStream_t GetNewStream();
void FreeEvent(hipEvent_t& event);
void FreeStream(hipStream_t& stream);
}// namespace rocm
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_ROCMMANAGEMENT_HPP_
