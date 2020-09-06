#ifndef HYDROGEN_DEVICE_GPU_CUDAMANAGEMENT_HPP_
#define HYDROGEN_DEVICE_GPU_CUDAMANAGEMENT_HPP_

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <thrust/complex.h>

namespace El
{
template <typename T>
class Complex;
}// namespace El

namespace hydrogen
{

using gpuEvent_t = cudaEvent_t;
using gpuStream_t = cudaStream_t;

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
    using type = cuComplex;
};

template <>
struct GPUStaticStorageTypeT<thrust::complex<double>>
{
    using type = cuDoubleComplex;
};

template <typename T>
struct GPUStaticStorageTypeT<El::Complex<T>>
    : GPUStaticStorageTypeT<NativeGPUType<T>>
{};

template <typename T>
using GPUStaticStorageType = typename GPUStaticStorageTypeT<T>::type;

template <typename T>
auto AsNativeGPUType(T* ptr)
{
    return reinterpret_cast<NativeGPUType<T>*>(ptr);
}

template <typename T>
auto AsNativeGPUType(T const* ptr)
{
    return reinterpret_cast<NativeGPUType<T> const*>(ptr);
}

namespace cuda
{
cudaEvent_t GetDefaultEvent() noexcept;
cudaStream_t GetDefaultStream() noexcept;
cudaEvent_t GetNewEvent();
cudaStream_t GetNewStream();
void FreeEvent(cudaEvent_t& event);
void FreeStream(cudaStream_t& stream);
}// namespace cuda
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_CUDAMANAGEMENT_HPP_
