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

/** @brief Metafunction to get a type suitable to device computation.
 *
 *  This type will support the "usual" arithmetic operators in device
 *  code. It should be bitwise-equivalent to the input type so that
 *  `reinterpret_cast` works sensibly.
 */
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

/** @brief Metafunction to get a type suitable to static device-side
 *         allocation.
 *
 *  The motivation for this type is complex numbers in CUDA kernel
 *  templates. One cannot create statically-allocated `__shared__`
 *  memory blocks with `thrust::complex`, but one cannot perform basic
 *  arithmetic with `cuComplex`. This solves the former problem, and
 *  NativeGPUTypeT solves the latter.
 *
 *  The returned type may or may not be the same as the corresponding
 *  NativeGPUTypeT. It should be bitwise-equivalent to the input type
 *  so that `reinterpret_cast` works sensibly.
 */
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
