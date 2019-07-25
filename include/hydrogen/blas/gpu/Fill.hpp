#ifndef HYDROGEN_BLAS_GPU_FILL_HPP_
#define HYDROGEN_BLAS_GPU_FILL_HPP_

/** @file
 *  @todo Write documentation!
 */

#include <hydrogen/Device.hpp>
#include <hydrogen/meta/MetaUtilities.hpp>

#ifdef HYDROGEN_HAVE_CUDA
#include <hydrogen/device/gpu/CUDA.hpp>
#elif defined(HYDROGEN_HAVE_ROCM)
#include <hydrogen/device/gpu/ROCm.hpp>
#endif

#include <stdexcept>

namespace hydrogen
{

template <typename T, typename=EnableWhen<IsStorageType<T,Device::GPU>>>
void Fill_GPU_impl(size_t height, size_t width,
                   T const& alpha, T* buffer, size_t ldim,
                   gpuStream_t stream);

template <typename T,
          typename=EnableUnless<IsDeviceValidType<T,Device::GPU>>,
          typename=void>
void Fill_GPU_impl(size_t const&, size_t const&,
                   T const&, T* const&, size_t const&,
                   gpuStream_t const&)
{
    throw std::logic_error("Fill: Type not valid on GPU.");
}

template <typename T, typename=EnableWhen<IsStorageType<T,Device::GPU>>>
void Fill_GPU_1D_impl(T* buffer, size_t const& size,
                      T const& alpha,
                      gpuStream_t stream)
{
    Fill_GPU_impl(size, 1, alpha, buffer, size, stream);
}

}// namespace hydrogen
#endif // HYDROGEN_BLAS_GPU_FILL_HPP_
