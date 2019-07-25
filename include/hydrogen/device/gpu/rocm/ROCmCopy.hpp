#ifndef HYDROGEN_DEVICE_GPU_ROCM_ROCMCOPY_HPP_
#define HYDROGEN_DEVICE_GPU_ROCM_ROCMCOPY_HPP_

#include "ROCmError.hpp"

#include <hydrogen/Device.hpp>
#include <hydrogen/SyncInfo.hpp>

#include <hydrogen/device/GPU.hpp>
#include <hydrogen/blas/gpu/Fill.hpp>

#include <hip/hip_runtime.h>

namespace hydrogen
{
namespace gpu
{

/** @todo Flesh out documentation
 *  @todo these are actually only valid for "packed" types
 */

// These functions are synchronous with respect to their SyncInfo
// objects (that is, they require explicit synchronization to the
// host).

template <typename T>
void Fill1DBuffer(T* buffer, size_t num_elements, T value,
                  SyncInfo<Device::GPU> const& si)
{
    Fill_GPU_1D_impl(buffer, num_elements, value, si.Stream());
}

template <typename T>
void Copy1DIntraDevice(T const* H_RESTRICT src, T* H_RESTRICT dest,
                       size_t num_elements,
                       SyncInfo<Device::GPU> const& si)
{
    H_CHECK_HIP(
        hipMemcpyAsync(
            dest, src, num_elements*sizeof(T),
            hipMemcpyDeviceToDevice, si.Stream()));
}

template <typename T>
void Copy1DToHost(T const* H_RESTRICT src, T* H_RESTRICT dest,
                  size_t num_elements,
                  SyncInfo<Device::GPU> const& src_si)
{
    H_CHECK_HIP(
        hipMemcpyAsync(
            dest, src, num_elements*sizeof(T),
            hipMemcpyDeviceToHost, src_si.Stream()));
}

template <typename T>
void Copy1DToDevice(T const* H_RESTRICT src, T* H_RESTRICT dest,
                    size_t num_elements,
                    SyncInfo<Device::GPU> const& dest_si)
{
    H_CHECK_HIP(
        hipMemcpyAsync(
            dest, src, num_elements*sizeof(T),
            hipMemcpyHostToDevice, dest_si.Stream()));
}


template <typename T>
void Copy2DIntraDevice(T const* src, size_t src_ldim,
                       T* dest, size_t dest_ldim,
                       size_t height, size_t width,
                       SyncInfo<Device::GPU> const& si)
{
    H_CHECK_HIP(
        hipMemcpy2DAsync(
            dest, dest_ldim*sizeof(T),
            src, src_ldim*sizeof(T),
            height*sizeof(T), width,
            hipMemcpyDeviceToDevice, si.Stream()));
}

template <typename T>
void Copy2DToHost(T const* src, size_t src_ldim,
                  T* dest, size_t dest_ldim,
                  size_t height, size_t width,
                  SyncInfo<Device::GPU> const& src_si)
{
    H_CHECK_HIP(
        hipMemcpy2DAsync(
            dest, dest_ldim*sizeof(T),
            src, src_ldim*sizeof(T),
            height*sizeof(T), width,
            hipMemcpyDeviceToHost, src_si.Stream()));
}

template <typename T>
void Copy2DToDevice(T const* src, size_t src_ldim,
                    T* dest, size_t dest_ldim,
                    size_t height, size_t width,
                    SyncInfo<Device::GPU> const& dest_si)
{
    H_CHECK_HIP(
        hipMemcpy2DAsync(
            dest, dest_ldim*sizeof(T),
            src, src_ldim*sizeof(T),
            height*sizeof(T), width,
            hipMemcpyHostToDevice, dest_si.Stream()));
}

}// namespace gpu
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_ROCM_ROCMCOPY_HPP_
