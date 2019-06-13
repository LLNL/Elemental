#ifndef HYDROGEN_UTILS_SIMPLEBUFFER_HPP_
#define HYDROGEN_UTILS_SIMPLEBUFFER_HPP_

#include <El/hydrogen_config.h>

#include <hydrogen/Device.hpp>
#include <hydrogen/SyncInfo.hpp>
#ifdef HYDROGEN_HAVE_GPU
#include <hydrogen/device/GPU.hpp>
#include <hydrogen/device/gpu/BasicCopy.hpp>
#endif // HYDROGEN_HAVE_GPU

#include <El/core/Memory/decl.hpp>

#include <algorithm>
#include <vector>

namespace hydrogen
{

// A simple data management class for temporary contiguous memory blocks
template <typename T, Device D>
class simple_buffer
{
public:
    simple_buffer() = default;

    // Construct uninitialized memory of a given size
    explicit simple_buffer(size_t size,
                           SyncInfo<D> const& = SyncInfo<D>{},
                           unsigned int mode = El::DefaultMemoryMode<D>());

    // Construct and initialize memory of a given size
    explicit simple_buffer(size_t size, T const& value,
                           SyncInfo<D> const& = SyncInfo<D>{},
                           unsigned int mode = El::DefaultMemoryMode<D>());
    // Enable moves
    simple_buffer(simple_buffer<T,D>&&) = default;

    // Disable copy
    simple_buffer(simple_buffer<T,D> const&) = delete;

    // Allow lazy allocation (use with default ctor)
    void allocate(size_t size);

    // Return the current memory size
    size_t size() const noexcept;

    // Buffer access
    T* data() noexcept;
    T const* data() const noexcept;

private:
    El::Memory<T,D> mem_;
    T* data_ = nullptr;
    size_t size_ = 0;
}; // class simple_buffer


namespace details
{

template <typename T>
void setBufferToValue(T* buffer, size_t size, T const& value,
                      SyncInfo<Device::CPU> const& = SyncInfo<Device::CPU>{})
{
    std::fill_n(buffer, size, value);
}

#ifdef HYDROGEN_HAVE_GPU
template <typename T>
void setBufferToValue(T* buffer, size_t size, T const& value,
                      SyncInfo<Device::GPU> const& syncInfo)
{
    gpu::Fill1DBuffer(buffer, size, value, syncInfo);
    AddSynchronizationPoint(syncInfo);
}
#endif // HYDROGEN_HAVE_CUDA
}// namespace details


template <typename T, Device D>
simple_buffer<T,D>::simple_buffer(
    size_t size, SyncInfo<D> const& syncInfo, unsigned int mode)
    : mem_{size, mode, syncInfo},
      data_{mem_.Buffer()},
      size_{mem_.Size()}
{}

template <typename T, Device D>
simple_buffer<T,D>::simple_buffer(
    size_t size, T const& value, SyncInfo<D> const& syncInfo, unsigned mode)
    : simple_buffer{size, syncInfo, mode}
{
    details::setBufferToValue(this->data(), size, value, syncInfo);
}

template <typename T, Device D>
void simple_buffer<T,D>::allocate(size_t size)
{
    data_ = mem_.Require(size);
    size_ = size;
}

template <typename T, Device D>
size_t simple_buffer<T,D>::size() const noexcept
{
    return size_;
}

template <typename T, Device D>
T* simple_buffer<T,D>::data() noexcept
{
    return data_;
}

template <typename T, Device D>
T const* simple_buffer<T,D>::data() const noexcept
{
    return data_;
}
}// namespace hydrogen
#endif // HYDROGEN_UTILS_SIMPLEBUFFER_HPP_
