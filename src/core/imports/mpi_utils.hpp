/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   Copyright (c) 2013, Jeff Hammond
   All rights reserved.

   Copyright (c) 2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/


#ifndef EL_IMPORTS_MPIUTILS_HPP
#define EL_IMPORTS_MPIUTILS_HPP

#include <El/hydrogen_config.h>

#ifdef HYDROGEN_HAVE_GPU
#include <hydrogen/device/gpu/BasicCopy.hpp>
#endif

namespace
{

template<typename T>
MPI_Op NativeOp( const El::mpi::Op& op )
{
    MPI_Op opC;
    if( op == El::mpi::SUM )
        opC = El::mpi::SumOp<T>().op;
    else if( op == El::mpi::PROD )
        opC = El::mpi::ProdOp<T>().op;
    else if( op == El::mpi::MAX )
        opC = El::mpi::MaxOp<T>().op;
    else if( op == El::mpi::MIN )
        opC = El::mpi::MinOp<T>().op;
    else
        opC = op.op;
    return opC;
}

template<typename T>
MPI_Op AlNativeOp( const El::mpi::Op& op )
{
    MPI_Op opC;
    if( op == El::mpi::SUM )
        opC = MPI_SUM;
    else if( op == El::mpi::PROD )
        opC = MPI_PROD;
    else if( op == El::mpi::MAX )
        opC = MPI_MAX;
    else if( op == El::mpi::MIN )
        opC = MPI_MIN;
    else
        throw std::logic_error("IDK what op is!");

    return opC;
}

}// namespace <anon>

// This is for handling the host-blocking host-transfer stuff
#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
namespace El
{
namespace mpi
{
namespace internal
{

template <typename T>
class PassthroughMemoryWrapper
{
public:
    PassthroughMemoryWrapper(T* buf) noexcept : data_{buf} {}
    T* data() const noexcept { return data_; }
private:
    T* data_;
};// struct PassThroughMemoryWrapper

/**
 * \class ManagedHostMemoryWrapper
 * \brief Transfer memory to the host on construction, to the device
 *     on destruction.
 */
template <typename T, Device D>
class ManagedHostMemoryWrapper;

#ifdef HYDROGEN_HAVE_GPU
template <typename T>
class ManagedHostMemoryWrapper<T,Device::GPU>
{
public:

    ManagedHostMemoryWrapper(
        T* buf, size_t totalsize,
        size_t initial_xfer_offset, size_t initial_xfer_size,
        size_t final_xfer_offset, size_t final_xfer_size,
        SyncInfo<Device::GPU> const& syncInfo)
        : host_data_{totalsize, SyncInfo<Device::CPU>{}, /*mode=*/1},
          syncInfo_{syncInfo},
          device_data_{buf},
          final_xfer_offset_{final_xfer_offset},
          final_xfer_size_{final_xfer_size}
    {
        if ((host_data_.size() > 0) && (initial_xfer_size > 0))
        {
            gpu::Copy1DToHost(
                device_data_+initial_xfer_offset,
                host_data_.data()+initial_xfer_offset,
                initial_xfer_size, syncInfo_);
        }
    }

    ~ManagedHostMemoryWrapper()
    {
        // Transfer stuff back to device
        try
        {
            if ((host_data_.size() > 0) && (final_xfer_size_ > 0))
            {
                gpu::Copy1DToDevice(
                    host_data_.data()+final_xfer_offset_,
                    device_data_+final_xfer_offset_,
                    final_xfer_size_, syncInfo_);
            }
            Synchronize(syncInfo_);
        }
        catch (std::exception const& e)
        {
            H_REPORT_DTOR_EXCEPTION_AND_TERMINATE(e);
        }
    }

    // Enable move construction/assignment
    ManagedHostMemoryWrapper(
        ManagedHostMemoryWrapper<T,Device::GPU>&&) = default;
    // Disable copy construction/assignment
    ManagedHostMemoryWrapper(
        ManagedHostMemoryWrapper<T,Device::GPU> const&) = delete;

    T* data() noexcept { return host_data_.data(); }
private:
    simple_buffer<T,Device::CPU> host_data_;
    SyncInfo<Device::GPU> syncInfo_;
    T* device_data_;
    size_t final_xfer_offset_;
    size_t final_xfer_size_;
};// class ManagedHostMemoryWrapper<T,Device::GPU>
#endif // HYDROGEN_HAVE_GPU

template <typename T>
auto MakeHostBuffer(T* buf, size_t const& size,
                    SyncInfo<Device::CPU> const& syncInfo)
    -> PassthroughMemoryWrapper<T>
{
    return PassthroughMemoryWrapper<T>(buf);
}

// Helper functions that make all this stuff useful.
template <typename T>
auto
MakeManagedHostBuffer(T* buf, size_t const&, size_t const&, size_t const&,
                      size_t const&, size_t const&,
                      SyncInfo<Device::CPU> const&)
    -> PassthroughMemoryWrapper<T>
{
    return PassthroughMemoryWrapper<T>(buf);
}
template <typename T>
struct type_check;

#ifdef HYDROGEN_HAVE_GPU
// This can't (shouldn't) just be std::vector<T> because we want
// pinned memory for GPUs. And I don't want to write a new Allocator
// for std::vector that uses pinned memory through CUDA. We can access
// pinned memory through the simple_buffer.
template <typename T>
auto MakeHostBuffer(T const* buf, size_t const& size,
                    SyncInfo<Device::GPU> const& syncInfo)
    -> simple_buffer<T,Device::CPU>
{
    simple_buffer<T,Device::CPU> locbuf(
        size, SyncInfo<Device::CPU>{}, /*mode=*/ 1);
    gpu::Copy1DToHost(buf, locbuf.data(), size, syncInfo);
    return locbuf;
}
#endif // HYDROGEN_HAVE_GPU

template <typename T, Device D>
auto MakeManagedHostBuffer(
    T* buf, size_t const& totalsize,
    size_t const& initial_xfer_offset, size_t const& initial_xfer_size,
    size_t const& final_xfer_offset, size_t const& final_xfer_size,
    SyncInfo<D> const& syncInfo)
    -> ManagedHostMemoryWrapper<T,D>
{
    return ManagedHostMemoryWrapper<T,D>(
        buf, totalsize,
        initial_xfer_offset, initial_xfer_size,
        final_xfer_offset, final_xfer_size,
        syncInfo);
}

// Transfer all on construction; transfer none at end
#define ENSURE_HOST_SEND_BUFFER(buf, size, syncinfo)                \
    auto host_sbuf = internal::MakeHostBuffer(buf, size, syncinfo); \
    buf = host_sbuf.data()

// No transfer on construction; transfer all at end
#define ENSURE_HOST_RECV_BUFFER(buf, size, syncinfo)                 \
    auto host_rbuf =                                                 \
        internal::MakeManagedHostBuffer(                             \
            buf, size, 0UL, 0UL, 0UL, size, syncinfo);               \
    buf = host_rbuf.data()

// General case: some transfer at construction, some transfer at end
#define ENSURE_HOST_BUFFER_PREPOST_XFER(buf, totalsize, preoffset, presize, postoffset, postsize, syncinfo) \
    auto host_buf =                                                     \
        internal::MakeManagedHostBuffer(                                \
            buf, totalsize,                                             \
            preoffset, presize, postoffset, postsize,                   \
            syncinfo);                                                  \
    buf = host_buf.data()

// Transfer all at construction and destruction
#define ENSURE_HOST_INPLACE_BUFFER(buf, size, syncinfo)                 \
    ENSURE_HOST_BUFFER_PREPOST_XFER(                                    \
        buf, size, 0UL, size, 0UL, size, syncinfo)

}// namespace internal
}// namespace mpi
}// namespace El
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

#endif // ifndef EL_IMPORTS_MPIUTILS_HPP
