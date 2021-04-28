#ifndef HYDROGEN_IMPORTS_ALUMINUM_HPP_
#define HYDROGEN_IMPORTS_ALUMINUM_HPP_

#include <hydrogen/Device.hpp>

#ifdef HYDROGEN_HAVE_ALUMINUM
#include <Al.hpp>

#ifdef HYDROGEN_HAVE_NVPROF
#include "nvToolsExt.h"
#include "nvToolsExtCuda.h"
#include "nvToolsExtCudaRt.h"
#endif // HYDROGEN_HAVE_NVPROF

#endif // HYDROGEN_HAVE_ALUMINUM

namespace El
{
// "Real" declaration is in include/El/core/environment/impl.hpp
extern void break_on_me();

// FIXME: This is a lame shortcut to save some
// metaprogramming. Deadlines are the worst.
enum class Collective
{
    ALLGATHER,
    ALLREDUCE,
    ALLTOALL,
    BROADCAST,
    GATHER,
    REDUCE,
    REDUCESCATTER,
    SCATTER,
    SENDRECV
};// enum class Collective

#ifndef HYDROGEN_HAVE_ALUMINUM

template <typename T> struct IsAluminumTypeT : std::false_type {};
template <typename T, Device D>
struct IsAluminumDeviceType : std::false_type {};
template <typename T, Device D, Collective C>
struct IsAluminumSupported : std::false_type {};

#else

// A function to convert an MPI MPI_Op into an Aluminum operator
Al::ReductionOperator MPI_Op2ReductionOperator(MPI_Op op);

#define ADD_ALUMINUM_TYPE(type, backend) \
    template <> struct IsAlTypeT<type,backend> : std::true_type {}
#define ADD_ALUMINUM_COLLECTIVE(coll, backend) \
    template <> struct IsBackendSupported<coll, backend> : std::true_type {}

//
// Setup type support
//

template <typename T, typename BackendT>
struct IsAlTypeT : std::false_type {};

ADD_ALUMINUM_TYPE(              char, Al::MPIBackend);
ADD_ALUMINUM_TYPE(       signed char, Al::MPIBackend);
ADD_ALUMINUM_TYPE(     unsigned char, Al::MPIBackend);
ADD_ALUMINUM_TYPE(             short, Al::MPIBackend);
ADD_ALUMINUM_TYPE(    unsigned short, Al::MPIBackend);
ADD_ALUMINUM_TYPE(               int, Al::MPIBackend);
ADD_ALUMINUM_TYPE(      unsigned int, Al::MPIBackend);
ADD_ALUMINUM_TYPE(          long int, Al::MPIBackend);
ADD_ALUMINUM_TYPE(     long long int, Al::MPIBackend);
ADD_ALUMINUM_TYPE( unsigned long int, Al::MPIBackend);
ADD_ALUMINUM_TYPE(unsigned long long, Al::MPIBackend);
ADD_ALUMINUM_TYPE(             float, Al::MPIBackend);
ADD_ALUMINUM_TYPE(            double, Al::MPIBackend);
ADD_ALUMINUM_TYPE(       long double, Al::MPIBackend);

#ifdef HYDROGEN_HAVE_NCCL2
ADD_ALUMINUM_TYPE(                  char, Al::NCCLBackend);
ADD_ALUMINUM_TYPE(         unsigned char, Al::NCCLBackend);
ADD_ALUMINUM_TYPE(                   int, Al::NCCLBackend);
ADD_ALUMINUM_TYPE(          unsigned int, Al::NCCLBackend);
ADD_ALUMINUM_TYPE(         long long int, Al::NCCLBackend);
ADD_ALUMINUM_TYPE(unsigned long long int, Al::NCCLBackend);
#ifdef HYDROGEN_GPU_USE_FP16
ADD_ALUMINUM_TYPE(         gpu_half_type, Al::NCCLBackend);
#endif // HYDROGEN_GPU_USE_FP16
ADD_ALUMINUM_TYPE(                 float, Al::NCCLBackend);
ADD_ALUMINUM_TYPE(                double, Al::NCCLBackend);
#endif // HYDROGEN_HAVE_NCCL2

#ifdef HYDROGEN_HAVE_AL_MPI_CUDA
template <typename T>
struct IsAlTypeT<T, Al::MPICUDABackend> : IsAlTypeT<T, Al::MPIBackend> {};
#endif // HYDROGEN_HAVE_AL_MPI_CUDA

#ifdef HYDROGEN_HAVE_AL_HOST_XFER
template <typename T>
struct IsAlTypeT<T, Al::HostTransferBackend> : IsAlTypeT<T, Al::MPIBackend> {};
#endif // HYDROGEN_HAVE_AL_HOST_XFER

//
// Setup collective support
//

template <Collective C, typename BackendT>
struct IsBackendSupported : std::false_type {};

// MPI backend now supports all collectives
ADD_ALUMINUM_COLLECTIVE(    Collective::ALLGATHER, Al::MPIBackend);
ADD_ALUMINUM_COLLECTIVE(    Collective::ALLREDUCE, Al::MPIBackend);
ADD_ALUMINUM_COLLECTIVE(     Collective::ALLTOALL, Al::MPIBackend);
ADD_ALUMINUM_COLLECTIVE(    Collective::BROADCAST, Al::MPIBackend);
ADD_ALUMINUM_COLLECTIVE(       Collective::GATHER, Al::MPIBackend);
ADD_ALUMINUM_COLLECTIVE(       Collective::REDUCE, Al::MPIBackend);
ADD_ALUMINUM_COLLECTIVE(Collective::REDUCESCATTER, Al::MPIBackend);
ADD_ALUMINUM_COLLECTIVE(      Collective::SCATTER, Al::MPIBackend);
ADD_ALUMINUM_COLLECTIVE(     Collective::SENDRECV, Al::MPIBackend);

#ifdef HYDROGEN_HAVE_NCCL2
// NCCL backend supports these
ADD_ALUMINUM_COLLECTIVE(    Collective::ALLGATHER, Al::NCCLBackend);
ADD_ALUMINUM_COLLECTIVE(    Collective::ALLREDUCE, Al::NCCLBackend);
ADD_ALUMINUM_COLLECTIVE(     Collective::ALLTOALL, Al::NCCLBackend);
ADD_ALUMINUM_COLLECTIVE(    Collective::BROADCAST, Al::NCCLBackend);
ADD_ALUMINUM_COLLECTIVE(       Collective::GATHER, Al::NCCLBackend);
ADD_ALUMINUM_COLLECTIVE(       Collective::REDUCE, Al::NCCLBackend);
ADD_ALUMINUM_COLLECTIVE(Collective::REDUCESCATTER, Al::NCCLBackend);
ADD_ALUMINUM_COLLECTIVE(      Collective::SCATTER, Al::NCCLBackend);
//ADD_ALUMINUM_COLLECTIVE(     Collective::SENDRECV, Al::NCCLBackend);
#endif // HYDROGEN_HAVE_NCCL2

#ifdef HYDROGEN_HAVE_AL_HOST_XFER
// MPICUDA backend now supports all collectives
ADD_ALUMINUM_COLLECTIVE(    Collective::ALLGATHER, Al::HostTransferBackend);
ADD_ALUMINUM_COLLECTIVE(    Collective::ALLREDUCE, Al::HostTransferBackend);
ADD_ALUMINUM_COLLECTIVE(     Collective::ALLTOALL, Al::HostTransferBackend);
ADD_ALUMINUM_COLLECTIVE(    Collective::BROADCAST, Al::HostTransferBackend);
ADD_ALUMINUM_COLLECTIVE(       Collective::GATHER, Al::HostTransferBackend);
ADD_ALUMINUM_COLLECTIVE(       Collective::REDUCE, Al::HostTransferBackend);
ADD_ALUMINUM_COLLECTIVE(Collective::REDUCESCATTER, Al::HostTransferBackend);
ADD_ALUMINUM_COLLECTIVE(      Collective::SCATTER, Al::HostTransferBackend);
ADD_ALUMINUM_COLLECTIVE(     Collective::SENDRECV, Al::HostTransferBackend);
#endif // HYDROGEN_HAVE_AL_HOST_XFER

template <Device D>
struct BackendsForDeviceT;

template <>
struct BackendsForDeviceT<Device::CPU>
{
    using type = hydrogen::TypeList<Al::MPIBackend>;
};// struct BackendsForDeviceT<Device::CPU>

namespace details
{
struct BackendNotDefined;
}

#ifdef HYDROGEN_HAVE_NCCL2
using AluminumNCCLBackendIfDefined = Al::NCCLBackend;
#else
using AluminumNCCLBackendIfDefined = details::BackendNotDefined;
#endif

#ifdef HYDROGEN_HAVE_AL_HOST_XFER
using AluminumHostTransferBackendIfDefined = Al::HostTransferBackend;
#else
using AluminumHostTransferBackendIfDefined = details::BackendNotDefined;
#endif

#ifdef HYDROGEN_HAVE_AL_MPI_CUDA
using AluminumMPICUDABackendIfDefined = Al::MPICUDABackend;
#else
using AluminumMPICUDABackendIfDefined = details::BackendNotDefined;
#endif

// Prefer the NCCL2, then host transfer, then MPI-CUDA backend
#ifdef HYDROGEN_HAVE_GPU
template <>
struct BackendsForDeviceT<Device::GPU>
{
    using type = hydrogen::RemoveAll<
        hydrogen::TypeList<
          AluminumNCCLBackendIfDefined,
          AluminumHostTransferBackendIfDefined,
          AluminumMPICUDABackendIfDefined>,
        details::BackendNotDefined>;
};// struct BackendsForDeviceT<Device::GPU>
#endif // HYDROGEN_HAVE_GPU

// Helper using statement
template <Device D>
using BackendsForDevice = typename BackendsForDeviceT<D>::type;

#ifdef HYDROGEN_HAVE_GPU
using AllAluminumBackends = Join<BackendsForDevice<Device::CPU>,
                                 BackendsForDevice<Device::GPU>>;
#else
using AllAluminumBackends = BackendsForDevice<Device::CPU>;
#endif // HYDROGEN_HAVE_GPU

template <typename BackendT>
struct DeviceForBackendT;

template <>
struct DeviceForBackendT<Al::MPIBackend>
{
    constexpr static Device value = Device::CPU;
};

#ifdef HYDROGEN_HAVE_GPU
#ifdef HYDROGEN_HAVE_NCCL2
template <>
struct DeviceForBackendT<Al::NCCLBackend>
{
    constexpr static Device value = Device::GPU;
};
#endif // HYDROGEN_HAVE_NCCL2
#ifdef HYDROGEN_HAVE_AL_HOST_XFER
template <>
struct DeviceForBackendT<Al::HostTransferBackend>
{
    constexpr static Device value = Device::GPU;
};
#endif // HYDROGEN_HAVE_AL_HOST_XFER
#ifdef HYDROGEN_HAVE_AL_MPI_CUDA
template <>
struct DeviceForBackendT<Al::MPICUDABackend>
{
    constexpr static Device value = Device::GPU;
};
#endif // HYDROGEN_HAVE_AL_MPI_CUDA
#endif // HYDROGEN_HAVE_GPU

template <typename BackendT>
constexpr Device DeviceForBackend()
{
    return DeviceForBackendT<BackendT>::value;
}

//
// Aluminum-specific predicates/metafunctions
//

template <typename T, Collective C, typename BackendT>
struct AluminumSupportsBackendAndCollective
    : And<IsAlTypeT<T,BackendT>,IsBackendSupported<C,BackendT>>
{};

template <typename T, Collective C, typename BackendList>
struct IsBackendSupportedByAny
    : Or<AluminumSupportsBackendAndCollective<T,C,Head<BackendList>>,
         IsBackendSupportedByAny<T,C,Tail<BackendList>>>
{};

template <typename T, Collective C>
struct IsBackendSupportedByAny<T,C,hydrogen::TypeList<>>
    : std::false_type
{};

template <typename T, Device D, Collective C>
struct IsAluminumSupported
    : IsBackendSupportedByAny<T,C,BackendsForDevice<D>>
{};


template <typename List, typename U,
          Collective C, template <class,Collective,class> class Pred>
struct SelectFirstOkBackend
    : std::conditional<Pred<U,C,Head<List>>::value,
                       HeadT<List>,
                       SelectFirstOkBackend<Tail<List>,U,C,Pred>>::type
{};

// The "best" backend is the first one in the list that supports our
// type T and implements our collective C.
template <typename T, Device D, Collective C>
struct BestBackendT
    : SelectFirstOkBackend<BackendsForDevice<D>,T,C,
                           AluminumSupportsBackendAndCollective>
{};

template <typename T, Device D, Collective C>
using BestBackend = typename BestBackendT<T,D,C>::type;

namespace mpi
{
namespace internal
{
template <Device D>
struct SyncInfoManager;

template <>
struct SyncInfoManager<Device::CPU>
{
    SyncInfoManager(std::string const&)
    {}

    SyncInfo<Device::CPU> si_;
};

#ifdef HYDROGEN_HAVE_GPU
template <>
struct SyncInfoManager<Device::GPU>
{
    SyncInfoManager(std::string const& backend_name)
        : si_{CreateNewSyncInfo<Device::GPU>()}
    {
#ifdef HYDROGEN_HAVE_NVPROF
        // Name the stream for debugging purposes
        std::string const stream_name
            = "H: Comm (" + backend_name + ")";
        nvtxNameCudaStreamA(si_.Stream(), stream_name.c_str());
#else
        (void) backend_name;
#endif // HYDROGEN_HAVE_NVPROF
    }
    ~SyncInfoManager()
    {
        try
        {
            DestroySyncInfo(si_);
        }
        catch (std::exception const& e)
        {
            std::cerr << "Error detected in ~SyncInfoManager():\n\n"
                      << e.what() << std::endl
                      << "std::terminate() will be called."
                      << std::endl;
            break_on_me();
            std::terminate();

        }
        catch (...)
        {
            std::cerr << "Unknown error detected in ~SyncInfoManager().\n\n"
                      << "std::terminate() will be called."
                      << std::endl;
            break_on_me();
            std::terminate();
        }
    }
    SyncInfoManager(SyncInfoManager const&) = delete;
    SyncInfoManager(SyncInfoManager&&) = delete;
    SyncInfoManager& operator=(SyncInfoManager const&) = delete;
    SyncInfoManager& operator=(SyncInfoManager &&) = delete;

    SyncInfo<Device::GPU> si_;
};
#endif // HYDROGEN_HAVE_GPU

template <typename BackendT>
SyncInfo<DeviceForBackend<BackendT>()> const& BackendSyncInfo()
{
    constexpr Device D = DeviceForBackend<BackendT>();
    static SyncInfoManager<D> si_mgr_(BackendT::Name());
    return si_mgr_.si_;
}

}// namespace internal
}// namespace mpi

#endif // ndefined(HYDROGEN_HAVE_ALUMINUM)

} // namespace El

#endif // HYDROGEN_IMPORTS_ALUMINUM_HPP_
