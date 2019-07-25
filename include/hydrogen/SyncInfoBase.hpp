#ifndef HYDROGEN_SYNCINFOBASE_HPP_
#define HYDROGEN_SYNCINFOBASE_HPP_

#include <El/hydrogen_config.h>

#include <hydrogen/Device.hpp>
#include <hydrogen/meta/IndexSequence.hpp>

#include <tuple>

namespace hydrogen
{

/** \class SyncInfo
 *  \brief Manage device-specific synchronization information.
 *
 *  Device-specific synchronization information. For CPUs, this is
 *  empty since all CPU operations are synchronous with respect to the
 *  host. For GPUs, this will be a stream and an associated event.
 *
 *  The use-case for this is to cope with the matrix-free part of the
 *  interface. Many of the copy routines have the paradigm that they
 *  take Matrix<T,D>s as arguments and then the host will organize and
 *  dispatch subkernels that operate on data buffers, i.e., T[]
 *  data. In the GPU case, for example, this provides a lightweight
 *  way to pass the CUDA stream through the T* interface without an
 *  entire matrix (which, semantically, may not make sense).
 *
 *  This also might be useful for interacting with
 *  Aluminum/MPI/NCCL/whatever. It essentially enables tagged
 *  dispatch, where the tags possibly contain some extra
 *  device-specific helpers.
 */
template <Device D> struct SyncInfo
{
    SyncInfo() = default;
    ~SyncInfo() = default;
};// struct SyncInfo<D>

template <Device D>
bool operator==(SyncInfo<D> const&, SyncInfo<D> const&)
{
    return true;
}

template <Device D>
bool operator!=(SyncInfo<D> const&, SyncInfo<D> const&)
{
    return false;
}

template <Device D1, Device D2>
bool operator==(SyncInfo<D1> const&, SyncInfo<D2> const&)
{
    return false;
}

template <Device D1, Device D2>
bool operator!=(SyncInfo<D1> const&, SyncInfo<D2> const&)
{
    return true;
}

// This synchronizes the additional SyncInfos to the "master". That
// is, the execution streams described by the "others" will wait
// for the "master" stream.
template <Device D, Device... Ds>
void AddSynchronizationPoint(
    SyncInfo<D> const& /* master */,
    SyncInfo<Ds>... /* others */)
{
}

// Synchronizing is a no-op by default
template <Device D>
void Synchronize(SyncInfo<D> const&)
{}

template <Device D, Device... Ds>
void AllWaitOnMaster(
    SyncInfo<D> const& master, SyncInfo<Ds> const&... others)
{
    AddSynchronizationPoint(master, others...);
}

template <Device D, Device... Ds>
void MasterWaitOnAll(
    SyncInfo<D> const& master,
    SyncInfo<Ds> const&... others)
{
    int dummy[] = {
        (AddSynchronizationPoint(others, master), 0)...};
    (void) dummy;
}

/** \class MultiSync
 *  \brief RAII class to wrap a bunch of SyncInfo objects.
 *
 *  Provides basic synchronization for the common case in which an
 *  operation may act upon objects that exist on multiple distinct
 *  synchronous processing elements (e.g., cudaStreams) but actual
 *  computation can only occur on one of them.
 *
 *  Constructing an object of this class will cause the master
 *  processing element to wait on the others, asynchronously with
 *  respect to the CPU, if possible. Symmetrically, destruction of
 *  this object will cause the other processing elements to wait on
 *  the master processing element, asynchronously with respect to the
 *  CPU, if possible.
 *
 *  The master processing element is assumed to be the first SyncInfo
 *  passed into the constructor.
 */
template <Device... Ds>
class MultiSync
{
public:
    MultiSync(SyncInfo<Ds> const&... syncInfos)
        : syncInfos_{syncInfos...}
    {
        MasterWaitOnAll(syncInfos...);
    }

    ~MultiSync()
    {
        DTorImpl_(MakeIndexSequence<sizeof...(Ds)>());
    }
private:
    template <size_t... Is>
    void DTorImpl_(IndexSequence<Is...>)
    {
        AllWaitOnMaster(std::get<Is>(syncInfos_)...);
    }

    std::tuple<SyncInfo<Ds>...> syncInfos_;
};// class MultiSync

template <Device... Ds>
auto MakeMultiSync(SyncInfo<Ds> const&... syncInfos) -> MultiSync<Ds...>
{
    return MultiSync<Ds...>(syncInfos...);
}

}// namespace hydrogen
#endif // HYDROGEN_SYNCINFOBASE_HPP_
