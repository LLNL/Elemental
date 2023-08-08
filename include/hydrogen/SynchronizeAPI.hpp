#ifndef HYDROGEN_SYNCHRONIZEAPI_HPP_
#define HYDROGEN_SYNCHRONIZEAPI_HPP_

#include "SyncInfo.hpp"

namespace hydrogen
{

// This synchronizes the additional SyncInfos to the "master". That
// is, the execution streams described by the "others" will wait
// for the "master" stream.
template <Device D, Device D2, Device... Ds>
void AddSynchronizationPoint(SyncInfo<D> const &master,
                             SyncInfo<D2> const &other,
                             SyncInfo<Ds> const &...others)
{
#ifdef HYDROGEN_HAVE_GPU
    if constexpr (D == Device::GPU && D == D2) {
        // When the streams are the same, there is no need to create
        // synchronization points. Skip "other" call recursively with the rest.
        if (master.Stream() == other.Stream())
        {
            if constexpr (sizeof...(others) > 0UL)
                AddSynchronizationPoint(master, others...);
            return;
        }
    }
#endif // HYDROGEN_HAVE_GPU

    AddSynchronizationPoint(master);
    int dummy[] = {(details::AddSyncPoint(master, other), 0),
                   (details::AddSyncPoint(master, others), 0)...};
    (void)dummy;
}

template <Device D, Device... Ds>
void AllWaitOnMaster(SyncInfo<D> const &master, SyncInfo<Ds> const &...others)
{
    AddSynchronizationPoint(master, others...);
}

template <Device D, Device... Ds>
void MasterWaitOnAll(SyncInfo<D> const &master, SyncInfo<Ds> const &...others)
{
    int dummy[] = {(AddSynchronizationPoint(others, master), 0)...};
    (void)dummy;
}

} // namespace hydrogen
#endif // HYDROGEN_SYNCHRONIZEAPI_HPP_
