#ifndef HYDROGEN_MULTISYNC_HPP_
#define HYDROGEN_MULTISYNC_HPP_

#include "Device.hpp"
#include "SyncInfoBase.hpp"
#include "SynchronizeAPI.hpp"
#include "meta/IndexSequence.hpp"

#include <tuple>

namespace hydrogen
{

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
    using sync_tuple_type = std::tuple<SyncInfo<Ds>...>;
    using sync_master_type =
        typename std::tuple_element<0, sync_tuple_type>::type;
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

    /** @brief Implicitly convert to the master.
     *
     *  This is to be able to pass a multisync in place of a SyncInfo
     *  object. It is common to create a MultiSync and then pass its
     *  master to a bunch of other calls. This simplifies things by
     *  not needing to store an external reference to the master
     *  SyncInfo.
     */
    operator sync_master_type const& () const noexcept
    {
        return std::get<0>(syncInfos_);
    }

private:
    template <size_t... Is>
    void DTorImpl_(IndexSequence<Is...>)
    {
        AllWaitOnMaster(std::get<Is>(syncInfos_)...);
    }

    sync_tuple_type syncInfos_;
};// class MultiSync

template <Device... Ds>
auto MakeMultiSync(SyncInfo<Ds> const&... syncInfos) -> MultiSync<Ds...>
{
    return MultiSync<Ds...>(syncInfos...);
}

}// namespace hydrogen
#endif // HYDROGEN_MULTISYNC_HPP_
