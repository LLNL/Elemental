#ifndef HYDROGEN_SYNCINFOBASE_HPP_
#define HYDROGEN_SYNCINFOBASE_HPP_

#include <El/hydrogen_config.h>

#include "Device.hpp"

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
template <Device D>
class SyncInfo;

template <>
class SyncInfo<Device::CPU>
{
public:
    SyncInfo() noexcept = default;
    ~SyncInfo() noexcept = default;
};// struct SyncInfo<Device::CPU>

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

/** @brief Get a new instance of a certain SyncInfo class.
 *
 *  For CPU, this will be empty, as usual. For GPU, this will have a
 *  *new* stream and event.
 */
template <Device D>
SyncInfo<D> CreateNewSyncInfo();

/** @brief Create a new CPU SyncInfo object. */
template <>
inline SyncInfo<Device::CPU> CreateNewSyncInfo<Device::CPU>()
{
    return SyncInfo<Device::CPU>{};
}

/** @brief Reset any internal state in the SyncInfo object.
 *
 *  For CPU, this will do nothing. For GPU, this will destroy the
 *  stream and event.
 */
template <Device D>
void DestroySyncInfo(SyncInfo<D>&);

/** @brief Destroy the CPU SyncInfo. */
inline void DestroySyncInfo(SyncInfo<Device::CPU>&) noexcept {}

/** @brief Synchronize the SyncInfo with the main (CPU) thread. */
template <Device D>
void Synchronize(SyncInfo<D> const&);

inline void Synchronize(SyncInfo<Device::CPU> const&) {}

/** @brief Add information to the SyncInfo object identifying this
 *         execution point.
 */
template <Device D, Device... Ds>
void AddSynchronizationPoint(
    SyncInfo<D> const& master,
    SyncInfo<Ds> const&... others);

inline void AddSynchronizationPoint(SyncInfo<Device::CPU> const&)
{}

inline void AddSynchronizationPoint(SyncInfo<Device::CPU> const&,
                                    SyncInfo<Device::CPU> const&)
{}

inline void AddSynchronizationPoint(SyncInfo<Device::CPU> const&,
                                    SyncInfo<Device::CPU> const&,
                                    SyncInfo<Device::CPU> const&)
{}

namespace details
{
template <Device D1, Device D2>
void AddSyncPoint(SyncInfo<D1> const&, SyncInfo<D2> const&);

inline void AddSyncPoint(SyncInfo<Device::CPU> const&,
                         SyncInfo<Device::CPU> const&) noexcept
{}

}// namespace details
}// namespace hydrogen
#endif // HYDROGEN_SYNCINFOBASE_HPP_
