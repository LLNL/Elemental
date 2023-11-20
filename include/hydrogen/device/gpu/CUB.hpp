#ifndef HYDROGEN_IMPORTS_CUB_HPP_
#define HYDROGEN_IMPORTS_CUB_HPP_

#include "El/hydrogen_config.h"
#include "hydrogen/PoolAllocator.hpp"

namespace hydrogen
{
namespace cub
{

    /** @brief Get singleton instance of CUB memory pool.
     *
     *  A new memory pool is constructed if one doesn't exist
     *  already. The following environment variables are used to
     *  control the construction of the memory pool:
     *
     *    - H_MEMPOOL_BIN_GROWTH: The growth factor. (Default: 2)
     *    - H_MEMPOOL_MIN_BIN: The minimum bin. (Default: 1)
     *    - H_MEMPOOL_MAX_BIN: The maximum bin. (Default: no max bin)
     *    - H_MEMPOOL_MAX_CACHED_SIZE: The maximum aggregate cached bytes
     *      per device. (Default: No maximum)
     *    - H_MEMPOOL_MAX_BIN_ALLOC_SIZE: The maximum cached bytes per
     *      allocation. (Default: No maximum)
     *    - H_MEMPOOL_BIN_MULT_THRESHOLD: The difference between two consecutive
     *      geometric bins from which linear binning is used. (Default: No
     *      linear binning)
     *    - H_MEMPOOL_BIN_MULT: The multiplier for linear bin sizes. (Default:
     *      No linear binning)
     *    - H_MEMPOOL_BIN_SIZES: Custom set of (comma-separated) bin sizes in
     *      bytes. (Default: None)
     *    - H_MEMPOOL_MALLOCASYNC: If nonzero, uses mallocAsync as the backend
     *      for non-binned allocation. (Default: 0)
     *    - H_MEMPOOL_DEBUG: If nonzero, prints debugging output.
     *
     *  Note that if debugging output is turned on, there is no
     *  synchronization across processes. Users should take care to
     *  redirect output on a per-rank basis, either through the
     *  features exposed by their MPI launcher or by some other means.
     */
    PooledDeviceAllocator& MemoryPool();
    /** Destroy singleton instance of CUB memory pool. */
    void DestroyMemoryPool();

} // namespace cub
} // namespace hydrogen

#endif // HYDROGEN_IMPORTS_CUB_HPP_
