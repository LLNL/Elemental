#ifndef HYDROGEN_IMPORTS_CUB_HPP_
#define HYDROGEN_IMPORTS_CUB_HPP_

#include <cuda_runtime.h>
#include <cub/util_allocator.cuh>

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
     *    - H_CUB_BIN_GROWTH: The growth factor. (Default: 2)
     *    - H_CUB_MIN_BIN: The minimum bin. (Default: 1)
     *    - H_CUB_MAX_BIN: The maximum bin. (Default: no max bin)
     *    - H_CUB_MAX_CACHED_SIZE: The maximum aggregate cached bytes
     *      per device. (Default: No maximum)
     *    - H_CUB_DEBUG: If nonzero, allow CUB to print debugging output.
     *
     *  Note that if debugging output is turned on, there is no
     *  synchronization across processes. Users should take care to
     *  redirect output on a per-rank basis, either through the
     *  features exposed by their MPI launcher or by some other means.
     */
    ::cub::CachingDeviceAllocator& MemoryPool();
    /** Destroy singleton instance of CUB memory pool. */
    void DestroyMemoryPool();

} // namespace cub
} // namespace hydrogen

#endif // HYDROGEN_IMPORTS_CUB_HPP_
