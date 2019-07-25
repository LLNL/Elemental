#ifndef HYDROGEN_MEMORYPOOL_HPP_
#define HYDROGEN_MEMORYPOOL_HPP_

#include <stddef.h>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <set>
#include <unordered_map>
#include <mutex>

#include "El/hydrogen_config.h"
#if defined(HYDROGEN_HAVE_CUDA)
#include <cuda_runtime.h>
#elif defined(HYDROGEN_HAVE_ROCM)
#include <hip/hip_runtime.h>
#endif

namespace El
{
namespace details
{
template <typename... Args>
void ThrowRuntimeError(Args&&... args)
{
    std::ostringstream oss;
    int dummy[sizeof...(Args)] = { (oss << args, 0)... };
    (void) dummy;
    throw std::runtime_error(oss.str());
}
}

/** Simple caching memory pool.
 *  This maintains a set of bins that contain allocations of a fixed size.
 *  Each allocation will use the smallest size greater than or equal to the
 *  requested size. If an allocation is larger than any bin, it is allocated
 *  and freed directly.
 *  This memory pool is thread-safe.
 *  @tparam Pinned Whether this pool allocates CUDA pinned memory.
 */
template <bool Pinned>
class MemoryPool
{
public:

    /** Initialize the memory pool.
     *  This sets up bins per specification, and additionally adds power-of-2
     *  bins.
     *  @param bin_growth Controls how fast bins grow.
     *  @param min_bin_size Smallest bin size (in bytes).
     *  @param max_bin_size Largest bin size (in bytes).
     */
    MemoryPool(float bin_growth = 1.6,
               size_t min_bin_size = 1,
               size_t max_bin_size = 1<<26)
    {
        std::set<size_t> bin_sizes;
        for (float bin_size = min_bin_size;
             bin_size <= max_bin_size;
             bin_size *= bin_growth)
            bin_sizes.insert((size_t) bin_size);
        // Additionally, add power-of-2 bins.
        if (bin_growth != 2.0f)
        {
            for (size_t bin_size = min_bin_size;
                 bin_size <= max_bin_size;
                 bin_size *= 2)
                bin_sizes.insert(bin_size);
        }
        // Copy into bin_sizes_.
        for (const auto& size : bin_sizes)
            bin_sizes_.push_back(size);
        // Set up bins.
        for (size_t i = 0; i < bin_sizes_.size(); ++i)
            free_data_.emplace_back();
    }
    ~MemoryPool()
    {
        FreeAllUnused();
    }

    /** Return memory of size bytes. */
    void* Allocate(size_t size)
    {
        size_t bin = get_bin(size);
        void* mem = nullptr;
        std::lock_guard<std::mutex> lock(mutex_);
        // size is too large, this will not be cached.
        if (bin == INVALID_BIN)
        {
            mem = do_allocation(size);
        }
        else
        {
            // Check if there is available memory in our bin.
            if (free_data_[bin].size() > 0)
            {
                mem = free_data_[bin].back();
                free_data_[bin].pop_back();
            }
            else
            {
                mem = do_allocation(bin_sizes_[bin]);
            }
        }
        alloc_to_bin_[mem] = bin;
        return mem;
    }
    /** Release previously allocated memory. */
    void Free(void* ptr)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto iter = alloc_to_bin_.find(ptr);
        if (iter == alloc_to_bin_.end())
        {
            details::ThrowRuntimeError("Tried to free unknown ptr");
        }
        else
        {
            size_t bin = iter->second;
            alloc_to_bin_.erase(iter);
            if (bin == INVALID_BIN)
            {
                do_free(ptr);
            }
            else
            {
                // Cache the pointer for reuse.
                free_data_[bin].push_back(ptr);
            }
        }
    }
    /** Release all unused memory. */
    void FreeAllUnused()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t bin = 0; bin < bin_sizes_.size(); ++bin)
            for (auto&& ptr : free_data_[bin])
                do_free(ptr);
    }

private:

    /** Index of an invalid bin. */
    static constexpr size_t INVALID_BIN = (size_t) -1;

    /** Serialize access from multiple threads. */
    std::mutex mutex_;

    /** Size in bytes of each bin. */
    std::vector<size_t> bin_sizes_;
    /** Data available to allocate.
     *  Each entry is a bin, and each bin has a vector of pointers to free
     *  memory of that size.
     */
    std::vector<std::vector<void*>> free_data_;
    /** Map used pointers to the associated bin index. */
    std::unordered_map<void*, size_t> alloc_to_bin_;

    /** Allocate size bytes. */
    inline void* do_allocation(size_t size);
    /** Free ptr. */
    inline void do_free(void* ptr);

    /** Return the bin index for size. */
    inline size_t get_bin(size_t size)
    {
        // Assuming we don't have that many bins, just do a linear search.
        // Could optimize with binary search if need be.
        for (size_t i = 0; i < bin_sizes_.size(); ++i)
        {
            if (bin_sizes_[i] >= size) return i;
        }
        return INVALID_BIN;
    }

};  // class MemoryPool

#ifdef HYDROGEN_HAVE_CUDA
template <>
inline void* MemoryPool<true>::do_allocation(size_t bytes)
{
    void* ptr;
    auto error = cudaMallocHost(&ptr, bytes);
    if (error != cudaSuccess)
    {
        details::ThrowRuntimeError(
            "Failed to allocate CUDA pinned memory with message: ",
            "\"", cudaGetErrorString(error), "\"");
    }
    return ptr;
}

template<>
inline void MemoryPool<true>::do_free(void* ptr)
{
    auto error = cudaFreeHost(ptr);
    if (error != cudaSuccess)
    {
        details::ThrowRuntimeError(
            "Failed to free CUDA pinned memory with message: ",
            "\"", cudaGetErrorString(error), "\"");
    }
}
#elif defined(HYDROGEN_HAVE_ROCM)
template <>
inline void* MemoryPool<true>::do_allocation(size_t bytes)
{
    void* ptr;
    auto error = hipHostMalloc(&ptr, bytes);
    if (error != hipSuccess)
    {
        details::ThrowRuntimeError(
            "Failed to allocate HIP pinned memory with message: ",
            "\"", hipGetErrorString(error), "\"");
    }
    return ptr;
}

template<>
inline void MemoryPool<true>::do_free(void* ptr)
{
    auto error = hipHostFree(ptr);
    if (error != hipSuccess)
    {
        details::ThrowRuntimeError(
            "Failed to free HIP pinned memory with message: ",
            "\"", hipGetErrorString(error), "\"");
    }
}
#endif  // HYDROGEN_HAVE_CUDA

template <>
inline void* MemoryPool<false>::do_allocation(size_t bytes) {
    void* ptr = std::malloc(bytes);
    if (ptr == nullptr)
    {
        details::ThrowRuntimeError("Failed to allocate memory");
    }
    return ptr;
}

template<>
inline void MemoryPool<false>::do_free(void* ptr)
{
    return std::free(ptr);
}

#ifdef HYDROGEN_HAVE_GPU
/** Get singleton instance of CUDA pinned host memory pool. */
MemoryPool<true>& PinnedHostMemoryPool();
/** Destroy singleton instance of CUDA pinned host memory pool. */
void DestroyPinnedHostMemoryPool();
#endif  // HYDROGEN_HAVE_CUDA
/** Get singleton instance of host memory pool. */
MemoryPool<false>& HostMemoryPool();
/** Destroy singleton instance of host memory pool. */
void DestroyHostMemoryPool();

}  // namespace El

#endif  // HYDROGEN_MEMORYPOOL_HPP_
