#ifndef HYDROGEN_MEMORYPOOL_HPP_
#define HYDROGEN_MEMORYPOOL_HPP_

#include "El/hydrogen_config.h"
#if defined(HYDROGEN_HAVE_CUDA)
#include <cuda_runtime.h>
#elif defined(HYDROGEN_HAVE_ROCM)
#include <hip/hip_runtime.h>
#endif

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

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

/** @brief Returns true iff env(H_MEMPOOL_DEBUG) is truthy.
 *
 *  Truthy values are non-empty strings that start with any character
 *  other than '0' (ASCII "zero"). So "true", "false", "1", "13",
 *  "-q", ":)", and " " are all truthy, while "", "0true", "0false",
 *  "0000", "0123", and "0:)" are all falsey.
 */
bool debug_mempool() noexcept;

/** @brief Check env(H_MEMPOOL_BIN_GROWTH). Default 1.6f. */
float default_mempool_bin_growth() noexcept;

/** @brief Check env(H_MEMPOOL_MIN_BIN). Default 1UL. */
size_t default_mempool_min_bin() noexcept;

/** @brief Check env(H_MEMPOOL_MAX_BIN). Default (1<<26). */
size_t default_mempool_max_bin() noexcept;

} // namespace details

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
     *  @param debug Print debugging messages.
     */
    MemoryPool(float const bin_growth = details::default_mempool_bin_growth(),
               size_t const min_bin_size = details::default_mempool_min_bin(),
               size_t const max_bin_size = details::default_mempool_max_bin(),
               bool const debug = details::debug_mempool())
        : debug_{debug}
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
        if (debug_)
        {
            std::clog << "==Mempool(" << this << ")== "
                      << "Created memory pool ("
                      << "pinned=" << (Pinned ? "t" : "f")
                      << ", growth=" << bin_growth
                      << ", min bin=" << bin_sizes_.front()
                      << ", max bin=" << bin_sizes_.back() << ")\n"
                      << "==Mempool(" << this << ")== "
                      << "Bin sizes: [";
            for (auto const& b : bin_sizes_)
                std::clog << " " << b;
            std::clog << " ]" << std::endl;
        }
    }
    ~MemoryPool()
    {
        FreeAllUnused();
        if (debug_)
            std::clog << "==Mempool(" << this << ")== "
                      << alloc_to_bin_.size()
                      << " dangling allocations\n"
                      << "==Mempool(" << this << ")== "
                      << "Destroyed memory pool"
                      << std::endl;
    }

    /** Return memory of size bytes. */
    void* Allocate(size_t size)
    {
        if (debug_)
            std::clog << "==Mempool(" << this << ")== "
                      << "Requesting allocation of "
                      << size << " bytes."
                      << std::endl;
        size_t const bin = get_bin(size);
        void* mem = nullptr;
        std::lock_guard<std::mutex> lock(mutex_);
        // size is too large, this will not be cached.
        if (bin == INVALID_BIN)
            mem = do_allocation(size);
        else
        {
            // Check if there is available memory in our bin.
            if (free_data_[bin].size() > 0)
            {
                mem = free_data_[bin].back();
                free_data_[bin].pop_back();
                --num_cached_blks_;
                if (debug_)
                    std::clog << "==Mempool(" << this << ")== "
                              << "Reusing cached pointer " << mem << "\n";
            }
            else
            {
                mem = do_allocation(bin_sizes_[bin]);
            }
        }
        alloc_to_bin_[mem] = bin;
        if (debug_)
            std::clog << "==Mempool(" << this << ")== "
                      << alloc_to_bin_.size()
                      << " blocks allocated; "
                      << num_cached_blks_
                      << " blocks cached"
                      << std::endl;

        return mem;
    }
    /** Release previously allocated memory. */
    void Free(void* ptr)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto const iter = alloc_to_bin_.find(ptr);
        if (iter == alloc_to_bin_.end())
            details::ThrowRuntimeError("Tried to free unknown ptr");

        size_t const& bin = iter->second;
        alloc_to_bin_.erase(iter);
        if (bin == INVALID_BIN)
            do_free(ptr);
        else
        {
            // Cache the pointer for reuse.
            free_data_[bin].push_back(ptr);
            ++num_cached_blks_;
            if (debug_)
                std::clog << "==Mempool(" << this << ")== "
                          << "Cached pointer " << ptr << "\n";
        }
        if (debug_)
            std::clog << "==Mempool(" << this << ")== "
                      << alloc_to_bin_.size()
                      << " blocks allocated; "
                      << num_cached_blks_
                      << " blocks cached"
                      << std::endl;
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

    /** Track the total number of available blocks. */
    size_t num_cached_blks_;

    /** Print debugging messages throughout lifetime. */
    bool debug_;

    /** Release all unused memory. */
    void FreeAllUnused()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t bin = 0; bin < bin_sizes_.size(); ++bin)
        {
            for (auto&& ptr : free_data_[bin])
                do_free(ptr);
            std::vector<void*>{}.swap(free_data_[bin]);
        }
        num_cached_blks_ = 0ul;
    }

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
inline void* MemoryPool<true>::do_allocation(size_t const bytes)
{
    void* ptr;
    auto error = cudaMallocHost(&ptr, bytes);
    if (error != cudaSuccess)
    {
        details::ThrowRuntimeError(
            "Failed to allocate CUDA pinned memory with message: ",
            "\"", cudaGetErrorString(error), "\"");
    }
    if (debug_)
        std::clog << "==Mempool(" << this << ")== "
                  << "Allocated pinned " << bytes << " bytes at " << ptr
                  << std::endl;
    return ptr;
}

template<>
inline void MemoryPool<true>::do_free(void* const ptr)
{
    auto error = cudaFreeHost(ptr);
    if (error != cudaSuccess)
    {
        details::ThrowRuntimeError(
            "Failed to free CUDA pinned memory with message: ",
            "\"", cudaGetErrorString(error), "\"");
    }
    if (debug_)
        std::clog << "==Mempool(" << this << ")== "
                  << "Freed pinned ptr " << ptr
                  << std::endl;
}
#elif defined(HYDROGEN_HAVE_ROCM)
template <>
inline void* MemoryPool<true>::do_allocation(size_t const bytes)
{
    void* ptr;
    auto error = hipHostMalloc(&ptr, bytes);
    if (error != hipSuccess)
        details::ThrowRuntimeError(
            "Failed to allocate HIP pinned memory with message: ",
            "\"", hipGetErrorString(error), "\"");
    if (debug_)
        std::clog << "==Mempool(" << this << ")== "
                  << "Allocated pinned " << bytes << " bytes at " << ptr
                  << std::endl;
    return ptr;
}

template<>
inline void MemoryPool<true>::do_free(void* const ptr)
{
    auto error = hipHostFree(ptr);
    if (error != hipSuccess)
        details::ThrowRuntimeError(
            "Failed to free HIP pinned memory with message: ",
            "\"", hipGetErrorString(error), "\"");
    if (debug_)
        std::clog << "==Mempool(" << this << ")== "
                  << "Freed pinned ptr " << ptr
                  << std::endl;
}
#endif  // HYDROGEN_HAVE_CUDA

template <>
inline void* MemoryPool<false>::do_allocation(size_t const bytes)
{
    void* ptr = std::malloc(bytes);
    if (ptr == nullptr)
        details::ThrowRuntimeError("Failed to allocate memory");
    if (debug_)
        std::clog << "==Mempool(" << this << ")== "
                  << "Allocated " << bytes << " bytes at " << ptr
                  << std::endl;
    return ptr;
}

template<>
inline void MemoryPool<false>::do_free(void* const ptr)
{
    std::free(ptr);
    if (debug_)
        std::clog << "==Mempool(" << this << ")== "
                  << "Freed ptr " << ptr
                  << std::endl;
}

#ifdef HYDROGEN_HAVE_GPU
/** Get singleton instance of CUDA pinned host memory pool. */
MemoryPool<true>& PinnedHostMemoryPool();
/** Destroy singleton instance of CUDA pinned host memory pool. */
void DestroyPinnedHostMemoryPool();
#endif  // HYDROGEN_HAVE_GPU
/** Get singleton instance of host memory pool. */
MemoryPool<false>& HostMemoryPool();
/** Destroy singleton instance of host memory pool. */
void DestroyHostMemoryPool();

}  // namespace El

#endif  // HYDROGEN_MEMORYPOOL_HPP_
