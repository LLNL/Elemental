#ifndef HYDROGEN_SYNCINFOPOOL_HPP_
#define HYDROGEN_SYNCINFOPOOL_HPP_

#include "hydrogen/Device.hpp"
#include "hydrogen/SyncInfo.hpp"

#ifdef HYDROGEN_HAVE_NVPROF
#include "nvToolsExt.h"
#include "nvToolsExtCuda.h"
#include "nvToolsExtCudaRt.h"
#include "cuda_runtime.h"
#endif // HYDROGEN_HAVE_NVPROF

namespace hydrogen
{

/** @class SyncInfoPool
 *  @brief A pool of SyncInfo objects across which tasks can be distributed.
 *
 *  The pool owns all of the contained SyncInfo objects. All
 *  references are invalid after the pool is destroyed.
 */
template <Device D>
class SyncInfoPool
{
    using sync_type = SyncInfo<D>;
    using container_type = std::vector<sync_type>;
    using iterator_type = typename container_type::const_iterator;

public:

    /** @brief Construct an empty pool. */
    SyncInfoPool() noexcept = default;

    /** @brief Construct a pool of a specific size. */
    SyncInfoPool(size_t pool_size)
    {
        this->EnsureSize(pool_size);
    }

    /** @brief Destructor */
    ~SyncInfoPool();

    /** @brief Move constructor. */
    SyncInfoPool(SyncInfoPool&& other) noexcept = default;

    /** @brief Move assignment. */
    SyncInfoPool& operator=(SyncInfoPool&& other) noexcept = default;

    /** @brief Copy construction is deleted. */
    SyncInfoPool(SyncInfoPool const&) = delete;

    /** @brief Copy assignment is deleted. */
    SyncInfoPool& operator=(SyncInfoPool const&) = delete;

    /** @name Queries */
    ///@{

    /** @brief Get the current size of the pool. */
    size_t Size() const noexcept
    {
        return pool_.size();
    }

    ///@}
    /** @name Modifiers */
    ///@{

    /** @brief Ensure that the pool has required size. */
    void EnsureSize(size_t pool_size);

    /** @brief Swap contents with another pool. */
    void Swap(SyncInfoPool& other) noexcept
    {
        std::swap(pool_, other.pool_);
        std::swap(pos_, other.pos_);
    }

    void Reset() const noexcept
    {
        pos_ = pool_.cbegin();
    }

    ///@}
    /** @name Access */
    ///@{

    /** @brief Access the next SyncInfo object */
    SyncInfo<D> const& Next() const
    {
        if (!this->Size())
            throw std::runtime_error(
                "SyncInfoPool: Cannot call Next() on empty pool.");

        // Handle circular condition
        if ((++pos_) == pool_.cend())
            pos_ = pool_.cbegin();

        return *pos_;
    }

    // auto begin() const { return pool_.begin(); }
    // auto end() const { return pool_.end(); }

    // auto cbegin() const { return pool_.cbegin(); }
    // auto cend() const { return pool_.cend(); }

    ///@}

private:

    /** @brief The underlying storage for the SyncInfo objects.
     *
     *  The pool can only grow. It cannot shrink.
     */
    container_type pool_;

    /** @brief The current position in the circular array.
     *
     *  This iterator is *INVALID* for empty pools. For nonempty
     *  pools, it is always valid with respect to the underlying
     *  storage.
     *
     *  If, when growing the size of the pool, the pool is ever
     *  reallocated, this is guaranteed to point to the same element
     *  as before the reallocation. The same guarantee is made after
     *  moving or swapping the pool.
     */
    mutable iterator_type pos_;

};// class SyncInfoPool

template <Device D>
void swap(SyncInfoPool<D>& a, SyncInfoPool<D>& b) noexcept
{
    a.Swap(b);
}

// CPU Implementation

// TODO

// GPU Implementation

template <>
SyncInfoPool<Device::GPU>::~SyncInfoPool()
{
#ifdef HYDROGEN_HAVE_CUDA
    using GPUErrorType = CUDAError;
#elif defined(HYDROGEN_HAVE_ROCM)
    using GPUErrorType = HIPError;
#endif
    try
    {
        for (auto& si : pool_)
        {
            DestroySyncInfo(si);
        }
    }
    catch (GPUErrorType const& e)
    {
        std::cerr << "Warning: GPU runtime error detected:\n\ne.what(): "
                  << e.what()
                  << std::endl;
    }
}

template <>
void SyncInfoPool<Device::GPU>::EnsureSize(size_t pool_size)
{
    if (pool_size <= this->Size())
        return;

    // Take care for reallocation:
    auto const initial_offset =
        Size() == 0UL ? 0UL : std::distance(pool_.cbegin(), pos_);

    pool_.reserve(pool_size);
    size_t const start_size = this->Size();
    size_t const new_elements = pool_size - start_size;
    for (auto ii = 0UL; ii < new_elements; ++ii)
    {
        auto si = CreateNewSyncInfo<Device::GPU>();
#ifdef HYDROGEN_HAVE_NVPROF
        // Name the stream for debugging purposes
        std::string const stream_name
            = "H: SP (" + std::to_string(start_size + ii) + ")";
        nvtxNameCudaStreamA(si.Stream(), stream_name.c_str());
#endif // HYDROGEN_HAVE_NVPROF
        pool_.emplace_back(std::move(si));
    }

    // Handle iterators:
    pos_ = pool_.cbegin() + initial_offset;
}

}// namespace hydrogen
#endif // HYDROGEN_SYNCINFOPOOL_HPP_
