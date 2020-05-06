#include <hydrogen/device/gpu/cuda/CUB.hpp>

#include <memory>

namespace hydrogen
{
namespace cub
{
namespace
{

unsigned int get_env_uint(char const* env_var_name,
                          unsigned int default_value = 0U) noexcept
{
    char const* env = std::getenv(env_var_name);
    return (env
            ? static_cast<unsigned>(std::stoi(env))
            : default_value);
}

unsigned int get_bin_growth() noexcept
{
    return get_env_uint("H_CUB_BIN_GROWTH", 2U);
}

unsigned int get_min_bin() noexcept
{
    return get_env_uint("H_CUB_MIN_BIN", 1U);
}

unsigned int get_max_bin() noexcept
{
    return get_env_uint("H_CUB_MAX_BIN",
                        cub_impl::CachingDeviceAllocator::INVALID_BIN);
}

size_t get_max_cached_size() noexcept
{
    char const* env = std::getenv("H_CUB_MAX_CACHED_SIZE");
    return (env
            ? static_cast<size_t>(std::stoul(env))
            : cub_impl::CachingDeviceAllocator::INVALID_SIZE);
}

bool get_debug() noexcept
{
    char const* env = std::getenv("H_CUB_DEBUG");
    return (env
            ? static_cast<bool>(std::stoi(env))
            : false);
}

/** Singleton instance of CUB memory pool. */
std::unique_ptr<cub_impl::CachingDeviceAllocator> memoryPool_;
} // namespace <anon>

cub_impl::CachingDeviceAllocator& MemoryPool()
{
    if (!memoryPool_)
        memoryPool_.reset(
            new cub_impl::CachingDeviceAllocator(
                get_bin_growth(),
                get_min_bin(),
                get_max_bin(),
                get_max_cached_size(),
                /*skip_cleanup=*/false,
                get_debug()));
    return *memoryPool_;
}

void DestroyMemoryPool()
{ memoryPool_.reset(); }

} // namespace CUBMemoryPool
} // namespace hydrogen
