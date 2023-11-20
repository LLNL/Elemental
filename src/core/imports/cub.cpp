#include "hydrogen/device/gpu/CUB.hpp"

#include <memory>
#include <set>
#include <sstream>

namespace hydrogen {
namespace cub {
namespace {

unsigned int get_env_uint(char const *env_var_name,
                          unsigned int default_value = 0U) noexcept {
  char const *env = std::getenv(env_var_name);
  return (env ? static_cast<unsigned>(std::stoi(env)) : default_value);
}

unsigned int get_bin_growth() noexcept {
  return get_env_uint("H_MEMPOOL_BIN_GROWTH", 2U);
}

unsigned int get_min_bin() noexcept {
  return get_env_uint("H_MEMPOOL_MIN_BIN", 1U);
}

unsigned int get_max_bin() noexcept {
  return get_env_uint("H_MEMPOOL_MAX_BIN", PooledDeviceAllocator::INVALID_BIN);
}

size_t get_max_cached_size() noexcept {
  char const *env = std::getenv("H_MEMPOOL_MAX_CACHED_SIZE");
  return (env ? static_cast<size_t>(std::stoul(env))
              : PooledDeviceAllocator::INVALID_SIZE);
}

size_t get_max_bin_alloc_size() noexcept {
  char const *env = std::getenv("H_MEMPOOL_MAX_BIN_ALLOC_SIZE");
  return (env ? static_cast<size_t>(std::stoul(env))
              : PooledDeviceAllocator::INVALID_SIZE);
}

unsigned int get_bin_mult_threshold() noexcept {
  return get_env_uint("H_MEMPOOL_BIN_MULT_THRESHOLD",
                      PooledDeviceAllocator::INVALID_BIN);
}

unsigned int get_bin_mult() noexcept {
  return get_env_uint("H_MEMPOOL_BIN_MULT", PooledDeviceAllocator::INVALID_BIN);
}

bool get_debug() noexcept {
  char const *env = std::getenv("H_MEMPOOL_DEBUG");
  return (env ? static_cast<bool>(std::stoi(env)) : false);
}

bool get_malloc_async() noexcept {
  char const *env = std::getenv("H_MEMPOOL_MALLOCASYNC");
  return (env ? static_cast<bool>(std::stoi(env)) : false);
}

std::set<size_t> get_bin_sizes() noexcept {
  std::set<size_t> result;
  char const *env = std::getenv("H_MEMPOOL_BIN_SIZES");
  if (!env)
    return result;

  std::string envstr{env};
  std::istringstream iss{envstr};
  std::string elem;
  while(std::getline(iss, elem, ',')) {
    result.insert(std::stoull(elem));
  }
  return result;
}

/** Singleton instance of CUB memory pool. */
std::unique_ptr<PooledDeviceAllocator> memoryPool_;
} // namespace

PooledDeviceAllocator &MemoryPool() {
  if (!memoryPool_)
    memoryPool_.reset(new PooledDeviceAllocator(
        get_bin_growth(), get_min_bin(), get_max_bin(), get_max_cached_size(),
        /*skip_cleanup=*/false, get_debug(), get_bin_mult_threshold(),
        get_bin_mult(), get_max_bin_alloc_size(), get_bin_sizes(),
        get_malloc_async()));

  return *memoryPool_;
}

void DestroyMemoryPool() { memoryPool_.reset(); }

} // namespace cub
} // namespace hydrogen
