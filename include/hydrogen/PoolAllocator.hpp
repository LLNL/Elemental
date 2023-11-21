// See LICENSE for Hydrogen license. Original license for CUB follows:
/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * Extended asynchronous pooling allocator with exponential, multiplicative, and
 * user-specified bin sizes. This allocator is based on CUB's pooling allocator
 * and can use {cuda,hip}MallocAsync as necessary. It also provides extensive
 * reporting for allocations, bins, and extraneous memory.
 ******************************************************************************/

#ifndef HYDROGEN_POOLALLOCATOR_HPP_
#define HYDROGEN_POOLALLOCATOR_HPP_

#include <cmath>
#include <iostream>
#include <map>
#include <mutex>
#include <set>

#include <El/hydrogen_config.h>

// Set up functions
#ifdef HYDROGEN_HAVE_CUDA
#include <cuda_runtime.h>

#define gpuMallocAsync cudaMallocAsync
#define gpuMalloc cudaMalloc
#define gpuFreeAsync cudaFreeAsync
#define gpuFree cudaFree
#define gpuSetDevice cudaSetDevice
#define gpuGetDevice cudaGetDevice
#define gpuEventQuery cudaEventQuery
#define gpuGetLastError cudaGetLastError
#define gpuEventDestroy cudaEventDestroy
#define gpuEventCreateWithFlags cudaEventCreateWithFlags
#define gpuEventDisableTiming cudaEventDisableTiming
#define gpuEventRecord cudaEventRecord
#define gpuGetErrorString cudaGetErrorString

#define gpuStream_t cudaStream_t
#define gpuEvent_t cudaEvent_t
#define gpuError_t cudaError_t

#define gpuSuccess cudaSuccess
#define gpuErrorNotReady cudaErrorNotReady
#define gpuErrorMemoryAllocation cudaErrorMemoryAllocation

#ifndef GPU_PTX_ARCH
#ifndef __CUDA_ARCH__
#define GPU_PTX_ARCH 0
#else
#define GPU_PTX_ARCH __CUDA_ARCH__
#endif
#endif

#elif defined(HYDROGEN_HAVE_ROCM)
#include <hip/hip_runtime.h>

#define gpuMallocAsync hipMallocAsync
#define gpuMalloc hipMalloc
#define gpuFreeAsync hipFreeAsync
#define gpuFree hipFree
#define gpuSetDevice hipSetDevice
#define gpuGetDevice hipGetDevice
#define gpuEventQuery hipEventQuery
#define gpuGetLastError hipGetLastError
#define gpuEventDestroy hipEventDestroy
#define gpuEventCreateWithFlags hipEventCreateWithFlags
#define gpuEventDisableTiming hipEventDisableTiming
#define gpuEventRecord hipEventRecord
#define gpuGetErrorString hipGetErrorString

#define gpuStream_t hipStream_t
#define gpuEvent_t hipEvent_t
#define gpuError_t hipError_t

#define gpuSuccess hipSuccess
#define gpuErrorNotReady hipErrorNotReady
#define gpuErrorMemoryAllocation hipErrorMemoryAllocation

#ifndef GPU_PTX_ARCH
#ifndef __HIP_ARCH__
#define GPU_PTX_ARCH 0
#else
#define GPU_PTX_ARCH 1
#endif
#endif
#else
#error "This file must be included with a GPU (CUDA/ROCm) environment"
#endif

namespace gpudebug {
/* Minimal copy of CubDebug */
__host__ __device__ __forceinline__ gpuError_t Debug(gpuError_t error,
                                                     const char *filename,
                                                     int line) {
  if (error) {
#if (GPU_PTX_ARCH == 0)
    fprintf(stderr, "GPU error %d [%s, %d]: %s\n", error, filename, line,
            gpuGetErrorString(error));
    fflush(stderr);
#else
    printf("GPU error %d [block (%d,%d,%d) thread (%d,%d,%d), %s, %d]\n",
           error, blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z, threadIdx.y,
           threadIdx.x, filename, line);
#endif
  }
  return error;
}
} // namespace gpudebug

/**
 * \brief Debug macro
 */
#ifndef gpuDebug
#define gpuDebug(e) gpudebug::Debug((gpuError_t)(e), __FILE__, __LINE__)
#endif

/**
 * Prints human-readable size (for reporting)
 */
static inline void HumanReadableSize(size_t bytes, std::ostream &os) {
  const std::string sizes[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  int unit = 0;
  float size = bytes;
  while (size > 1024) {
    size /= 1024;
    ++unit;
  }
  auto oldprec = os.precision(unit > 0 ? 2 : 0);
  auto oldf = os.setf(std::ios_base::fixed, std::ios_base::floatfield);
  os << size << " " << sizes[unit];
  os.precision(oldprec);
  os.setf(oldf);
}

namespace hydrogen {

/******************************************************************************
 * PooledDeviceAllocator
 ******************************************************************************/

/**
 * \brief A simple caching allocator for device memory allocations.
 *
 * \par Overview
 * The allocator is thread-safe and stream-safe and is capable of managing
 * cached device allocations on multiple devices.  It behaves as follows:
 *
 * \par
 * - Allocations from the allocator are associated with an \p active_stream.
 * Once freed, the allocation becomes available immediately for reuse within the
 * \p active_stream with which it was associated with during allocation, and it
 * becomes available for reuse within other streams when all prior work
 * submitted to \p active_stream has completed.
 * - Allocations are categorized and cached by bin size.  A new allocation
 * request of a given size will only consider cached allocations within the
 * corresponding bin.
 * - (EXTENDED) Bin limits have a combined geometric/linear progression; or can
 *   be given as a set of sizes. It behaves as follows:
 *     - If a set of sizes is given in \p bin_sizes, they are used to construct
 *       the allocation bins. If an allocation is larger than the largest bin, the
 *       behavior matches the rest of the algorithm. Allocations in [0, bin_min]
 *       will allocate ``bin_min`` bytes.
 *     - Bin limits progress geometrically in accordance with the (integer)
 *       growth factor \p bin_growth provided during construction. Unused device
 *       allocations within a larger bin cache are not reused for allocation
 *       requests that categorize to smaller bin sizes.
 *       Allocation requests below (\p bin_growth ^ \p min_bin) are rounded up
 * to ( \p bin_growth ^ \p min_bin ).
 *     - If two consecutive geometric bins exceed \p bin_mult_threshold and
 *       \p bin_mult is given, a linear binning scheme is created where bins
 *       follow \p bin_growth ^ some_bin + \p bin_mult * n
 *     - Allocations above min( \p bin_growth ^ \p max_bin , \p
 *       max_bin_alloc_size ) are not rounded up to the nearest bin and are simply
 *       freed when they are deallocated instead of being returned to a bin-cache.
 * - If the total storage of cached allocations on a given device will exceed
 *   \p max_cached_bytes, allocations for that device are simply freed when they
 *   are deallocated instead of being returned to their bin-cache.
 *
 */
struct PooledDeviceAllocator {

  //---------------------------------------------------------------------
  // Constants
  //---------------------------------------------------------------------

  /// Out-of-bounds bin
  static const unsigned int INVALID_BIN = (unsigned int)-1;

  /// Invalid size
  static const size_t INVALID_SIZE = (size_t)-1;

  /// Invalid device ordinal
  static const int INVALID_DEVICE_ORDINAL = -1;

  //---------------------------------------------------------------------
  // Type definitions and helper types
  //---------------------------------------------------------------------

  /**
   * Descriptor for device memory allocations
   */
  struct BlockDescriptor {
    void *d_ptr;                   // Device pointer
    size_t bytes;                  // Size of allocation in bytes
    size_t requested_bytes;        // Size of true allocation in bytes
    bool binned;                   // Whether the block is part of the pool bins
    int device;                    // device ordinal
    gpuStream_t associated_stream; // Associated associated_stream
    gpuEvent_t ready_event; // Signal when associated stream has run to the
                            // point at which this block was freed

    // Constructor (suitable for searching maps for a specific block, given its
    // pointer and device)
    BlockDescriptor(void *d_ptr, int device)
        : d_ptr(d_ptr), bytes(0), requested_bytes(0), binned(false),
          device(device), associated_stream(0), ready_event(0) {}

    // Constructor (suitable for searching maps for a range of suitable blocks,
    // given a device)
    BlockDescriptor(int device)
        : d_ptr(NULL), bytes(0), requested_bytes(0), binned(false),
          device(device), associated_stream(0), ready_event(0) {}

    // Comparison functor for comparing device pointers
    static bool PtrCompare(const BlockDescriptor &a, const BlockDescriptor &b) {
      if (a.device == b.device)
        return (a.d_ptr < b.d_ptr);
      else
        return (a.device < b.device);
    }

    // Comparison functor for comparing allocation sizes
    static bool SizeCompare(const BlockDescriptor &a,
                            const BlockDescriptor &b) {
      if (a.device == b.device)
        return (a.bytes < b.bytes);
      else
        return (a.device < b.device);
    }
  };

  /// BlockDescriptor comparator function interface
  typedef bool (*Compare)(const BlockDescriptor &, const BlockDescriptor &);

  class TotalBytes {
  public:
    size_t free;
    size_t live;
    TotalBytes() { free = live = 0; }
  };

  /// Set type for cached blocks (ordered by size)
  typedef std::multiset<BlockDescriptor, Compare> CachedBlocks;

  /// Set type for live blocks (ordered by ptr)
  typedef std::multiset<BlockDescriptor, Compare> BusyBlocks;

  /// Map type of device ordinals to the number of cached bytes cached by each
  /// device
  typedef std::map<int, TotalBytes> GpuCachedBytes;

  //---------------------------------------------------------------------
  // Utility functions
  //---------------------------------------------------------------------

  /**
   * Integer pow function for unsigned base and exponent
   */
  static unsigned int IntPow(unsigned int base, unsigned int exp) {
    unsigned int retval = 1;
    while (exp > 0) {
      if (exp & 1) {
        retval = retval * base; // multiply the result by the current base
      }
      base = base * base; // square the base
      exp = exp >> 1;     // divide the exponent in half
    }
    return retval;
  }

  /**
   * Round up to the nearest power-of
   */
  void NearestPowerOf(unsigned int &power, size_t &rounded_bytes,
                      unsigned int base, size_t value) {
    power = 0;
    rounded_bytes = 1;

    if (value * base < value) {
      // Overflow
      power = sizeof(size_t) * 8;
      rounded_bytes = size_t(0) - 1;
      return;
    }

    while (rounded_bytes < value) {
      rounded_bytes *= base;
      power++;
    }
  }

  size_t NearestMultOf(unsigned int mult, size_t value) {
    // Ceiling division followed by multiplication
    return ((value + mult - 1) / mult) * mult;
  }

  unsigned int ComputeLinearBinIndex(unsigned int bin_growth,
                                     unsigned int bin_mult_threshold) {
    if (bin_mult_threshold == INVALID_BIN || bin_growth == 0 ||
        bin_mult_threshold == 0)
      return INVALID_BIN;

    return static_cast<unsigned int>(std::log(bin_mult_threshold) /
                                     std::log(bin_growth));
  }

  size_t ComputeMaxBinBytes(unsigned int bin_growth, unsigned int max_bin,
                            size_t max_bin_alloc_size) {
    size_t result = INVALID_SIZE;
    if (max_bin != INVALID_BIN) {
      result = IntPow(bin_growth, max_bin);
    }
    if (max_bin_alloc_size != INVALID_SIZE) {
      result = std::min(result, max_bin_alloc_size);
    }
    return result;
  }

  //---------------------------------------------------------------------
  // Fields
  //---------------------------------------------------------------------

  std::mutex mutex; /// Mutex for thread-safety

  unsigned int bin_growth; /// Geometric growth factor for bin-sizes
  unsigned int min_bin;    /// Minimum bin enumeration
  unsigned int max_bin;    /// Maximum bin enumeration

  // Extensions
  unsigned int bin_mult_threshold; /// Threshold to switch between geometric and
                                   /// linear growth
  unsigned int bin_mult;           /// Linear bin scaling size
  size_t max_bin_alloc_size;       /// Maximal binned allocation size
  std::set<size_t> bin_sizes;      /// Explicit control over bin sizes

  unsigned int linear_bin_index; /// Geometric bin to consider linear binning
                                 /// from (computed)
  size_t min_bin_bytes;          /// Minimum bin size
  size_t max_bin_bytes;          /// Maximum bin size
  size_t max_cached_bytes;       /// Maximum aggregate cached bytes per device

  const bool
      skip_cleanup;  /// Whether or not to skip a call to FreeAllCached() when
                     /// destructor is called.  (The runtime may have already
                     /// shut down for statically declared allocators)
  bool debug;        /// Whether or not to print (de)allocation events to stdout
  bool malloc_async; /// Use {cuda,hip}MallocAsync

  std::set<size_t> actual_bin_sizes; /// Bin sizes used by the allocator
  GpuCachedBytes cached_bytes; /// Map of device ordinal to aggregate cached
                               /// bytes on that device
  CachedBlocks
      cached_blocks; /// Set of cached device allocations available for reuse
  BusyBlocks live_blocks; /// Set of live device allocations currently in use

  //---------------------------------------------------------------------
  // Methods
  //---------------------------------------------------------------------

  /**
   * \brief Constructor.
   */
  PooledDeviceAllocator(
      unsigned int bin_growth,  ///< Geometric growth factor for bin-sizes
      unsigned int min_bin = 1, ///< Minimum bin (default is bin_growth ^ 1)
      unsigned int max_bin =
          INVALID_BIN, ///< Maximum bin (default is no max bin)
      size_t max_cached_bytes =
          INVALID_SIZE, ///< Maximum aggregate cached bytes per device (default
                        ///< is no limit)
      bool skip_cleanup =
          false, ///< Whether or not to skip a call to \p FreeAllCached() when
                 ///< the destructor is called (default is to deallocate)
      bool debug = false, ///< Whether or not to print (de)allocation events to
                          ///< stdout (default is no stderr output)
      unsigned int bin_mult_threshold =
          INVALID_BIN, ///< Threshold to switch between geometric and linear
                       ///< growth
      unsigned int bin_mult = INVALID_BIN, ///< Linear bin scaling size
      size_t max_bin_alloc_size =
          INVALID_SIZE,                ///< Maximal binned allocation size
      std::set<size_t> bin_sizes = {}, ///< Explicit control over bin size
      bool use_malloc_async = false)   ///< Use asynchronous malloc/free calls
      : bin_growth(bin_growth), min_bin(min_bin), max_bin(max_bin),
        bin_mult_threshold(bin_mult_threshold), bin_mult(bin_mult),
        bin_sizes(bin_sizes),
        linear_bin_index(ComputeLinearBinIndex(bin_growth, bin_mult_threshold)),
        min_bin_bytes(IntPow(bin_growth, min_bin)),
        max_bin_bytes(
            ComputeMaxBinBytes(bin_growth, max_bin, max_bin_alloc_size)),
        max_cached_bytes(max_cached_bytes), skip_cleanup(skip_cleanup),
        debug(debug), malloc_async(use_malloc_async),
        cached_blocks(BlockDescriptor::SizeCompare),
        live_blocks(BlockDescriptor::PtrCompare) {}

  /**
   * \brief Default constructor.
   *
   * Configured with:
   * \par
   * - \p bin_growth          = 8
   * - \p min_bin             = 3
   * - \p max_bin             = 7
   * - \p max_cached_bytes    = (\p bin_growth ^ \p max_bin) * 3) - 1 =
   * 6,291,455 bytes
   *
   * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB and
   * sets a maximum of 6,291,455 cached bytes per device
   */
  PooledDeviceAllocator(bool skip_cleanup = false, bool debug = false)
      : bin_growth(8), min_bin(3), max_bin(7), bin_mult_threshold(INVALID_BIN),
        bin_mult(INVALID_BIN), bin_sizes{}, linear_bin_index(INVALID_BIN),
        min_bin_bytes(IntPow(bin_growth, min_bin)),
        max_bin_bytes(IntPow(bin_growth, max_bin)),
        max_cached_bytes((max_bin_bytes * 3) - 1), skip_cleanup(skip_cleanup),
        debug(debug), malloc_async(false),
        cached_blocks(BlockDescriptor::SizeCompare),
        live_blocks(BlockDescriptor::PtrCompare) {}

  /**
   * \brief Sets the limit on the number bytes this allocator is allowed to
   * cache per device.
   *
   * Changing the ceiling of cached bytes does not cause any allocations (in-use
   * or cached-in-reserve) to be freed.  See \p FreeAllCached().
   */
  gpuError_t SetMaxCachedBytes(size_t max_cached_bytes_) {
    // Lock
    mutex.lock();

    if (debug)
      printf("Changing max_cached_bytes (%lld -> %lld)\n",
             (long long)this->max_cached_bytes, (long long)max_cached_bytes_);

    this->max_cached_bytes = max_cached_bytes_;

    // Unlock
    mutex.unlock();

    return gpuSuccess;
  }

  /**
   * \brief Implements the bin-finding algorithm described in the class
   * documentation. Returns true if a bin was found, or false otherwise.
   */
  bool FindBin(BlockDescriptor &search_key) {
    size_t bytes = search_key.requested_bytes;
    search_key.bytes = bytes;

    if (bytes > max_bin_bytes) {
      // Size is greater than our preconfigured maximum: allocate the request
      // exactly and give out-of-bounds bin.  It will not be cached
      // for reuse when returned.
      return false;
    }

    // If a custom bin histogram is given, use that
    auto it = bin_sizes.lower_bound(bytes);
    if (it != bin_sizes.end()) {
      search_key.bytes = *it;
      return true;
    }

    // Find geometric bin
    unsigned int geobin;
    NearestPowerOf(geobin, search_key.bytes, bin_growth, bytes);
    // Minimum bin
    if (geobin < min_bin) {
      // Bin is less than minimum bin: round up
      search_key.bytes = min_bin_bytes;
      return true;
    }

    // Test for linear binning; if so, find linear bin
    if (linear_bin_index != INVALID_BIN && geobin >= linear_bin_index) {
      search_key.bytes = NearestMultOf(bin_mult, bytes);
      return true;
    }

    // Otherwise, use geometric bin
    if (geobin > max_bin) {
      // Bin is greater than our maximum bin: allocate the request
      // exactly and give out-of-bounds bin.  It will not be cached
      // for reuse when returned.
      return false;
    }

    // search_key.bytes was set above by NearestPowerOf
    return true;
  }

  /**
   * \brief Provides a suitable allocation of device memory for the given size
   * on the specified device.
   *
   * Once freed, the allocation becomes available immediately for reuse within
   * the \p active_stream with which it was associated with during allocation,
   * and it becomes available for reuse within other streams when all prior work
   * submitted to \p active_stream has completed.
   */
  gpuError_t DeviceAllocate(
      int device,   ///< [in] Device on which to place the allocation
      void **d_ptr, ///< [out] Reference to pointer to the allocation
      size_t bytes, ///< [in] Minimum number of bytes for the allocation
      gpuStream_t active_stream =
          0) ///< [in] The stream to be associated with this allocation
  {
    *d_ptr = NULL;
    int entrypoint_device = INVALID_DEVICE_ORDINAL;
    gpuError_t error = gpuSuccess;

    if (device == INVALID_DEVICE_ORDINAL) {
      if (gpuDebug(error = gpuGetDevice(&entrypoint_device)))
        return error;
      device = entrypoint_device;
    }

    // Create a block descriptor for the requested allocation
    bool found = false;
    BlockDescriptor search_key(device);
    search_key.associated_stream = active_stream;
    search_key.requested_bytes = bytes;
    bool binned = FindBin(search_key);
    search_key.binned = binned;

    if (binned) {
      // Search for a suitable cached allocation: lock
      mutex.lock();

      // Add bin size to created bin sizes
      actual_bin_sizes.insert(search_key.bytes);

      // Iterate through the range of cached blocks on the same device in the
      // same bin
      CachedBlocks::iterator block_itr = cached_blocks.lower_bound(search_key);
      while ((block_itr != cached_blocks.end()) &&
             (block_itr->device == device) &&
             (block_itr->bytes == search_key.bytes)) {
        // To prevent races with reusing blocks returned by the host but still
        // in use by the device, only consider cached blocks that are
        // either (from the active stream) or (from an idle stream)
        bool is_reusable = false;
        if (active_stream == block_itr->associated_stream) {
          is_reusable = true;
        } else {
          const gpuError_t event_status = gpuEventQuery(block_itr->ready_event);
          if (event_status != gpuErrorNotReady) {
            gpuDebug(event_status);
            is_reusable = true;
          }
        }

        if (is_reusable) {
          // Reuse existing cache block.  Insert into live blocks.
          found = true;
          search_key = *block_itr;
          search_key.requested_bytes = bytes;
          search_key.associated_stream = active_stream;
          live_blocks.insert(search_key);

          // Remove from free blocks
          cached_bytes[device].free -= search_key.bytes;
          cached_bytes[device].live += search_key.bytes;

          if (debug)
            printf("\tDevice %d reused cached block at %p (%lld bytes) for "
                   "stream %lld (previously associated with stream %lld).\n",
                   device, search_key.d_ptr, (long long)search_key.bytes,
                   (long long)search_key.associated_stream,
                   (long long)block_itr->associated_stream);

          cached_blocks.erase(block_itr);

          break;
        }
        block_itr++;
      }

      // Done searching: unlock
      mutex.unlock();
    }

    // Allocate the block if necessary
    if (!found) {
      // Set runtime's current device to specified device (entrypoint may not be
      // set)
      if (device != entrypoint_device) {
        if (gpuDebug(error = gpuGetDevice(&entrypoint_device)))
          return error;
        if (gpuDebug(error = gpuSetDevice(device)))
          return error;
      }

      // Attempt to allocate
      if (gpuDebug(error = MallocInternal(&search_key.d_ptr, search_key.bytes,
                                          active_stream)) ==
          gpuErrorMemoryAllocation) {
        // The allocation attempt failed: free all cached blocks on device and
        // retry
        if (debug)
          printf("\tDevice %d failed to allocate %lld bytes for stream %lld, "
                 "retrying after freeing cached allocations",
                 device, (long long)search_key.bytes,
                 (long long)search_key.associated_stream);

        error = gpuSuccess; // Reset the error we will return
        gpuGetLastError();  // Reset error

        // Lock
        mutex.lock();

        // Iterate the range of free blocks on the same device
        BlockDescriptor free_key(device);
        CachedBlocks::iterator block_itr = cached_blocks.lower_bound(free_key);

        while ((block_itr != cached_blocks.end()) &&
               (block_itr->device == device)) {
          // No need to worry about synchronization with the device: gpuFree is
          // blocking and will synchronize across all kernels executing
          // on the current device

          // Free device memory and destroy stream event.
          if (gpuDebug(error = FreeInternal(block_itr->d_ptr,
                                            block_itr->associated_stream)))
            break;
          if (gpuDebug(error = gpuEventDestroy(block_itr->ready_event)))
            break;

          // Reduce balance and erase entry
          cached_bytes[device].free -= block_itr->bytes;

          if (debug)
            printf("\tDevice %d freed %lld bytes.\n\t\t  %lld available blocks "
                   "cached (%lld bytes), %lld live blocks (%lld bytes) "
                   "outstanding.\n",
                   device, (long long)block_itr->bytes,
                   (long long)cached_blocks.size(),
                   (long long)cached_bytes[device].free,
                   (long long)live_blocks.size(),
                   (long long)cached_bytes[device].live);

          block_itr = cached_blocks.erase(block_itr);
        }

        // Unlock
        mutex.unlock();

        // Return under error
        if (error)
          return error;

        // Try to allocate again
        if (gpuDebug(error = MallocInternal(&search_key.d_ptr, search_key.bytes,
                                            active_stream)))
          return error;
      }

      search_key.associated_stream = active_stream;

      // Create ready event
      if (gpuDebug(error = gpuEventCreateWithFlags(&search_key.ready_event,
                                                   gpuEventDisableTiming)))
        return error;

      // Insert into live blocks
      mutex.lock();
      live_blocks.insert(search_key);
      cached_bytes[device].live += search_key.bytes;
      mutex.unlock();

      if (debug)
        printf("\tDevice %d allocated new device block at %p (%lld bytes "
               "associated with stream %lld).\n",
               device, search_key.d_ptr, (long long)search_key.bytes,
               (long long)search_key.associated_stream);

      // Attempt to revert back to previous device if necessary
      if ((entrypoint_device != INVALID_DEVICE_ORDINAL) &&
          (entrypoint_device != device)) {
        if (gpuDebug(error = gpuSetDevice(entrypoint_device)))
          return error;
      }
    }

    // Copy device pointer to output parameter
    *d_ptr = search_key.d_ptr;

    if (debug)
      printf(
          "\t\t%lld available blocks cached (%lld bytes), %lld live blocks "
          "outstanding(%lld bytes).\n",
          (long long)cached_blocks.size(), (long long)cached_bytes[device].free,
          (long long)live_blocks.size(), (long long)cached_bytes[device].live);

    return error;
  }

  /**
   * \brief Provides a suitable allocation of device memory for the given size
   * on the current device.
   *
   * Once freed, the allocation becomes available immediately for reuse within
   * the \p active_stream with which it was associated with during allocation,
   * and it becomes available for reuse within other streams when all prior work
   * submitted to \p active_stream has completed.
   */
  gpuError_t DeviceAllocate(
      void **d_ptr, ///< [out] Reference to pointer to the allocation
      size_t bytes, ///< [in] Minimum number of bytes for the allocation
      gpuStream_t active_stream =
          0) ///< [in] The stream to be associated with this allocation
  {
    return DeviceAllocate(INVALID_DEVICE_ORDINAL, d_ptr, bytes, active_stream);
  }

  /**
   * \brief Frees a live allocation of device memory on the specified device,
   * returning it to the allocator.
   *
   * Once freed, the allocation becomes available immediately for reuse within
   * the \p active_stream with which it was associated with during allocation,
   * and it becomes available for reuse within other streams when all prior work
   * submitted to \p active_stream has completed.
   */
  gpuError_t DeviceFree(int device, void *d_ptr) {
    int entrypoint_device = INVALID_DEVICE_ORDINAL;
    gpuError_t error = gpuSuccess;

    if (device == INVALID_DEVICE_ORDINAL) {
      if (gpuDebug(error = gpuGetDevice(&entrypoint_device)))
        return error;
      device = entrypoint_device;
    }

    // Lock
    mutex.lock();

    // Find corresponding block descriptor
    bool recached = false;
    BlockDescriptor search_key(d_ptr, device);
    BusyBlocks::iterator block_itr = live_blocks.find(search_key);
    if (block_itr != live_blocks.end()) {
      // Remove from live blocks
      search_key = *block_itr;
      live_blocks.erase(block_itr);
      cached_bytes[device].live -= search_key.bytes;

      // Keep the returned allocation if bin is valid and we won't exceed the
      // max cached threshold
      if (search_key.binned &&
          (cached_bytes[device].free + search_key.bytes <= max_cached_bytes)) {
        // Insert returned allocation into free blocks
        recached = true;
        cached_blocks.insert(search_key);
        cached_bytes[device].free += search_key.bytes;

        if (debug)
          printf("\tDevice %d returned %lld bytes from associated stream "
                 "%lld.\n\t\t %lld available blocks cached (%lld bytes), %lld "
                 "live blocks outstanding. (%lld bytes)\n",
                 device, (long long)search_key.bytes,
                 (long long)search_key.associated_stream,
                 (long long)cached_blocks.size(),
                 (long long)cached_bytes[device].free,
                 (long long)live_blocks.size(),
                 (long long)cached_bytes[device].live);
      }
    }

    // Unlock
    mutex.unlock();

    // First set to specified device (entrypoint may not be set)
    if (device != entrypoint_device) {
      if (gpuDebug(error = gpuGetDevice(&entrypoint_device)))
        return error;
      if (gpuDebug(error = gpuSetDevice(device)))
        return error;
    }

    if (recached) {
      // Insert the ready event in the associated stream (must have current
      // device set properly)
      if (gpuDebug(error = gpuEventRecord(search_key.ready_event,
                                          search_key.associated_stream)))
        return error;
    }

    if (!recached) {
      // Free the allocation from the runtime and cleanup the event.
      if (gpuDebug(error = FreeInternal(d_ptr, search_key.associated_stream)))
        return error;
      if (gpuDebug(error = gpuEventDestroy(search_key.ready_event)))
        return error;

      if (debug)
        printf("\tDevice %d freed %lld bytes from associated stream "
               "%lld.\n\t\t  %lld available blocks cached (%lld bytes), %lld "
               "live blocks (%lld bytes) outstanding.\n",
               device, (long long)search_key.bytes,
               (long long)search_key.associated_stream,
               (long long)cached_blocks.size(),
               (long long)cached_bytes[device].free,
               (long long)live_blocks.size(),
               (long long)cached_bytes[device].live);
    }

    // Reset device
    if ((entrypoint_device != INVALID_DEVICE_ORDINAL) &&
        (entrypoint_device != device)) {
      if (gpuDebug(error = gpuSetDevice(entrypoint_device)))
        return error;
    }

    return error;
  }

  /**
   * \brief Frees a live allocation of device memory on the current device,
   * returning it to the allocator.
   *
   * Once freed, the allocation becomes available immediately for reuse within
   * the \p active_stream with which it was associated with during allocation,
   * and it becomes available for reuse within other streams when all prior work
   * submitted to \p active_stream has completed.
   */
  gpuError_t DeviceFree(void *d_ptr) {
    return DeviceFree(INVALID_DEVICE_ORDINAL, d_ptr);
  }

  /**
   * \brief Frees all cached device allocations on all devices
   */
  gpuError_t FreeAllCached() {
    gpuError_t error = gpuSuccess;
    int entrypoint_device = INVALID_DEVICE_ORDINAL;
    int current_device = INVALID_DEVICE_ORDINAL;

    mutex.lock();

    while (!cached_blocks.empty()) {
      // Get first block
      CachedBlocks::iterator begin = cached_blocks.begin();

      // Get entry-point device ordinal if necessary
      if (entrypoint_device == INVALID_DEVICE_ORDINAL) {
        if (gpuDebug(error = gpuGetDevice(&entrypoint_device)))
          break;
      }

      // Set current device ordinal if necessary
      if (begin->device != current_device) {
        if (gpuDebug(error = gpuSetDevice(begin->device)))
          break;
        current_device = begin->device;
      }

      // Free device memory
      if (gpuDebug(error =
                       FreeInternal(begin->d_ptr, begin->associated_stream)))
        break;
      if (gpuDebug(error = gpuEventDestroy(begin->ready_event)))
        break;

      // Reduce balance and erase entry
      const size_t block_bytes = begin->bytes;
      cached_bytes[current_device].free -= block_bytes;
      cached_blocks.erase(begin);

      if (debug)
        printf(
            "\tDevice %d freed %lld bytes.\n\t\t  %lld available blocks cached "
            "(%lld bytes), %lld live blocks (%lld bytes) outstanding.\n",
            current_device, (long long)block_bytes,
            (long long)cached_blocks.size(),
            (long long)cached_bytes[current_device].free,
            (long long)live_blocks.size(),
            (long long)cached_bytes[current_device].live);
    }

    mutex.unlock();

    // Attempt to revert back to entry-point device if necessary
    if (entrypoint_device != INVALID_DEVICE_ORDINAL) {
      if (gpuDebug(error = gpuSetDevice(entrypoint_device)))
        return error;
    }

    return error;
  }

  /**
   * \brief Destructor
   */
  virtual ~PooledDeviceAllocator() {
    if (!skip_cleanup)
      FreeAllCached();
  }

  /* Inspection and reporting methods */

  size_t TotalAllocatedMemory(int device = INVALID_DEVICE_ORDINAL) const {
    size_t result = 0;
    for (auto const &[dev, totals] : cached_bytes) {
      if (device != INVALID_DEVICE_ORDINAL && device != dev)
        continue;
      result += totals.live + totals.free;
    }
    return result;
  }

  size_t FreeMemory(int device = INVALID_DEVICE_ORDINAL) const {
    size_t result = 0;
    for (auto const &[dev, totals] : cached_bytes) {
      if (device != INVALID_DEVICE_ORDINAL && device != dev)
        continue;
      result += totals.free;
    }
    return result;
  }

  size_t GetBinFreeMemory(int device = INVALID_DEVICE_ORDINAL,
                          size_t bin_size = INVALID_SIZE) const {
    size_t result = 0;
    for (BlockDescriptor const &desc : cached_blocks) {
      if (device != INVALID_DEVICE_ORDINAL && desc.device != device)
        continue;
      if (desc.bytes == bin_size)
        result += desc.bytes;
    }
    return result;
  }

  size_t LiveMemory(int device = INVALID_DEVICE_ORDINAL) const {
    size_t result = 0;
    for (auto const &[dev, totals] : cached_bytes) {
      if (device != INVALID_DEVICE_ORDINAL && device != dev)
        continue;
      result += totals.live;
    }
    return result;
  }

  size_t GetBinLiveMemory(int device = INVALID_DEVICE_ORDINAL,
                          size_t bin_size = INVALID_SIZE) const {
    size_t result = 0;
    for (BlockDescriptor const &desc : live_blocks) {
      if (device != INVALID_DEVICE_ORDINAL && desc.device != device)
        continue;
      if (desc.bytes == bin_size)
        result += desc.bytes;
    }
    return result;
  }

  size_t NonbinnedMemory(int device = INVALID_DEVICE_ORDINAL) const {
    size_t result = 0;
    for (BlockDescriptor const &desc : live_blocks) {
      if (device != INVALID_DEVICE_ORDINAL && desc.device != device)
        continue;
      if (!desc.binned)
        result += desc.bytes;
    }
    return result;
  }

  size_t ExcessMemory(int device = INVALID_DEVICE_ORDINAL,
                      size_t bin_size = INVALID_SIZE) const {
    size_t result = 0;
    for (BlockDescriptor const &desc : live_blocks) {
      if (device != INVALID_DEVICE_ORDINAL && desc.device != device)
        continue;
      if (bin_size != INVALID_SIZE && desc.bytes != bin_size)
        continue;
      result += desc.bytes - desc.requested_bytes;
    }
    return result;
  }

  size_t GetNumBuffers(int device = INVALID_DEVICE_ORDINAL, bool cached = true,
                       bool live = true) const {
    size_t result = 0;
    if (cached) {
      for (BlockDescriptor const &desc : cached_blocks) {
        if (device != INVALID_DEVICE_ORDINAL && desc.device != device)
          continue;
        ++result;
      }
    }
    if (live) {
      for (BlockDescriptor const &desc : live_blocks) {
        if (device != INVALID_DEVICE_ORDINAL && desc.device != device)
          continue;
        ++result;
      }
    }
    return result;
  }

  void Report(std::ostream &os, bool report_bins = true) const {
    os << "Memory pool configuration:" << std::endl;
    os << "  Geometric bins - " << bin_growth << " ^ (" << min_bin << "-"
       << ((max_bin == INVALID_BIN) ? "inf" : std::to_string(max_bin))
       << ")" << std::endl;
    if (bin_mult_threshold == INVALID_BIN) {
      os << "  Linear bins - DISABLED" << std::endl;
    } else {
      os << "  Linear bins - when geometric bin difference > "
         << bin_mult_threshold << ", allocate in multiples of " << bin_mult
         << std::endl;
    }

    if (bin_sizes.size() == 0) {
      os << "  Custom bins - NONE" << std::endl;
    } else {
      os << "  Custom bins - ";
      bool first = true;
      for (auto const &bin : bin_sizes) {
        if (!first)
          os << ", ";
        HumanReadableSize(bin, os);
        first = false;
      }
      os << std::endl;
    }
    os << "  mallocAsync: " << (malloc_async ? "enabled" : "disabled")
       << ", debug: " << (debug ? "enabled" : "disabled")
       << ", skip cleanup: " << (skip_cleanup ? "yes" : "no") << std::endl;

    for (auto const &[dev, totals] : cached_bytes) {
      os << "Memory pool allocation report (Device " << dev
         << "):" << std::endl;
      os << "  Allocated memory: ";
      HumanReadableSize(totals.live + totals.free, os);
      os << " (Live: ";
      HumanReadableSize(totals.live, os);
      os << ", free: ";
      HumanReadableSize(totals.free, os);
      os << "). Buffers: " << GetNumBuffers() << std::endl;
      os << "  Total excess memory due to binning: ";
      HumanReadableSize(ExcessMemory(dev), os);
      os << std::endl;

      if (report_bins) {
        os << "  Detailed bin report:" << std::endl;
        for (auto const &bin : actual_bin_sizes) {
          os << "    ";
          HumanReadableSize(bin, os);
          os << ": Live = ";
          HumanReadableSize(GetBinLiveMemory(dev, bin), os);
          os << ", Free = ";
          HumanReadableSize(GetBinFreeMemory(dev, bin), os);
          os << ", Excess = ";
          HumanReadableSize(ExcessMemory(dev, bin), os);
          os << std::endl;
        }
        os << "    Non-binned: ";
        HumanReadableSize(NonbinnedMemory(dev), os);
        os << std::endl;
      }
    }
  }

private:
  gpuError_t MallocInternal(void **ptr, size_t size,
                            gpuStream_t active_stream) {
    if (malloc_async) {
      return gpuMallocAsync(ptr, size, active_stream);
    } else {
      return gpuMalloc(ptr, size);
    }
  }

  gpuError_t FreeInternal(void *ptr, gpuStream_t active_stream) {
    if (malloc_async) {
      return gpuFreeAsync(ptr, active_stream);
    } else {
      return gpuFree(ptr);
    }
  }
};

} // namespace hydrogen

#undef gpuMallocAsync
#undef gpuMalloc
#undef gpuFreeAsync
#undef gpuFree
#undef gpuSetDevice
#undef gpuGetDevice
#undef gpuEventQuery
#undef gpuGetLastError
#undef gpuEventDestroy
#undef gpuEventCreateWithFlags
#undef gpuEventDisableTiming
#undef gpuEventRecord
#undef gpuGetErrorString
#undef gpuStream_t
#undef gpuEvent_t
#undef gpuError_t
#undef gpuSuccess
#undef gpuErrorNotReady
#undef gpuErrorMemoryAllocation
#undef GPU_ARCH

#endif // HYDROGEN_POOLALLOCATOR_HPP_
