#include <catch2/catch.hpp>

#include "El.hpp"
#include "hydrogen/PoolAllocator.hpp"

TEST_CASE("Testing hydrogen::PooledDeviceAllocator", "[memory][utils][gpu]") {
  SECTION("Basic pool behavior and geometric binning") {
    hydrogen::PooledDeviceAllocator alloc{/*bin_growth=*/2,
                                          /*min_bin=*/1, /*max_bin=*/20};
    void *ptr;
    alloc.DeviceAllocate(-1, &ptr, sizeof(float) * 201);
    REQUIRE(alloc.TotalAllocatedMemory() == 256 * sizeof(float));
    CHECK(alloc.ExcessMemory() == 55 * sizeof(float));
    alloc.DeviceFree(ptr);
    CHECK(alloc.TotalAllocatedMemory() == 256 * sizeof(float));
    alloc.DeviceAllocate(-1, &ptr, sizeof(float) * 203);
    CHECK(alloc.TotalAllocatedMemory() == 256 * sizeof(float));
    alloc.DeviceFree(ptr);

    // Unbinned memory
    alloc.DeviceAllocate(-1, &ptr, sizeof(float) * 1048576);
    CHECK(alloc.TotalAllocatedMemory() == (256 + 1048576) * sizeof(float));
    alloc.DeviceFree(ptr);
    CHECK(alloc.TotalAllocatedMemory() == 256 * sizeof(float));
  }
  SECTION("Linear binning test") {
    hydrogen::PooledDeviceAllocator alloc{
        /*bin_growth=*/2,
        /*min_bin=*/1,
        /*max_bin=*/20,
        /*max_cached_size=*/hydrogen::PooledDeviceAllocator::INVALID_SIZE,
        /*skip_cleanup=*/false,
        /*debug=*/false,
        /*bin_mult_threshold=*/1024 * sizeof(float),
        /*bin_mult=*/6,
        /*max_binned_alloc_size=*/4007 * sizeof(float)};

    void *ptr;
    alloc.DeviceAllocate(-1, &ptr, sizeof(float) * 201);
    CHECK(alloc.TotalAllocatedMemory() == 256 * sizeof(float));
    alloc.DeviceFree(ptr);

    alloc.DeviceAllocate(-1, &ptr, sizeof(float) * 3001);
    CHECK(alloc.TotalAllocatedMemory() == 256 * sizeof(float) + 12006);
    alloc.DeviceFree(ptr);

    alloc.DeviceAllocate(-1, &ptr, sizeof(float) * 4009);
    CHECK(alloc.TotalAllocatedMemory() ==
          ((256 + 4009) * sizeof(float) + 12006));
    alloc.DeviceFree(ptr);
    CHECK(alloc.TotalAllocatedMemory() == (256 * sizeof(float) + 12006));
  }
  SECTION("Custom binning test") {
    hydrogen::PooledDeviceAllocator alloc{
        /*bin_growth=*/2,
        /*min_bin=*/1,
        /*max_bin=*/20,
        /*max_cached_size=*/hydrogen::PooledDeviceAllocator::INVALID_SIZE,
        /*skip_cleanup=*/false,
        /*debug=*/false,
        /*bin_mult_threshold=*/hydrogen::PooledDeviceAllocator::INVALID_BIN,
        /*bin_mult=*/hydrogen::PooledDeviceAllocator::INVALID_BIN,
        /*max_cached_size=*/hydrogen::PooledDeviceAllocator::INVALID_SIZE,
        /*bin_sizes=*/{1, 21, 39, 100}};

    void *ptr;
    alloc.DeviceAllocate(-1, &ptr, 32);
    CHECK(alloc.TotalAllocatedMemory() == 39);
    alloc.DeviceFree(ptr);

    alloc.DeviceAllocate(-1, &ptr, 201);
    CHECK(alloc.TotalAllocatedMemory() == (39 + 256));
    alloc.DeviceFree(ptr);
  }
}
