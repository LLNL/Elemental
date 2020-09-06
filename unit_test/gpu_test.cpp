#include <catch2/catch.hpp>

#include <hydrogen/device/GPU.hpp>

TEST_CASE("Testing core GPU functionality", "[seq][gpu][init]")
{
    // The "main()" function should handle initialization.
    REQUIRE(hydrogen::gpu::IsInitialized());
    REQUIRE_FALSE(hydrogen::gpu::IsFinalized());
}
