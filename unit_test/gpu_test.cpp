#include <catch2/catch.hpp>

#include <hydrogen/device/GPU.hpp>

TEST_CASE("Testing core GPU functionality", "[seq][gpu][init]")
{
    REQUIRE_FALSE(hydrogen::gpu::IsInitialized());
    REQUIRE(hydrogen::gpu::IsFinalized());

    REQUIRE_NOTHROW(hydrogen::gpu::Initialize());

    REQUIRE(hydrogen::gpu::IsInitialized());
    REQUIRE_FALSE(hydrogen::gpu::IsFinalized());

    REQUIRE_NOTHROW(hydrogen::gpu::Finalize());

    REQUIRE_FALSE(hydrogen::gpu::IsInitialized());
    REQUIRE(hydrogen::gpu::IsFinalized());
}
