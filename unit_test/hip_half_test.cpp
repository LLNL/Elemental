#include <catch2/catch.hpp>

#include <hydrogen/meta/MetaUtilities.hpp>
#include <hydrogen/device/GPU.hpp>

#include <hip/hip_fp16.h>

namespace hydrogen
{

class HipGPUHalf
{
    __half val_;

public:
    template <typename T, EnableWhen<std::is_integral<T>, int> = 0>
    HipGPUHalf(T val)
        : val_{val}
    {}
    template <typename T, EnableWhen<std::is_floating_point<T>, int> = 0>
    HipGPUHalf(T val)
        : val_{val}
    {}
};

}

using gpu_half_type = hydrogen::HipGPUHalf;

// Test static properties here. Just use static_assert, since it's
// generally sufficient.

// Construct by integral types
static_assert(std::is_constructible<gpu_half_type, int>::value,
              "gpu_half_type should be constructible by an int.");
static_assert(std::is_constructible<gpu_half_type, long>::value,
              "gpu_half_type should be constructible by an long.");
static_assert(std::is_constructible<gpu_half_type, size_t>::value,
              "gpu_half_type should be constructible by an size_t.");

// Construct by IEEE-754 floating point types
static_assert(std::is_constructible<gpu_half_type, float>::value,
              "gpu_half_type should be constructible by a float.");
static_assert(std::is_constructible<gpu_half_type, double>::value,
              "gpu_half_type should be constructible by a double.");

// Now runtime testing here.

TEST_CASE("Testing HipGPUHalf operations", "[seq][gpu][half][hip]")
{
    // Add SECTIONs here.
}
