// MUST include this
#include <catch2/catch.hpp>

// Other includes
#include <El.hpp>
#include <hydrogen/meta/TypeTraits.hpp>

// File being tested
#include "SyncInfoPool.hpp"

using namespace hydrogen;

TEST_CASE(
    "Testing SyncInfo pool","[tdd][gemm][util]")
{

    SECTION("A default-constructed SyncInfo pool is empty")
    {
        SyncInfoPool<Device::GPU> pool;
        CHECK(pool.Size() == 0UL);

        SECTION("Calling Next() on an empty pool is an error")
        {
            CHECK_THROWS(pool.Next());
        }
    }

    SECTION("A size-constructed pool has the given size")
    {
        SyncInfoPool<Device::GPU> pool(4UL);
        CHECK(pool.Size() == 4UL);

        SECTION("Ensuring a smaller size has no effect")
        {
            pool.EnsureSize(2UL);
            CHECK(pool.Size() == 4UL);
        }

        SECTION("Ensuring the current size has no effect")
        {
            pool.EnsureSize(pool.Size());
            CHECK(pool.Size() == 4UL);
        }

        SECTION("Ensuring a larger size grows the pool")
        {
            pool.EnsureSize(6UL);
            CHECK(pool.Size() == 6UL);
        }

        SECTION("Two pools can be swapped")
        {
            SyncInfoPool<Device::GPU> pool2(2UL);
            pool.Swap(pool2);
            CHECK(pool.Size() == 2UL);
            CHECK(pool2.Size() == 4UL);
        }

        SECTION("The pool is circular")
        {
            std::vector<SyncInfo<Device::GPU>> tmp(4UL);
            for (auto& si : tmp)
                si = pool.Next();

            for (auto const& si : tmp)
            {
                auto const& pool_si = pool.Next();
                CHECK(si.Stream() == pool_si.Stream());
                CHECK(si.Event() == pool_si.Event());
            }

            SECTION("Moving the pool preserves iterators")
            {
                // Move the iterator one slot
                pool.Next();

                // Move-construct a new pool
                SyncInfoPool<Device::GPU> pool_mv(std::move(pool));

                auto const& pool_si = pool_mv.Next();
                auto const& si = tmp[1];
                CHECK(si.Stream() == pool_si.Stream());
                CHECK(si.Event() == pool_si.Event());
            }

            SECTION("Growing the pool preserves iterators")
            {
                // Move the iterator one slot
                pool.Next();

                // Grow the pool
                pool.EnsureSize(6UL);

                REQUIRE(pool.Size() == 6UL);

                auto const& pool_si = pool.Next();
                auto const& si = tmp[1];
                CHECK(si.Stream() == pool_si.Stream());
                CHECK(si.Event() == pool_si.Event());
            }
        }
        SECTION("Resetting the pool returns to the same point")
        {
            auto const& first = pool.Next();
            pool.Reset();
            auto const& after_reset = pool.Next();
            CHECK(first.Event() == after_reset.Event());
            CHECK(first.Stream() == after_reset.Stream());
        }
    }

    SECTION("A pool can be move-constructed")
    {
        SyncInfoPool<Device::GPU> pool(2UL);
        SyncInfoPool<Device::GPU> pool_mv(std::move(pool));

        CHECK(pool.Size() == 0UL);
        CHECK(pool_mv.Size() == 2UL);

        SECTION("The moved-from pool is reusable")
        {
            pool.EnsureSize(3UL);
            CHECK(pool.Size() == 3UL);
            CHECK_NOTHROW(pool.Next());
        }
    }

}
