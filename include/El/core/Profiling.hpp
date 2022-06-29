#pragma once
#ifndef EL_CORE_PROFILING_HPP_
#define EL_CORE_PROFILING_HPP_

#include <string>

#include "El-lite.hpp"
#include "hydrogen/Device.hpp"
#include "hydrogen/SyncInfo.hpp"

namespace El
{

// These are no-ops if VTune is not enabled at compile time
void EnableVTune() noexcept;
void DisableVTune() noexcept;

// These are no-ops if NVProf is not enabled at compile time
void EnableNVProf() noexcept;
void DisableNVProf() noexcept;

// These are no-ops if roctracer is not enabled at compile time
void EnableROCTX() noexcept;
void DisableROCTX() noexcept;

/** \brief A selection of colors to use with the profiling interface.
 *
 *  It seems unlikely that a user will ever need to access these by
 *  name, but being the good strongly-typed C++ people that we are, it
 *  seemed like the thing to do.
 */
enum class Color
{
    INVALID=0x7FFFFFFF,
    ROYAL_BLUE=0x3366CC,
    HARLEY_DAVIDSON_ORANGE=0xDC3912,
    ORANGE_PEEL=0xFF9900,
    FOREST_GREEN=0x109618,
    DARK_MAGENTA=0x990099,
    GOVERNOR_BAY=0x3B3EAC,
    PACIFIC_BLUE=0x0099C6,
    CRANBERRY=0xDD4477,
    CHRISTI=0x66AA00,
    FIRE_BRICK=0xB82E2E,
    LOCHMARA=0x316395,
    VIOLET_BLUE=0x994499,
    NIAGARA=0x22AA99,
    CITRUS=0xAAAA11,
    PURPLE_HEART=0x6633CC,
    MANGO_TANGO=0xE67300,
    DARK_RED=0x8B0707,
    EUCALYPTUS=0x329262,
    SAN_MARINO=0x5574A6
};// enum class Color

inline Color IntToColor(int c)
{
    switch (c)
    {
    case 0x3366CC: return Color::ROYAL_BLUE;
    case 0xDC3912: return Color::HARLEY_DAVIDSON_ORANGE;
    case 0xFF9900: return Color::ORANGE_PEEL;
    case 0x109618: return Color::FOREST_GREEN;
    case 0x990099: return Color::DARK_MAGENTA;
    case 0x3B3EAC: return Color::GOVERNOR_BAY;
    case 0x0099C6: return Color::PACIFIC_BLUE;
    case 0xDD4477: return Color::CRANBERRY;
    case 0x66AA00: return Color::CHRISTI;
    case 0xB82E2E: return Color::FIRE_BRICK;
    case 0x316395: return Color::LOCHMARA;
    case 0x994499: return Color::VIOLET_BLUE;
    case 0x22AA99: return Color::NIAGARA;
    case 0xAAAA11: return Color::CITRUS;
    case 0x6633CC: return Color::PURPLE_HEART;
    case 0xE67300: return Color::MANGO_TANGO;
    case 0x8B0707: return Color::DARK_RED;
    case 0x329262: return Color::EUCALYPTUS;
    case 0x5574A6: return Color::SAN_MARINO;
    }
    return Color::INVALID;
}

inline int ColorToInt(Color c)
{
    return static_cast<int>(c);
}

inline std::string ColorToString(Color c)
{
    switch (c)
    {
    case Color::INVALID: return "INVALID";
    case Color::ROYAL_BLUE: return "ROYAL_BLUE";
    case Color::HARLEY_DAVIDSON_ORANGE: return "HARLEY_DAVIDSON_ORANGE";
    case Color::ORANGE_PEEL: return "ORANGE_PEEL";
    case Color::FOREST_GREEN: return "FOREST_GREEN";
    case Color::DARK_MAGENTA: return "DARK_MAGENTA";
    case Color::GOVERNOR_BAY: return "GOVERNOR_BAY";
    case Color::PACIFIC_BLUE: return "PACIFIC_BLUE";
    case Color::CRANBERRY: return "CRANBERRY";
    case Color::CHRISTI: return "CHRISTI";
    case Color::FIRE_BRICK: return "FIRE_BRICK";
    case Color::LOCHMARA: return "LOCHMARA";
    case Color::VIOLET_BLUE: return "VIOLET_BLUE";
    case Color::NIAGARA: return "NIAGARA";
    case Color::CITRUS: return "CITRUS";
    case Color::PURPLE_HEART: return "PURPLE_HEART";
    case Color::MANGO_TANGO: return "MANGO_TANGO";
    case Color::DARK_RED: return "DARK_RED";
    case Color::EUCALYPTUS: return "EUCALYPTUS";
    case Color::SAN_MARINO: return "SAN_MARINO";
    }
    return "UNKNOWN";
}

/** \brief Circularly loops through a list of colors returning a new
 *      color each call.
 *
 *  While not guaranteed to remain this way, the is currently 20
 *  colors, 19 unique with one repeat. The list is here:
 *  http://there4.io/2012/05/02/google-chart-color-list/
 *
 *  The current color is currently an atomic size_t with relaxed
 *  memory ordering, so multiple threads could see the same value. I
 *  don't see this as an issue since the colors are just sugar on a
 *  visualization tool.
 *
 *  \return A color.
*/

Color GetNextProfilingColor() noexcept;

/** \brief Begin a profiling region.
 *
 *  \param desc A name or description for the profiling region.
 *  \param color The color for the region.
 */
void BeginRegionProfile(char const *desc, Color color) noexcept;

/** \brief End a profiling region
 *
 *  \param desc The name or description for the profiling region to be
 *      stopped. Must match the one given when the region was started.
 */
void EndRegionProfile(const char *desc) noexcept;

/** \brief Synchronize and begin a profiling region
 *
 *  \param desc A name or description for the profiling region.
 *  \param color The color for the region.
 *  \param si The SyncInfo object to be synchronized.
 */
template <Device D>
void BeginRegionProfile(
    char const *desc, Color color, SyncInfo<D> const& si) noexcept
{
    Synchronize(si);
    BeginRegionProfile(desc, color);
}

/** \brief Syncrhonize and end a profiling region
 *
 *  \param desc The name or description for the profiling region to be
 *      stopped. Must match the one given when the region was started.
 *  \param si The SyncInfo object to be synchronized.
 */
template <Device D>
void EndRegionProfile(const char* desc, SyncInfo<D> const& si) noexcept
{
    Synchronize(si);
    EndRegionProfile(desc);
}

/** \class ProfileRegion
 *  \brief RAII profiler.
 *
 *  This wrapper class annotates the region between the construction
 *  and the destruction of the object. No thread/stream/device/etc
 *  synchronization is performed.
 */
struct ProfileRegion
{
    ProfileRegion(std::string desc, Color color) noexcept
        : desc_{std::move(desc)}
    {
        BeginRegionProfile(desc_.c_str(), color);
    }
    ~ProfileRegion() noexcept
    {
        if (desc_.size())
            EndRegionProfile(desc_.c_str());
    }

    // Disable copy
    ProfileRegion(ProfileRegion const&) = delete;
    ProfileRegion& operator=(ProfileRegion const&) = delete;

    // Allow move -- allows the Make function to work
    ProfileRegion(ProfileRegion&&) noexcept = default;
    ProfileRegion& operator=(ProfileRegion&&) = default;

    std::string desc_;
};// struct ProfileRegion

inline ProfileRegion MakeProfileRegion(std::string desc, Color color)
{
    return ProfileRegion(std::move(desc), color);
}

/** \class SyncProfileRegion
 *  \brief Synchronous RAII profiler.
 *
 *  This wrapper class synchronizes the SyncInfo object at
 *  construction and destruction. It annotates the region between the
 *  construction and the destruction of the object.
 */
template <Device D>
struct SyncProfileRegion
{
    SyncProfileRegion(std::string desc, Color color, SyncInfo<D> si) noexcept
        : desc_{std::move(desc)}, si_{std::move(si)}
    {
        BeginRegionProfile(desc_.c_str(), color, si_);
    }

    ~SyncProfileRegion() noexcept
    {
        if (desc_.size())
            EndRegionProfile(desc_.c_str(), si_);
    }

    // Allow move -- allows the Make function to work
    SyncProfileRegion(SyncProfileRegion&&) noexcept = default;
    SyncProfileRegion& operator=(SyncProfileRegion&&) noexcept = default;

    // Disable copy
    SyncProfileRegion(SyncProfileRegion const&) = delete;
    SyncProfileRegion& operator=(SyncProfileRegion const&) = delete;

    std::string desc_;
    SyncInfo<D> si_;
};// struct SyncProfileRegion

template <Device D>
auto MakeSyncProfileRegion(
    std::string desc, Color color, SyncInfo<D> si) noexcept
    -> SyncProfileRegion<D>
{
    return SyncProfileRegion<D>(std::move(desc), color, std::move(si));
}


#define AUTO_NOSYNC_PROFILE_REGION(description)                         \
    auto region_nosync_profiler__ =                                     \
        MakeProfileRegion(description, GetNextProfilingColor())

#define AUTO_SYNC_PROFILE_REGION(description,syncinfo)                  \
    auto region_sync_profiler__ =                                       \
        MakeSyncProfileRegion(                                          \
            description, GetNextProfilingColor(), syncinfo)

#ifdef HYDROGEN_DEFAULT_SYNC_PROFILING
#define AUTO_PROFILE_REGION(description, syncinfo) \
    AUTO_SYNC_PROFILE_REGION(description, syncinfo)
#else
#define AUTO_PROFILE_REGION(description, syncinfo) \
    AUTO_NOSYNC_PROFILE_REGION(description)
#endif

}// namespace El
#endif /* EL_CORE_PROFILING_HPP_ */
