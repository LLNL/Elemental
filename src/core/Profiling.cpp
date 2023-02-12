#include <array>
#include <atomic>
#include <stdexcept>

#include "El/hydrogen_config.h"
#include "El/core/Profiling.hpp"

#ifdef HYDROGEN_HAVE_NVPROF
#include "nvToolsExt.h"
#include "nvToolsExtCuda.h"
#include "nvToolsExtCudaRt.h"
#include "cuda_runtime.h"
#endif // HYDROGEN_HAVE_NVPROF

#ifdef HYDROGEN_HAVE_ROCTRACER
#include "roctx.h"
#endif

#ifdef HYDROGEN_HAVE_VTUNE
#include <ittnotify.h>
#endif // HYDROGEN_HAVE_VTUNE

namespace El
{

namespace
{

// Some global data members to help with colors
constexpr size_t num_prof_colors = 20;
std::array<Color, num_prof_colors> list_of_colors =
{
    Color::ROYAL_BLUE,   Color::HARLEY_DAVIDSON_ORANGE,
    Color::ORANGE_PEEL,  Color::FOREST_GREEN,
    Color::DARK_MAGENTA, Color::GOVERNOR_BAY,
    Color::PACIFIC_BLUE, Color::CRANBERRY,
    Color::CHRISTI,      Color::FIRE_BRICK,
    Color::LOCHMARA,     Color::VIOLET_BLUE,
    Color::NIAGARA,      Color::CITRUS,
    Color::PURPLE_HEART, Color::MANGO_TANGO,
    Color::DARK_RED,     Color::EUCALYPTUS,
    Color::SAN_MARINO,   Color::GOVERNOR_BAY
};

std::atomic<size_t> current_color;

// Some handles/etc for VTune
#ifdef HYDROGEN_HAVE_VTUNE
__itt_domain* vtune_domain = __itt_domain_create("Profiling.Hydrogen");
std::unordered_map<std::string, __itt_string_handle*> vtune_handles;
bool vtune_runtime_enabled = true;

bool VTuneRuntimeEnabled() noexcept { return vtune_runtime_enabled; }
__itt_domain* GetVTuneDomain() noexcept { return vtune_domain; }
#endif // HYDROGEN_HAVE_VTUNE

// Some variables for NVPROF
#ifdef HYDROGEN_HAVE_NVPROF
bool nvprof_runtime_enabled = true;

bool NVProfRuntimeEnabled() noexcept { return nvprof_runtime_enabled; }
#endif

// Some variables for roctx
#ifdef HYDROGEN_HAVE_ROCTRACER
bool roctx_runtime_enabled = true;

bool roctxRuntimeEnabled() noexcept { return roctx_runtime_enabled; }
#endif

}// namespace <anon>

void EnableVTune() noexcept
{
#ifdef HYDROGEN_HAVE_VTUNE
    vtune_runtime_enabled = true;
#endif // HYDROGEN_HAVE_VTUNE
}

void EnableNVProf() noexcept
{
#ifdef HYDROGEN_HAVE_NVPROF
    nvprof_runtime_enabled = true;
#endif // HYDROGEN_HAVE_NVPROF
}

void EnableROCTX() noexcept
{
#ifdef HYDROGEN_HAVE_ROCTRACER
    roctx_runtime_enabled = true;
#endif // HYDROGEN_HAVE_ROCTRACER
}

void DisableVTune() noexcept
{
#ifdef HYDROGEN_HAVE_VTUNE
    vtune_runtime_enabled = false;
#endif // HYDROGEN_HAVE_VTUNE
}

void DisableNVProf() noexcept
{
#ifdef HYDROGEN_HAVE_NVPROF
    nvprof_runtime_enabled = false;
#endif // HYDROGEN_HAVE_NVPROF
}

void DisableROCTX() noexcept
{
#ifdef HYDROGEN_HAVE_ROCTRACER
    roctx_runtime_enabled = false;
#endif // HYDROGEN_HAVE_ROCTRACER
}

Color GetNextProfilingColor() noexcept
{
    auto id = current_color.fetch_add(1, std::memory_order_relaxed);
    return list_of_colors[id % num_prof_colors];
}

void BeginRegionProfile(char const* s, Color c) noexcept
{
#ifdef HYDROGEN_HAVE_ROCTRACER
    if (roctxRuntimeEnabled())
        roctxRangePush(s);
#endif // HYDROGEN_HAVE_ROCTRACER

#ifdef HYDROGEN_HAVE_NVPROF
    if (NVProfRuntimeEnabled())
    {
        // Doesn't work with gcc 4.9
        // nvtxEventAttributes_t ev = {0};
        nvtxEventAttributes_t ev;
        memset(&ev, 0, sizeof(nvtxEventAttributes_t));
        ev.version = NVTX_VERSION;
        ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        ev.colorType = NVTX_COLOR_ARGB;
        ev.color = ColorToInt(c);
        ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
        ev.message.ascii = s;
        nvtxRangePushEx(&ev);
    }
#endif // HYDROGEN_HAVE_NVPROF

#ifdef HYDROGEN_HAVE_VTUNE
    if (VTuneRuntimeEnabled())
    {
        std::string str(s);
        auto& handle = vtune_handles[str];
        if (!handle)
            handle = __itt_string_handle_create(s);

        __itt_task_begin(GetVTuneDomain(), __itt_null, __itt_null, handle);
    }
#endif // HYDROGEN_HAVE_VTUNE

    // Just so there are no nasty compiler warnings
    (void) s;
    (void) c;
}

void EndRegionProfile(const char *) noexcept
{
#ifdef HYDROGEN_HAVE_ROCTRACER
    if (roctxRuntimeEnabled())
        roctxRangePop();
#endif // HYDROGEN_HAVE_ROCTRACER

#ifdef HYDROGEN_HAVE_NVPROF
    if (NVProfRuntimeEnabled())
        nvtxRangePop();
#endif // HYDROGEN_HAVE_NVPROF

#ifdef HYDROGEN_HAVE_VTUNE
    if (VTuneRuntimeEnabled())
        __itt_task_end(GetVTuneDomain());
#endif // HYDROGEN_HAVE_VTUNE
}
} // namespace El
