#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include <El/hydrogen_config.h>
#ifdef HYDROGEN_HAVE_GPU
#include <hydrogen/device/GPU.hpp>
#endif // HYDROGEN_HAVE_GPU

int main(int argc, char* argv[])
{
#ifdef HYDROGEN_HAVE_GPU
    hydrogen::gpu::Initialize();
#endif // HYDROGEN_HAVE_GPU

    int result = Catch::Session().run(argc, argv);

#ifdef HYDROGEN_HAVE_GPU
    hydrogen::gpu::Finalize();
#endif // HYDROGEN_HAVE_GPU
    return result;
}
