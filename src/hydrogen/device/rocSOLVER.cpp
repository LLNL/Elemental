#include <hydrogen/device/gpu/rocm/rocSOLVER.hpp>

namespace hydrogen
{
namespace rocsolver
{
namespace // <anon>
{
bool rocsolver_is_initialized_ = false;
}// namespace <anon>

rocblas_handle GetDenseLibraryHandle() noexcept
{
    return rocblas::GetLibraryHandle();
}

bool IsDenseInitialized() noexcept
{
    return rocsolver_is_initialized_;
}

void InitializeDense(rocblas_handle handle)
{
    rocblas::Initialize(handle);
    rocsolver_is_initialized_ = true;
}

void FinalizeDense()
{
    rocblas::Finalize();
    rocsolver_is_initialized_ = false;
}

void ReplaceDenseLibraryHandle(rocblas_handle handle)
{
    rocblas::ReplaceLibraryHandle(handle);
}

}// namespace rocsolver
}// namespace hydrogen
