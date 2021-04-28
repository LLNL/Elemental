#include <hydrogen/device/gpu/cuda/cuSOLVER.hpp>

// Helper macro for converting enums to strings.
#define H_ADD_CUSOLVER_ENUM_TO_STRING_CASE(enum_value) \
    case enum_value:                        \
        return #enum_value

namespace
{

std::string GetcuSOLVERErrorString(cusolverStatus_t status)
{
    switch (status)
    {
        H_ADD_CUSOLVER_ENUM_TO_STRING_CASE(CUSOLVER_STATUS_SUCCESS);
        H_ADD_CUSOLVER_ENUM_TO_STRING_CASE(CUSOLVER_STATUS_NOT_INITIALIZED);
        H_ADD_CUSOLVER_ENUM_TO_STRING_CASE(CUSOLVER_STATUS_ALLOC_FAILED);
        H_ADD_CUSOLVER_ENUM_TO_STRING_CASE(CUSOLVER_STATUS_INVALID_VALUE);
        H_ADD_CUSOLVER_ENUM_TO_STRING_CASE(CUSOLVER_STATUS_ARCH_MISMATCH);
        H_ADD_CUSOLVER_ENUM_TO_STRING_CASE(CUSOLVER_STATUS_EXECUTION_FAILED);
        H_ADD_CUSOLVER_ENUM_TO_STRING_CASE(CUSOLVER_STATUS_INTERNAL_ERROR);
        H_ADD_CUSOLVER_ENUM_TO_STRING_CASE(
            CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    default:
        return "Unknown cuSOLVER error.";
    }
}
}// namespace <anon>

namespace hydrogen
{
namespace cusolver
{
namespace // <anon2>
{
bool cusolver_dense_is_initialized_ = false;
cusolverDnHandle_t default_cusolver_dense_handle_;
}// namespace <anon2>

cusolverDnHandle_t GetDenseLibraryHandle() noexcept
{
    return default_cusolver_dense_handle_;
}

bool IsDenseInitialized() noexcept
{
    return cusolver_dense_is_initialized_;
}

void InitializeDense(cusolverDnHandle_t handle)
{
    if (!IsDenseInitialized())
    {
        if (!handle)
            H_CHECK_CUSOLVER(
                cusolverDnCreate(&default_cusolver_dense_handle_));
        else
            default_cusolver_dense_handle_ = handle;

        H_CHECK_CUSOLVER(
            cusolverDnSetStream(
                GetDenseLibraryHandle(), cuda::GetDefaultStream()));

        cusolver_dense_is_initialized_ = true;
    }
}

void FinalizeDense()
{
    if (default_cusolver_dense_handle_)
        H_CHECK_CUSOLVER(cusolverDnDestroy(default_cusolver_dense_handle_));
    default_cusolver_dense_handle_ = nullptr;
    cusolver_dense_is_initialized_ = false;
}

void ReplaceDenseLibraryHandle(cusolverDnHandle_t handle)
{
    H_ASSERT_FALSE(handle == nullptr,
                   std::logic_error,
                   "hydrogen::cusolver::ReplaceDenseLibraryHandle(): "
                   "Detected a null cuSOLVER dense handle.");

    H_ASSERT(IsDenseInitialized(),
             std::logic_error,
             "hydrogen::cusolver::ReplaceDenseLibraryHandle(): "
             "cuSOLVER Dense must be initialized "
             "before calling this function.");

    if (default_cusolver_dense_handle_)
        H_CHECK_CUSOLVER(cusolverDnDestroy(default_cusolver_dense_handle_));
    default_cusolver_dense_handle_ = handle;
}

SyncManager::SyncManager(cusolverDnHandle_t handle,
                         SyncInfo<Device::GPU> const& si)
{
    H_CHECK_CUSOLVER(
        cusolverDnGetStream(handle, &orig_stream_));
    H_CHECK_CUSOLVER(
        cusolverDnSetStream(handle, si.Stream()));
}

SyncManager::~SyncManager()
{
    try
    {
        H_CHECK_CUSOLVER(
            cusolverDnSetStream(
                GetDenseLibraryHandle(), orig_stream_));
    }
    catch (std::exception const& e)
    {
        H_REPORT_DTOR_EXCEPTION_AND_TERMINATE(e);
    }
}

std::string BuildcuSOLVERErrorMessage(
    std::string const& cmd, cusolverStatus_t error_code)
{
    std::ostringstream oss;
    oss << "cuSOLVER error detected in command: \"" << cmd << "\"\n\n"
        << "    Error Code: " << error_code << "\n"
        << "    Error Name: " << GetcuSOLVERErrorString(error_code);
    return oss.str();
}

}// namespace cusolver
}// namespace hydrogen
