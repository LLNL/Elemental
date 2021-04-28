#include <hydrogen/device/gpu/rocm/rocBLAS.hpp>
#include <hydrogen/device/gpu/rocm/rocSOLVER.hpp>

// Helper macro for converting enums to strings.
#define H_ADD_ROCBLAS_ENUM_TO_STRING_CASE(enum_value) \
    case enum_value:                        \
        return #enum_value


namespace
{

std::string GetrocBLASErrorString(rocblas_status status)
{
    switch (status)
    {
        H_ADD_ROCBLAS_ENUM_TO_STRING_CASE(rocblas_status_success);
        H_ADD_ROCBLAS_ENUM_TO_STRING_CASE(rocblas_status_invalid_handle);
        H_ADD_ROCBLAS_ENUM_TO_STRING_CASE(rocblas_status_not_implemented);
        H_ADD_ROCBLAS_ENUM_TO_STRING_CASE(rocblas_status_invalid_pointer);
        H_ADD_ROCBLAS_ENUM_TO_STRING_CASE(rocblas_status_invalid_size);
        H_ADD_ROCBLAS_ENUM_TO_STRING_CASE(rocblas_status_memory_error);
        H_ADD_ROCBLAS_ENUM_TO_STRING_CASE(rocblas_status_internal_error);
    default:
        return "Unknown rocBLAS error.";
    }
}

}

namespace hydrogen
{

namespace rocblas
{
namespace // <anon2>
{
bool rocblas_is_initialized_ = false;
rocblas_handle default_rocblas_handle_;
}// namespace <anon2>

rocblas_handle GetLibraryHandle() noexcept
{
    return default_rocblas_handle_;
}

bool IsInitialized() noexcept
{
    return rocblas_is_initialized_;
}

void Initialize(rocblas_handle handle)
{
    if (!IsInitialized())
    {
        rocblas_initialize();

        if (!handle)
            H_CHECK_ROCBLAS(rocblas_create_handle(&default_rocblas_handle_));
        else
            default_rocblas_handle_ = handle;

        H_CHECK_ROCBLAS(
            rocblas_set_stream(
                GetLibraryHandle(), rocm::GetDefaultStream()));
        H_CHECK_ROCBLAS(
            rocblas_set_pointer_mode(
                GetLibraryHandle(), rocblas_pointer_mode_host));

        rocblas_is_initialized_ = true;

        rocsolver::InitializeDense();
    }
}

void Finalize()
{
    if (default_rocblas_handle_)
        H_CHECK_ROCBLAS(rocblas_destroy_handle(default_rocblas_handle_));
    default_rocblas_handle_ = nullptr;
    rocblas_is_initialized_ = false;
}

void ReplaceLibraryHandle(rocblas_handle handle)
{
    H_ASSERT_FALSE(handle == nullptr,
                   std::logic_error,
                   "hydrogen::rocblas::ReplaceLibraryHandle(): "
                   "Detected a null rocBLAS handle.");

    H_ASSERT(IsInitialized(),
             std::logic_error,
             "hydrogen::rocblas::ReplaceLibraryHandle(): "
             "rocBLAS must be initialized before calling this function.");

    if (default_rocblas_handle_)
        H_CHECK_ROCBLAS(rocblas_destroy_handle(default_rocblas_handle_));
    default_rocblas_handle_ = handle;
}

SyncManager::SyncManager(rocblas_handle handle,
                         SyncInfo<Device::GPU> const& si)
{
    H_CHECK_ROCBLAS(
        rocblas_get_stream(handle, &orig_stream_));
    H_CHECK_ROCBLAS(
        rocblas_set_stream(handle, si.Stream()));
}

SyncManager::~SyncManager()
{
    try
    {
        H_CHECK_ROCBLAS(
            rocblas_set_stream(
                GetLibraryHandle(), orig_stream_));
    }
    catch (std::exception const& e)
    {
        H_REPORT_DTOR_EXCEPTION_AND_TERMINATE(e);
    }
}

std::string BuildrocBLASErrorMessage(
    std::string const& cmd, rocblas_status error_code)
{
    std::ostringstream oss;
    oss << "rocBLAS error detected in command: \"" << cmd << "\"\n\n"
        << "    Error Code: " << error_code << "\n"
        << "    Error Name: " << GetrocBLASErrorString(error_code);
    return oss.str();
}

}// namespace rocblas

namespace gpu_blas
{
void SetPointerMode(PointerMode mode)
{
    H_CHECK_ROCBLAS(
        rocblas_set_pointer_mode(rocblas::GetLibraryHandle(),
                                 (mode == PointerMode::HOST
                                  ? rocblas_pointer_mode_host
                                  : rocblas_pointer_mode_device)));
}
void RequestTensorOperations()
{
// Nothing to do here.
}

}// namespace gpu_blas

}// namespace hydrogen
