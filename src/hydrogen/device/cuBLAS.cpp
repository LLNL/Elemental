#include <hydrogen/device/gpu/cuda/cuBLAS.hpp>
#include <hydrogen/device/gpu/cuda/cuSOLVER.hpp>

// Helper macro for converting enums to strings.
#define H_ADD_CUBLAS_ENUM_TO_STRING_CASE(enum_value) \
    case enum_value:                        \
        return #enum_value


namespace
{

std::string GetcuBLASErrorString(cublasStatus_t status)
{
    switch (status)
    {
        H_ADD_CUBLAS_ENUM_TO_STRING_CASE(CUBLAS_STATUS_SUCCESS);
        H_ADD_CUBLAS_ENUM_TO_STRING_CASE(CUBLAS_STATUS_NOT_INITIALIZED);
        H_ADD_CUBLAS_ENUM_TO_STRING_CASE(CUBLAS_STATUS_ALLOC_FAILED);
        H_ADD_CUBLAS_ENUM_TO_STRING_CASE(CUBLAS_STATUS_INVALID_VALUE);
        H_ADD_CUBLAS_ENUM_TO_STRING_CASE(CUBLAS_STATUS_ARCH_MISMATCH);
        H_ADD_CUBLAS_ENUM_TO_STRING_CASE(CUBLAS_STATUS_MAPPING_ERROR);
        H_ADD_CUBLAS_ENUM_TO_STRING_CASE(CUBLAS_STATUS_EXECUTION_FAILED);
        H_ADD_CUBLAS_ENUM_TO_STRING_CASE(CUBLAS_STATUS_INTERNAL_ERROR);
        H_ADD_CUBLAS_ENUM_TO_STRING_CASE(CUBLAS_STATUS_NOT_SUPPORTED);
        H_ADD_CUBLAS_ENUM_TO_STRING_CASE(CUBLAS_STATUS_LICENSE_ERROR);
    default:
        return "Unknown cuBLAS error.";
    }
}

}

namespace hydrogen
{
namespace cublas
{
namespace // <anon2>
{
bool cublas_is_initialized_ = false;
cublasHandle_t default_cublas_handle_;
}// namespace <anon2>

cublasHandle_t GetLibraryHandle() noexcept
{
    return default_cublas_handle_;
}

bool IsInitialized() noexcept
{
    return cublas_is_initialized_;
}

void Initialize(cublasHandle_t handle)
{
    if (!IsInitialized())
    {
        if (!handle)
            H_CHECK_CUBLAS(cublasCreate(&default_cublas_handle_));
        else
            default_cublas_handle_ = handle;

        H_CHECK_CUBLAS(
            cublasSetStream(
                GetLibraryHandle(), cuda::GetDefaultStream()));
        H_CHECK_CUBLAS(
            cublasSetPointerMode(
                GetLibraryHandle(), CUBLAS_POINTER_MODE_HOST));
#ifdef HYDROGEN_GPU_USE_TENSOR_OP_MATH
        H_CHECK_CUBLAS(
            cublasSetMathMode(GetLibraryHandle(),CUBLAS_TENSOR_OP_MATH));
#else
        H_CHECK_CUBLAS(
            cublasSetMathMode(GetLibraryHandle(),CUBLAS_DEFAULT_MATH));
#endif // HYDROGEN_GPU_USE_TENSOR_OP_MATH

        cublas_is_initialized_ = true;

        // At this moment in time, cuSOLVER support in Hydrogen should
        // be viewed as inseparable from cuBLAS support. This ensures
        // that the library gets initialized. If cuSOLVER support
        // expands beyond the one function currently supported, we
        // should move this somewhere people will actually see it...
        cusolver::InitializeDense();
    }
}

void Finalize()
{
    if (default_cublas_handle_)
        H_CHECK_CUBLAS(cublasDestroy(default_cublas_handle_));
    default_cublas_handle_ = nullptr;
    cublas_is_initialized_ = false;
}

void ReplaceLibraryHandle(cublasHandle_t handle)
{
    H_ASSERT_FALSE(handle == nullptr,
                   std::logic_error,
                   "hydrogen::cublas::ReplaceLibraryHandle(): "
                   "Detected a null cuBLAS handle.");

    H_ASSERT(IsInitialized(),
             std::logic_error,
             "hydrogen::cublas::ReplaceLibraryHandle(): "
             "cuBLAS must be initialized before calling this function.");

    if (default_cublas_handle_)
        H_CHECK_CUBLAS(cublasDestroy(default_cublas_handle_));
    default_cublas_handle_ = handle;
}

SyncManager::SyncManager(cublasHandle_t handle,
                         SyncInfo<Device::GPU> const& si)
{
    H_CHECK_CUBLAS(
        cublasGetStream(handle, &orig_stream_));
    H_CHECK_CUBLAS(
        cublasSetStream(handle, si.Stream()));
}

SyncManager::~SyncManager()
{
    try
    {
        H_CHECK_CUBLAS(
            cublasSetStream(
                GetLibraryHandle(), orig_stream_));
    }
    catch (std::exception const& e)
    {
        H_REPORT_DTOR_EXCEPTION_AND_TERMINATE(e);
    }
}

std::string BuildcuBLASErrorMessage(
    std::string const& cmd, cublasStatus_t error_code)
{
    std::ostringstream oss;
    oss << "cuBLAS error detected in command: \"" << cmd << "\"\n\n"
        << "    Error Code: " << error_code << "\n"
        << "    Error Name: " << GetcuBLASErrorString(error_code);
    return oss.str();
}

}// namespace cublas

namespace gpu_blas
{
void SetPointerMode(PointerMode mode)
{
    H_CHECK_CUBLAS(
        cublasSetPointerMode(cublas::GetLibraryHandle(),
                             (mode == PointerMode::HOST
                              ? CUBLAS_POINTER_MODE_HOST
                              : CUBLAS_POINTER_MODE_DEVICE)));
}
void RequestTensorOperations()
{
    H_CHECK_CUBLAS(
        cublasSetMathMode(cublas::GetLibraryHandle(),
                          CUBLAS_TENSOR_OP_MATH));
}

}// namespace gpu_blas
}// namespace hydrogen
