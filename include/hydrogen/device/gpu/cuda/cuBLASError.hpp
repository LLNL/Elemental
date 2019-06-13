#ifndef HYDROGEN_DEVICE_GPU_CUDA_CUBLASERROR_HPP_
#define HYDROGEN_DEVICE_GPU_CUDA_CUBLASERROR_HPP_

#include <El/hydrogen_config.h>

#include <hydrogen/Error.hpp>
#include <hydrogen/device/gpu/GPUError.hpp>

#include <cublas_v2.h>

// Helper error-checking macro.
#define H_CHECK_CUBLAS(cmd)                                             \
    do                                                                  \
    {                                                                   \
        H_SYNC_CUDA();                                                  \
        auto h_check_cublas_err_code__ = cmd;                           \
        H_ASSERT(h_check_cublas_err_code__ == CUBLAS_STATUS_SUCCESS,    \
                 cuBLASError,                                           \
                 (cudaDeviceReset(),                                    \
                  cublas::BuildcuBLASErrorMessage(                      \
                      #cmd,                                             \
                      h_check_cublas_err_code__)));                     \
        H_SYNC_CUDA();                                                  \
    } while (false)

namespace hydrogen
{

/** @class cuBLASError
 *  @brief Exception representing errors detected by cuBLAS library.
 */
H_ADD_BASIC_EXCEPTION_CLASS(cuBLASError,GPUError);

namespace cublas
{

/** @brief Write an error message describing the error detected in CUDA.
 *  @param[in] cmd The expression that raised the error.
 *  @param[in] error_code The error code reported by CUDA.
 *  @returns A string describing the error.
 */
std::string BuildcuBLASErrorMessage(
    std::string const& cmd, cublasStatus_t error_code);

}// namespace cublas

}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_CUDA_CUBLASERROR_HPP_
