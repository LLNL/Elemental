#ifndef HYDROGEN_DEVICE_GPU_ROCM_ROCBLASERROR_HPP_
#define HYDROGEN_DEVICE_GPU_ROCM_ROCBLASERROR_HPP_

#include <El/hydrogen_config.h>

#include <hydrogen/Error.hpp>
#include <hydrogen/device/gpu/GPUError.hpp>

#include <rocblas/rocblas.h>

// Helper error-checking macro.
#define H_CHECK_ROCBLAS(cmd)                                            \
    do                                                                  \
    {                                                                   \
        H_SYNC_HIP();                                                   \
        auto h_check_rocblas_err_code__ = cmd;                          \
        H_ASSERT(h_check_rocblas_err_code__ == rocblas_status_success,  \
                 rocBLASError,                                          \
                 rocblas::BuildrocBLASErrorMessage(                     \
                     #cmd,                                              \
                     h_check_rocblas_err_code__));                      \
        H_SYNC_HIP();                                                   \
    } while (false)

namespace hydrogen
{

/** @class rocBLASError
 *  @brief Exception representing errors detected by rocBLAS library.
 */
H_ADD_BASIC_EXCEPTION_CLASS(rocBLASError,GPUError);

namespace rocblas
{

/** @brief Write an error message describing the error detected in rocBLAS.
 *  @param[in] cmd The expression that raised the error.
 *  @param[in] error_code The error code reported by rocBLAS.
 *  @returns A string describing the error.
 */
std::string BuildrocBLASErrorMessage(
    std::string const& cmd, rocblas_status error_code);

}// namespace rocblas

}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_ROCM_ROCBLASERROR_HPP_
