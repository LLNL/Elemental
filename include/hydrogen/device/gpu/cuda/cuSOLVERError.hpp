#ifndef HYDROGEN_DEVICE_GPU_CUDA_CUSOLVERERROR_HPP_
#define HYDROGEN_DEVICE_GPU_CUDA_CUSOLVERERROR_HPP_

#include <El/hydrogen_config.h>

#include <hydrogen/Error.hpp>
#include <hydrogen/device/gpu/GPUError.hpp>

#include <cusolverDn.h>

// Helper error-checking macro.
#define H_CHECK_CUSOLVER(cmd)                                           \
    do                                                                  \
    {                                                                   \
        H_SYNC_CUDA();                                                  \
        auto h_check_cusolver_err_code__ = cmd;                           \
        H_ASSERT(h_check_cusolver_err_code__ == CUSOLVER_STATUS_SUCCESS,  \
                 cuSOLVERError,                                         \
                 (cudaDeviceReset(),                                    \
                  cusolver::BuildcuSOLVERErrorMessage(                    \
                      #cmd,                                             \
                      h_check_cusolver_err_code__)));                     \
        H_SYNC_CUDA();                                                  \
    } while (false)

namespace hydrogen
{

/** @class cuSOLVERError
 *  @brief Exception representing errors detected by cuSOLVER library.
 */
H_ADD_BASIC_EXCEPTION_CLASS(cuSOLVERError,GPUError);

namespace cusolver
{

/** @brief Write an error message describing the error detected in CUDA.
 *  @param[in] cmd The expression that raised the error.
 *  @param[in] error_code The error code reported by CUDA.
 *  @returns A string describing the error.
 */
std::string BuildcuSOLVERErrorMessage(
    std::string const& cmd, cusolverStatus_t error_code);

}// namespace cusolver

}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_CUDA_CUSOLVERERROR_HPP_
