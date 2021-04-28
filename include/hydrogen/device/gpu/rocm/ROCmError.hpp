#ifndef HYDROGEN_DEVICE_GPU_ROCMERROR_HPP_
#define HYDROGEN_DEVICE_GPU_ROCMERROR_HPP_

#include <El/hydrogen_config.h>

#include <hip/hip_runtime.h>

#include <hydrogen/device/gpu/GPUError.hpp>

#ifdef HYDROGEN_GPU_CALLS_ARE_SYNCHRONOUS
#define H_SYNC_HIP() hipDeviceSynchronize()
#else
#define H_SYNC_HIP()
#endif

// Error handling macro
#define H_CHECK_HIP(cmd)                                                \
    do                                                                  \
    {                                                                   \
        H_SYNC_HIP();                                                   \
        auto h_check_hip_error_code__ = cmd;                            \
        H_ASSERT(h_check_hip_error_code__ == hipSuccess,                \
                 ::hydrogen::HIPError,                                  \
                 ::hydrogen::rocm::BuildHipErrorMessage(                \
                     #cmd, h_check_hip_error_code__));                  \
        H_SYNC_HIP();                                                   \
    } while (false)

namespace hydrogen
{

/** @class HipError
 *  @brief Exception class describing an error in the HIP environment
 */
H_ADD_BASIC_EXCEPTION_CLASS(HIPError, GPUError);

namespace rocm
{

/** @brief Write an error message describing the error detected in HIP.
 *  @param[in] cmd The expression that raised the error.
 *  @param[in] hipError_T The error code reported by HIP.
 *  @return A string describing the error.
 */
std::string BuildHipErrorMessage(
    std::string const& cmd, hipError_t error_code);

}// namespace rocm
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_ROCMERROR_HPP_
