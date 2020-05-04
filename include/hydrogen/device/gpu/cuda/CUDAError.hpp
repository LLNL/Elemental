#ifndef HYDROGEN_DEVICE_GPU_CUDAERROR_HPP_
#define HYDROGEN_DEVICE_GPU_CUDAERROR_HPP_

#include <El/hydrogen_config.h>

#include <cuda_runtime_api.h>

#include <hydrogen/device/gpu/GPUError.hpp>

#ifdef HYDROGEN_GPU_CALLS_ARE_SYNCHRONOUS
#define H_SYNC_CUDA() cudaDeviceSynchronize()
#else
#define H_SYNC_CUDA()
#endif

// Error handling macro
#define H_CHECK_CUDA(cmd)                                               \
    do                                                                  \
    {                                                                   \
        H_SYNC_CUDA();                                                  \
        auto h_check_cuda_error_code__ = cmd;                           \
        H_ASSERT(h_check_cuda_error_code__ == cudaSuccess,              \
                 ::hydrogen::CUDAError,                                 \
                 (cudaDeviceReset(),                                    \
                  ::hydrogen::cuda::BuildCUDAErrorMessage(              \
                      #cmd, h_check_cuda_error_code__)));               \
        H_SYNC_CUDA();                                                  \
    } while (false)

namespace hydrogen
{

/** @class CUDAError
 *  @brief Exception class representing an error detected by the CUDA
 *         runtime.
 */
H_ADD_BASIC_EXCEPTION_CLASS(CUDAError, GPUError);

namespace cuda
{

/** @brief Write an error message describing the error detected in CUDA.
 *  @param[in] cmd The expression that raised the error.
 *  @param[in] error_code The error code reported by CUDA.
 *  @returns A string describing the error.
 */
std::string BuildCUDAErrorMessage(
    std::string const& cmd, cudaError_t error_code);

}// namespace cuda
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_CUDAERROR_HPP_
