#ifndef HYDROGEN_DEVICE_GPU_ROCM_ROCSOLVERERROR_HPP_
#define HYDROGEN_DEVICE_GPU_ROCM_ROCSOLVERERROR_HPP_

#include "rocBLASError.hpp"

// Helper error-checking macro. Since the statuses are just rocBLAS
// statuses, just use that macro. Yes, we could get a more precise
// error class, e.g., "rocSOLVERError", but the command is still
// printed verbatim, so any user will see that the rocSOLVER call is
// causing the error. Errrr detecting the error...
#define H_CHECK_ROCSOLVER(cmd)                                          \
    H_CHECK_ROCBLAS(cmd)

#endif // HYDROGEN_DEVICE_GPU_ROCM_ROCSOLVERERROR_HPP_
