#ifndef HYDROGEN_DEVICE_GPU_ROCM_ROCBLASMANAGEMENT_HPP_
#define HYDROGEN_DEVICE_GPU_ROCM_ROCBLASMANAGEMENT_HPP_

#include "rocBLASError.hpp"

#include <hydrogen/Device.hpp>
#include <hydrogen/SyncInfo.hpp>

#include <hip/hip_runtime.h>
#include <rocblas.h>

namespace hydrogen
{
namespace rocblas
{

/** @name rocBLAS management functions. */
///@{

/** @brief Initialize rocBLAS.
 *
 *  Creates the default library instance for rocBLAS.
 *
 *  \param[in] handle The handle to use for rocBLAS. If null, a new
 *                    handle will be created. If not null, it is
 *                    assumed that the handle has been created with a
 *                    user-side call to rocblas_create_handle().
 */
void Initialize(rocblas_handle handle=nullptr);

/** @brief Finalize the rocBLAS library.
 *
 *  Destroys the default library handle.
 *
 *  \throws rocBLASError If the rocBLAS library detects any errors.
 */
void Finalize();

/** @brief Replace the default rocBLAS library handle.
 *
 *  This will destroy the current default rocBLAS library handle and
 *  assume control of the input handle. The rocBLAS library must be
 *  initialized in order to call this function.
 *
 *  \param[in] handle The new library handle. Hydrogen will take
 *                    ownership of the new handle and destroy it in
 *                    Finalize().
 *
 *  \throws std::logic_error If the input handle is null or the
 *                           library isn't initialized.
 */
void ReplaceLibraryHandle(rocblas_handle handle);

/** @brief Get the rocBLAS library handle. */
rocblas_handle GetLibraryHandle() noexcept;

/** @class SyncManager
 *  @brief Manage stream synchronization within rocBLAS.
 */
class SyncManager
{
public:
    SyncManager(rocblas_handle handle, SyncInfo<Device::GPU> const& si);
    ~SyncManager();
private:
    hipStream_t orig_stream_;
};// class SyncManager

///@}

}// namespace rocblas
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_ROCM_ROCBLASMANAGEMENT_HPP_
