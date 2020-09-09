#ifndef HYDROGEN_DEVICE_GPU_CUDA_CUSOLVERMANAGEMENT_HPP_
#define HYDROGEN_DEVICE_GPU_CUDA_CUSOLVERMANAGEMENT_HPP_

#include <El/hydrogen_config.h>

#include "cuSOLVERError.hpp"

#include <hydrogen/Device.hpp>
#include <hydrogen/SyncInfo.hpp>

#include <cusolverDn.h>

namespace hydrogen
{

namespace cusolver
{

/** @name cuSOLVER management functions. */
///@{

/** @brief Initialize cuSOLVER's dense interface.
 *
 *  Creates the default library instance for cuSolverDN.
 *
 *  @note This must be called after `MPI_Init` is called with
 *  MVAPICH2-GDR.
 *
 *  \param[in] handle The handle to use for cuSolverDN. If null, a new
 *                    handle will be created. If not null, it is
 *                    assumed that the handle has been created with a
 *                    user-side call to cusolverDnCreate().
 */
void InitializeDense(cusolverDnHandle_t handle=nullptr);

/** @brief Finalize the cuSolverDn library.
 *
 *  Destroys the default library handle.
 *
 *  \throws cuSOLVERError If the cuSOLVER library detects any errors.
 */
void FinalizeDense();

/** @brief Replace the default cuSolverDn library handle.
 *
 *  This will destroy the current default cuSOLVER library handle and
 *  assume control of the input handle. The cuSOLVER library must be
 *  initialized in order to call this function.
 *
 *  \param[in] handle The new library handle. Hydrogen will take
 *                    ownership of the new handle and destroy it in
 *                    Finalize().
 *
 *  \throws std::logic_error If the input handle is null or the
 *                           library isn't initialized.
 */
void ReplaceDenseLibraryHandle(cusolverDnHandle_t handle);

/** @brief Get the cuSolverDn library handle. */
cusolverDnHandle_t GetDenseLibraryHandle() noexcept;

/** @class SyncManager
 *  @brief Manage stream synchronization within cuSolverDn.
 */
class SyncManager
{
public:
    SyncManager(cusolverDnHandle_t handle, SyncInfo<Device::GPU> const& si);
    ~SyncManager();
private:
    cudaStream_t orig_stream_;
};// class SyncManager

///@}

}// namespace cusolver
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_CUDA_CUSOLVERMANAGEMENT_HPP_
