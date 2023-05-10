#ifndef HYDROGEN_DEVICE_GPU_ROCM_ROCSOLVERMANAGEMENT_HPP_
#define HYDROGEN_DEVICE_GPU_ROCM_ROCSOLVERMANAGEMENT_HPP_

#include <El/hydrogen_config.h>

#include "rocBLASManagement.hpp"
#include <rocsolver/rocsolver.h>

namespace hydrogen
{
namespace rocsolver
{

/** @name rocSOLVER management functions. */
///@{

/** @brief Initialize rocSOLVER's dense interface.
 *
 *  Creates the default library instance for rocSolver.
 *
 *  rocSOLVER is interesting because it just recycles objects from
 *  rocBLAS. These are all just pass-throughs to the corresponding
 *  calls in the hydrogen::rocblas interface, and only one
 *  rocblas_handle object is used for both rocBLAS and rocSOLVER.
 *
 *  @note This must be called after `MPI_Init` is called with
 *  MVAPICH2-GDR.
 *
 *  \param[in] handle The handle to use for rocSolver. If null, a new
 *                    handle will be created. If not null, it is
 *                    assumed that the handle has been created with a
 *                    user-side call to rocblas_create_handle().
 */
void InitializeDense(rocblas_handle handle=nullptr);

/** @brief Finalize the rocSolver library.
 *
 *  Destroys the default library handle.
 *
 *  \throws rocSOLVERError If the rocSOLVER library detects any errors.
 */
void FinalizeDense();

/** @brief Replace the default rocSolver library handle.
 *
 *  This will destroy the current default rocSOLVER library handle and
 *  assume control of the input handle. The rocSOLVER library must be
 *  initialized in order to call this function.
 *
 *  \param[in] handle The new library handle. Hydrogen will take
 *                    ownership of the new handle and destroy it in
 *                    Finalize().
 *
 *  \throws std::logic_error If the input handle is null or the
 *                           library isn't initialized.
 */
void ReplaceDenseLibraryHandle(rocblas_handle handle);

/** @brief Get the rocSolver library handle. */
rocblas_handle GetDenseLibraryHandle() noexcept;

/** @class SyncManager
 *  @brief Manage stream synchronization within rocSolver.
 */
using SyncManager = rocblas::SyncManager;

///@}

}// namespace rocsolver
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_ROCM_ROCSOLVERMANAGEMENT_HPP_
