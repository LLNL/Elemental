#ifndef HYDROGEN_DEVICE_GPU_CUDA_CUBLASMANAGEMENT_HPP_
#define HYDROGEN_DEVICE_GPU_CUDA_CUBLASMANAGEMENT_HPP_

#include <El/hydrogen_config.h>

#include "cuBLASError.hpp"

#include <hydrogen/Device.hpp>
#include <hydrogen/SyncInfo.hpp>

#include <cublas_v2.h>

namespace hydrogen
{

namespace cublas
{

/** @name cuBLAS management functions. */
///@{

/** @brief Initialize cuBLAS.
 *
 *  Creates the default library instance for cuBLAS.
 *
 *  @note This must be called after `MPI_Init` is called with
 *  MVAPICH2-GDR. cuBLAS initialization allocates some device memory
 *  chunks, which MVAPICH-GDR attempts to intercept but fails if
 *  MPI_Init is not called yet. So, the correct ordering of
 *  initialization seems to be first CUDA, then MPI, and then any
 *  libraries that depend on CUDA or MPI.
 *
 *  \param[in] handle The handle to use for cuBLAS. If null, a new
 *                    handle will be created. If not null, it is
 *                    assumed that the handle has been created with a
 *                    user-side call to cublasCreate().
 */
void Initialize(cublasHandle_t handle=nullptr);

/** @brief Finalize the cuBLAS library.
 *
 *  Destroys the default library handle.
 *
 *  \throws cuBLASError If the cuBLAS library detects any errors.
 */
void Finalize();

/** @brief Replace the default cuBLAS library handle.
 *
 *  This will destroy the current default cuBLAS library handle and
 *  assume control of the input handle. The cuBLAS library must be
 *  initialized in order to call this function.
 *
 *  \param[in] handle The new library handle. Hydrogen will take
 *                    ownership of the new handle and destroy it in
 *                    Finalize().
 *
 *  \throws std::logic_error If the input handle is null or the
 *                           library isn't initialized.
 */
void ReplaceLibraryHandle(cublasHandle_t handle);

/** @brief Get the cuBLAS library handle. */
cublasHandle_t GetLibraryHandle() noexcept;

/** @class SyncManager
 *  @brief Manage stream synchronization within cuBLAS.
 */
class SyncManager
{
public:
    SyncManager(cublasHandle_t handle, SyncInfo<Device::GPU> const& si);
    ~SyncManager();
private:
    cudaStream_t orig_stream_;
};// class SyncManager

///@}

}// namespace cublas
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_CUDA_CUBLASMANAGEMENT_HPP_
