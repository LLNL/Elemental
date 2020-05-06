#ifndef HYDROGEN_DEVICE_GPU_HPP_
#define HYDROGEN_DEVICE_GPU_HPP_

/** @defgroup gpu_mgmt GPU device interaction and management
 *
 *  These functions provide a runtime-agnostic API for basic
 *  interaction with GPUs. The exposed functionality is deliberately
 *  quite basic and represents the functions needed for Hydrogen.
 */

#include <El/hydrogen_config.h>

#include <hydrogen/Device.hpp>
#include <hydrogen/SyncInfo.hpp>

#include <stdexcept>

namespace hydrogen
{

/** @namespace gpu
 *  @brief Interface functions for interacting with the GPU.
 *
 *  This is basically a "backended" system where the backends are
 *  mutually exclusive and therefore largely hidden from view. At time
 *  of writing, the backends are CUDA and ROCm/HIP. This will be
 *  determined at configure time based on user-input configure options
 *  and/or system interrogation.
 *
 *  @note Since HIP is a compatibility layer, it should be
 *  possible to just universally use HIP. However, we wish to allow
 *  the two backends to evolve independently. Thus, it should
 *  theoretically be possible to just universally use the HIP
 *  backend. However, in the current implementation, CUDA-specific
 *  optimizations will be lost if compiling under HIP (as they will
 *  likely be protected by "HYDROGEN_HAVE_CUDA", which will not be
 *  defined in this case).
 */
namespace gpu
{

/** @name Environment management */
///@{

/** @brief Initialize the GPU driver and runtime.
 *
 *  This incorporates anything that needs to be done before kernels
 *  can be dispatched to the GPU. In CUDA terms, this establishes a
 *  CUDA context.
 *
 *  @ingroup gpu_mgmt
 */
void Initialize();

/** @brief Cleanup and shutdown any GPU driver/runtime state.
 *
 *  This performs any tasks that are required to close the GPU
 *  environment and leave a clean state.
 *
 *  @ingroup gpu_mgmt
 */
void Finalize();

/** @brief Query if the GPU environment is initialized.
 *  @ingroup gpu_mgmt
 */
bool IsInitialized() noexcept;

/** @brief Query if the GPU environment is finalized.
 *
 *  Finalized means "not initialized", so an environment that has
 *  never been initialized is, in this sense, "finalized".
 *
 *  @ingroup gpu_mgmt
 */
inline bool IsFinalized() noexcept { return !IsInitialized(); }

///@}
/** @name Device management */
///@{

/** @brief Get the number of GPUs visible to this process.
 *  @throws GPUError If the runtime detects any errors.
 *  @ingroup gpu_mgmt
 */
size_t DeviceCount();

/** @brief Get the ID of the currently selected GPU.
 *  @throws GPUError If the runtime detects any errors.
 *  @ingroup gpu_mgmt
 */
int CurrentDevice();

/** @brief Get the ID of the default GPU.
 *  @throws GPUError If the runtime detects any errors.
 *  @ingroup gpu_mgmt
 */
int DefaultDevice();

/** @brief Get the device ID we should be using.
 *  @details This uses environment variables set by most MPI libraries
 *      and/or launchers (slurm,lsf) to determine a device ID. Devices
 *      are assigned round-robin based on local rank.
 *  @param[in] device_count Number of visible devices.
 *  @ingroup gpu_mgmt
 */
int ComputeDeviceId(unsigned int device_count) noexcept;

/** @brief Select the given device.
 *
 *  @param[in] device_id The ID of the device to select. Must be less
 *                       than the number of available GPUs.
 *
 *  @throws GPUError If the runtime detects any errors.
 *  @ingroup gpu_mgmt
 */
void SetDevice(int device_id);

/** @brief Block the host until all device execution has completed.
 *  @throws GPUError If the runtime detects any errors.
 *  @ingroup gpu_mgmt
 */
void SynchronizeDevice();

///@}
/** @name Execution control */
///@{

/** @brief Get the default SyncInfo object for this session.
 *
 *  Note that Hydrogen will use this SyncInfo by default. On CUDA
 *  platforms, for example, it will be different from the "default
 *  CUDA stream".
 *
 *  This SyncInfo object will persist for as long as
 *  IsInitialized(). Note that if the GPU environment is finalized and
 *  reinitialized, this SyncInfo object in the new environment may
 *  differ from the previous environment.
 *
 *  @throws GPUError If the runtime detects any errors.
 *
 *  @ingroup gpu_mgmt
 */
SyncInfo<Device::GPU> const& DefaultSyncInfo() noexcept;

///@}

}// namespace gpu

/** @name SyncInfo management */
///@{

/** @brief Create a new CPU SyncInfo object. */
template <>
SyncInfo<Device::GPU> CreateNewSyncInfo<Device::GPU>();

/** @brief Destroy the GPU SyncInfo. */
void DestroySyncInfo(SyncInfo<Device::GPU>&);

///@}

}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_HPP_
