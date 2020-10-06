#ifndef HYDROGEN_DEVICE_GPU_CUDALAUNCHKERNEL_HPP_
#define HYDROGEN_DEVICE_GPU_CUDALAUNCHKERNEL_HPP_

#include <cuda_runtime.h>

#include <hydrogen/Device.hpp>
#include <hydrogen/SyncInfo.hpp>

#include "CUDAError.hpp"

#include <array>
#include <tuple>
#include <utility>

namespace hydrogen
{
namespace gpu
{
namespace details
{

template <typename... Ts, size_t... Is>
auto pack_arguments_impl(std::tuple<Ts...>& args,
                         std::index_sequence<Is...>)
{
    return std::array<void*, sizeof...(Ts)>{{&std::get<Is>(args)...}};
}

template <typename... Ts>
auto pack_arguments(std::tuple<Ts...>& args)
{
    return pack_arguments_impl(args,
                               std::make_index_sequence<sizeof...(Ts)>{});
}

}// namespace details

inline constexpr int Default2DTileSize() { return 32; }

/** @brief Launch the specified kernel.
 *
 *  The implementation has a few oddities and subtleties.
 *
 *   -# The kernel must be by function pointer. std::function
 *      won't work, and it's not clear to me if nvstd::function
 *      would be ok to hold a kernel. But let's not find out.
 *   -# As such, any overloading must be fully specified at the
 *      call site.
 *   -# The variadic argument pack passed to this function does
 *      not need to exactly match the formal kernel arguments,
 *      but they must be implicitly convertible to the formal
 *      kernel arguments, as though assigned as `formal = input`.
 *   -# Kernel arguments will likely be copied; take care if
 *      passing non-POD inputs.
 *
 *  @tparam KernelFormalArgs (Inferred) The formal argument types
 *                                      expected for the kernel.
 *  @tparam InputArgs (Inferred) The argument types passed into
 *                               the launch function.
 *
 *  @param[in] kernel      A pointer to the kernel to dispatch.
 *  @param[in] gridDim     The dimensions of the grid in thread blocks.
 *  @param[in] blkDim      The dimensions of each thread block.
 *  @param[in] sharedMem   Dynamic shared memory requirement.
 *  @param[in] si          The synchronization object for this call.
 *  @param[in] kernel_args The arguments to forward to the kernel.
 *
 *  @todo This could be improved in a few ways. We could elide the
 *        copy-to-tuple if KernelFormalArgs and InputArgs are amenable
 *        to that. We could assert that the copy-to-tuple doesn't wrap
 *        around for integer types. We might have to care about
 *        casting pointer-to-const (though that hasn't been an issue
 *        yet) to plain-ol'-void-star.
 */
template <typename... KernelFormalArgs, typename... InputArgs>
void LaunchKernel(void (*kernel)(KernelFormalArgs...),
                  dim3 const& gridDim, dim3 const& blkDim,
                  size_t sharedMem, SyncInfo<Device::GPU> const& si,
                  InputArgs... kernel_args)
{
    static_assert(sizeof...(KernelFormalArgs) == sizeof...(InputArgs),
                  "Number of provided arguments to LaunchKernel "
                  "must match the number of formal arguments to "
                  "the kernel.");

    auto formal_args = std::tuple<KernelFormalArgs...>{kernel_args...};
    auto formal_arg_pack = details::pack_arguments(formal_args);
    H_CHECK_CUDA(
        cudaLaunchKernel((void const*) kernel,
                         gridDim,
                         blkDim,
                         formal_arg_pack.data(),
                         sharedMem,
                         si.Stream()));
}

}// namespace gpu
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_CUDALAUNCHKERNEL_HPP_
