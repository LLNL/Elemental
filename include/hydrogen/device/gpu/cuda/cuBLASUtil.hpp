#ifndef HYDROGEN_DEVICE_GPU_CUDA_CUBLASUTIL_HPP_
#define HYDROGEN_DEVICE_GPU_CUDA_CUBLASUTIL_HPP_

#include <El/hydrogen_config.h>

#include <cublas_v2.h>

namespace hydrogen
{
namespace cublas
{

/** @brief cuBLAS uses ints to represent sizes. */
using SizeT = int;

/** @brief Convert a value to the size type expected by the cuBLAS
 *         library.
 *
 *  If `HYDROGEN_DO_BOUNDS_CHECKING` is defined, this will do a
 *  "safe cast" (it will verify that `val` is in the dynamic range of
 *  `int`. Otherwise it will do a regular static_cast.
 */
template <typename T>
#ifdef HYDROGEN_DO_BOUNDS_CHECKING
SizeT ToSizeT(T const& val)
{
    return narrow_cast<SizeT>(val);
}
#else
SizeT ToSizeT(T const& val) noexcept
{
    return static_cast<SizeT>(val);
}
#endif // HYDROGEN_DO_BOUNDS_CHECKING

/** @brief Overload to prevent extra work in the case of dynamic range
 *         checking.
 */
inline SizeT ToSizeT(SizeT const& val) noexcept
{
    return val;
}

/** @brief Convert an TransposeMode to the cuBLAS operation type. */
inline cublasOperation_t
ToNativeTransposeMode(TransposeMode const& orient) noexcept
{
    switch (orient)
    {
    case TransposeMode::TRANSPOSE:
        return CUBLAS_OP_T;
    case TransposeMode::CONJ_TRANSPOSE:
        return CUBLAS_OP_C;
    default: // TransposeMode::NORMAL
        return CUBLAS_OP_N;
    }
}

/** @brief Convert a SideMode to the cuBLAS side mode type. */
inline cublasSideMode_t
ToNativeSideMode(SideMode const& side) noexcept
{
    if (side == SideMode::LEFT)
        return CUBLAS_SIDE_LEFT;

    return CUBLAS_SIDE_RIGHT;
}

}// namespace cublas
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_CUDA_CUBLASUTIL_HPP_
