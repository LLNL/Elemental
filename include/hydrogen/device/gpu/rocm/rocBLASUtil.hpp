#ifndef HYDROGEN_DEVICE_GPU_ROCM_ROCBLASUTIL_HPP_
#define HYDROGEN_DEVICE_GPU_ROCM_ROCBLASUTIL_HPP_

#include <El/hydrogen_config.h>

#include <rocblas/rocblas.h>

namespace hydrogen
{
namespace rocblas
{

/** @brief rocBLAS uses its own int typedef to represent sizes. */
using SizeT = rocblas_int;

/** @brief Convert a value to the size type expected by the rocBLAS
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

/** @brief Convert an TransposeMode to the rocBLAS operation type. */
inline rocblas_operation
ToNativeTransposeMode(TransposeMode const& orient) noexcept
{
    switch (orient)
    {
    case TransposeMode::TRANSPOSE:
        return rocblas_operation_transpose;
    case TransposeMode::CONJ_TRANSPOSE:
        return rocblas_operation_conjugate_transpose;
    default: // TransposeMode::NORMAL
        return rocblas_operation_none;
    }
}

/** @brief Convert a SideMode to the rocBLAS side mode type. */
inline rocblas_side
ToNativeSideMode(SideMode const& side) noexcept
{
    if (side == SideMode::LEFT)
        return rocblas_side_left;

    return rocblas_side_right;
}

inline rocblas_fill
ToNativeFillMode(FillMode const& uplo) noexcept
{
    switch (uplo)
    {
    case FillMode::UPPER_TRIANGLE:
        return rocblas_fill_upper;
    case FillMode::LOWER_TRIANGLE:
        return rocblas_fill_lower;
    case FillMode::FULL:
        return rocblas_fill_full;
    }
    return rocblas_fill_full;
}

inline rocblas_diagonal
ToNativeDiagType(DiagType const& diag) noexcept
{
    switch (diag)
    {
    case DiagType::UNIT:
        return rocblas_diagonal_unit;
    case DiagType::NON_UNIT:
        return rocblas_diagonal_non_unit;
    }
    return rocblas_diagonal_unit;
}

}// namespace rocblas
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_ROCM_ROCBLASUTIL_HPP_
