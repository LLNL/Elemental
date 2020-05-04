#ifndef HYDROGEN_BLAS_COMMON_HPP_
#define HYDROGEN_BLAS_COMMON_HPP_

#include <stdexcept>

/** @file
 *
 *  Contains common components for device BLAS integration.
 */
namespace hydrogen
{

/** @brief Enable metaprogramming on the BLAS operation. */
enum class BLAS_Op
{
    AXPY,
    COPY,
    DGMM,
    DOT,
    GEAM,
    GEMM,
    GEMMSTRIDEDBATCHED,
    GEMV,
    NRM2,
    SCAL,
    /** @brief Axpy for 2D data with leading dimension */
    AXPY2D,
    /** @brief Copy for 2D data with leading dimension */
    COPY2D,
    /** @brief Copy for 2D data with row and column strides */
    COPY2DSTRIDED,
    /** @brief In-place scale for 2D data with leading dimension */
    SCALE2D,
};

/** @brief Describes transpose operations for BLAS. */
enum class TransposeMode
{
    NORMAL,
    TRANSPOSE,
    CONJ_TRANSPOSE,
};

// Interop with old-school BLAS
inline char TransposeModeToChar(TransposeMode mode)
{
    switch (mode)
    {
    case TransposeMode::NORMAL:
        return 'N';
    case TransposeMode::TRANSPOSE:
        return 'T';
    case TransposeMode::CONJ_TRANSPOSE:
        return 'C';
    }
    return 'N'; // Silence potential compiler warning
}

inline TransposeMode CharToTransposeMode(char c)
{
    switch (c)
    {
    case 'N':
        return TransposeMode::NORMAL;
    case 'T':
        return TransposeMode::TRANSPOSE;
    case 'C':
        return TransposeMode::CONJ_TRANSPOSE;
    }
    throw std::logic_error("Bad char");
}

/** @brief Describes placement of diagonal in DGMM. */
enum class SideMode
{
    LEFT,
    RIGHT,
};

/** @brief Describes where pointers point. */
enum class PointerMode
{
    HOST,
    DEVICE,
};// enum class PointerMode

namespace gpu_blas
{
/** @brief Set the pointer mode of the underlying library. */
void SetPointerMode(PointerMode mode);
}
}// namespace hydrogen
#endif // HYDROGEN_BLAS_COMMON_HPP_
