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
    HERK,
    NRM2,
    SCAL,
    SYRK,
    TRSM,
    /** @brief Axpy for 2D data with leading dimension */
    AXPY2D,
    /** @brief Copy for 2D data with leading dimension */
    COPY2D,
    /** @brief Copy for 2D data with row and column strides */
    COPY2DSTRIDED,
    /** @brief In-place scale for 2D data with leading dimension */
    SCALE2D,
}; // enum class BLAS_Op

enum class LAPACK_Op
{
    POTRF,
}; // enum class LAPACK_Op

/** @brief Describes the fill mode for BLAS. */
enum class FillMode
{
    UPPER_TRIANGLE,
    LOWER_TRIANGLE,
    FULL,
};

inline char FillModeToChar(FillMode mode)
{
    switch (mode)
    {
    case FillMode::UPPER_TRIANGLE:
        return 'U';
    case FillMode::LOWER_TRIANGLE:
        return 'L';
    case FillMode::FULL:
        return 'F';
    }
    return 'F';
}

inline FillMode CharToFillMode(char c)
{
    switch (c)
    {
    case 'U':
        return FillMode::UPPER_TRIANGLE;
    case 'L':
        return FillMode::LOWER_TRIANGLE;
    case 'F':
        return FillMode::FULL;
    }
    throw std::logic_error("Bad char");
}

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

enum class DiagType
{
    UNIT,
    NON_UNIT,
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

/** @brief Request that the underlying library use specialized tensor
 *         instructions.
 *
 *  This is not a guarantee that such operations are available or will
 *  be used. However, if the library/hardware does expose such
 *  features, this will suggest to the library that they be used
 *  whenever possible.
 */
void RequestTensorOperations();
}
}// namespace hydrogen
#endif // HYDROGEN_BLAS_COMMON_HPP_
