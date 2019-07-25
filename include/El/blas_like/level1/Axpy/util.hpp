/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_AXPY_UTIL_HPP
#define EL_BLAS_AXPY_UTIL_HPP

#ifdef HYDROGEN_HAVE_GPU
#include <hydrogen/blas/gpu/Axpy.hpp>
#endif

namespace El
{
namespace axpy
{
namespace util
{

template <typename T>
void InterleaveMatrixUpdate(
    T alpha, Int height, Int width,
    T const* A, Int colStrideA, Int rowStrideA,
    T* B, Int colStrideB, Int rowStrideB,
    SyncInfo<Device::CPU>)
{
    // TODO: Add OpenMP parallelization and/or optimize
    for( Int j=0; j<width; ++j )
        blas::Axpy(
            height, alpha,
            &A[rowStrideA*j], colStrideA,
            &B[rowStrideB*j], colStrideB);
}

#ifdef HYDROGEN_HAVE_GPU
template <typename T>
void InterleaveMatrixUpdate(
    T alpha, Int height, Int width,
    T const* A, Int colStrideA, Int rowStrideA,
    T* B, Int colStrideB, Int rowStrideB,
    SyncInfo<Device::GPU> syncInfo)
{
    EL_DEBUG_CSE;
    hydrogen::Axpy_GPU_impl(height, width, alpha,
                            A, colStrideA, rowStrideA,
                            B, colStrideB, rowStrideB,
                            syncInfo.Stream());
}
#endif // HYDROGEN_HAVE_GPU

template<typename T, Device D>
void UpdateWithLocalData(
    T alpha, ElementalMatrix<T> const& A,
    DistMatrix<T,STAR,STAR,ELEMENT,D>& B)
{
    EL_DEBUG_CSE

    if (A.GetLocalDevice() != D)
        LogicError("axpy::util::UpdateWithLocalData: Bad device.");

    auto syncInfoA = SyncInfoFromMatrix(
        static_cast<Matrix<T,D> const&>(A.LockedMatrix()));
    auto syncInfoB = SyncInfoFromMatrix(B.LockedMatrix());
    auto syncHelper = MakeMultiSync(syncInfoB, syncInfoA);

    InterleaveMatrixUpdate(
        alpha, A.LocalHeight(), A.LocalWidth(),
        A.LockedBuffer(),
        1,             A.LDim(),
        B.Buffer(A.ColShift(),A.RowShift()),
        A.ColStride(), A.RowStride()*B.LDim(), syncInfoB);
}

} // namespace util
} // namespace axpy
} // namespace El

#endif // ifndef EL_BLAS_AXPY_UTIL_HPP
