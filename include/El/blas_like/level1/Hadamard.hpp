/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_HADAMARD_HPP
#define EL_BLAS_HADAMARD_HPP

#ifdef HYDROGEN_HAVE_GPU
#include <hydrogen/blas/gpu/Hadamard.hpp>
#endif // HYDROGEN_HAVE_GPU

// C(i,j) := A(i,j) B(i,j)

namespace El
{

template<typename T>
void Hadamard(AbstractMatrix<T> const& A, AbstractMatrix<T> const& B,
              AbstractMatrix<T>& C )
{
    EL_DEBUG_CSE
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        LogicError("Hadamard product requires equal dimensions");
    if ((A.GetDevice() != B.GetDevice()) || (B.GetDevice() != C.GetDevice()))
        LogicError("Hadamard product requires all matrices on same device");
    C.Resize( A.Height(), A.Width() );

    const Int height = A.Height();
    const Int width = A.Width();
    const T* ABuf = A.LockedBuffer();
    const T* BBuf = B.LockedBuffer();
    T* CBuf = C.Buffer();
    const Int ALDim = A.LDim();
    const Int BLDim = B.LDim();
    const Int CLDim = C.LDim();

    switch (A.GetDevice())
    {
    case Device::CPU:
        // Iterate over single loop if memory is contiguous. Otherwise
        // iterate over double loop.
        if( ALDim == height && BLDim == height && CLDim == height )
        {
            // Check if output matrix is equal to either input matrix
            if( CBuf == BBuf )
            {
                EL_PARALLEL_FOR
                    for( Int i=0; i<height*width; ++i )
                        CBuf[i] *= ABuf[i];
            }
            else if( CBuf == ABuf )
            {
                EL_PARALLEL_FOR
                    for( Int i=0; i<height*width; ++i )
                        CBuf[i] *= BBuf[i];
            }
            else
            {
                EL_PARALLEL_FOR
                    for( Int i=0; i<height*width; ++i )
                        CBuf[i] = ABuf[i] * BBuf[i];
            }
        }
        else
        {
            EL_SIMD_COLLAPSE2
            for( Int j=0; j<width; ++j )
            {
                for( Int i=0; i<height; ++i )
                {
                    CBuf[i+j*CLDim] = ABuf[i+j*ALDim] * BBuf[i+j*BLDim];
                }
            }
        }
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
    {
        auto si_A = SyncInfoFromMatrix(
            static_cast<Matrix<T,Device::GPU> const&>(A));
        auto si_B = SyncInfoFromMatrix(
            static_cast<Matrix<T,Device::GPU> const&>(B));
        auto si_C = SyncInfoFromMatrix(
            static_cast<Matrix<T,Device::GPU>&>(C));

        auto syncHelper = MakeMultiSync(si_C, si_A, si_B);

        hydrogen::Hadamard_GPU_impl(height, width,
                                    ABuf, 1, ALDim, BBuf, 1, BLDim,
                                    CBuf, 1, CLDim,
                                    si_C);
    }
    break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("Bad device type for Hadamard.");
    }

}

template<typename T>
void Hadamard
( const AbstractDistMatrix<T>& A,
  const AbstractDistMatrix<T>& B,
        AbstractDistMatrix<T>& C )
{
    EL_DEBUG_CSE
    const DistData& ADistData = A.DistData();
    const DistData& BDistData = B.DistData();
    DistData CDistData = C.DistData();
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        LogicError("Hadamard product requires equal dimensions");
    AssertSameGrids( A, B );
    if( ADistData.colDist != BDistData.colDist ||
        ADistData.rowDist != BDistData.rowDist ||
        BDistData.colDist != CDistData.colDist ||
        BDistData.rowDist != CDistData.rowDist )
        LogicError("A, B, and C must share the same distribution");
    if( A.ColAlign() != B.ColAlign() || A.RowAlign() != B.RowAlign() )
        LogicError("A and B must be aligned");
    if ( A.BlockHeight() != B.BlockHeight() ||
         A.BlockWidth() != B.BlockWidth())
      LogicError("A and B must have the same block size");
    C.AlignWith( A.DistData() );
    C.Resize( A.Height(), A.Width() );
    Hadamard( A.LockedMatrix(), B.LockedMatrix(), C.Matrix() );
}

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void Hadamard \
  ( const AbstractMatrix<T>& A, const AbstractMatrix<T>& B, AbstractMatrix<T>& C ); \
  EL_EXTERN template void Hadamard \
  ( const AbstractDistMatrix<T>& A, \
    const AbstractDistMatrix<T>& B, \
          AbstractDistMatrix<T>& C );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_HADAMARD_HPP
