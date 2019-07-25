/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_DOT_HPP
#define EL_BLAS_DOT_HPP

namespace El {

template<typename T, Device D>
T Dot( const Matrix<T, D>& A, const Matrix<T, D>& B )
{
    EL_DEBUG_CSE
    return HilbertSchmidt( A, B );
}

template<typename T>
T Dot( const AbstractMatrix<T>& A, const AbstractMatrix<T>& B )
{
    if (A.GetDevice() != B.GetDevice())
        LogicError("Dot requires matching device types.");

    T sum(0);
    switch(A.GetDevice()) {
    case Device::CPU:
      sum = Dot(static_cast<const Matrix<T,Device::CPU>&>(A),
                static_cast<const Matrix<T,Device::CPU>&>(B));
      break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
      sum = Dot(static_cast<const Matrix<T,Device::GPU>&>(A),
                static_cast<const Matrix<T,Device::GPU>&>(B));
      break;
#endif // HYDROGEN_HAVE_GPU
    default:
      LogicError("Unsupported device type.");
    }
    return sum;
}

template<typename T>
T Dot( const AbstractDistMatrix<T>& A, const AbstractDistMatrix<T>& B )
{
    EL_DEBUG_CSE
    return HilbertSchmidt( A, B );
}

// TODO(poulson): Think about using a more stable accumulation algorithm?

template<typename T>
T Dotu( const Matrix<T>& A, const Matrix<T>& B )
{
    EL_DEBUG_CSE
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        LogicError("Matrices must be the same size");
    T sum(0);
    const Int width = A.Width();
    const Int height = A.Height();
    for( Int j=0; j<width; ++j )
        for( Int i=0; i<height; ++i )
            sum += A(i,j)*B(i,j);
    return sum;
}

template<typename T>
T Dotu( const ElementalMatrix<T>& A, const ElementalMatrix<T>& B )
{
    EL_DEBUG_CSE
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        LogicError("Matrices must be the same size");
    AssertSameGrids( A, B );
    if( A.DistData().colDist != B.DistData().colDist ||
        A.DistData().rowDist != B.DistData().rowDist )
        LogicError("Matrices must have the same distribution");
    if( A.ColAlign() != B.ColAlign() ||
        A.RowAlign() != B.RowAlign() )
        LogicError("Matrices must be aligned");
    if ((A.GetLocalDevice() != Device::CPU)
        || (B.GetLocalDevice() != Device::CPU))
    {
        LogicError("Dotu: Only implemented for CPU matrices.");
    }

    auto syncInfoA = SyncInfoFromMatrix(
        static_cast<Matrix<T,Device::CPU> const&>(A.LockedMatrix()));

    T innerProd;
    if( A.Participating() )
    {
        T localInnerProd(0);
        auto& ALoc = dynamic_cast<Matrix<T,Device::CPU> const&>(A.LockedMatrix());
        auto& BLoc = dynamic_cast<Matrix<T,Device::CPU> const&>(B.LockedMatrix());
        const Int localHeight = A.LocalHeight();
        const Int localWidth = A.LocalWidth();
        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
            for( Int iLoc=0; iLoc<localHeight; ++iLoc )
                localInnerProd += ALoc(iLoc,jLoc)*BLoc(iLoc,jLoc);
        innerProd = mpi::AllReduce(
            localInnerProd, A.DistComm(), syncInfoA);
    }
    mpi::Broadcast(innerProd, A.Root(), A.CrossComm(), syncInfoA);
    return innerProd;
}


#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template T Dot \
  ( const Matrix<T>& A, const Matrix<T>& B ); \
  EL_EXTERN template T Dot \
  ( const AbstractMatrix<T>& A, const AbstractMatrix<T>& B ); \
  EL_EXTERN template T Dot \
  ( const AbstractDistMatrix<T>& A, const AbstractDistMatrix<T>& B ); \
  EL_EXTERN template T Dotu \
  ( const Matrix<T>& A, const Matrix<T>& B ); \
  EL_EXTERN template T Dotu \
  ( const ElementalMatrix<T>& A, const ElementalMatrix<T>& B ); \

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_DOT_HPP
