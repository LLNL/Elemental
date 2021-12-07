/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

namespace El {

template<typename Ring>
Base<Ring> MaxNorm( const Matrix<Ring>& A )
{
    EL_DEBUG_CSE
    typedef Base<Ring> Real;
    const Int height = A.Height();
    const Int width = A.Width();

    Real maxAbs = 0;
    for( Int j=0; j<width; ++j )
        for( Int i=0; i<height; ++i )
            maxAbs = Max( maxAbs, Abs(A(i,j)) );

    return maxAbs;
}

template<typename Ring>
Base<Ring> HermitianMaxNorm( UpperOrLower uplo, const Matrix<Ring>& A )
{
    EL_DEBUG_CSE
    if( A.Height() != A.Width() )
        LogicError("Hermitian matrices must be square.");

    typedef Base<Ring> Real;
    const Int height = A.Height();
    const Int width = A.Width();

    Real maxAbs = 0;
    if( uplo == UPPER )
    {
        for( Int j=0; j<width; ++j )
            for( Int i=0; i<=j; ++i )
                maxAbs = Max( maxAbs, Abs(A(i,j)) );
    }
    else
    {
        for( Int j=0; j<width; ++j )
            for( Int i=j; i<height; ++i )
                maxAbs = Max( maxAbs, Abs(A(i,j)) );
    }
    return maxAbs;
}

#if 0 // TOM

template<typename Ring>
Base<Ring> SymmetricMaxNorm( UpperOrLower uplo, const Matrix<Ring>& A )
{
    EL_DEBUG_CSE
    return HermitianMaxNorm( uplo, A );
}

template<typename Ring>
Base<Ring> MaxNorm( const AbstractDistMatrix<Ring>& A )
{
    EL_DEBUG_CSE
    Base<Ring> norm=0;
    if( A.Participating() )
    {
        Base<Ring> localMaxAbs = MaxNorm( A.LockedMatrix() );
        norm = mpi::AllReduce( localMaxAbs, mpi::MAX, A.DistComm() );
    }
    mpi::Broadcast( norm, A.Root(), A.CrossComm() );
    return norm;
}

template<typename Ring>
Base<Ring>
HermitianMaxNorm( UpperOrLower uplo, const AbstractDistMatrix<Ring>& A )
{
    EL_DEBUG_CSE
    if( A.Height() != A.Width() )
        LogicError("Hermitian matrices must be square.");
    typedef Base<Ring> Real;

    Real norm;
    if( A.Participating() )
    {
        const Int localWidth = A.LocalWidth();
        const Int localHeight = A.LocalHeight();
        const Matrix<Ring>& ALoc = A.LockedMatrix();

        Real localMaxAbs = 0;
        if( uplo == UPPER )
        {
            for( Int jLoc=0; jLoc<localWidth; ++jLoc )
            {
                const Int j = A.GlobalCol(jLoc);
                const Int numUpperRows = A.LocalRowOffset(j+1);
                for( Int iLoc=0; iLoc<numUpperRows; ++iLoc )
                    localMaxAbs = Max(localMaxAbs,Abs(ALoc(iLoc,jLoc)));
            }
        }
        else
        {
            for( Int jLoc=0; jLoc<localWidth; ++jLoc )
            {
                const Int j = A.GlobalCol(jLoc);
                const Int numStrictlyUpperRows = A.LocalRowOffset(j);
                for( Int iLoc=numStrictlyUpperRows; iLoc<localHeight; ++iLoc )
                    localMaxAbs = Max(localMaxAbs,Abs(ALoc(iLoc,jLoc)));
            }
        }
        norm = mpi::AllReduce( localMaxAbs, mpi::MAX, A.DistComm() );
    }
    mpi::Broadcast( norm, A.Root(), A.CrossComm() );
    return norm;
}

template<typename Ring>
Base<Ring>
SymmetricMaxNorm( UpperOrLower uplo, const AbstractDistMatrix<Ring>& A )
{
    EL_DEBUG_CSE
    return HermitianMaxNorm( uplo, A );
}

#endif // 0 TOM

#define PROTO(Ring) \
  template Base<Ring> MaxNorm( const Matrix<Ring>& A ); \
  template Base<Ring> HermitianMaxNorm \
  ( UpperOrLower uplo, const Matrix<Ring>& A );

/*
  template Base<Ring> MaxNorm ( const AbstractDistMatrix<Ring>& A ); \
  template Base<Ring> HermitianMaxNorm \
  ( UpperOrLower uplo, const AbstractDistMatrix<Ring>& A ); \
  template Base<Ring> SymmetricMaxNorm \
  ( UpperOrLower uplo, const Matrix<Ring>& A ); \
  template Base<Ring> SymmetricMaxNorm \
  ( UpperOrLower uplo, const AbstractDistMatrix<Ring>& A );
*/

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
/*#undef EL_ENABLE_HALF*/
#include <El/macros/Instantiate.h>

} // namespace El
