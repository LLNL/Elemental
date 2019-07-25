/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

namespace El {

template <typename Field>
Base<Field> FrobeniusNorm(AbstractMatrix<Field> const& A)
{
    switch (A.GetDevice())
    {
    case Device::CPU:
        return FrobeniusNorm(static_cast<Matrix<Field,Device::CPU> const&>(A));
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
    {
        AbstractMatrixReadDeviceProxy<Field,Device::CPU> ALocProxy{A};
        return FrobeniusNorm(ALocProxy.GetLocked());
    }
#endif //HYDROGEN_HAVE_GPU
    default:
        LogicError("FrobeniusNorm: Bad Device.");
    }
}

template<typename Field>
Base<Field> FrobeniusNorm(const Matrix<Field>& A)
{
    EL_DEBUG_CSE
    typedef Base<Field> Real;
    Real scale = TypeTraits<Real>::Zero();
    Real scaledSquare = TypeTraits<Real>::One();
    const Int width = A.Width();
    const Int height = A.Height();
    for (Int j=0; j<width; ++j)
        for (Int i=0; i<height; ++i)
            UpdateScaledSquare(A(i,j), scale, scaledSquare);
    return scale*Sqrt(scaledSquare);
}

template<typename Field>
Base<Field> HermitianFrobeniusNorm(UpperOrLower uplo, const Matrix<Field>& A)
{
    EL_DEBUG_CSE
    if (A.Height() != A.Width())
        LogicError("Hermitian matrices must be square.");

    typedef Base<Field> Real;
    Real scale = TypeTraits<Real>::Zero();
    Real scaledSquare = TypeTraits<Real>::One();
    const Int height = A.Height();
    const Int width = A.Width();
    if (uplo == UPPER)
    {
        for (Int j=0; j<width; ++j)
        {
            for (Int i=0; i<j; ++i)
            {
                UpdateScaledSquare(A(i,j), scale, scaledSquare);
                UpdateScaledSquare(A(i,j), scale, scaledSquare);
            }
            UpdateScaledSquare(A(j,j), scale, scaledSquare);
        }
    }
    else
    {
        for (Int j=0; j<width; ++j)
        {
            for (Int i=j+1; i<height; ++i)
            {
                UpdateScaledSquare(A(i,j), scale, scaledSquare);
                UpdateScaledSquare(A(i,j), scale, scaledSquare);
            }
            UpdateScaledSquare(A(j,j), scale, scaledSquare);
        }
    }
    return scale*Sqrt(scaledSquare);
}

template<typename Field>
Base<Field> SymmetricFrobeniusNorm(UpperOrLower uplo, const Matrix<Field>& A)
{
    EL_DEBUG_CSE
    return HermitianFrobeniusNorm(uplo, A);
}

template<typename Real>
Real NormFromScaledSquare
(Real localScale, Real localScaledSquare, mpi::Comm const& comm)
{
    // Find the maximum relative scale
    const Real scale = mpi::AllReduce(localScale, mpi::MAX, comm,
                                      SyncInfo<Device::CPU>());

    if (scale != TypeTraits<Real>::Zero())
    {
        // Equilibrate our local scaled sum to the maximum scale
        Real relScale = localScale/scale;
        localScaledSquare *= relScale*relScale;

        // The scaled square is now the sum of the local contributions
        const Real scaledSquare = mpi::AllReduce(localScaledSquare, comm,
                                                 SyncInfo<Device::CPU>());
        return scale*Sqrt(scaledSquare);
    }
    else
        return TypeTraits<Real>::Zero();
}

template<typename Field>
Base<Field> FrobeniusNorm(const AbstractDistMatrix<Field>& A)
{
    EL_DEBUG_CSE
    typedef Base<Field> Real;
    Real norm;
    if (A.Participating())
    {
        Real localScale = TypeTraits<Real>::Zero(),
            localScaledSquare = TypeTraits<Real>::One();
        const Int localHeight = A.LocalHeight();
        const Int localWidth = A.LocalWidth();

        AbstractMatrixReadDeviceProxy<Field,Device::CPU>
            ALocProxy{A.LockedMatrix()};

        auto const& ALoc = ALocProxy.GetLocked();

        for (Int jLoc=0; jLoc<localWidth; ++jLoc)
            for (Int iLoc=0; iLoc<localHeight; ++iLoc)
                UpdateScaledSquare
                    (ALoc(iLoc,jLoc), localScale, localScaledSquare);
        norm = NormFromScaledSquare
            (localScale, localScaledSquare, A.DistComm());
    }
    mpi::Broadcast(norm, A.Root(), A.CrossComm(), SyncInfo<Device::CPU>{});
    return norm;
}

template<typename Field>
Base<Field> HermitianFrobeniusNorm
(UpperOrLower uplo, const AbstractDistMatrix<Field>& A)
{
    EL_DEBUG_CSE
    if (A.Height() != A.Width())
        LogicError("Hermitian matrices must be square.");

    typedef Base<Field> Real;
    Real norm;
    if (A.Participating())
    {
        Real localScale = TypeTraits<Real>::Zero();
        Real localScaledSquare = TypeTraits<Real>::One();
        const Int localWidth = A.LocalWidth();
        const Int localHeight = A.LocalHeight();
        const Matrix<Field>& ALoc =
            dynamic_cast<Matrix<Field,Device::CPU> const&>(A.LockedMatrix());
        if (uplo == UPPER)
        {
            for (Int jLoc=0; jLoc<localWidth; ++jLoc)
            {
                const Int j = A.GlobalCol(jLoc);
                const Int numUpperRows = A.LocalRowOffset(j+1);
                for (Int iLoc=0; iLoc<numUpperRows; ++iLoc)
                {
                    const Int i = A.GlobalRow(iLoc);
                    UpdateScaledSquare
                    (ALoc(iLoc,jLoc), localScale, localScaledSquare);
                    if (i != j)
                        UpdateScaledSquare
                        (ALoc(iLoc,jLoc), localScale, localScaledSquare);
                }
            }
        }
        else
        {
            for (Int jLoc=0; jLoc<localWidth; ++jLoc)
            {
                const Int j = A.GlobalCol(jLoc);
                const Int numStrictlyUpperRows = A.LocalRowOffset(j);
                for (Int iLoc=numStrictlyUpperRows; iLoc<localHeight; ++iLoc)
                {
                    const Int i = A.GlobalRow(iLoc);
                    UpdateScaledSquare
                    (ALoc(iLoc,jLoc), localScale, localScaledSquare);
                    if (i != j)
                        UpdateScaledSquare
                        (ALoc(iLoc,jLoc), localScale, localScaledSquare);
                }
            }
        }

        norm = NormFromScaledSquare
          (localScale, localScaledSquare, A.DistComm());
    }
    mpi::Broadcast(norm, A.Root(), A.CrossComm(), SyncInfo<Device::CPU>{});
    return norm;
}

template<typename Field>
Base<Field> SymmetricFrobeniusNorm
(UpperOrLower uplo, const AbstractDistMatrix<Field>& A)
{
    EL_DEBUG_CSE
    return HermitianFrobeniusNorm(uplo, A);
}

#define PROTO(Field) \
  template Base<Field> FrobeniusNorm(const Matrix<Field>& A); \
  template Base<Field> FrobeniusNorm (const AbstractDistMatrix<Field>& A); \
  template Base<Field> HermitianFrobeniusNorm \
  (UpperOrLower uplo, const Matrix<Field>& A); \
  template Base<Field> HermitianFrobeniusNorm \
  (UpperOrLower uplo, const AbstractDistMatrix<Field>& A); \
  template Base<Field> SymmetricFrobeniusNorm \
  (UpperOrLower uplo, const Matrix<Field>& A); \
  template Base<Field> SymmetricFrobeniusNorm \
  (UpperOrLower uplo, const AbstractDistMatrix<Field>& A);

#ifdef HYDROGEN_GPU_USE_FP16
PROTO(gpu_half_type)
#endif

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#define EL_ENABLE_HALF
#include <El/macros/Instantiate.h>

} // namespace El
