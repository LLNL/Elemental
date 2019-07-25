/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_DIAGONALSCALE_HPP
#define EL_BLAS_DIAGONALSCALE_HPP

#include <El/hydrogen_config.h>

namespace El {

#ifdef HYDROGEN_HAVE_GPU
template <typename T, typename>
void DiagonalScale(
    LeftOrRight side, Orientation orientation,
    Matrix<T,Device::GPU> const& d, Matrix<T,Device::GPU>& A)
{
    EL_DEBUG_CSE;
    const Int m = A.Height();
    const Int n = A.Width();
    const Int lda = A.LDim();
    const Int incd = 1;

    if (orientation != NORMAL)
        LogicError("DiagonalScale: Only NORMAL mode supported on GPU");

    auto master_sync = SyncInfoFromMatrix(A);
    auto syncManager = MakeMultiSync(master_sync, SyncInfoFromMatrix(d));
    gpu_blas::Dgmm(
        (side == LEFT ? SideMode::LEFT : SideMode::RIGHT),
        m, n,
        A.LockedBuffer(), lda,
        d.LockedBuffer(), incd,
        A.Buffer(), lda,
        master_sync);
}

template <typename T, typename, typename>
void DiagonalScale(
    LeftOrRight, Orientation,
    Matrix<T,Device::GPU> const&, Matrix<T,Device::GPU>&)
{
    LogicError("DiagonalScale: Bad device type.");
}

#endif // HYDROGEN_HAVE_GPU

template<typename TDiag,typename T>
void DiagonalScale(
    LeftOrRight side,
    Orientation orientation,
    Matrix<TDiag> const& d,
    Matrix<T>& A)
{
    EL_DEBUG_CSE;
    const Int m = A.Height();
    const Int n = A.Width();
    const bool conj = (orientation == ADJOINT);

    if (side == LEFT)
    {
#ifndef EL_RELEASE
        if (d.Height() != m)
            LogicError("Invalid left diagonal scaling dimension");
#endif
        for(Int i=0; i<m; ++i)
        {
            const T delta = (conj ? Conj(d(i)) : d(i));
            for(Int j=0; j<n; ++j)
                A(i,j) *= delta;
        }
    }
    else
    {
#ifndef EL_RELEASE
        if (d.Height() != n)
            LogicError("Invalid right diagonal scaling dimension");
#endif
        for(Int j=0; j<n; ++j)
        {
            const T delta = (conj ? Conj(d(j)) : d(j));
            for(Int i=0; i<m; ++i)
                A(i,j) *= delta;
        }
    }
}

template <typename T>
void DiagonalScale(LeftOrRight side,
                   Orientation orientation,
                   AbstractMatrix<T> const& d,
                   AbstractMatrix<T>& A)
{
    if (d.GetDevice() != A.GetDevice())
        LogicError("DiagonalScale: d and A must be on the same device!");
    switch (A.GetDevice())
    {
    case Device::CPU:
        DiagonalScale(side, orientation,
                      static_cast<Matrix<T,Device::CPU> const&>(d),
                      static_cast<Matrix<T,Device::CPU>&>(A));
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        DiagonalScale(side, orientation,
                      static_cast<Matrix<T,Device::GPU> const&>(d),
                      static_cast<Matrix<T,Device::GPU>&>(A));
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("DiagonalScale: Bad device.");
    }
}

template<typename TDiag,typename T,Dist U,Dist V,DistWrap wrapType,Device D,typename>
void DiagonalScale(
    LeftOrRight side,
    Orientation orientation,
    AbstractDistMatrix<TDiag> const& dPre,
    DistMatrix<T,U,V,wrapType,D>& A)
{
    EL_DEBUG_CSE;
    if (dPre.GetLocalDevice() != D)
        LogicError("DiagonalScale: dPre must have same device as A");

    if (wrapType == ELEMENT)
    {
        if (side == LEFT)
        {
            ElementalProxyCtrl ctrl;
            ctrl.rootConstrain = true;
            ctrl.colConstrain = true;
            ctrl.root = A.Root();
            ctrl.colAlign = A.ColAlign();

            DistMatrixReadProxy<TDiag,TDiag,U,Collect<V>(),ELEMENT,D> dProx(dPre, ctrl);
            auto& d = dProx.GetLocked();

            DiagonalScale(LEFT, orientation, d.LockedMatrix(), A.Matrix());
        }
        else
        {
            ElementalProxyCtrl ctrl;
            ctrl.rootConstrain = true;
            ctrl.colConstrain = true;
            ctrl.root = A.Root();
            ctrl.colAlign = A.RowAlign();
            DistMatrixReadProxy<TDiag,TDiag,V,Collect<U>(),ELEMENT,D> dProx(dPre, ctrl);
            auto& d = dProx.GetLocked();

            DiagonalScale(RIGHT, orientation, d.LockedMatrix(), A.Matrix());
        }
    }
    else
    {
        ProxyCtrl ctrl;
        ctrl.rootConstrain = true;
        ctrl.colConstrain = true;
        ctrl.root = A.Root();

        if (side == LEFT)
        {
            ctrl.colAlign = A.ColAlign();
            ctrl.blockHeight = A.BlockHeight();
            ctrl.colCut = A.ColCut();

            DistMatrixReadProxy<TDiag,TDiag,U,Collect<V>(),BLOCK>
              dProx(dPre, ctrl);
            auto& d = dProx.GetLocked();

            DiagonalScale(LEFT, orientation, d.LockedMatrix(), A.Matrix());
        }
        else
        {
            ctrl.colAlign = A.RowAlign();
            ctrl.blockHeight = A.BlockWidth();
            ctrl.colCut = A.RowCut();

            DistMatrixReadProxy<TDiag,TDiag,V,Collect<U>(),BLOCK>
              dProx(dPre, ctrl);
            auto& d = dProx.GetLocked();

            DiagonalScale(RIGHT, orientation, d.LockedMatrix(), A.Matrix());
        }
    }
}

template<typename TDiag, typename T>
void DiagonalScale(
    LeftOrRight side,
    Orientation orientation,
    AbstractDistMatrix<TDiag> const& d,
    AbstractDistMatrix<T>& A)
{
    EL_DEBUG_CSE
    #define GUARD(CDIST,RDIST,WRAP,DEVICE) \
      A.ColDist() == CDIST && A.RowDist() == RDIST && A.Wrap() == WRAP && A.GetLocalDevice() == DEVICE
    #define PAYLOAD(CDIST,RDIST,WRAP,DEVICE) \
        auto& ACast = static_cast<DistMatrix<T,CDIST,RDIST,WRAP,DEVICE>&>(A); \
        DiagonalScale(side, orientation, d, ACast);
    #include <El/macros/DeviceGuardAndPayload.h>
}

template<typename TDiag, typename T,
         Dist U, Dist V, DistWrap wrapType, Device D,
         typename, typename>
void DiagonalScale(
    LeftOrRight, Orientation,
    AbstractDistMatrix<TDiag> const&, DistMatrix<T,U,V,wrapType,D>&)
{
    LogicError("DiagonalScale: Bad device type combo.");
}

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void DiagonalScale \
  (LeftOrRight side, \
    Orientation orientation, \
    AbstractMatrix<T> const& d, \
    AbstractMatrix<T>& A);             \
  EL_EXTERN template void DiagonalScale \
  (LeftOrRight side, \
    Orientation orientation, \
    const AbstractDistMatrix<T>& d, \
          AbstractDistMatrix<T>& A);

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_DIAGONALSCALE_HPP
