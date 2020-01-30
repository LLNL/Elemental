#ifndef EL_BLAS_LIKE_LEVEL1_COPYASYNCDISTMATRIX_HPP_
#define EL_BLAS_LIKE_LEVEL1_COPYASYNCDISTMATRIX_HPP_

#include <hydrogen/Device.hpp>

#include "CopyLocal.hpp"
#include "CopyDistMatrix.hpp"
#include "CopyAsyncLocal.hpp"

namespace El
{

template <typename S, typename T, Dist U, Dist V, Device D1, Device D2>
void CopyAsync(DistMatrix<S,U,V,ELEMENT,D1> const& A,
               DistMatrix<T,U,V,ELEMENT,D2>& B)
{
    EL_DEBUG_CSE;
#ifndef EL_RELEASE
    auto const Adata = A.DistData(), Bdata = B.DistData();
    if (!((Adata.blockHeight == Bdata.blockHeight) &&
          (Adata.blockWidth == Bdata.blockWidth) &&
          (Adata.colAlign == Bdata.colAlign) &&
          (Adata.rowAlign == Bdata.rowAlign) &&
          (Adata.colCut == Bdata.colCut) &&
          (Adata.rowCut == Bdata.rowCut) &&
          (Adata.root == Bdata.root) &&
          (Adata.grid == Bdata.grid)))
    {
        LogicError("CopyAsync: "
                   "A and B must have the same DistData, except device.");
    }
#endif // !defined(EL_RELEASE)
    B.Resize(A.Height(), A.Width());
    CopyAsync(A.LockedMatrix(), B.Matrix());
}

template <typename S, typename T, Dist U, Dist V, Device D>
void CopyAsync(ElementalMatrix<S> const& A, DistMatrix<T,U,V,ELEMENT,D>& B)
{
    EL_DEBUG_CSE;
    if ((A.ColDist() == U) && (A.RowDist() == V))
    {
        switch (A.GetLocalDevice())
        {
        case Device::CPU:
            CopyAsync(
                static_cast<DistMatrix<S,U,V,ELEMENT,Device::CPU> const&>(A),
                B);
            break;
#ifdef HYDROGEN_HAVE_CUDA
        case Device::GPU:
            CopyAsync(
                static_cast<DistMatrix<S,U,V,ELEMENT,Device::GPU> const&>(A),
                B);
            break;
#endif // HYDROGEN_HAVE_CUDA
        default:
            LogicError("CopyAsync: Unknown device type.");
        }
    }
    else
        LogicError("CopyAsync requires A and B to have the same distribution.");
}


template <typename T, typename U>
void CopyAsync(ElementalMatrix<T> const& A, ElementalMatrix<U>& B)
{
    EL_DEBUG_CSE;
#define GUARD(CDIST,RDIST,WRAP,DEVICE)                              \
    (B.ColDist() == CDIST) && (B.RowDist() == RDIST)                \
        && (B.Wrap() == WRAP) && (B.GetLocalDevice() == DEVICE)
#define PAYLOAD(CDIST,RDIST,WRAP,DEVICE)                            \
    auto& BCast =                                                   \
        static_cast<DistMatrix<U,CDIST,RDIST,ELEMENT,DEVICE>&>(B);  \
    CopyAsync(A, BCast);
    #include <El/macros/DeviceGuardAndPayload.h>
}

template <typename T, typename U>
void CopyAsync(AbstractDistMatrix<T> const& A, AbstractDistMatrix<U>& B)
{
    EL_DEBUG_CSE;
    if (A.Wrap() == ELEMENT && B.Wrap() == ELEMENT)
    {
        auto& ACast = static_cast<const ElementalMatrix<T>&>(A);
        auto& BCast = static_cast<ElementalMatrix<U>&>(B);
        CopyAsync(ACast, BCast);
    }
    else
    {
        LogicError("CopyAsync is only supported for ElementalMatrices.");
    }
}

}// namespace El
#endif // EL_BLAS_LIKE_LEVEL1_COPYASYNCDISTMATRIX_HPP_
