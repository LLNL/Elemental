#ifndef EL_BLAS_LIKE_LEVEL1_COPYDISTMATRIX_HPP_
#define EL_BLAS_LIKE_LEVEL1_COPYDISTMATRIX_HPP_

namespace El
{

template <typename T, Dist U, Dist V, Device D,
          EnableWhen<IsStorageType<T,D>, int> = 0>
void Copy(ElementalMatrix<T> const& A,
          DistMatrix<T,U,V,ELEMENT,D>& B)
{
    EL_DEBUG_CSE;
    B = A;
}


template <typename T, Dist U, Dist V, Device D,
          EnableUnless<IsStorageType<T,D>, int> = 0>
void Copy(ElementalMatrix<T> const&,
          DistMatrix<T,U,V,ELEMENT,D>&)
{
    EL_DEBUG_CSE;
    LogicError("Copy: bad data/device combination.");
}

// Datatype conversions should not be very common, and so it is likely best to
// avoid explicitly instantiating every combination
template <typename S,typename T, Dist U, Dist V, Device D,
          EnableWhen<And<IsStorageType<S,D>,
                         IsStorageType<T,D>>, int> = 0>
void Copy(ElementalMatrix<S> const& A,
          DistMatrix<T,U,V,ELEMENT,D>& B)
{
    EL_DEBUG_CSE;
    if (A.Grid() == B.Grid() && A.ColDist() == U && A.RowDist() == V
        && A.GetLocalDevice() == D)
    {
        if (!B.RootConstrained())
            B.SetRoot(A.Root());
        if (!B.ColConstrained())
            B.AlignCols(A.ColAlign());
        if (!B.RowConstrained())
            B.AlignRows(A.RowAlign());
        if (A.Root() == B.Root() &&
            A.ColAlign() == B.ColAlign() && A.RowAlign() == B.RowAlign())
        {
            B.Resize(A.Height(), A.Width());
            Copy(A.LockedMatrix(), B.Matrix());
            return;
        }
    }
    DistMatrix<S,U,V,ELEMENT,D> BOrig(A.Grid());
    BOrig.AlignWith(B);
    BOrig = A;
    B.Resize(A.Height(), A.Width());
    Copy(BOrig.LockedMatrix(), B.Matrix());
}

template <typename S,typename T, Dist U,Dist V,Device D,
          EnableUnless<And<IsStorageType<S,D>,
                         IsStorageType<T,D>>, int> = 0>
void Copy(const ElementalMatrix<S>&, DistMatrix<T,U,V,ELEMENT,D>&)
{
    EL_DEBUG_CSE;
    LogicError("Copy: bad data/device combination.");
}

template <typename T,Dist U,Dist V>
void Copy(const BlockMatrix<T>& A, DistMatrix<T,U,V,BLOCK>& B)
{
    EL_DEBUG_CSE;
    B = A;
}

// Datatype conversions should not be very common, and so it is likely best to
// avoid explicitly instantiating every combination
template <typename S, typename T, Dist U, Dist V>
void Copy(const BlockMatrix<S>& A, DistMatrix<T,U,V,BLOCK>& B)
{
    EL_DEBUG_CSE;
    if (A.Grid() == B.Grid() && A.ColDist() == U && A.RowDist() == V)
    {
        if (!B.RootConstrained())
            B.SetRoot(A.Root());
        if (!B.ColConstrained())
            B.AlignColsWith(A.DistData());
        if (!B.RowConstrained())
            B.AlignRowsWith(A.DistData());
        if (A.Root() == B.Root() &&
            A.ColAlign() == B.ColAlign() &&
            A.RowAlign() == B.RowAlign() &&
            A.ColCut() == B.ColCut() &&
            A.RowCut() == B.RowCut())
        {
            B.Resize(A.Height(), A.Width());
            Copy(A.LockedMatrix(), B.Matrix());
            return;
        }
    }
    DistMatrix<S,U,V,BLOCK> BOrig(A.Grid());
    BOrig.AlignWith(B);
    BOrig = A;
    B.Resize(A.Height(), A.Width());
    Copy(BOrig.LockedMatrix(), B.Matrix());
}

template <typename T, typename U>
void Copy(const ElementalMatrix<T>& A, ElementalMatrix<U>& B)
{
    EL_DEBUG_CSE;
#define GUARD(CDIST,RDIST,WRAP,DEVICE)                                  \
        (B.ColDist() == CDIST) && (B.RowDist() == RDIST)                \
            && (B.Wrap() == WRAP) && (B.GetLocalDevice() == DEVICE)
#define PAYLOAD(CDIST,RDIST,WRAP,DEVICE)                                \
        auto& BCast =                                                   \
            static_cast<DistMatrix<U,CDIST,RDIST,ELEMENT,DEVICE>&>(B);  \
        Copy(A, BCast);
    #include <El/macros/DeviceGuardAndPayload.h>
}

template<typename S,typename T,
         typename/*=EnableIf<CanCast<S,T>>*/>
void Copy(const BlockMatrix<S>& A, BlockMatrix<T>& B)
{
    EL_DEBUG_CSE;
    #define GUARD(CDIST,RDIST,WRAP) \
        B.ColDist() == CDIST && B.RowDist() == RDIST && B.Wrap() == WRAP \
            && B.GetLocalDevice() == Device::CPU
    #define PAYLOAD(CDIST,RDIST,WRAP) \
      auto& BCast = static_cast<DistMatrix<T,CDIST,RDIST,BLOCK>&>(B); \
      Copy(A, BCast);
    #include <El/macros/GuardAndPayload.h>
}

template <typename T>
void Copy(AbstractDistMatrix<T> const& A, AbstractDistMatrix<T>& B)
{
    EL_DEBUG_CSE;
    DistWrap const wrapA=A.Wrap(), wrapB=B.Wrap();
    if (wrapB == ELEMENT)
    {
        auto& ACast = static_cast<ElementalMatrix<T> const&>(A);
        auto& BCast = static_cast<ElementalMatrix<T>&>(B);
        Copy(ACast, BCast);
    }
    else if (wrapA == BLOCK && wrapB == BLOCK)
    {
        auto& ACast = static_cast<BlockMatrix<T> const&>(A);
        auto& BCast = static_cast<BlockMatrix<T>&>(B);
        Copy(ACast, BCast);
    }
    else
    {
        LogicError("If you see this error, please tell Tom.");
        copy::GeneralPurpose(A, B);
    }
}

template <typename T, typename U>
void Copy(AbstractDistMatrix<T> const& A, AbstractDistMatrix<U>& B)
{
    EL_DEBUG_CSE;
    DistWrap const wrapA=A.Wrap(), wrapB=B.Wrap();
    if (wrapB == ELEMENT)
    {
        auto& ACast = static_cast<ElementalMatrix<T> const&>(A);
        auto& BCast = static_cast<ElementalMatrix<U>&>(B);
        Copy(ACast, BCast);
    }
    else if (wrapA == BLOCK && wrapB == BLOCK)
    {
        auto& ACast = static_cast<BlockMatrix<T> const&>(A);
        auto& BCast = static_cast<BlockMatrix<U>&>(B);
        Copy(ACast, BCast);
    }
    else
    {
        LogicError("If you see this error, please tell Tom.");
    }
}

}// namespace El
#endif // EL_BLAS_LIKE_LEVEL1_COPYDISTMATRIX_HPP_
