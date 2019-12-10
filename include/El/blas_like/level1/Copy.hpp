/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_COPY_HPP
#define EL_BLAS_COPY_HPP

#ifdef _OPENMP
#include <omp.h>
#endif

#include <El/blas_like/level1/Copy/internal_decl.hpp>
#include <El/blas_like/level1/Copy/GeneralPurpose.hpp>
#include <El/blas_like/level1/Copy/util.hpp>

namespace El {
namespace details {

template <template <typename> class X, typename... Ts>
using Expand = TypeList<X<Ts>...>;

// This is replaced by a generic multiple dispatch engine in
// DiHydrogen; this is a one-off use-case for now, so there's no need
// to backport a robust implementation.
template <typename FunctorT, typename LHSList, typename RHSList>
struct CopyDispatcher
{
    static void Do(FunctorT f,
                   BaseDistMatrix const& src, BaseDistMatrix& tgt)
    {
        using LHead = Head<LHSList>;
        using LTail = Tail<LHSList>;
        if (auto const* ptr = dynamic_cast<LHead const*>(&src))
            return CopyDispatcher<FunctorT, LHSList, RHSList>::DoRHS(
                f, *ptr, tgt);
        else
            return CopyDispatcher<FunctorT, LTail, RHSList>::Do(f, src, tgt);
    }

    template <typename LHSType>
    static void DoRHS(FunctorT f, LHSType const& src, BaseDistMatrix& tgt)
    {
        using RHead = Head<RHSList>;
        using RTail = Tail<RHSList>;
        if (auto* ptr = dynamic_cast<RHead*>(&tgt))
            return f(src, *ptr);
        else
            return CopyDispatcher<FunctorT, LHSList, RTail>::DoRHS(f, src, tgt);
    }
};// struct CopyDispatcher

template <typename FunctorT, typename RHSList>
struct CopyDispatcher<FunctorT, TypeList<>, RHSList>
{
    static void Do(FunctorT const&,
                   BaseDistMatrix const&, BaseDistMatrix const&)
    {
        LogicError("Source matrix type not found.");
    }
};

template <typename FunctorT, typename LHSList>
struct CopyDispatcher<FunctorT, LHSList, TypeList<>>
{
    static void DoRHS(FunctorT const&,
                      BaseDistMatrix const&, BaseDistMatrix const&)
    {
        LogicError("Target matrix type not found.");
    }
};

struct CopyFunctor
{
    template <typename T, typename U>
    void operator()(AbstractDistMatrix<T> const& src,
                    AbstractDistMatrix<U>& tgt) const
    {
        return Copy(src, tgt);
    }
};// CopyFunctor

struct CopyAsyncFunctor
{
    template <typename T, typename U>
    void operator()(AbstractDistMatrix<T> const& src,
                    AbstractDistMatrix<U>& tgt) const
    {
        return CopyAsync(src, tgt);
    }
};// CopyAsyncFunctor

}// namespace details

inline void Copy(BaseDistMatrix const& Source, BaseDistMatrix& Target)
{
    using FunctorT = details::CopyFunctor;
    using MatrixTs = details::Expand<AbstractDistMatrix, float, double>;
    using Dispatcher = details::CopyDispatcher<FunctorT, MatrixTs, MatrixTs>;
    details::CopyFunctor f;
    return Dispatcher::Do(f, Source, Target);
}

inline void CopyAsync(BaseDistMatrix const& Source, BaseDistMatrix& Target)
{
    using FunctorT = details::CopyAsyncFunctor;
    using MatrixTs = details::Expand<AbstractDistMatrix, float, double>;
    using Dispatcher = details::CopyDispatcher<FunctorT, MatrixTs, MatrixTs>;
    details::CopyAsyncFunctor f;
    return Dispatcher::Do(f, Source, Target);
}

template <typename T>
void Copy(AbstractMatrix<T> const& A, AbstractMatrix<T>& B)
{
    switch (A.GetDevice())
    {
    case Device::CPU:
        switch (B.GetDevice())
        {
        case Device::CPU:
            Copy(static_cast<Matrix<T,Device::CPU> const&>(A),
                 static_cast<Matrix<T,Device::CPU>&>(B));
            break;
#ifdef HYDROGEN_HAVE_CUDA
        case Device::GPU:
            Copy(static_cast<Matrix<T,Device::CPU> const&>(A),
                 static_cast<Matrix<T,Device::GPU>&>(B));
            break;
#endif // HYDROGEN_HAVE_CUDA
        default:
            LogicError("Copy: Bad device.");
        }
        break;
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
        switch (B.GetDevice())
        {
        case Device::CPU:
            Copy(static_cast<Matrix<T,Device::GPU> const&>(A),
                 static_cast<Matrix<T,Device::CPU>&>(B));
            break;
        case Device::GPU:
            Copy(static_cast<Matrix<T,Device::GPU> const&>(A),
                 static_cast<Matrix<T,Device::GPU>&>(B));
            break;
        default:
            LogicError("Copy: Bad device.");
        }
        break;
#endif //  HYDROGEN_HAVE_CUDA
    default:
        LogicError("Copy: Bad device.");
    }
}

template <typename T, typename U>
void Copy(AbstractMatrix<T> const& A, AbstractMatrix<U>& B)
{
    switch (A.GetDevice())
    {
    case Device::CPU:
        switch (B.GetDevice())
        {
        case Device::CPU:
            Copy(static_cast<Matrix<T,Device::CPU> const&>(A),
                 static_cast<Matrix<U,Device::CPU>&>(B));
            break;
#ifdef HYDROGEN_HAVE_CUDA
        case Device::GPU:
            Copy(static_cast<Matrix<T,Device::CPU> const&>(A),
                 static_cast<Matrix<U,Device::GPU>&>(B));
            break;
#endif // HYDROGEN_HAVE_CUDA
        default:
            LogicError("Copy: Bad device.");
        }
        break;
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
        switch (B.GetDevice())
        {
        case Device::CPU:
            Copy(static_cast<Matrix<T,Device::GPU> const&>(A),
                 static_cast<Matrix<U,Device::CPU>&>(B));
            break;
        case Device::GPU:
            Copy(static_cast<Matrix<T,Device::GPU> const&>(A),
                 static_cast<Matrix<U,Device::GPU>&>(B));
            break;
        default:
            LogicError("Copy: Bad device.");
        }
        break;
#endif //  HYDROGEN_HAVE_CUDA
    default:
        LogicError("Copy: Bad device.");
    }
}

template<typename T>
void Copy( const Matrix<T>& A, Matrix<T>& B )
{
    EL_DEBUG_CSE
    const Int height = A.Height();
    const Int width = A.Width();
    const Int size = height * width;
    B.Resize( height, width );
    const Int ldA = A.LDim();
    const Int ldB = B.LDim();
    const T* EL_RESTRICT ABuf = A.LockedBuffer();
          T* EL_RESTRICT BBuf = B.Buffer();

    if( ldA == height && ldB == height )
    {
#ifdef _OPENMP
#if defined(HYDROGEN_HAVE_OMP_TASKLOOP)
        const Int numThreads = omp_get_num_threads();
        #pragma omp taskloop default(shared)
        for(Int thread = 0; thread < numThreads; ++thread)
        {
#else
        #pragma omp parallel
        {
            const Int numThreads = omp_get_num_threads();
            const Int thread = omp_get_thread_num();
#endif
            const Int chunk = (size + numThreads - 1) / numThreads;
            const Int start = Min(chunk * thread, size);
            const Int end = Min(chunk * (thread + 1), size);
            MemCopy( &BBuf[start], &ABuf[start], end - start );
        }
#else
        MemCopy( BBuf, ABuf, size );
#endif
    }
    else
    {
        EL_PARALLEL_FOR
        for( Int j=0; j<width; ++j )
        {
            MemCopy(&BBuf[j*ldB], &ABuf[j*ldA], height);
        }
    }
}

#ifdef HYDROGEN_HAVE_CUDA
template<typename T>
void Copy(const Matrix<T,Device::GPU>& A, Matrix<T,Device::GPU>& B)
{
    EL_DEBUG_CSE
    const Int height = A.Height();
    const Int width = A.Width();
    B.Resize(height, width);
    const Int ldA = A.LDim();
    const Int ldB = B.LDim();
    const T* ABuf = A.LockedBuffer();
    T* BBuf = B.Buffer();

    SyncInfo<Device::GPU> syncInfoA = SyncInfoFromMatrix(A), syncInfoB = SyncInfoFromMatrix(B);
    auto syncHelper = MakeMultiSync(syncInfoB, syncInfoA);

    // Launch the copy
    H_CHECK_CUDA(
        cudaMemcpy2DAsync(BBuf, ldB*sizeof(T),
                          ABuf, ldA*sizeof(T),
                          height*sizeof(T), width,
                          cudaMemcpyDeviceToDevice,
                          syncInfoB.stream_));
}

// These inter-device copy functions are SYNCHRONOUS with respect to
// the host.
template <typename T>
void Copy(Matrix<T,Device::CPU> const& A, Matrix<T,Device::GPU>& B)
{
    EL_DEBUG_CSE
    const Int height = A.Height();
    const Int width = A.Width();
    B.Resize(height, width);
    const Int ldA = A.LDim();
    const Int ldB = B.LDim();
    const T* EL_RESTRICT ABuf = A.LockedBuffer();
    T* EL_RESTRICT BBuf = B.Buffer();

    SyncInfo<Device::GPU> syncInfoB = SyncInfoFromMatrix(B);
    InterDeviceCopy<Device::CPU,Device::GPU>::MemCopy2DAsync(
        BBuf, ldB, ABuf, ldA, height, width, syncInfoB.stream_);
    Synchronize(syncInfoB); // Is this necessary??
}

template <typename T>
void Copy(Matrix<T,Device::GPU> const& A, Matrix<T,Device::CPU>& B)
{
    EL_DEBUG_CSE
    const Int height = A.Height();
    const Int width = A.Width();
    B.Resize(height, width);
    const Int ldA = A.LDim();
    const Int ldB = B.LDim();
    const T* EL_RESTRICT ABuf = A.LockedBuffer();
    T* EL_RESTRICT BBuf = B.Buffer();

    SyncInfo<Device::GPU> syncInfoA = SyncInfoFromMatrix(A);
    InterDeviceCopy<Device::GPU,Device::CPU>::MemCopy2DAsync(
        BBuf, ldB, ABuf, ldA, height, width, syncInfoA.stream_);
    Synchronize(syncInfoA); // Is this necessary??
}

// These inter-device copy functions are ASYNCHRONOUS with respect to
// the host.
template <typename T>
void CopyAsync(Matrix<T,Device::CPU> const& A, Matrix<T,Device::GPU>& B)
{
    EL_DEBUG_CSE
    const Int height = A.Height();
    const Int width = A.Width();
    B.Resize(height, width);
    const Int ldA = A.LDim();
    const Int ldB = B.LDim();
    const T* EL_RESTRICT ABuf = A.LockedBuffer();
    T* EL_RESTRICT BBuf = B.Buffer();

    InterDeviceCopy<Device::CPU, Device::GPU>::MemCopy2DAsync(
        BBuf, ldB, ABuf, ldA, height, width, B.Stream());
}

template <typename T>
void CopyAsync(Matrix<T,Device::GPU> const& A, Matrix<T,Device::CPU>& B)
{
    EL_DEBUG_CSE
    const Int height = A.Height();
    const Int width = A.Width();
    B.Resize(height, width);
    const Int ldA = A.LDim();
    const Int ldB = B.LDim();
    const T* EL_RESTRICT ABuf = A.LockedBuffer();
    T* EL_RESTRICT BBuf = B.Buffer();

    InterDeviceCopy<Device::GPU, Device::CPU>::MemCopy2DAsync(
        BBuf, ldB, ABuf, ldA, height, width, A.Stream());
}

#endif // HYDROGEN_HAVE_CUDA

template<typename S,typename T,
         typename/*=EnableIf<CanCast<S,T>>*/>
void Copy(const Matrix<S>& A, Matrix<T>& B)
{
    EL_DEBUG_CSE
    EntrywiseMap(A, B, MakeFunction(Caster<S,T>::Cast));
}

template<typename T,Dist U,Dist V,Device D,
         typename = EnableIf<IsDeviceValidType<T,D>>>
void Copy(const ElementalMatrix<T>& A,
          DistMatrix<T,U,V,ELEMENT,D>& B)
{
    EL_DEBUG_CSE
    B = A;
}

template<typename T,Dist U,Dist V,Device D,
         typename = DisableIf<IsDeviceValidType<T,D>>,
         typename = void>
void Copy(const ElementalMatrix<T>& A,
          DistMatrix<T,U,V,ELEMENT,D>& B)
{
    EL_DEBUG_CSE
    LogicError("Copy: bad data/device combination.");
}

// Datatype conversions should not be very common, and so it is likely best to
// avoid explicitly instantiating every combination
template<typename S,typename T,Dist U,Dist V,Device D,
         typename = EnableIf<IsDeviceValidType<T,D>>>
void Copy( const ElementalMatrix<S>& A, DistMatrix<T,U,V,ELEMENT,D>& B )
{
    EL_DEBUG_CSE
    if (A.Grid() == B.Grid() && A.ColDist() == U && A.RowDist() == V
        && A.GetLocalDevice() == D)
    {
        if( !B.RootConstrained() )
            B.SetRoot( A.Root() );
        if( !B.ColConstrained() )
            B.AlignCols( A.ColAlign() );
        if( !B.RowConstrained() )
            B.AlignRows( A.RowAlign() );
        if( A.Root() == B.Root() &&
            A.ColAlign() == B.ColAlign() && A.RowAlign() == B.RowAlign() )
        {
            B.Resize( A.Height(), A.Width() );
            Copy( A.LockedMatrix(), B.Matrix() );
            return;
        }
    }
    DistMatrix<S,U,V,ELEMENT,D> BOrig(A.Grid());
    BOrig.AlignWith( B );
    BOrig = A;
    B.Resize( A.Height(), A.Width() );
    Copy( BOrig.LockedMatrix(), B.Matrix() );
}

template<typename S,typename T,Dist U,Dist V,Device D,
         typename=DisableIf<IsDeviceValidType<T,D>>,
         typename=void>
void Copy( const ElementalMatrix<S>& A, DistMatrix<T,U,V,ELEMENT,D>& B )
{
    EL_DEBUG_CSE
    LogicError("Copy: bad data/device combination.");
}

template<typename T,Dist U,Dist V>
void Copy( const BlockMatrix<T>& A, DistMatrix<T,U,V,BLOCK>& B )
{
    EL_DEBUG_CSE
    B = A;
}

// Datatype conversions should not be very common, and so it is likely best to
// avoid explicitly instantiating every combination
template<typename S,typename T,Dist U,Dist V>
void Copy( const BlockMatrix<S>& A, DistMatrix<T,U,V,BLOCK>& B )
{
    EL_DEBUG_CSE
    if( A.Grid() == B.Grid() && A.ColDist() == U && A.RowDist() == V )
    {
        if( !B.RootConstrained() )
            B.SetRoot( A.Root() );
        if( !B.ColConstrained() )
            B.AlignColsWith( A.DistData() );
        if( !B.RowConstrained() )
            B.AlignRowsWith( A.DistData() );
        if( A.Root() == B.Root() &&
            A.ColAlign() == B.ColAlign() &&
            A.RowAlign() == B.RowAlign() &&
            A.ColCut() == B.ColCut() &&
            A.RowCut() == B.RowCut() )
        {
            B.Resize( A.Height(), A.Width() );
            Copy( A.LockedMatrix(), B.Matrix() );
            return;
        }
    }
    DistMatrix<S,U,V,BLOCK> BOrig(A.Grid());
    BOrig.AlignWith( B );
    BOrig = A;
    B.Resize( A.Height(), A.Width() );
    Copy( BOrig.LockedMatrix(), B.Matrix() );
}

template<typename S,typename T,
         typename/*=EnableIf<CanCast<S,T>>*/>
void Copy( const ElementalMatrix<S>& A, ElementalMatrix<T>& B )
{
    EL_DEBUG_CSE
#define GUARD(CDIST,RDIST,WRAP,DEVICE)                                        \
        (B.ColDist() == CDIST) && (B.RowDist() == RDIST)                \
            && (B.Wrap() == WRAP) && (B.GetLocalDevice() == DEVICE)
#define PAYLOAD(CDIST,RDIST,WRAP,DEVICE)                                \
        auto& BCast =                                                   \
            static_cast<DistMatrix<T,CDIST,RDIST,ELEMENT,DEVICE>&>(B);  \
        Copy( A, BCast );
    #include <El/macros/DeviceGuardAndPayload.h>
}

template<typename T>
void Copy( const AbstractDistMatrix<T>& A, AbstractDistMatrix<T>& B )
{
    EL_DEBUG_CSE
    const DistWrap wrapA=A.Wrap(), wrapB=B.Wrap();
    if( wrapA == ELEMENT && wrapB == ELEMENT )
    {
        auto& ACast = static_cast<const ElementalMatrix<T>&>(A);
        auto& BCast = static_cast<ElementalMatrix<T>&>(B);
        Copy( ACast, BCast );
    }
    else if( wrapA == BLOCK && wrapB == BLOCK )
    {
        auto& ACast = static_cast<const BlockMatrix<T>&>(A);
        auto& BCast = static_cast<BlockMatrix<T>&>(B);
        Copy( ACast, BCast );
    }
    else
    {
        copy::GeneralPurpose( A, B );
    }
}

template <typename T, Dist U, Dist V, Device D1, Device D2>
void CopyAsync(DistMatrix<T,U,V,ELEMENT,D1> const& A,
               DistMatrix<T,U,V,ELEMENT,D2>& B)
{
    EL_DEBUG_CSE
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

template <typename T, Dist U, Dist V, Device D>
void CopyAsync(DistMatrix<T,U,V,ELEMENT,D> const& A,
               DistMatrix<T,U,V,ELEMENT,D>& B)
{
    LogicError("CopyAsync: Both matrices on same device (D=",
               DeviceName<D>(), ").");
}

template <typename T, Dist U, Dist V, Device D>
void CopyAsync(ElementalMatrix<T> const& A, DistMatrix<T,U,V,ELEMENT,D>& B)
{
    EL_DEBUG_CSE
    if ((A.ColDist() == U) && (A.RowDist() == V))
    {
        switch (A.GetLocalDevice())
        {
        case Device::CPU:
            CopyAsync(
                static_cast<DistMatrix<T,U,V,ELEMENT,Device::CPU> const&>(A),
                B);
            break;
#ifdef HYDROGEN_HAVE_CUDA
        case Device::GPU:
            CopyAsync(
                static_cast<DistMatrix<T,U,V,ELEMENT,Device::GPU> const&>(A),
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

template <typename T>
void CopyAsync(ElementalMatrix<T> const& A, ElementalMatrix<T>& B)
{
    EL_DEBUG_CSE
#define GUARD(CDIST,RDIST,WRAP,DEVICE)                              \
    (B.ColDist() == CDIST) && (B.RowDist() == RDIST)                \
        && (B.Wrap() == WRAP) && (B.GetLocalDevice() == DEVICE)
#define PAYLOAD(CDIST,RDIST,WRAP,DEVICE)                            \
    auto& BCast =                                                   \
        static_cast<DistMatrix<T,CDIST,RDIST,ELEMENT,DEVICE>&>(B);  \
    CopyAsync(A, BCast);
    #include <El/macros/DeviceGuardAndPayload.h>
}


template <typename T>
void CopyAsync(AbstractDistMatrix<T> const& A, AbstractDistMatrix<T>& B)
{
    EL_DEBUG_CSE
    const DistWrap wrapA = A.Wrap(), wrapB = B.Wrap();
    if (wrapA == ELEMENT && wrapB == ELEMENT)
    {
        auto& ACast = static_cast<const ElementalMatrix<T>&>(A);
        auto& BCast = static_cast<ElementalMatrix<T>&>(B);
        CopyAsync(ACast, BCast);
    }
    else
        LogicError("CopyAsync only implemented for ElementalMatrix.");
}

template<typename S,typename T,
         typename/*=EnableIf<CanCast<S,T>>*/>
void Copy( const AbstractDistMatrix<S>& A, AbstractDistMatrix<T>& B )
{
    EL_DEBUG_CSE
    const DistWrap wrapA=A.Wrap(), wrapB=B.Wrap();
    if( wrapA == ELEMENT && wrapB == ELEMENT )
    {
        auto& ACast = static_cast<const ElementalMatrix<S>&>(A);
        auto& BCast = static_cast<ElementalMatrix<T>&>(B);
        Copy( ACast, BCast );
    }
    else if( wrapA == BLOCK && wrapB == BLOCK )
    {
        auto& ACast = static_cast<const BlockMatrix<S>&>(A);
        auto& BCast = static_cast<BlockMatrix<T>&>(B);
        Copy( ACast, BCast );
    }
    else
    {
        LogicError("If you see this error, please tell Tom.");
        copy::GeneralPurpose( A, B );
    }
}

template<typename S,typename T,
         typename/*=EnableIf<CanCast<S,T>>*/>
void Copy( const BlockMatrix<S>& A, BlockMatrix<T>& B )
{
    EL_DEBUG_CSE
    #define GUARD(CDIST,RDIST,WRAP) \
        B.ColDist() == CDIST && B.RowDist() == RDIST && B.Wrap() == WRAP \
            && B.GetLocalDevice() == Device::CPU
    #define PAYLOAD(CDIST,RDIST,WRAP) \
      auto& BCast = static_cast<DistMatrix<T,CDIST,RDIST,BLOCK>&>(B); \
      Copy( A, BCast );
    #include <El/macros/GuardAndPayload.h>
}

template<typename T>
void CopyFromRoot
( const Matrix<T>& A, DistMatrix<T,CIRC,CIRC>& B, bool includingViewers )
{
    EL_DEBUG_CSE
    if( B.CrossRank() != B.Root() )
        LogicError("Called CopyFromRoot from non-root");
    B.Resize( A.Height(), A.Width() );
    B.MakeSizeConsistent( includingViewers );
    B.Matrix() = A;
}

template<typename T>
void CopyFromNonRoot( DistMatrix<T,CIRC,CIRC>& B, bool includingViewers )
{
    EL_DEBUG_CSE
    if( B.CrossRank() == B.Root() )
        LogicError("Called CopyFromNonRoot from root");
    B.MakeSizeConsistent( includingViewers );
}

template<typename T>
void CopyFromRoot
( const Matrix<T>& A, DistMatrix<T,CIRC,CIRC,BLOCK>& B,
  bool includingViewers )
{
    EL_DEBUG_CSE
    if( B.CrossRank() != B.Root() )
        LogicError("Called CopyFromRoot from non-root");
    B.Resize( A.Height(), A.Width() );
    B.MakeSizeConsistent( includingViewers );
    B.Matrix() = A;
}

template<typename T>
void CopyFromNonRoot
( DistMatrix<T,CIRC,CIRC,BLOCK>& B, bool includingViewers )
{
    EL_DEBUG_CSE
    if( B.CrossRank() == B.Root() )
        LogicError("Called CopyFromNonRoot from root");
    B.MakeSizeConsistent( includingViewers );
}

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T)                                                        \
    EL_EXTERN template void Copy(                                       \
        AbstractMatrix<T> const& A,                                     \
        AbstractMatrix<T>& B);                                          \
    EL_EXTERN template void Copy(                                       \
        Matrix<T> const& A,                                             \
        Matrix<T>& B);                                                  \
    EL_EXTERN template void Copy(                                       \
        AbstractDistMatrix<T> const& A,                                 \
        AbstractDistMatrix<T>& B);                                      \
    EL_EXTERN template void CopyFromRoot(                               \
        Matrix<T> const& A,                                             \
        DistMatrix<T,CIRC,CIRC>& B,                                     \
        bool includingViewers);                                         \
    EL_EXTERN template void CopyFromNonRoot(                            \
        DistMatrix<T,CIRC,CIRC>& B,                                     \
        bool includingViewers);                                         \
    EL_EXTERN template void CopyFromRoot(                               \
        Matrix<T> const& A,                                             \
        DistMatrix<T,CIRC,CIRC,BLOCK>& B,                               \
        bool includingViewers);                                         \
    EL_EXTERN template void CopyFromNonRoot(                            \
        DistMatrix<T,CIRC,CIRC,BLOCK>& B,                               \
        bool includingViewers);                                         \
    EL_EXTERN template void CopyAsync(                                  \
        AbstractDistMatrix<T> const& A,                                 \
        AbstractDistMatrix<T>& B);


#ifdef HYDROGEN_HAVE_CUDA
#ifdef HYDROGEN_GPU_USE_FP16
EL_EXTERN template void Copy(
    const AbstractMatrix<gpu_half_type>& A,
    AbstractMatrix<gpu_half_type>& B);
EL_EXTERN template void Copy(
    const AbstractDistMatrix<gpu_half_type>& A,
    AbstractDistMatrix<gpu_half_type>& B);
EL_EXTERN template void Copy(
    const Matrix<gpu_half_type,Device::GPU>& A,
    Matrix<gpu_half_type,Device::GPU>& B);
EL_EXTERN template void Copy(
    const Matrix<gpu_half_type,Device::GPU>& A,
    Matrix<gpu_half_type,Device::CPU>& B);
EL_EXTERN template void Copy(
    const Matrix<gpu_half_type,Device::CPU>& A,
    Matrix<gpu_half_type,Device::GPU>& B);
EL_EXTERN template void CopyAsync(
    const Matrix<gpu_half_type,Device::GPU>& A,
    Matrix<gpu_half_type,Device::CPU>& B);
EL_EXTERN template void CopyAsync(
    const Matrix<gpu_half_type,Device::CPU>& A,
    Matrix<gpu_half_type,Device::GPU>& B);
#endif
EL_EXTERN template void Copy(
    Matrix<float,Device::GPU> const& A,
    Matrix<float,Device::GPU>& B);
EL_EXTERN template void Copy(
    Matrix<float,Device::GPU> const& A,
    Matrix<float,Device::CPU>& B);
EL_EXTERN template void Copy(
    Matrix<float,Device::CPU> const& A,
    Matrix<float,Device::GPU>& B);
EL_EXTERN template void CopyAsync(
    Matrix<float,Device::GPU> const& A,
    Matrix<float,Device::CPU>& B);
EL_EXTERN template void CopyAsync(
    Matrix<float,Device::CPU> const& A,
    Matrix<float,Device::GPU>& B);
EL_EXTERN template void Copy(
    Matrix<double,Device::GPU> const& A,
    Matrix<double,Device::GPU>& B);
EL_EXTERN template void Copy(
    Matrix<double,Device::GPU> const& A,
    Matrix<double,Device::CPU>& B);
EL_EXTERN template void Copy(
    Matrix<double,Device::CPU> const& A,
    Matrix<double,Device::GPU>& B);
EL_EXTERN template void CopyAsync(
    Matrix<double,Device::GPU> const& A,
    Matrix<double,Device::CPU>& B );
EL_EXTERN template void CopyAsync(
    Matrix<double,Device::CPU> const& A,
    Matrix<double,Device::GPU>& B );
#endif // HYDROGEN_HAVE_CUDA

#ifdef HYDROGEN_HAVE_HALF
EL_EXTERN template void Copy(
    AbstractMatrix<cpu_half_type> const& A,
    AbstractMatrix<cpu_half_type>& B );
EL_EXTERN template void Copy(
    AbstractDistMatrix<cpu_half_type> const& A,
    AbstractDistMatrix<cpu_half_type>& B );
EL_EXTERN template void Copy(
    Matrix<cpu_half_type> const& A,
    Matrix<cpu_half_type>& B );
#endif // HYDROGEN_HAVE_HALF

EL_EXTERN template void Copy(
    AbstractMatrix<uint8_t> const& A,
    AbstractMatrix<uint8_t>& B );
EL_EXTERN template void Copy(
    Matrix<uint8_t> const& A,
    Matrix<uint8_t>& B );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_COPY_HPP
