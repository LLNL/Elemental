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

#include <El/hydrogen_config.h>

#include <El/blas_like/level1/Copy/internal_decl.hpp>
#include <El/blas_like/level1/Copy/GeneralPurpose.hpp>
#include <El/blas_like/level1/Copy/util.hpp>

#ifdef HYDROGEN_HAVE_GPU
#include <hydrogen/device/gpu/BasicCopy.hpp>
#endif

namespace El {

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
#ifdef HYDROGEN_HAVE_GPU
        case Device::GPU:
            Copy(static_cast<Matrix<T,Device::CPU> const&>(A),
                 static_cast<Matrix<T,Device::GPU>&>(B));
            break;
#endif // HYDROGEN_HAVE_GPU
        default:
            LogicError("Copy: Bad device.");
        }
        break;
#ifdef HYDROGEN_HAVE_GPU
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
#endif //  HYDROGEN_HAVE_GPU
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

#ifdef HYDROGEN_HAVE_GPU
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

    SyncInfo<Device::GPU> syncInfoA = SyncInfoFromMatrix(A),
        syncInfoB = SyncInfoFromMatrix(B);
    auto syncHelper = MakeMultiSync(syncInfoB, syncInfoA);

    gpu::Copy2DIntraDevice(
        ABuf, ldA,
        BBuf, ldB,
        height, width, syncInfoB);
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

    SyncInfo<Device::GPU> syncInfoB = B.GetSyncInfo();SyncInfoFromMatrix(B);
    gpu::Copy2DToDevice(
        ABuf, ldA, BBuf, ldB,
        height, width, syncInfoB);
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

    gpu::Copy2DToHost(
        ABuf, ldA, BBuf, ldB,
        height, width, A.GetSyncInfo());
    Synchronize(A.GetSyncInfo()); // Is this necessary??
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

    gpu::Copy2DToDevice(
        ABuf, ldA, BBuf, ldB,
        height, width, B.GetSyncInfo());
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

    gpu::Copy2DToDevice(
        ABuf, ldA, BBuf, ldB,
        height, width, A.GetSyncInfo());
}

#endif // HYDROGEN_HAVE_GPU

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
#ifdef HYDROGEN_HAVE_GPU
        case Device::GPU:
            CopyAsync(
                static_cast<DistMatrix<T,U,V,ELEMENT,Device::GPU> const&>(A),
                B);
            break;
#endif // HYDROGEN_HAVE_GPU
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

#define PROTO(T) \
  EL_EXTERN template void Copy \
  ( const AbstractMatrix<T>& A, AbstractMatrix<T>& B ); \
  EL_EXTERN template void Copy \
  ( const Matrix<T>& A, Matrix<T>& B ); \
  EL_EXTERN template void Copy \
  ( const AbstractDistMatrix<T>& A, AbstractDistMatrix<T>& B ); \
  EL_EXTERN template void CopyFromRoot \
  ( const Matrix<T>& A, DistMatrix<T,CIRC,CIRC>& B, bool includingViewers ); \
  EL_EXTERN template void CopyFromNonRoot \
  ( DistMatrix<T,CIRC,CIRC>& B, bool includingViewers ); \
  EL_EXTERN template void CopyFromRoot \
  ( const Matrix<T>& A, DistMatrix<T,CIRC,CIRC,BLOCK>& B, \
    bool includingViewers ); \
  EL_EXTERN template void CopyFromNonRoot \
  ( DistMatrix<T,CIRC,CIRC,BLOCK>& B, bool includingViewers ); \
  EL_EXTERN template void CopyAsync \
  ( const AbstractDistMatrix<T>& A, AbstractDistMatrix<T>& B ); \


#ifdef HYDROGEN_HAVE_GPU
#ifdef HYDROGEN_GPU_USE_FP16
EL_EXTERN template void Copy(
    const AbstractMatrix<gpu_half_type>& A, AbstractMatrix<gpu_half_type>& B );
EL_EXTERN template void Copy
( const Matrix<gpu_half_type,Device::GPU>& A, Matrix<gpu_half_type,Device::GPU>& B );
EL_EXTERN template void Copy
( const Matrix<gpu_half_type,Device::GPU>& A, Matrix<gpu_half_type,Device::CPU>& B );
EL_EXTERN template void Copy
( const Matrix<gpu_half_type,Device::CPU>& A, Matrix<gpu_half_type,Device::GPU>& B );
EL_EXTERN template void CopyAsync
( const Matrix<gpu_half_type,Device::GPU>& A, Matrix<gpu_half_type,Device::CPU>& B );
EL_EXTERN template void CopyAsync
( const Matrix<gpu_half_type,Device::CPU>& A, Matrix<gpu_half_type,Device::GPU>& B );
#endif
EL_EXTERN template void Copy
( const Matrix<float,Device::GPU>& A, Matrix<float,Device::GPU>& B );
EL_EXTERN template void Copy
( const Matrix<float,Device::GPU>& A, Matrix<float,Device::CPU>& B );
EL_EXTERN template void Copy
( const Matrix<float,Device::CPU>& A, Matrix<float,Device::GPU>& B );
EL_EXTERN template void CopyAsync
( const Matrix<float,Device::GPU>& A, Matrix<float,Device::CPU>& B );
EL_EXTERN template void CopyAsync
( const Matrix<float,Device::CPU>& A, Matrix<float,Device::GPU>& B );
EL_EXTERN template void Copy
( const Matrix<double,Device::GPU>& A, Matrix<double,Device::GPU>& B );
EL_EXTERN template void Copy
( const Matrix<double,Device::GPU>& A, Matrix<double,Device::CPU>& B );
EL_EXTERN template void Copy
( const Matrix<double,Device::CPU>& A, Matrix<double,Device::GPU>& B );
EL_EXTERN template void CopyAsync
( const Matrix<double,Device::GPU>& A, Matrix<double,Device::CPU>& B );
EL_EXTERN template void CopyAsync
( const Matrix<double,Device::CPU>& A, Matrix<double,Device::GPU>& B );
#endif // HYDROGEN_HAVE_GPU

#ifdef HYDROGEN_HAVE_HALF
EL_EXTERN template void Copy(
    const AbstractMatrix<cpu_half_type>& A, AbstractMatrix<cpu_half_type>& B );
EL_EXTERN template void Copy(
    const Matrix<cpu_half_type>& A, Matrix<cpu_half_type>& B );
#endif // HYDROGEN_HAVE_HALF
EL_EXTERN template void Copy(
    const AbstractMatrix<uint8_t>& A, AbstractMatrix<uint8_t>& B );
EL_EXTERN template void Copy(
    const Matrix<uint8_t>& A, Matrix<uint8_t>& B );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_COPY_HPP
