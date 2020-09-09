/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_ENTRYWISEMAP_HPP
#define EL_BLAS_ENTRYWISEMAP_HPP

#include "El/core/DistMatrix/AbstractDistMatrix.hpp"
#if defined HYDROGEN_HAVE_GPU
#include "hydrogen/blas/gpu/CombineImpl.hpp"
#include "hydrogen/blas/gpu/EntrywiseMapImpl.hpp"
#endif // defined HYDROGEN_HAVE_GPU

namespace El {

template<typename T>
void EntrywiseMap(AbstractMatrix<T>& A, function<T(const T&)> func)
{
    EL_DEBUG_CSE

    if (A.GetDevice() != Device::CPU)
        LogicError("EntrywiseMap not allowed on non-CPU matrices.");

    const Int m = A.Height();
    const Int n = A.Width();
    T* ABuf = A.Buffer();
    const Int ALDim = A.LDim();

    // Iterate over single loop if memory is contiguous. Otherwise
    // iterate over double loop.
    if (ALDim == m)
    {
        EL_PARALLEL_FOR
        for(Int i=0; i<m*n; ++i)
        {
            ABuf[i] = func(ABuf[i]);
        }
    }
    else
    {
        EL_PARALLEL_FOR
        for(Int j=0; j<n; ++j)
        {
            EL_SIMD
            for(Int i=0; i<m; ++i)
            {
                ABuf[i+j*ALDim] = func(ABuf[i+j*ALDim]);
            }
        }
    }
}

template<typename T>
void EntrywiseMap(AbstractDistMatrix<T>& A, function<T(const T&)> func)
{ EntrywiseMap(A.Matrix(), func); }

template<typename S,typename T>
void EntrywiseMap
(const AbstractMatrix<S>& A, AbstractMatrix<T>& B, function<T(const S&)> func)
{
    EL_DEBUG_CSE

    if ((A.GetDevice() != Device::CPU) || (B.GetDevice() != Device::CPU))
        LogicError("EntrywiseMap not allowed on non-CPU matrices.");

    const Int m = A.Height();
    const Int n = A.Width();
    B.Resize(m, n);
    const S* ABuf = A.LockedBuffer();
    T* BBuf = B.Buffer();
    const Int ALDim = A.LDim();
    const Int BLDim = B.LDim();
    EL_PARALLEL_FOR
    for(Int j=0; j<n; ++j)
    {
        EL_SIMD
        for(Int i=0; i<m; ++i)
        {
            BBuf[i+j*BLDim] = func(ABuf[i+j*ALDim]);
        }
    }
}

template <Dist U, Dist V, DistWrap W, Device D, typename S, typename T,
          typename=EnableIf<IsDeviceValidType<S,D>>>
void EntrywiseMap_payload(
    AbstractDistMatrix<S> const& A,
    AbstractDistMatrix<T>& B,
    function<T(const S&)> func)
{
    DistMatrix<S,U,V,W,D> AProx(B.Grid());
    AProx.AlignWith(B.DistData());
    Copy(A, AProx);
    EntrywiseMap(AProx.Matrix(), B.Matrix(), func);
}

template <Dist U, Dist V, DistWrap W, Device D, typename S, typename T,
          typename=DisableIf<IsDeviceValidType<S,D>>, typename=void>
void EntrywiseMap_payload(
    AbstractDistMatrix<S> const&,
    AbstractDistMatrix<T>&,
    function<T(const S&)>)
{
    LogicError("EntrywiseMap: Bad device/type combination.");
}

template<typename S,typename T>
void EntrywiseMap
(const AbstractDistMatrix<S>& A,
        AbstractDistMatrix<T>& B,
        function<T(const S&)> func)
{
    if (A.DistData().colDist == B.DistData().colDist &&
        A.DistData().rowDist == B.DistData().rowDist &&
        A.Wrap() == B.Wrap())
    {
        B.AlignWith(A.DistData());
        B.Resize(A.Height(), A.Width());
        EntrywiseMap(A.LockedMatrix(), B.Matrix(), func);
    }
    else
    {
        B.Resize(A.Height(), A.Width());
        #define GUARD(CDIST,RDIST,WRAP,DEVICE) \
          B.DistData().colDist == CDIST && B.DistData().rowDist == RDIST && \
              B.Wrap() == WRAP && B.GetLocalDevice() == DEVICE
        #define PAYLOAD(CDIST,RDIST,WRAP,DEVICE) \
            EntrywiseMap_payload<CDIST,RDIST,WRAP,DEVICE>(A,B,func);
        #include <El/macros/DeviceGuardAndPayload.h>
        #undef GUARD
        #undef PAYLOAD
    }
}

#if defined HYDROGEN_HAVE_GPU
// This section only valid when device-compiling.
#if defined __CUDACC__ || defined __HIPCC__
/** @brief Entrywise map function for GPU matrices.
 *
 *  This function handles only the high-level Resize and
 *  Synchronization tasks. The kernel launch is done elsewhere.
 *
 *  @param A The source matrix.
 *  @param B The target matrix.
 *  @param func The functor to apply entrywise to elements of A. The
 *              signature should be `T(S const&)` or equivalent. It
 *              must be device-executable code.
 */
template <typename S, typename T, typename FunctorT>
void EntrywiseMap(Matrix<S, Device::GPU> const& A,
                  Matrix<T, Device::GPU>& B,
                  FunctorT func)
{
    B.Resize(A.Height(), A.Width());

    auto multisync = hydrogen::MakeMultiSync(
        SyncInfoFromMatrix(B), SyncInfoFromMatrix(A));
    hydrogen::device::EntrywiseMapImpl(
        A.Height(), A.Width(),
        A.LockedBuffer(), A.LDim(),
        B.Buffer(), B.LDim(),
        func,
        multisync);
}

/** @brief Combine function for GPU matrices.
 *
 *  The operation here is `Bij <- func(Aij, Bij)`. This _could_ be
 *  very hackily implemented with EntrywiseMap or with the right
 *  closure being passed into an IndexDependentMap function, but it's
 *  probably better to have the dedicated API.
 *
 *  This function handles only the high-level Resize and
 *  Synchronization tasks. The kernel launch is done elsewhere.
 *
 *  @param A The source matrix.
 *  @param B The target matrix.
 *  @param func The functor to apply entrywise to elements of A. The
 *              signature should be `T(S const&, T const&)` or equivalent. It
 *              must be device-executable code.
 */
template <typename S, typename T, typename FunctorT>
void Combine(Matrix<S, Device::GPU> const& A,
             Matrix<T, Device::GPU>& B,
             FunctorT func)
{
    if (A.Height() != B.Height() || A.Width() != B.Width())
        RuntimeError("A and B must be the same size for Combine.");

    auto multisync = hydrogen::MakeMultiSync(
        SyncInfoFromMatrix(B), SyncInfoFromMatrix(A));
    hydrogen::device::CombineImpl(
        A.Height(), A.Width(),
        A.LockedBuffer(), A.LDim(),
        B.Buffer(), B.LDim(),
        func,
        multisync);
}
#else
// Just declare the template prototypes if not in device compilation.
template <typename S, typename T, typename FunctorT>
void Combine(Matrix<S, Device::GPU> const& A,
             Matrix<T, Device::GPU>& B,
             FunctorT func);
template <typename S, typename T, typename FunctorT>
void EntrywiseMap(Matrix<S, Device::GPU> const& A,
                  Matrix<T, Device::GPU>& B,
                  FunctorT func);
#endif // defined __CUDACC__ || defined __HIPCC__
#endif // defined HYDROGEN_HAVE_GPU

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T)                                \
    EL_EXTERN template void EntrywiseMap        \
    (AbstractMatrix<T>& A,                      \
     function<T(const T&)> func);               \
    EL_EXTERN template void EntrywiseMap        \
    (AbstractDistMatrix<T>& A,                  \
     function<T(const T&)> func);               \
    EL_EXTERN template void EntrywiseMap        \
    (const AbstractMatrix<T>& A,                \
     AbstractMatrix<T>& B,                      \
     function<T(const T&)> func);               \
    EL_EXTERN template void EntrywiseMap        \
    (const AbstractDistMatrix<T>& A,            \
     AbstractDistMatrix<T>& B,                  \
     function<T(const T&)> func);

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_ENTRYWISEMAP_HPP
