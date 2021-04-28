/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like/level1.hpp>
#include <El/blas_like/level3.hpp>

#include "./Syrk/LN.hpp"
#include "./Syrk/LT.hpp"
#include "./Syrk/UN.hpp"
#include "./Syrk/UT.hpp"

namespace El {
namespace {
template <typename T, typename=EnableWhen<IsComputeType<T,Device::CPU>>>
void SyrkImpl_(
    UpperOrLower uplo, Orientation orientation,
    T alpha, Matrix<T, Device::CPU> const& A,
    T beta, Matrix<T, Device::CPU>& C,
    bool conjugate)
{
    const char uploChar = UpperOrLowerToChar(uplo);
    const char transChar = OrientationToChar(orientation);
    const Int k = (orientation == NORMAL ? A.Width() : A.Height());
    if (conjugate)
    {
        blas::Herk(uploChar, transChar, C.Height(), k,
                   RealPart(alpha), A.LockedBuffer(), A.LDim(),
                   RealPart(beta),  C.Buffer(),       C.LDim());
    }
    else
    {
        blas::Syrk(uploChar, transChar, C.Height(), k,
          alpha, A.LockedBuffer(), A.LDim(),
          beta,  C.Buffer(),       C.LDim());
    }
}

template <typename T,
          typename=EnableUnless<IsComputeType<T,Device::CPU>>,
          typename=void>
void SyrkImpl_(
    UpperOrLower, Orientation ,
    T, Matrix<T, Device::CPU> const&,
    T, Matrix<T, Device::CPU>&,
    bool)
{
    RuntimeError("Function not implemented for type on CPU.");
}

#if defined HYDROGEN_HAVE_GPU
template <typename T, typename=EnableWhen<IsComputeType<T,Device::GPU>>>
void SyrkImpl_(
    UpperOrLower uplo_in, Orientation orientation,
    T alpha, Matrix<T, Device::GPU> const& A,
    T beta, Matrix<T, Device::GPU>& C,
    bool conjugate)
{
    auto multisync = MakeMultiSync(
        SyncInfoFromMatrix(C), SyncInfoFromMatrix(A));
    Int const k = (orientation == NORMAL ? A.Width() : A.Height());
    FillMode const uplo = UpperOrLowerToFillMode(uplo_in);
    TransposeMode const trans = OrientationToTransposeMode(orientation);
    if (conjugate)
    {
        gpu_blas::Herk(uplo, trans, C.Height(), k,
                       RealPart(alpha), A.LockedBuffer(), A.LDim(),
                       RealPart(beta),  C.Buffer(),       C.LDim(),
                       multisync);
    }
    else
    {
        gpu_blas::Syrk(uplo, trans, C.Height(), k,
                       alpha, A.LockedBuffer(), A.LDim(),
                       beta,  C.Buffer(),       C.LDim(),
                       multisync);
    }
}

template <typename T,
          typename=EnableUnless<IsComputeType<T,Device::GPU>>,
          typename=void>
void SyrkImpl_(
    UpperOrLower, Orientation ,
    T, Matrix<T, Device::GPU> const&,
    T, Matrix<T, Device::GPU>&,
    bool)
{
    RuntimeError("Function not implemented for type on GPU.");
}
#endif // defined HYDROGEN_HAVE_GPU
}// namespace

template <typename T, Device D>
void Syrk(
    UpperOrLower uplo, Orientation orientation,
    T alpha, Matrix<T, D> const& A,
    T beta, Matrix<T, D>& C,
    bool conjugate)
{
    EL_DEBUG_CSE;
#ifndef EL_RELEASE
    if (orientation == NORMAL)
    {
        if (A.Height() != C.Height() || A.Height() != C.Width())
            LogicError("Nonconformal Syrk");
    }
    else
    {
        if (A.Width() != C.Height() || A.Width() != C.Width())
            LogicError("Nonconformal Syrk");
    }
#endif // not defined EL_RELEASE
    SyrkImpl_(uplo, orientation, alpha, A, beta, C, conjugate);
}

template <typename T, Device D>
void Syrk(
    UpperOrLower uplo, Orientation orientation,
    T alpha, Matrix<T,D> const& A,
    Matrix<T,D>& C,
    bool conjugate)
{
    EL_DEBUG_CSE;
    Int const n = (orientation==NORMAL ? A.Height() : A.Width());
    C.Resize(n, n);
    // FIXME(trb 07/20/2020): I don't think we need this. According to
    // cuBLAS docs, if beta==0, the input doesn't have to be
    // valid. According to netlib source code, the same is
    // true. Assuming other implementations are similarly robust, we
    // should be ok with out this.
    //
    // Zero(C);
    Syrk(uplo, orientation, alpha, A, T(0), C, conjugate);
}

template <typename T>
void Syrk(
    UpperOrLower uplo, Orientation orientation,
    T alpha, AbstractMatrix<T> const& A,
    AbstractMatrix<T>& C,
    bool conjugate)
{
    switch (A.GetDevice())
    {
    case Device::CPU:
        Syrk(uplo, orientation,
             alpha,
             static_cast<Matrix<T,Device::CPU> const&>(A),
             static_cast<Matrix<T,Device::CPU>&>(C),
             conjugate);
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        Syrk(uplo, orientation,
             alpha,
             static_cast<Matrix<T,Device::GPU> const&>(A),
             static_cast<Matrix<T,Device::GPU>&>(C),
             conjugate);
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        RuntimeError("Invalid device.");
    }
}

template <typename T>
void Syrk(
    UpperOrLower uplo, Orientation orientation,
    T alpha, AbstractMatrix<T> const& A,
    T beta, AbstractMatrix<T>& C,
    bool conjugate)
{
    switch (A.GetDevice())
    {
    case Device::CPU:
        Syrk(uplo, orientation,
             alpha, static_cast<Matrix<T,Device::CPU> const&>(A),
             beta, static_cast<Matrix<T,Device::CPU>&>(C),
             conjugate);
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        Syrk(uplo, orientation,
             alpha, static_cast<Matrix<T,Device::GPU> const&>(A),
             beta, static_cast<Matrix<T,Device::GPU>&>(C),
             conjugate);
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        RuntimeError("Invalid device.");
    }
}

template <typename T>
void Syrk(
    UpperOrLower uplo, Orientation orientation,
    T alpha, AbstractDistMatrix<T> const& A,
    T beta, AbstractDistMatrix<T>& C,
    bool conjugate)
{
    EL_DEBUG_CSE;
    ScaleTrapezoid(beta, uplo, C);
    if (uplo == LOWER && orientation == NORMAL)
        syrk::LN(alpha, A, C, conjugate);
    else if (uplo == LOWER)
        syrk::LT(alpha, A, C, conjugate);
    else if (orientation == NORMAL)
        syrk::UN(alpha, A, C, conjugate);
    else
        syrk::UT(alpha, A, C, conjugate);
}

template <typename T>
void Syrk(
    UpperOrLower uplo, Orientation orientation,
    T alpha, AbstractDistMatrix<T> const& A,
    AbstractDistMatrix<T>& C, bool conjugate)
{
    EL_DEBUG_CSE
    const Int n = (orientation==NORMAL ? A.Height() : A.Width());
    C.Resize(n, n);
    Zero(C);
    Syrk(uplo, orientation, alpha, A, T(0), C, conjugate);
}

#define PROTO(T)                                                        \
    template void Syrk(                                                 \
        UpperOrLower uplo, Orientation orientation,                     \
        T alpha, AbstractMatrix<T> const& A,                            \
        T beta, AbstractMatrix<T>& C, bool conjugate);                  \
    template void Syrk(                                                 \
        UpperOrLower uplo, Orientation orientation,                     \
        T alpha, AbstractMatrix<T> const& A,                            \
        AbstractMatrix<T>& C, bool conjugate);                          \
    template void Syrk(                                                 \
        UpperOrLower uplo, Orientation orientation,                     \
        T alpha, AbstractDistMatrix<T> const& A,                        \
        T beta, AbstractDistMatrix<T>& C, bool conjugate);              \
    template void Syrk(                                                 \
        UpperOrLower uplo, Orientation orientation,                     \
        T alpha, AbstractDistMatrix<T> const& A,                        \
        AbstractDistMatrix<T>& C, bool conjugate);

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#define EL_ENABLE_HALF
#include <El/macros/Instantiate.h>

} // namespace El
