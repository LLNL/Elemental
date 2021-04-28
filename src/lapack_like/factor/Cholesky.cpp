/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

// HACK
#ifdef HYDROGEN_HAVE_GPU
namespace {
template <typename T>
void LocalGPUCholesky(hydrogen::FillMode uplo,
                      El::Matrix<T, hydrogen::Device::GPU>& A)
{
#ifndef EL_RELEASE
    if (A.Height() != A.Width())
        El::LogicError("Can only compute Cholesky factor of square matrices");
#endif // EL_RELEASE
    hydrogen::gpu_lapack::CholeskyFactorize(
        uplo, A.Height(), A.Buffer(), A.LDim(),
        El::SyncInfoFromMatrix(A));
}
}// namespace
#endif // HYDROGEN_HAVE_GPU

#include "./Cholesky/LowerVariant3.hpp"
#include "./Cholesky/UpperVariant3.hpp"
#include "./Cholesky/ReverseLowerVariant3.hpp"
#include "./Cholesky/ReverseUpperVariant3.hpp"
#include "./Cholesky/PivotedLowerVariant3.hpp"
#include "./Cholesky/PivotedUpperVariant3.hpp"
#include "./Cholesky/SolveAfter.hpp"

#include "./Cholesky/LowerMod.hpp"
#include "./Cholesky/UpperMod.hpp"

namespace El {

// TODO: Pivoted Reverse Cholesky?

#ifdef HYDROGEN_HAVE_GPU
template <typename F>
void Cholesky(UpperOrLower uplo, Matrix<F,Device::GPU>& A)
{
    LocalGPUCholesky(UpperOrLowerToFillMode(uplo), A);
}
#endif // HYDROGEN_HAVE_GPU

template <typename F>
void Cholesky(UpperOrLower uplo, Matrix<F,Device::CPU>& A)
{
    EL_DEBUG_CSE;
#ifndef EL_RELEASE
    if (A.Height() != A.Width())
        LogicError("A must be square");
#endif // EL_RELEASE
    if (uplo == LOWER)
        cholesky::LowerVariant3Blocked(A);
    else
        cholesky::UpperVariant3Blocked(A);
}

template <typename F>
void Cholesky(UpperOrLower uplo, AbstractMatrix<F>& A)
{
    switch (A.GetDevice())
    {
    case Device::CPU:
        Cholesky(uplo, static_cast<Matrix<F,Device::CPU>&>(A));
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        Cholesky(uplo, static_cast<Matrix<F,Device::GPU>&>(A));
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        RuntimeError("Bad device.");
    }
}

#ifdef HYDROGEN_ENABLE_PIVOTED_CHOLESKY
template <typename F>
void Cholesky(UpperOrLower uplo, Matrix<F>& A, Permutation& p)
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
      if (A.Height() != A.Width())
          LogicError("A must be square");
   )
    if (uplo == LOWER)
        cholesky::PivotedLowerVariant3Blocked(A, p);
    else
        cholesky::PivotedUpperVariant3Blocked(A, p);
}
#endif // HYDROGEN_ENABLE_PIVOTED_CHOLESKY

#ifdef HYDROGEN_ENABLE_REVERSE_CHOLESKY
template <typename F>
void ReverseCholesky(UpperOrLower uplo, Matrix<F>& A)
{
    EL_DEBUG_CSE;
#ifndef EL_RELEASE
      if (A.Height() != A.Width())
          LogicError("A must be square");
#endif // EL_RELEASE
    if (uplo == LOWER)
        cholesky::ReverseLowerVariant3Blocked(A);
    else
        cholesky::ReverseUpperVariant3Blocked(A);
}
#endif // HYDROGEN_ENABLE_REVERSE_CHOLESKY

namespace cholesky {

template <typename F,typename=EnableIf<IsBlasScalar<F>>>
void ScaLAPACKHelper(UpperOrLower uplo, AbstractDistMatrix<F>& A)
{
    AssertScaLAPACKSupport();
#ifdef EL_HAVE_SCALAPACK
    // TODO: Add support for optionally timing the proxy redistribution
    DistMatrixReadWriteProxy<F,F,MC,MR,BLOCK> ABlockProx(A);
    auto& ABlock = ABlockProx.Get();

    const Int n = ABlock.Height();
    const char uploChar = UpperOrLowerToChar(uplo);

    auto descA = FillDesc(ABlock);
    scalapack::Cholesky(uploChar, n, ABlock.Buffer(), descA.data());
#endif
}

template <typename F,typename=DisableIf<IsBlasScalar<F>>,typename=void>
void ScaLAPACKHelper(UpperOrLower uplo, AbstractDistMatrix<F>& A)
{
    RuntimeError("There is no ScaLAPACK support for this datatype");
}

} // anonymous namespace

template <typename F>
void Cholesky(UpperOrLower uplo, AbstractDistMatrix<F>& A, bool scalapack)
{
    EL_DEBUG_CSE;
    if (scalapack)
    {
        cholesky::ScaLAPACKHelper(uplo, A);
    }
    else
    {
        if (uplo == LOWER)
            cholesky::LowerVariant3Blocked(A);
        else
            cholesky::UpperVariant3Blocked(A);
    }
}

#ifdef HYDROGEN_ENABLE_PIVOTED_CHOLESKY
template <typename F>
void Cholesky
(UpperOrLower uplo, AbstractDistMatrix<F>& A, DistPermutation& p)
{
    EL_DEBUG_CSE;
    if (uplo == LOWER)
        cholesky::PivotedLowerVariant3Blocked(A, p);
    else
        cholesky::PivotedUpperVariant3Blocked(A, p);
}
#endif // HYDROGEN_ENABLE_PIVOTED_CHOLESKY

template <typename F>
void Cholesky
(UpperOrLower uplo, DistMatrix<F,STAR,STAR>& A)
{ Cholesky(uplo, A.Matrix()); }

#ifdef HYDROGEN_ENABLE_REVERSE_CHOLESKY
template <typename F>
void ReverseCholesky(UpperOrLower uplo, AbstractDistMatrix<F>& A)
{
    EL_DEBUG_CSE;
    if (uplo == LOWER)
        cholesky::ReverseLowerVariant3Blocked(A);
    else
        cholesky::ReverseUpperVariant3Blocked(A);
}

template <typename F>
void ReverseCholesky
(UpperOrLower uplo, DistMatrix<F,STAR,STAR>& A)
{ ReverseCholesky(uplo, A.Matrix()); }
#endif // HYDROGEN_ENABLE_REVERSE_CHOLESKY

#ifdef HYDROGEN_ENABLE_CHOLESKY_MOD
// Either
//         L' L'^H := L L^H + alpha V V^H
// or
//         U'^H U' := U^H U + alpha V V^H
template <typename F>
void CholeskyMod(UpperOrLower uplo, Matrix<F>& T, Base<F> alpha, Matrix<F>& V)
{
    EL_DEBUG_CSE;
    if (alpha == Base<F>(0))
        return;
    if (uplo == LOWER)
        cholesky::LowerMod(T, alpha, V);
    else
        cholesky::UpperMod(T, alpha, V);
}

template <typename F>
void CholeskyMod
(UpperOrLower uplo,
  AbstractDistMatrix<F>& T,
  Base<F> alpha,
  AbstractDistMatrix<F>& V)
{
    EL_DEBUG_CSE;
    if (alpha == Base<F>(0))
        return;
    if (uplo == LOWER)
        cholesky::LowerMod(T, alpha, V);
    else
        cholesky::UpperMod(T, alpha, V);
}

template <typename F>
void HPSDCholesky(UpperOrLower uplo, Matrix<F>& A)
{
    EL_DEBUG_CSE;
    HPSDSquareRoot(uplo, A);
    MakeHermitian(uplo, A);

    if (uplo == LOWER)
        lq::ExplicitTriang(A);
    else
        qr::ExplicitTriang(A);
}

template <typename F>
void HPSDCholesky(UpperOrLower uplo, AbstractDistMatrix<F>& APre)
{
    EL_DEBUG_CSE;

    // NOTE: This should be removed once HPSD, LQ, and QR have been generalized
    DistMatrixReadWriteProxy<F,F,MC,MR> AProx(APre);
    auto& A = AProx.Get();

    HPSDSquareRoot(uplo, A);
    MakeHermitian(uplo, A);

    if (uplo == LOWER)
        lq::ExplicitTriang(A);
    else
        qr::ExplicitTriang(A);
}
#endif // HYDROGEN_ENABLE_CHOLESKY_MOD

#define PROTO(F)                                                        \
    template void Cholesky(UpperOrLower uplo, AbstractMatrix<F>& A);    \
    template void Cholesky(                                             \
        UpperOrLower uplo, AbstractDistMatrix<F>& A, bool scalapack);   \
    template void Cholesky(                                             \
        UpperOrLower uplo, DistMatrix<F,STAR,STAR>& A);

#ifdef HYDROGEN_ENABLE_ALL_CHOLESKY
#define PROTO_BASE(F) \
  template void Cholesky(UpperOrLower uplo, Matrix<F>& A); \
  template void Cholesky(                                        \
      UpperOrLower uplo, AbstractDistMatrix<F>& A, bool scalapack);     \
  template void Cholesky(UpperOrLower uplo, DistMatrix<F,STAR,STAR>& A); \
  template void ReverseCholesky(UpperOrLower uplo, Matrix<F>& A); \
  template void ReverseCholesky \
  (UpperOrLower uplo, AbstractDistMatrix<F>& A); \
  template void ReverseCholesky \
  (UpperOrLower uplo, DistMatrix<F,STAR,STAR>& A); \
  template void Cholesky(UpperOrLower uplo, Matrix<F>& A, Permutation& p); \
  template void Cholesky \
  (UpperOrLower uplo, \
    AbstractDistMatrix<F>& A, \
    DistPermutation& p); \
  template void CholeskyMod \
  (UpperOrLower uplo, Matrix<F>& T, Base<F> alpha, Matrix<F>& V); \
  template void CholeskyMod \
  (UpperOrLower uplo, \
    AbstractDistMatrix<F>& T, \
    Base<F> alpha, \
    AbstractDistMatrix<F>& V); \
  template void cholesky::SolveAfter \
  (UpperOrLower uplo, Orientation orientation, \
    const Matrix<F>& A, \
          Matrix<F>& B); \
  template void cholesky::SolveAfter \
  (UpperOrLower uplo, Orientation orientation, \
    const AbstractDistMatrix<F>& A, \
          AbstractDistMatrix<F>& B); \
  template void cholesky::SolveAfter \
  (UpperOrLower uplo, Orientation orientation, \
    const Matrix<F>& A, \
    const Permutation& p, \
          Matrix<F>& B); \
  template void cholesky::SolveAfter \
  (UpperOrLower uplo, Orientation orientation, \
    const AbstractDistMatrix<F>& A, \
    const DistPermutation& p, \
          AbstractDistMatrix<F>& B);

#define PROTO(F) \
  PROTO_BASE(F) \
  template void HPSDCholesky(UpperOrLower uplo, Matrix<F>& A); \
  template void HPSDCholesky(UpperOrLower uplo, AbstractDistMatrix<F>& A);

#define PROTO_DOUBLEDOUBLE PROTO_BASE(DoubleDouble)
#define PROTO_QUADDOUBLE PROTO_BASE(QuadDouble)
#define PROTO_COMPLEX_DOUBLEDOUBLE PROTO_BASE(Complex<DoubleDouble>)
#define PROTO_COMPLEX_QUADDOUBLE PROTO_BASE(Complex<QuadDouble>)
#define PROTO_QUAD PROTO_BASE(Quad)
#define PROTO_COMPLEX_QUAD PROTO_BASE(Complex<Quad>)
#define PROTO_BIGFLOAT PROTO_BASE(BigFloat)
#define PROTO_COMPLEX_BIGFLOAT PROTO_BASE(Complex<BigFloat>)
// #define PROTO_COMPLEX_HALF PROTO_BASE(Complex<cpu_half_type>)
#endif // HYDROGEN_ENABLE_ALL_CHOLESKY

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#define EL_ENABLE_HALF
#include <El/macros/Instantiate.h>

} // namespace El
