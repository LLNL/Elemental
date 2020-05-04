/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/

#ifdef HYDROGEN_HAVE_MS_GEMM
#include "TN_Multistream.hpp"
#endif // HYDROGEN_HAVE_MS_GEMM

namespace El {
namespace gemm {

// Transpose Normal Gemm that avoids communicating the matrix A
template <Device D, typename T, typename=EnableIf<IsDeviceValidType<T,D>>>
void SUMMA_TNA_impl
(Orientation orientA,
  T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE
    const Int n = CPre.Width();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();

    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> AProx(APre);
    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> BProx(BPre);
    DistMatrixReadWriteProxy<T,T,MC,MR,ELEMENT,D> CProx(CPre);
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();
    auto& C = CProx.Get();

    // Temporary distributions
    DistMatrix<T,MC,STAR,ELEMENT,D> B1_MC_STAR(g);
    DistMatrix<T,MR,STAR,ELEMENT,D> D1_MR_STAR(g);
    DistMatrix<T,MR,MC  ,ELEMENT,D> D1_MR_MC(g);

    B1_MC_STAR.AlignWith(A);
    D1_MR_STAR.AlignWith(A);

    for(Int k=0; k<n; k+=bsize)
    {
        const Int nb = Min(bsize,n-k);
        auto B1 = B(ALL, IR(k,k+nb));
        auto C1 = C(ALL, IR(k,k+nb));

        // D1[MR,*] := alpha (A1[MC,MR])^T B1[MC,*]
        //           = alpha (A1^T)[MR,MC] B1[MC,*]
        B1_MC_STAR = B1;
        LocalGemm(orientA, NORMAL, alpha, A, B1_MC_STAR, D1_MR_STAR);

        // C1[MC,MR] += scattered & transposed D1[MR,*] summed over grid cols
        Contract(D1_MR_STAR, D1_MR_MC);
        Axpy(TypeTraits<T>::One(), D1_MR_MC, C1);
    }
}

template <Device D, typename T,
          typename=DisableIf<IsDeviceValidType<T,D>>, typename=void>
void SUMMA_TNA_impl
(Orientation orientA,
  T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre)
{
    LogicError("SUMMA_TNA_impl type-device combo not supported.");
}

template<typename T>
void SUMMA_TNA
(Orientation orientA,
  T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE

    switch (CPre.GetLocalDevice())
    {
    case Device::CPU:
        SUMMA_TNA_impl<Device::CPU>(orientA, alpha, APre, BPre, CPre);
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        SUMMA_TNA_impl<Device::GPU>(orientA, alpha, APre, BPre, CPre);
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("SUMMA_TNA: Bad device.");
    }
}

template<typename T>
void SUMMA_TNA_MS(
    Orientation orientA,
    T alpha,
    AbstractDistMatrix<T> const& APre,
    AbstractDistMatrix<T> const& BPre,
    AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE;
#ifndef HYDROGEN_HAVE_MS_GEMM
    OutputFromRoot(CPre.Grid().Comm(),
                   "WARNING: Multistream support not available; "
                   "requires GPU and Aluminum.");
    SUMMA_TNA(orientA, alpha, APre, BPre, CPre);
#else

    switch (CPre.GetLocalDevice())
    {
    case Device::CPU:
        OutputFromRoot(
            CPre.Grid().Comm(),
            "WARNING: CPU doesn't support \"multistream\" variants.");
        SUMMA_TNA_impl<Device::CPU>(orientA, alpha, APre, BPre, CPre);
        break;
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
        SUMMA_TNA_impl_multistream(orientA, alpha, APre, BPre, CPre);
        break;
#endif // HYDROGEN_HAVE_CUDA
    default:
        LogicError("SUMMA_TNA_MS: Bad device.");
    }
#endif // HYDROGEN_HAVE_MS_GEMM
}

// Transpose Normal Gemm that avoids communicating the matrix B
template <Device D, typename T, typename=EnableIf<IsDeviceValidType<T,D>>>
void SUMMA_TNB_impl
(Orientation orientA,
  T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE
    const Int m = CPre.Height();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();
    const bool conjugate = (orientA == ADJOINT);

    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> AProx(APre);
    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> BProx(BPre);
    DistMatrixReadWriteProxy<T,T,MC,MR,ELEMENT,D> CProx(CPre);
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();
    auto& C = CProx.Get();

    // Temporary distributions
    DistMatrix<T,MC,STAR,ELEMENT,D> A1_MC_STAR(g);
    DistMatrix<T,MR,STAR,ELEMENT,D> D1Trans_MR_STAR(g);

    A1_MC_STAR.AlignWith(B);
    D1Trans_MR_STAR.AlignWith(B);

    for(Int k=0; k<m; k+=bsize)
    {
        const Int nb = Min(bsize,m-k);
        auto A1 = A(ALL,        IR(k,k+nb));
        auto C1 = C(IR(k,k+nb), ALL       );

        // D1[*,MR] := alpha (A1[MC,*])^[T/H] B[MC,MR]
        //           = alpha (A1^[T/H])[*,MC] B[MC,MR]
        A1_MC_STAR = A1; // A1[MC,*] <- A1[MC,MR]
        LocalGemm(orientA, NORMAL, TypeTraits<T>::One(), B, A1_MC_STAR, D1Trans_MR_STAR);
        TransposeAxpyContract(alpha, D1Trans_MR_STAR, C1, conjugate);
    }
}

template <Device D, typename T,
          typename=DisableIf<IsDeviceValidType<T,D>>, typename=void>
void SUMMA_TNB_impl
(Orientation orientA,
  T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre)
{
    LogicError("SUMMA_TNB_impl type-device combo not supported.");
}

template<typename T>
void SUMMA_TNB
(Orientation orientA,
  T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE

    switch (CPre.GetLocalDevice())
    {
    case Device::CPU:
        SUMMA_TNB_impl<Device::CPU>(orientA, alpha, APre, BPre, CPre);
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        SUMMA_TNB_impl<Device::GPU>(orientA, alpha, APre, BPre, CPre);
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("SUMMA_TNA: Bad device.");
    }
}

template<typename T>
void SUMMA_TNB_MS(
    Orientation orientA,
    T alpha,
    AbstractDistMatrix<T> const& APre,
    AbstractDistMatrix<T> const& BPre,
    AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE;
#ifndef HYDROGEN_HAVE_MS_GEMM
    OutputFromRoot(CPre.Grid().Comm(),
                   "WARNING: Multistream support not available; "
                   "requires GPU and Aluminum.");
    SUMMA_TNB(orientA, alpha, APre, BPre, CPre);
#else

    switch (CPre.GetLocalDevice())
    {
    case Device::CPU:
        OutputFromRoot(
            CPre.Grid().Comm(),
            "WARNING: CPU doesn't support \"multistream\" variants.");
        SUMMA_TNB_impl<Device::CPU>(orientA, alpha, APre, BPre, CPre);
        break;
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
        SUMMA_TNB_impl_multistream(orientA, alpha, APre, BPre, CPre);
        break;
#endif // HYDROGEN_HAVE_CUDA
    default:
        LogicError("SUMMA_TNB_MS: Bad device.");
    }
#endif // HYDROGEN_HAVE_MS_GEMM
}

// Transpose Normal Gemm that avoids communicating the matrix C
template <Device D, typename T, typename=EnableIf<IsDeviceValidType<T,D>>>
void SUMMA_TNC_impl
(Orientation orientA,
  T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE
    const Int sumDim = BPre.Height();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();

    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> AProx(APre);
    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> BProx(BPre);
    DistMatrixReadWriteProxy<T,T,MC,MR,ELEMENT,D> CProx(CPre);
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();
    auto& C = CProx.Get();

    // Temporary distributions
    DistMatrix<T,STAR,MC,ELEMENT,D> A1_STAR_MC(g);
    DistMatrix<T,MR,STAR,ELEMENT,D> B1Trans_MR_STAR(g);

    A1_STAR_MC.AlignWith(C);
    B1Trans_MR_STAR.AlignWith(C);

    for(Int k=0; k<sumDim; k+=bsize)
    {
        const Int nb = Min(bsize,sumDim-k);
        auto A1 = A(IR(k,k+nb), ALL);
        auto B1 = B(IR(k,k+nb), ALL);

        // C[MC,MR] += alpha (A1[*,MC])^T B1[*,MR]
        //           = alpha (A1^T)[MC,*] B1[*,MR]
        A1_STAR_MC = A1;
        Transpose(B1, B1Trans_MR_STAR);
        LocalGemm
        (orientA, TRANSPOSE, alpha, A1_STAR_MC, B1Trans_MR_STAR, TypeTraits<T>::One(), C);
    }
}

template <Device D, typename T,
          typename=DisableIf<IsDeviceValidType<T,D>>, typename=void>
void SUMMA_TNC_impl
(Orientation orientA,
  T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre)
{
    LogicError("SUMMA_TNC_impl type-device combo not supported.");
}

template<typename T>
void SUMMA_TNC
(Orientation orientA,
  T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE

    switch (CPre.GetLocalDevice())
    {
    case Device::CPU:
        SUMMA_TNC_impl<Device::CPU>(orientA, alpha, APre, BPre, CPre);
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        SUMMA_TNC_impl<Device::GPU>(orientA, alpha, APre, BPre, CPre);
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("SUMMA_TNA: Bad device.");
    }
}

template<typename T>
void SUMMA_TNC_MS(
    Orientation orientA,
    T alpha,
    AbstractDistMatrix<T> const& APre,
    AbstractDistMatrix<T> const& BPre,
    AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE;
#ifndef HYDROGEN_HAVE_MS_GEMM
    OutputFromRoot(CPre.Grid().Comm(),
                   "WARNING: Multistream support not available; "
                   "requires GPU and Aluminum.");
    SUMMA_TNC(orientA, alpha, APre, BPre, CPre);
#else

    switch (CPre.GetLocalDevice())
    {
    case Device::CPU:
        OutputFromRoot(
            CPre.Grid().Comm(),
            "WARNING: CPU doesn't support \"multistream\" variants.");
        SUMMA_TNC_impl<Device::CPU>(orientA, alpha, APre, BPre, CPre);
        break;
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
        SUMMA_TNC_impl_multistream(orientA, alpha, APre, BPre, CPre);
        break;
#endif // HYDROGEN_HAVE_CUDA
    default:
        LogicError("SUMMA_TNC_MS: Bad device.");
    }
#endif // HYDROGEN_HAVE_MS_GEMM
}

// Transpose Normal Gemm for panel-panel dot products
//
// Use summations of local multiplications from a 1D distribution of A and B
// to update blockSize x blockSize submatrices of C
//
template <Device D, typename T, typename=EnableIf<IsDeviceValidType<T,D>>>
void SUMMA_TNDot_impl
(Orientation orientA,
  T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre,
  Int blockSize)
{
    EL_DEBUG_CSE
    const Int m = CPre.Height();
    const Int n = CPre.Width();
    const Grid& g = APre.Grid();

    DistMatrixReadProxy<T,T,VC,STAR,ELEMENT,D> AProx(APre);
    auto& A = AProx.GetLocked();

    ElementalProxyCtrl BCtrl;
    BCtrl.colConstrain = true;
    BCtrl.colAlign = A.ColAlign();
    DistMatrixReadProxy<T,T,VC,STAR,ELEMENT,D> BProx(BPre, BCtrl);
    auto& B = BProx.GetLocked();

    DistMatrixReadWriteProxy<T,T,MC,MR,ELEMENT,D> CProx(CPre);
    auto& C = CProx.Get();

    DistMatrix<T,STAR,STAR,ELEMENT,D> C11_STAR_STAR(g);
    for(Int kOuter=0; kOuter<m; kOuter+=blockSize)
    {
        const Int nbOuter = Min(blockSize,m-kOuter);
        const Range<Int> indOuter(kOuter, kOuter+nbOuter);

        auto A1 = A(ALL, indOuter);

        for(Int kInner=0; kInner<n; kInner+=blockSize)
        {
            const Int nbInner = Min(blockSize,n-kInner);
            const Range<Int> indInner(kInner, kInner+nbInner);

            auto B1  = B(ALL,      indInner);
            auto C11 = C(indOuter, indInner);

            LocalGemm(orientA, NORMAL, alpha, A1, B1, C11_STAR_STAR);
            AxpyContract(TypeTraits<T>::One(), C11_STAR_STAR, C11);
        }
    }
}

template <Device D, typename T,
          typename=DisableIf<IsDeviceValidType<T,D>>, typename=void>
void SUMMA_TNDot_impl
(Orientation, T,
  const AbstractDistMatrix<T>&,
  const AbstractDistMatrix<T>&,
  AbstractDistMatrix<T>&,
  Int)
{
    LogicError("SUMMA_TNDot_impl type-device combo not supported.");
}

template<typename T>
void SUMMA_TNDot
(Orientation orientA,
  T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
  AbstractDistMatrix<T>& CPre,
    Int blockSize=2000)
{
    EL_DEBUG_CSE

    switch (CPre.GetLocalDevice())
    {
    case Device::CPU:
        SUMMA_TNDot_impl<Device::CPU>(
            orientA, alpha, APre, BPre, CPre, blockSize);
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        SUMMA_TNDot_impl<Device::GPU>(
            orientA, alpha, APre, BPre, CPre, blockSize);
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("SUMMA_TNA: Bad device.");
    }
}

template<typename T>
void SUMMA_TN
(Orientation orientA,
  T alpha,
  const AbstractDistMatrix<T>& A,
  const AbstractDistMatrix<T>& B,
        AbstractDistMatrix<T>& C,
  GemmAlgorithm alg=GEMM_DEFAULT)
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      AssertSameGrids(A, B, C);
      if(orientA == NORMAL)
          LogicError("A must be (Conjugate)Transposed");
      if(A.Width() != C.Height() ||
          B.Width() != C.Width() ||
          A.Height() != B.Height())
          LogicError
          ("Nonconformal matrices:\n",
           DimsString(A,"A"),"\n",
           DimsString(B,"B"),"\n",
           DimsString(C,"C"));
   )

    const Int m = C.Height();
    const Int n = C.Width();
    const Int sumDim = A.Height();
    const double weightTowardsC = 2.;
    const double weightAwayFromDot = 10.;

    // TODO(poulson): Make this tunable
    const Int blockSizeDot = 2000;

    if (alg == GEMM_DEFAULT)
    {
#ifdef HYDROGEN_HAVE_MS_GEMM
        bool const multistream =
            (C.GetLocalDevice() == Device::GPU
             && GetSyncInfoPool(C.Grid()).Size() > 1);
#else
        bool constexpr multistream = false;
#endif // HYDROGEN_HAVE_MS_GEMM
        if(weightAwayFromDot*m <= sumDim && weightAwayFromDot*n <= sumDim)
            alg = GEMM_SUMMA_DOT;
        else if(m <= n && weightTowardsC*m <= sumDim)
            alg = (multistream ? GEMM_SUMMA_B_MS : GEMM_SUMMA_B);
        else if(n <= m && weightTowardsC*n <= sumDim)
            alg = (multistream ? GEMM_SUMMA_A_MS : GEMM_SUMMA_A);
        else
            alg = (multistream ? GEMM_SUMMA_C_MS : GEMM_SUMMA_C);
    }

    switch(alg)
    {
    case GEMM_DEFAULT:
        LogicError("This shouldn't happen.");
        break;
    case GEMM_SUMMA_A_MS: SUMMA_TNA_MS(orientA, alpha, A, B, C); break;
    case GEMM_SUMMA_A:    SUMMA_TNA(orientA, alpha, A, B, C); break;
    case GEMM_SUMMA_B_MS: SUMMA_TNB_MS(orientA, alpha, A, B, C); break;
    case GEMM_SUMMA_B:    SUMMA_TNB(orientA, alpha, A, B, C); break;
    case GEMM_SUMMA_C_MS: SUMMA_TNC_MS(orientA, alpha, A, B, C); break;
    case GEMM_SUMMA_C:    SUMMA_TNC(orientA, alpha, A, B, C); break;
    case GEMM_SUMMA_DOT:
        SUMMA_TNDot(orientA, alpha, A, B, C, blockSizeDot);
        break;
    default:
        LogicError("Unsupported Gemm option");
    }
}

} // namespace gemm
} // namespace El
