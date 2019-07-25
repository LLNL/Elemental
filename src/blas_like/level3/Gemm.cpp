/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like/level3.hpp>
#include "El/core/Profiling.hpp"

#include "./Gemm/NN.hpp"
#include "./Gemm/NT.hpp"
#include "./Gemm/TN.hpp"
#include "./Gemm/TT.hpp"

namespace El
{

template <typename T>
void Gemm(Orientation orientA, Orientation orientB,
          T alpha, AbstractMatrix<T> const& A, AbstractMatrix<T> const& B,
          T beta, AbstractMatrix<T>& C)
{
    if ((A.GetDevice() != B.GetDevice()) || (A.GetDevice() != C.GetDevice()))
        LogicError("Must call gemm with matrices on same device.");

    switch (A.GetDevice())
    {
    case Device::CPU:
        Gemm(orientA, orientB, alpha,
             static_cast<Matrix<T,Device::CPU> const&>(A),
             static_cast<Matrix<T,Device::CPU> const&>(B),
             beta,
             static_cast<Matrix<T,Device::CPU>&>(C));
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        Gemm(orientA, orientB, alpha,
             static_cast<Matrix<T,Device::GPU> const&>(A),
             static_cast<Matrix<T,Device::GPU> const&>(B),
             beta,
             static_cast<Matrix<T,Device::GPU>&>(C));
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("Bad device type.");
    }
}

template <typename T>
void Gemm(Orientation orientA, Orientation orientB,
          T alpha, AbstractMatrix<T> const& A, AbstractMatrix<T> const& B,
          AbstractMatrix<T>& C)
{
    Gemm(orientA, orientB, alpha, A, B, T{0}, C);
}

namespace
{
template <typename T>
static void Gemm_impl(
    Orientation orientA, Orientation orientB,
    T alpha,
    Matrix<T,Device::CPU> const& A, Matrix<T,Device::CPU> const& B,
    T beta, Matrix<T,Device::CPU>& C)
{
    AUTO_PROFILE_REGION("Gemm_impl.CPU", SyncInfoFromMatrix(C));

    const char transA = OrientationToChar(orientA);
    const char transB = OrientationToChar(orientB);
    const Int m = C.Height();
    const Int n = C.Width();
    const Int k = (orientA == NORMAL ? A.Width() : A.Height());

    blas::Gemm(transA, transB, m, n, k,
               alpha, A.LockedBuffer(), A.LDim(),
               B.LockedBuffer(), B.LDim(),
               beta, C.Buffer(), C.LDim());
}

#ifdef HYDROGEN_HAVE_GPU
template <typename T>
static void Gemm_impl(
    Orientation orientA, Orientation orientB,
    T alpha,
    Matrix<T,Device::GPU> const& A, Matrix<T,Device::GPU> const& B,
    T beta, Matrix<T,Device::GPU>& C)
{
    AUTO_PROFILE_REGION("Gemm_impl.GPU", SyncInfoFromMatrix(C));

    auto const transA = OrientationToTransposeMode(orientA);
    auto const transB = OrientationToTransposeMode(orientB);
    const Int m = C.Height();
    const Int n = C.Width();
    const Int k = (orientA == NORMAL ? A.Width() : A.Height());

    auto master_sync = SyncInfoFromMatrix(C);
    auto SyncManager = MakeMultiSync(
        master_sync, SyncInfoFromMatrix(A), SyncInfoFromMatrix(B));

    gpu_blas::Gemm(transA, transB, m, n, k,
                   alpha, A.LockedBuffer(), A.LDim(),
                   B.LockedBuffer(), B.LDim(),
                   beta, C.Buffer(), C.LDim(), master_sync);
}

#endif // HYDROGEN_HAVE_GPU

}// namespace <anon>

template<typename T, Device D, typename>
void Gemm
(Orientation orientA, Orientation orientB,
  T alpha, Matrix<T,D> const& A, Matrix<T,D> const& B,
  T beta, Matrix<T,D>& C)
{
    EL_DEBUG_CSE
    if(orientA == NORMAL && orientB == NORMAL)
    {
        if (A.Height() != C.Height() ||
            B.Width()  != C.Width()  ||
            A.Width()  != B.Height())
            LogicError("Nonconformal GemmNN. Matrix dimensions are:\n"
                       "  A: ", A.Height(), "x", A.Width(), '\n',
                       "  B: ", B.Height(), "x", B.Width(), '\n',
                       "  C: ", C.Height(), "x", C.Width());
    }
    else if (orientA == NORMAL)
    {
        if (A.Height() != C.Height() ||
            B.Height() != C.Width()  ||
            A.Width()  != B.Width())
            LogicError("Nonconformal GemmN(T/C). Matrix dimensions are:\n"
                       "  A: ", A.Height(), "x", A.Width(), '\n',
                       "  B: ", B.Height(), "x", B.Width(), '\n',
                       "  C: ", C.Height(), "x", C.Width());
    }
    else if (orientB == NORMAL)
    {
        if (A.Width()  != C.Height() ||
            B.Width()  != C.Width()  ||
            A.Height() != B.Height())
            LogicError("Nonconformal Gemm(T/C)N. Matrix dimensions are:\n"
                       "  A: ", A.Height(), "x", A.Width(), '\n',
                       "  B: ", B.Height(), "x", B.Width(), '\n',
                       "  C: ", C.Height(), "x", C.Width());
    }
    else
    {
        if (A.Width()  != C.Height() ||
            B.Height() != C.Width()  ||
            A.Height() != B.Width())
            LogicError("Nonconformal Gemm(T/C)(T/C). Matrix dimensions are:\n"
                       "  A: ", A.Height(), "x", A.Width(), '\n',
                       "  B: ", B.Height(), "x", B.Width(), '\n',
                       "  C: ", C.Height(), "x", C.Width());
    }

    const Int k = (orientA == NORMAL ? A.Width() : A.Height());
    if (k != 0)
    {
        Gemm_impl(orientA, orientB, alpha, A, B, beta, C);
    }
    else
    {
        Scale(beta, C);
    }

}

template<typename T, Device D, typename, typename>
void Gemm
(Orientation, Orientation, T, Matrix<T,D> const&, Matrix<T,D> const&,
  T, Matrix<T,D>&)
{
    LogicError("Gemm: Bad device/type combination.");
}

template <typename T, Device D>
void Gemm (
    Orientation orientA, Orientation orientB,
    T alpha, Matrix<T,D> const& A, Matrix<T,D> const& B,
    Matrix<T,D>& C)
{
    EL_DEBUG_CSE
    const Int m = (orientA==NORMAL ? A.Height() : A.Width());
    const Int n = (orientB==NORMAL ? B.Width() : B.Height());
    C.Resize(m, n);
    Gemm(orientA, orientB, alpha, A, B, TypeTraits<T>::Zero(), C);
}

template<typename T>
void Gemm
(Orientation orientA, Orientation orientB,
  T alpha, const AbstractDistMatrix<T>& A,
           const AbstractDistMatrix<T>& B,
  T beta,        AbstractDistMatrix<T>& C,
  GemmAlgorithm alg)
{
    EL_DEBUG_CSE;
    Scale(beta, C);
    if(orientA == NORMAL && orientB == NORMAL)
    {
        if(alg == GEMM_CANNON)
            gemm::Cannon_NN(alpha, A, B, C);
        else
            gemm::SUMMA_NN(alpha, A, B, C, alg);
    }
    else if(orientA == NORMAL)
    {
        gemm::SUMMA_NT(orientB, alpha, A, B, C, alg);
    }
    else if(orientB == NORMAL)
    {
        gemm::SUMMA_TN(orientA, alpha, A, B, C, alg);
    }
    else
    {
        gemm::SUMMA_TT(orientA, orientB, alpha, A, B, C, alg);
    }
}

template<typename T>
void Gemm
(Orientation orientA, Orientation orientB,
  T alpha, const AbstractDistMatrix<T>& A,
           const AbstractDistMatrix<T>& B,
                 AbstractDistMatrix<T>& C, GemmAlgorithm alg)
{
    EL_DEBUG_CSE
    const Int m = (orientA==NORMAL ? A.Height() : A.Width());
    const Int n = (orientB==NORMAL ? B.Width() : B.Height());
    C.Resize(m, n);
    Gemm(orientA, orientB, alpha, A, B, TypeTraits<T>::Zero(), C, alg);
}

template<typename T>
void LocalGemm
(Orientation orientA, Orientation orientB,
  T alpha, const AbstractDistMatrix<T>& A,
           const AbstractDistMatrix<T>& B,
  T beta,        AbstractDistMatrix<T>& C)
{
    EL_DEBUG_CSE
#ifndef EL_RELEASE
    if(orientA == NORMAL && orientB == NORMAL)
    {
        if(A.ColDist() != C.ColDist() ||
           A.RowDist() != B.ColDist() ||
           B.RowDist() != C.RowDist())
            LogicError
                ("Tried to form C[",C.ColDist(),",",C.RowDist(),"] := "
                 "A[",A.ColDist(),",",A.RowDist(),"] "
                 "B[",B.ColDist(),",",B.RowDist(),"]");
        if(A.ColAlign() != C.ColAlign())
            LogicError("A's cols must align with C's rows");
        if(A.RowAlign() != B.ColAlign())
            LogicError("A's rows must align with B's cols");
        if(B.RowAlign() != C.RowAlign())
            LogicError("B's rows must align with C's rows");
        if(A.Height() != C.Height() ||
           A.Width() != B.Height() ||
           B.Width() != C.Width())
            LogicError
                ("Nonconformal LocalGemmNN:\n",
                 DimsString(A,"A"),"\n",
                 DimsString(B,"B"),"\n",
                 DimsString(C,"C"));
    }
    else if(orientA == NORMAL)
    {
        if(A.ColDist() != C.ColDist() ||
           A.RowDist() != B.RowDist() ||
           B.ColDist() != C.RowDist())
            LogicError
                ("Tried to form C[",C.ColDist(),",",C.RowDist(),"] := "
                 "A[",A.ColDist(),",",A.RowDist(),"] "
                 "B[",B.ColDist(),",",B.RowDist(),"]'");
        if(A.ColAlign() != C.ColAlign())
            LogicError("A's cols must align with C's rows");
        if(A.RowAlign() != B.RowAlign())
            LogicError("A's rows must align with B's rows");
        if(B.ColAlign() != C.RowAlign())
            LogicError("B's cols must align with C's rows");
        if(A.Height() != C.Height() ||
           A.Width() != B.Width() ||
           B.Height() != C.Width())
            LogicError
                ("Nonconformal LocalGemmNT:\n",
                 DimsString(A,"A"),"\n",
                 DimsString(B,"B"),"\n",
                 DimsString(C,"C"));
    }
    else if(orientB == NORMAL)
    {
        if(A.RowDist() != C.ColDist() ||
           A.ColDist() != B.ColDist() ||
           B.RowDist() != C.RowDist())
            LogicError
                ("Tried to form C[",C.ColDist(),",",C.RowDist(),"] := "
                 "A[",A.ColDist(),",",A.RowDist(),"]' "
                 "B[",B.ColDist(),",",B.RowDist(),"]");
        if(A.RowAlign() != C.ColAlign())
            LogicError("A's rows must align with C's cols");
        if(A.ColAlign() != B.ColAlign())
            LogicError("A's cols must align with B's cols");
        if(B.RowAlign() != C.RowAlign())
            LogicError("B's rows must align with C's rows");
        if(A.Width() != C.Height() ||
           A.Height() != B.Height() ||
           B.Width() != C.Width())
            LogicError
                ("Nonconformal LocalGemmTN:\n",
                 DimsString(A,"A"),"\n",
                 DimsString(B,"B"),"\n",
                 DimsString(C,"C"));
    }
    else
    {
        if(A.RowDist() != C.ColDist() ||
           A.ColDist() != B.RowDist() ||
           B.ColDist() != C.RowDist())
            LogicError
                ("Tried to form C[",C.ColDist(),",",C.RowDist(),"] := "
                 "A[",A.ColDist(),",",A.RowDist(),"]' "
                 "B[",B.ColDist(),",",B.RowDist(),"]'");
        if(A.RowAlign() != C.ColAlign())
            LogicError("A's rows must align with C's cols");
        if(A.ColAlign() != B.RowAlign())
            LogicError("A's cols must align with B's rows");
        if(B.ColAlign() != C.RowAlign())
            LogicError("B's cols must align with C's rows");
        if(A.Width() != C.Height() ||
           A.Height() != B.Width() ||
           B.Height() != C.Width())
            LogicError
                ("Nonconformal LocalGemmTT:\n",
                 DimsString(A,"A"),"\n",
                 DimsString(B,"B"),"\n",
                 DimsString(C,"C"));
    }
#endif // !EL_RELEASE
    Gemm(orientA , orientB,
         alpha, A.LockedMatrix(), B.LockedMatrix(), beta, C.Matrix());
}

template<typename T>
void LocalGemm
(Orientation orientA, Orientation orientB,
  T alpha, const AbstractDistMatrix<T>& A,
           const AbstractDistMatrix<T>& B,
                 AbstractDistMatrix<T>& C)
{
    EL_DEBUG_CSE
    const Int m = (orientA==NORMAL ? A.Height() : A.Width());
    const Int n = (orientB==NORMAL ? B.Width() : B.Height());
    C.Resize(m, n);
    LocalGemm(orientA, orientB, alpha, A, B, TypeTraits<T>::Zero(), C);
}

#ifdef HYDROGEN_HAVE_GPU
template void Gemm(Orientation orientA, Orientation orientB,
                   float alpha,
                   Matrix<float,Device::GPU> const& A,
                   Matrix<float,Device::GPU> const& B,
                   float beta,
                   Matrix<float,Device::GPU>& C);
template void Gemm(Orientation orientA, Orientation orientB,
                   double alpha,
                   Matrix<double,Device::GPU> const& A,
                   Matrix<double,Device::GPU> const& B,
                   double beta,
                   Matrix<double,Device::GPU>& C);
#ifdef HYDROGEN_GPU_USE_FP16
template void Gemm(Orientation orientA, Orientation orientB,
                   gpu_half_type alpha,
                   Matrix<gpu_half_type,Device::GPU> const& A,
                   Matrix<gpu_half_type,Device::GPU> const& B,
                   gpu_half_type beta,
                   Matrix<gpu_half_type,Device::GPU>& C);
#endif // HYDROGEN_GPU_USE_FP16
#endif // HYDROGEN_HAVE_GPU

#define ABSTRACT_PROTO(T)                                               \
    template void Gemm(                                                 \
        Orientation, Orientation, T,                                    \
        AbstractMatrix<T> const&, AbstractMatrix<T> const&,             \
        T, AbstractMatrix<T>&);                                         \
    template void Gemm(                                                 \
        Orientation orientA, Orientation orientB,                       \
        T alpha, const AbstractDistMatrix<T>& A,                        \
        const AbstractDistMatrix<T>& B,                                 \
        T beta, AbstractDistMatrix<T>& C, GemmAlgorithm alg);           \
    template void Gemm(                                                 \
        Orientation orientA, Orientation orientB,                       \
        T alpha, const AbstractDistMatrix<T>& A,                        \
        const AbstractDistMatrix<T>& B,                                 \
        AbstractDistMatrix<T>& C, GemmAlgorithm alg);                   \
    template void LocalGemm(                                            \
        Orientation orientA, Orientation orientB,                       \
        T alpha, const AbstractDistMatrix<T>& A,                        \
        const AbstractDistMatrix<T>& B,                                 \
        T beta,        AbstractDistMatrix<T>& C);                       \
    template void LocalGemm(                                            \
        Orientation orientA, Orientation orientB,                       \
        T alpha, const AbstractDistMatrix<T>& A,                        \
        const AbstractDistMatrix<T>& B,                                 \
        AbstractDistMatrix<T>& C)

#define PROTO(T)                                        \
    ABSTRACT_PROTO(T);                                  \
    template void Gemm(                                 \
        Orientation orientA, Orientation orientB,       \
        T alpha, const Matrix<T,Device::CPU>& A,        \
        const Matrix<T,Device::CPU>& B,                 \
        T beta,        Matrix<T,Device::CPU>& C);       \
    template void Gemm(                                 \
        Orientation orientA, Orientation orientB,       \
        T alpha, const Matrix<T,Device::CPU>& A,        \
        const Matrix<T,Device::CPU>& B,                 \
        Matrix<T,Device::CPU>& C);

#ifdef HYDROGEN_GPU_USE_FP16
ABSTRACT_PROTO(gpu_half_type);
#endif // HYDROGEN_GPU_USE_FP16

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#define EL_ENABLE_HALF
#include <El/macros/Instantiate.h>

} // namespace El
