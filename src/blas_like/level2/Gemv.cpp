/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like/level2.hpp>

#include "./Gemv/Normal.hpp"
#include "./Gemv/Transpose.hpp"

#include <El/hydrogen_config.h>

namespace El {

template <typename T>
void Gemv(Orientation orientA,
          T alpha, AbstractMatrix<T> const& A, AbstractMatrix<T> const& B,
          T beta, AbstractMatrix<T>& C)
{
    if ((A.GetDevice() != B.GetDevice()) || (A.GetDevice() != C.GetDevice()))
        LogicError("Must call gemm with matrices on same device.");

    switch (A.GetDevice())
    {
    case Device::CPU:
        Gemv(orientA, alpha,
             static_cast<Matrix<T,Device::CPU> const&>(A),
             static_cast<Matrix<T,Device::CPU> const&>(B),
             beta,
             static_cast<Matrix<T,Device::CPU>&>(C));
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        Gemv(orientA, alpha,
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
void Gemv(Orientation orientA,
          T alpha, AbstractMatrix<T> const& A, AbstractMatrix<T> const& B,
          AbstractMatrix<T>& C)
{
    Gemv(orientA, alpha, A, B, T{0}, C);
}

namespace
{

template <typename... Ts>
void DispatchGemv(SyncInfo<Device::CPU> const&, Ts&&... args)
{
    blas::Gemv(std::forward<Ts>(args)...);
}

#ifdef HYDROGEN_HAVE_GPU
template <typename... Ts>
void DispatchGemv(SyncInfo<Device::GPU> const& si,
                  char trans,
                  Ts&&... args)
{
    gpu_blas::Gemv(CharToTransposeMode(trans), std::forward<Ts>(args)..., si);
}
#endif // HYDROGEN_HAVE_GPU

}// namespace <anon>


template<typename T, Device D, typename>
void Gemv
(Orientation orientation,
  T alpha, const Matrix<T,D>& A,
           const Matrix<T,D>& x,
  T beta,        Matrix<T,D>& y)
{
    EL_DEBUG_CSE;
#ifdef HYDROGEN_DO_BOUNDS_CHECKING
    if ((x.Height() != 1 && x.Width() != 1) ||
        (y.Height() != 1 && y.Width() != 1))
        LogicError
            ("Nonconformal: \n",DimsString(x,"x"),"\n",DimsString(y,"y"));
    const Int xLength = (x.Width()==1 ? x.Height() : x.Width());
    const Int yLength = (y.Width()==1 ? y.Height() : y.Width());
    if (orientation == NORMAL)
    {
        if (A.Height() != yLength || A.Width() != xLength)
            LogicError
                ("Nonconformal: \n",DimsString(A,"A"),"\n",
                 DimsString(x,"x"),"\n",DimsString(y,"y"));
    }
    else
    {
        if (A.Width() != yLength || A.Height() != xLength)
            LogicError
                ("Nonconformal: \n",DimsString(A,"A"),"\n",
                 DimsString(x,"x"),"\n",DimsString(y,"y"));
    }
#endif // HYDROGEN_DO_BOUNDS_CHECKING

    auto master_sync = SyncInfoFromMatrix(y);
    auto SyncManager = MakeMultiSync(
        master_sync, SyncInfoFromMatrix(A), SyncInfoFromMatrix(x));

    const char transChar = OrientationToChar(orientation);
    const Int m = A.Height();
    const Int n = A.Width();
    const Int xDim = (transChar == 'N' ? n : m);
    const Int yDim = (transChar == 'N' ? m : n);
    const Int incx = (x.Width()==1 ? 1 : x.LDim());
    const Int incy = (y.Width()==1 ? 1 : y.LDim());
    if (xDim != 0)
    {
        if (yDim != 0)
        {
            DispatchGemv(
                master_sync,
                transChar, m, n,
                alpha, A.LockedBuffer(), A.LDim(), x.LockedBuffer(), incx,
                beta,  y.Buffer(), incy);
        }
    }
    else
    {
        Scale(beta, y);
    }
}

template<typename T, Device D, typename, typename>
void Gemv
(Orientation, T, Matrix<T,D> const&, Matrix<T,D> const&,
  T, Matrix<T,D>&)
{
    LogicError("Gemv: Bad device/type combination.");
}

template<typename T, Device D>
void Gemv
(Orientation orientation,
  T alpha, const Matrix<T,D>& A,
           const Matrix<T,D>& x,
           Matrix<T,D>& y)
{
    EL_DEBUG_CSE
    if (orientation == NORMAL)
        y.Resize(A.Height(), 1);
    else
        y.Resize(A.Width(), 1);
    Zero(y);
    Gemv(orientation, alpha, A, x, T(0), y);
}

template<typename T>
void Gemv
(Orientation orientation,
  T alpha, const AbstractDistMatrix<T>& A,
           const AbstractDistMatrix<T>& x,
  T beta,        AbstractDistMatrix<T>& y)
{
    EL_DEBUG_CSE
    if (orientation == NORMAL)
        gemv::Normal(alpha, A, x, beta, y);
    else
        gemv::Transpose(orientation, alpha, A, x, beta, y);
}

template<typename T>
void Gemv
(Orientation orientation,
  T alpha, const AbstractDistMatrix<T>& A,
           const AbstractDistMatrix<T>& x,
                 AbstractDistMatrix<T>& y)
{
    EL_DEBUG_CSE
    y.AlignWith(A);
    if (orientation == NORMAL)
        y.Resize(A.Height(), 1);
    else
        y.Resize(A.Width(), 1);
    Zero(y);
    Gemv(orientation, alpha, A, x, T(0), y);
}

template<typename T>
void LocalGemv
(Orientation orientation,
  T alpha, const AbstractDistMatrix<T>& A,
           const AbstractDistMatrix<T>& x,
  T beta,        AbstractDistMatrix<T>& y)
{
    EL_DEBUG_CSE
    // TODO(poulson): Add error checking here
    Gemv
    (orientation ,
      alpha, A.LockedMatrix(), x.LockedMatrix(),
      beta,                    y.Matrix());
}

namespace gemv {

template<typename T,typename=EnableIf<IsBlasScalar<T>>>
void ScaLAPACKHelper
(Orientation orientation,
  T alpha, const DistMatrix<T,MC,MR,BLOCK>& A,
           const DistMatrix<T,MC,MR,BLOCK>& x,
  T beta,        DistMatrix<T,MC,MR,BLOCK>& y)
{
    AssertScaLAPACKSupport();
#ifdef EL_HAVE_SCALAPACK
    const Int m = A.Height();
    const Int n = A.Width();
    const char orientChar = OrientationToChar(orientation);

    auto descA = FillDesc(A);
    auto descx = FillDesc(x);
    auto descy = FillDesc(y);
    pblas::Gemv
    (orientChar, m, n,
      alpha,
      A.LockedBuffer(), descA.data(),
      x.LockedBuffer(), descx.data(), 1,
      beta,
      y.Buffer(),       descy.data(), 1);
#endif
}

template<typename T,typename=DisableIf<IsBlasScalar<T>>,typename=void>
void ScaLAPACKHelper
(Orientation orientation,
  T alpha, const DistMatrix<T,MC,MR,BLOCK>& A,
           const DistMatrix<T,MC,MR,BLOCK>& x,
  T beta,        DistMatrix<T,MC,MR,BLOCK>& y)
{
    LogicError("ScaLAPACK does not support this datatype");
}

} // namespace gemv

template<typename T>
void Gemv
(Orientation orientation,
  T alpha, const DistMatrix<T,MC,MR,BLOCK>& A,
           const DistMatrix<T,MC,MR,BLOCK>& x,
  T beta,        DistMatrix<T,MC,MR,BLOCK>& y)
{
    EL_DEBUG_CSE
    gemv::ScaLAPACKHelper(orientation, alpha, A, x, beta, y);
}

template<>
void Gemv
(Orientation orientation,
  Int alpha, const DistMatrix<Int,MC,MR,BLOCK>& A,
             const DistMatrix<Int,MC,MR,BLOCK>& x,
  Int beta,        DistMatrix<Int,MC,MR,BLOCK>& y)
{
    EL_DEBUG_CSE
    LogicError("ScaLAPACK does not support integer data");
}

#ifdef HYDROGEN_HAVE_QUADMATH
template<>
void Gemv
(Orientation orientation,
  Quad alpha, const DistMatrix<Quad,MC,MR,BLOCK>& A,
              const DistMatrix<Quad,MC,MR,BLOCK>& x,
  Quad beta,        DistMatrix<Quad,MC,MR,BLOCK>& y)
{
    EL_DEBUG_CSE
    LogicError("ScaLAPACK does not support quad-precision data");
}

template<>
void Gemv
(Orientation orientation,
  Complex<Quad> alpha, const DistMatrix<Complex<Quad>,MC,MR,BLOCK>& A,
                       const DistMatrix<Complex<Quad>,MC,MR,BLOCK>& x,
  Complex<Quad> beta,        DistMatrix<Complex<Quad>,MC,MR,BLOCK>& y)
{
    EL_DEBUG_CSE
    LogicError("ScaLAPACK does not support quad-precision data");
}
#endif // ifdef HYDROGEN_HAVE_QUADMATH

template<typename T>
void Gemv
(Orientation orientation,
  T alpha, const DistMatrix<T,MC,MR,BLOCK>& A,
           const DistMatrix<T,MC,MR,BLOCK>& x,
                 DistMatrix<T,MC,MR,BLOCK>& y)
{
    EL_DEBUG_CSE
    y.AlignWith(A);
    if (orientation == NORMAL)
        y.Resize(A.Height(), 1);
    else
        y.Resize(A.Width(), 1);
    Zero(y);
    Gemv(orientation, alpha, A, x, T(0), y);
}

#ifdef HYDROGEN_HAVE_GPU
template void Gemv(Orientation orientA,
                   float alpha,
                   Matrix<float,Device::GPU> const& A,
                   Matrix<float,Device::GPU> const& B,
                   float beta,
                   Matrix<float,Device::GPU>& C);
template void Gemv(Orientation orientA,
                   double alpha,
                   Matrix<double,Device::GPU> const& A,
                   Matrix<double,Device::GPU> const& B,
                   double beta,
                   Matrix<double,Device::GPU>& C);
#ifdef HYDROGEN_GPU_USE_FP16
template void Gemv(Orientation orientA,
                   gpu_half_type alpha,
                   Matrix<gpu_half_type,Device::GPU> const& A,
                   Matrix<gpu_half_type,Device::GPU> const& B,
                   gpu_half_type beta,
                   Matrix<gpu_half_type,Device::GPU>& C);
template void Gemv(Orientation, gpu_half_type,
                   AbstractMatrix<gpu_half_type> const&,
                   AbstractMatrix<gpu_half_type> const&,
                   gpu_half_type,
                   AbstractMatrix<gpu_half_type>&);
#endif // HYDROGEN_GPU_USE_FP16
#endif // HYDROGEN_HAVE_GPU

#define PROTO(T) \
  template void Gemv                                   \
  (Orientation, T,                                     \
   AbstractMatrix<T> const&, AbstractMatrix<T> const&, \
   T, AbstractMatrix<T>&);                             \
  template void Gemv \
  (Orientation orientation, \
    T alpha, const Matrix<T,Device::CPU>& A,       \
             const Matrix<T,Device::CPU>& x, \
    T beta,        Matrix<T,Device::CPU>& y); \
  template void Gemv \
  (Orientation orientation, \
    T alpha, const Matrix<T,Device::CPU>& A, \
             const Matrix<T,Device::CPU>& x, \
                   Matrix<T,Device::CPU>& y); \
  template void Gemv \
  (Orientation orientation, \
    T alpha, const AbstractDistMatrix<T>& A, \
             const AbstractDistMatrix<T>& x, \
    T beta,        AbstractDistMatrix<T>& y); \
  template void Gemv \
  (Orientation orientation, \
    T alpha, const AbstractDistMatrix<T>& A, \
             const AbstractDistMatrix<T>& x, \
                   AbstractDistMatrix<T>& y); \
  template void Gemv \
  (Orientation orientation, \
    T alpha, const DistMatrix<T,MC,MR,BLOCK>& A, \
             const DistMatrix<T,MC,MR,BLOCK>& x, \
                   DistMatrix<T,MC,MR,BLOCK>& y); \
  template void LocalGemv \
  (Orientation orientation, \
    T alpha, const AbstractDistMatrix<T>& A, \
             const AbstractDistMatrix<T>& x, \
    T beta,        AbstractDistMatrix<T>& y);

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#define EL_ENABLE_HALF
#include <El/macros/Instantiate.h>

// Fix "explicit instantiation after explicit specialization" warnings
#undef PROTO
#define PROTO(T)                                        \
    template void Gemv                                  \
    (Orientation orientation,                           \
     T alpha, const DistMatrix<T,MC,MR,BLOCK>& A,       \
     const DistMatrix<T,MC,MR,BLOCK>& x,                \
     T beta,        DistMatrix<T,MC,MR,BLOCK>& y);

#define EL_NO_INT_PROTO
#undef EL_ENABLE_QUAD
#include <El/macros/Instantiate.h>

} // namespace El
