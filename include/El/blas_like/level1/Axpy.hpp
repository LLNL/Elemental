/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_AXPY_HPP
#define EL_BLAS_AXPY_HPP

#include <El/blas_like/level1/Axpy/util.hpp>

namespace El {

template <typename T, typename S>
void Axpy(S alphaS, AbstractMatrix<T> const& X, AbstractMatrix<T>& Y)
{
    if (X.GetDevice() != Y.GetDevice())
        LogicError("Axpy: Incompatible devices!");

    switch (X.GetDevice())
    {
    case Device::CPU:
        Axpy(alphaS,
             static_cast<Matrix<T,Device::CPU> const&>(X),
             static_cast<Matrix<T,Device::CPU>&>(Y));
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        Axpy(alphaS,
             static_cast<Matrix<T,Device::GPU> const&>(X),
             static_cast<Matrix<T,Device::GPU>&>(Y));
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("Axpy: Bad device.");
    }
}

template<typename T,typename S>
void Axpy(S alphaS, const Matrix<T,Device::CPU>& X, Matrix<T,Device::CPU>& Y)
{
    EL_DEBUG_CSE;

    const T alpha = T(alphaS);
    const Int mX = X.Height();
    const Int nX = X.Width();
    const Int nY = Y.Width();
    const Int ldX = X.LDim();
    const Int ldY = Y.LDim();
    const T* XBuf = X.LockedBuffer();
          T* YBuf = Y.Buffer();


    // If X and Y are vectors, we can allow one to be a column and the other
    // to be a row. Otherwise we force X and Y to be the same dimension.
    if (mX == 1 || nX == 1)
    {
        const Int XLength = (nX==1 ? mX : nX);
        const Int XStride = (nX==1 ? 1  : ldX);
        const Int YStride = (nY==1 ? 1  : ldY);
        EL_DEBUG_ONLY(
          const Int mY = Y.Height();
          const Int YLength = (nY==1 ? mY : nY);
          if (XLength != YLength)
              LogicError("Nonconformal Axpy");
       )
        EL_PARALLEL_FOR
        for(Int i=0; i<XLength; ++i)
            YBuf[i*YStride] += alpha*XBuf[i*XStride];
    }
    else
    {
        // Iterate over single loop if X and Y are both contiguous in
        // memory. Otherwise iterate over double loop.
        if (ldX == mX && ldY == mX)
        {
            EL_PARALLEL_FOR
            for(Int i=0; i<mX*nX; ++i)
                YBuf[i] += alpha*XBuf[i];
        }
        else
        {
            EL_PARALLEL_FOR
            for(Int j=0; j<nX; ++j)
            {
                EL_SIMD
                for(Int i=0; i<mX; ++i)
                {
                    YBuf[i+j*ldY] += alpha*XBuf[i+j*ldX];
                }
            }
        }
    }
}

#ifdef HYDROGEN_HAVE_GPU
template<typename T,typename S,
         typename=DisableIf<IsDeviceValidType<T,Device::GPU>>,
         typename=void>
void Axpy(S alphaS, const Matrix<T,Device::GPU>& X, Matrix<T,Device::GPU>& Y)
{
    LogicError("Axpy: Invalid type-device combination.");
}

template<typename T,typename S,
         typename=EnableIf<IsDeviceValidType<T,Device::GPU>>>
void Axpy(S alphaS, Matrix<T,Device::GPU> const& X, Matrix<T,Device::GPU>& Y)
{
    EL_DEBUG_CSE;

    T const alpha = T(alphaS);
    Int const mX = X.Height();
    Int const nX = X.Width();
    Int const nY = Y.Width();
    Int const ldX = X.LDim();
    Int const ldY = Y.LDim();
    T const* XBuf = X.LockedBuffer();
    T* YBuf = Y.Buffer();

    SyncInfo<Device::GPU>
        syncInfoX = SyncInfoFromMatrix(X),
        syncInfoY = SyncInfoFromMatrix(Y);
    auto syncHelper = MakeMultiSync(syncInfoY, syncInfoX);

    // If X and Y are vectors, we can allow one to be a column and the other
    // to be a row. Otherwise we force X and Y to be the same dimension.
    if (mX == 1 || nX == 1)
    {
        const Int XLength = (nX==1 ? mX : nX);
        const Int XStride = (nX==1 ? 1  : ldX);
        const Int YStride = (nY==1 ? 1  : ldY);
#ifndef EL_RELEASE
        const Int mY = Y.Height();
        const Int YLength = (nY==1 ? mY : nY);
        if (XLength != YLength)
            LogicError("Nonconformal Axpy");
#endif // !EL_RELEASE
        gpu_blas::Axpy(
            XLength, alpha, XBuf, XStride, YBuf, YStride, syncInfoY);
    }
    else
    {
        gpu_blas::Axpy(
            mX, nX, alpha, XBuf, ldX, YBuf, ldY, syncInfoY);
    }
}
#endif // HYDROGEN_HAVE_GPU

template<typename T,typename S>
void Axpy(S alphaS, const ElementalMatrix<T>& X, ElementalMatrix<T>& Y)
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(AssertSameGrids(X, Y))
    const T alpha = T(alphaS);

    const DistData& XDistData = X.DistData();
    const DistData& YDistData = Y.DistData();

    if (XDistData == YDistData)
    {
        Axpy(alpha, X.LockedMatrix(), Y.Matrix());
    }
    else
    {
        // TODO(poulson):
        // Consider what happens if one is a row vector and the other
        // is a column vector...
        unique_ptr<ElementalMatrix<T>> XCopy(Y.Construct(Y.Grid(),Y.Root()));
        XCopy->AlignWith(YDistData);
        Copy(X, *XCopy);
        Axpy(alpha, XCopy->LockedMatrix(), Y.Matrix());
    }
}

template<typename T,typename S>
void Axpy(S alphaS, const BlockMatrix<T>& X, BlockMatrix<T>& Y)
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(AssertSameGrids(X, Y))
    const T alpha = T(alphaS);

    const DistData XDistData = X.DistData();
    const DistData YDistData = Y.DistData();

    if (XDistData == YDistData)
    {
        Axpy(alpha, X.LockedMatrix(), Y.Matrix());
    }
    else
    {
        unique_ptr<BlockMatrix<T>>
          XCopy(Y.Construct(Y.Grid(),Y.Root()));
        XCopy->AlignWith(YDistData);
        Copy(X, *XCopy);
        Axpy(alpha, XCopy->LockedMatrix(), Y.Matrix());
    }
}

template<typename T,typename S>
void Axpy(S alphaS, const AbstractDistMatrix<T>& X, AbstractDistMatrix<T>& Y)
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(AssertSameGrids(X, Y))
    const T alpha = T(alphaS);

    if (X.Wrap() == ELEMENT && Y.Wrap() == ELEMENT)
    {
        const auto& XCast = static_cast<const ElementalMatrix<T>&>(X);
              auto& YCast = static_cast<      ElementalMatrix<T>&>(Y);
        Axpy(alpha, XCast, YCast);
    }
    else if (X.Wrap() == BLOCK && Y.Wrap() == BLOCK)
    {
        const auto& XCast = static_cast<const BlockMatrix<T>&>(X);
              auto& YCast = static_cast<      BlockMatrix<T>&>(Y);
        Axpy(alpha, XCast, YCast);
    }
    else if (X.Wrap() == ELEMENT)
    {
        const auto& XCast = static_cast<const ElementalMatrix<T>&>(X);
              auto& YCast = static_cast<      BlockMatrix<T>&>(Y);
        unique_ptr<BlockMatrix<T>>
          XCopy(YCast.Construct(Y.Grid(),Y.Root()));
        XCopy->AlignWith(YCast.DistData());
        Copy(XCast, *XCopy);
        Axpy(alpha, XCopy->LockedMatrix(), Y.Matrix());
    }
    else
    {
        const auto& XCast = static_cast<const BlockMatrix<T>&>(X);
              auto& YCast = static_cast<      ElementalMatrix<T>&>(Y);
        unique_ptr<ElementalMatrix<T>>
          XCopy(YCast.Construct(Y.Grid(),Y.Root()));
        XCopy->AlignWith(YCast.DistData());
        Copy(XCast, *XCopy);
        Axpy(alpha, XCopy->LockedMatrix(), Y.Matrix());
    }
}

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void Axpy \
  (T alpha, const AbstractMatrix<T>& X, AbstractMatrix<T>& Y); \
  EL_EXTERN template void Axpy \
  (T alpha, const Matrix<T>& X, Matrix<T>& Y); \
  EL_EXTERN template void Axpy \
  (T alpha, const ElementalMatrix<T>& X, ElementalMatrix<T>& Y); \
  EL_EXTERN template void Axpy \
  (T alpha, const BlockMatrix<T>& X, BlockMatrix<T>& Y); \
  EL_EXTERN template void Axpy \
  (T alpha, const AbstractDistMatrix<T>& X, AbstractDistMatrix<T>& Y);

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_AXPY_HPP
