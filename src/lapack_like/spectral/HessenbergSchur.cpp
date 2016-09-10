/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

#include "./HessenbergSchur/Simple.hpp"
#include "./HessenbergSchur/MultiBulge.hpp"
#include "./HessenbergSchur/AED.hpp"

namespace El {

template<typename F>
HessenbergSchurInfo
HessenbergSchur
( Matrix<F>& H,
  Matrix<Complex<Base<F>>>& w,
  Matrix<F>& Z,
  const HessenbergSchurCtrl& ctrl )
{
    const Int n = H.Height();
    auto ctrlMod( ctrl );
    ctrlMod.winBeg = ( ctrl.winBeg==END ? n : ctrl.winBeg );
    ctrlMod.winEnd = ( ctrl.winEnd==END ? n : ctrl.winEnd );
    ctrlMod.wantSchurVecs = true;

    if( ctrl.alg == HESSENBERG_SCHUR_AED )
    {
        return hess_schur::AED( H, w, Z, ctrlMod );
    }
    else if( ctrl.alg == HESSENBERG_SCHUR_MULTIBULGE )
    {
        return hess_schur::MultiBulge( H, w, Z, ctrlMod );
    }
    else
    {
        return hess_schur::Simple( H, w, Z, ctrlMod );
    }
}

template<typename F>
HessenbergSchurInfo
HessenbergSchur
( Matrix<F>& H,
  Matrix<Complex<Base<F>>>& w,
  const HessenbergSchurCtrl& ctrl )
{
    const Int n = H.Height();
    auto ctrlMod( ctrl );
    ctrlMod.winBeg = ( ctrl.winBeg==END ? n : ctrl.winBeg );
    ctrlMod.winEnd = ( ctrl.winEnd==END ? n : ctrl.winEnd );
    ctrlMod.wantSchurVecs = false;

    Matrix<F> Z;
    if( ctrl.alg == HESSENBERG_SCHUR_AED )
    {
        return hess_schur::AED( H, w, Z, ctrlMod );
    }
    else if( ctrl.alg == HESSENBERG_SCHUR_MULTIBULGE )
    {
        return hess_schur::MultiBulge( H, w, Z, ctrlMod );
    }
    else
    {
        return hess_schur::Simple( H, w, Z, ctrlMod );
    }
}

#define PROTO(F) \
  template HessenbergSchurInfo HessenbergSchur \
  ( Matrix<F>& H, \
    Matrix<Complex<Base<F>>>& w, \
    const HessenbergSchurCtrl& ctrl ); \
  template HessenbergSchurInfo HessenbergSchur \
  ( Matrix<F>& H, \
    Matrix<Complex<Base<F>>>& w, \
    Matrix<F>& Z, \
    const HessenbergSchurCtrl& ctrl );

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El
