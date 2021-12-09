/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/lapack_like/factor.hpp>
#include <El/matrices.hpp>

namespace El {

template<typename F>
void Haar( Matrix<F>& A, Int n )
{
    EL_DEBUG_CSE
    // TODO: Replace this with a quadratic scheme similar to Stewart's, which
    //       essentially generates random Householder reflectors
    Gaussian( A, n, n );
    qr::ExplicitUnitary( A );
}

template<typename F>
void ImplicitHaar( Matrix<F>& A, Matrix<F>& t, Matrix<Base<F>>& d, Int n )
{
    EL_DEBUG_CSE
    // TODO: Replace this with a quadratic scheme similar to Stewart's, which
    //       essentially generates random Householder reflectors
    Gaussian( A, n, n );
    QR( A, t, d );
}

template<typename F>
void Haar( ElementalMatrix<F>& A, Int n )
{
    EL_DEBUG_CSE
    // TODO: Replace this with a quadratic scheme similar to Stewart's, which
    //       essentially generates random Householder reflectors
    Gaussian( A, n, n );
    qr::ExplicitUnitary( A );
}

template<typename F>
void ImplicitHaar
( ElementalMatrix<F>& A,
  ElementalMatrix<F>& t,
  ElementalMatrix<Base<F>>& d, Int n )
{
    EL_DEBUG_CSE
    // TODO: Replace this with a quadratic scheme similar to Stewart's, which
    //       essentially generates random Householder reflectors
    Gaussian( A, n, n );
    QR( A, t, d );
}

#if 0 // TOM
  template void Haar( Matrix<F>& A, Int n );
  template void Haar( ElementalMatrix<F>& A, Int n );
#endif // 0 TOM

#define PROTO(F)                                                               \
    template void ImplicitHaar(Matrix<F>& A,                                   \
                               Matrix<F>& t,                                   \
                               Matrix<Base<F>>& d,                             \
                               Int n);
#if 0 // TOM
  template void ImplicitHaar \
  ( ElementalMatrix<F>& A, \
    ElementalMatrix<F>& t, \
    ElementalMatrix<Base<F>>& d, Int n );
#endif // 0 TOM

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
/*#undef EL_ENABLE_HALF*/
#include <El/macros/Instantiate.h>

} // namespace El
