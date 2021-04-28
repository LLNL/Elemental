/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like/level3.hpp>

namespace El {

template <typename T>
void Herk(
    UpperOrLower uplo, Orientation orientation,
    Base<T> alpha, AbstractMatrix<T> const& A,
    Base<T> beta, AbstractMatrix<T>& C)
{
    EL_DEBUG_CSE;
    Syrk(uplo, orientation, T(alpha), A, T(beta), C, true);
}

template<typename T>
void Herk(
    UpperOrLower uplo, Orientation orientation,
    Base<T> alpha, AbstractMatrix<T> const& A,
    AbstractMatrix<T>& C)
{
    EL_DEBUG_CSE;
    const Int n = (orientation==NORMAL ? A.Height() : A.Width());
    C.Resize(n, n);
    Zero(C);
    Syrk(uplo, orientation, T(alpha), A, T(0), C, true);
}

template<typename T>
void Herk(
    UpperOrLower uplo, Orientation orientation,
    Base<T> alpha, AbstractDistMatrix<T> const& A,
    Base<T> beta, AbstractDistMatrix<T>& C)
{
    EL_DEBUG_CSE;
    Syrk(uplo, orientation, T(alpha), A, T(beta), C, true);
}

template<typename T>
void Herk(
    UpperOrLower uplo, Orientation orientation,
    Base<T> alpha, AbstractDistMatrix<T> const& A,
    AbstractDistMatrix<T>& C)
{
    EL_DEBUG_CSE;
    const Int n = (orientation==NORMAL ? A.Height() : A.Width());
    C.Resize(n, n);
    Zero(C);
    Syrk(uplo, orientation, T(alpha), A, T(0), C, true);
}

#define PROTO(T)                                        \
    template void Herk(                                 \
        UpperOrLower uplo, Orientation orientation,     \
        Base<T> alpha, AbstractMatrix<T> const& A,      \
        Base<T> beta, AbstractMatrix<T>& C);            \
    template void Herk(                                 \
        UpperOrLower uplo, Orientation orientation,     \
        Base<T> alpha, AbstractMatrix<T> const& A,      \
        AbstractMatrix<T>& C);                          \
    template void Herk(                                 \
        UpperOrLower uplo, Orientation orientation,     \
        Base<T> alpha, AbstractDistMatrix<T> const& A,  \
        AbstractDistMatrix<T>& C);                      \
    template void Herk(                                 \
        UpperOrLower uplo, Orientation orientation,     \
        Base<T> alpha, AbstractDistMatrix<T> const& A,  \
        Base<T> beta, AbstractDistMatrix<T>& C);

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#define EL_ENABLE_HALF
#include <El/macros/Instantiate.h>

} // namespace El
