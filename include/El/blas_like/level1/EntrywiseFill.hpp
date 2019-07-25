/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_ENTRYWISEFILL_HPP
#define EL_BLAS_ENTRYWISEFILL_HPP

namespace El {

template<typename T>
void EntrywiseFill( Matrix<T, Device::CPU>& A, function<T(void)> func )
{
    EL_DEBUG_CSE
    const Int m = A.Height();
    const Int n = A.Width();
    for( Int j=0; j<n; ++j )
        for( Int i=0; i<m; ++i )
            A(i,j) = func();
}

// FIXME: Make proper kernel
#ifdef HYDROGEN_HAVE_GPU
template <typename T>
void EntrywiseFill(Matrix<T,Device::GPU> &A, function<T(void)> func)
{
    EL_DEBUG_CSE
    Matrix<T,Device::CPU> CPU_Mat(A.Height(),A.Width(),A.LDim());
    EntrywiseFill(CPU_Mat, std::move(func));
    A = CPU_Mat;
}
#endif // HYDROGEN_HAVE_GPU

template<typename T>
void EntrywiseFill( AbstractDistMatrix<T>& A, function<T(void)> func )
{ EntrywiseFill( dynamic_cast<Matrix<T,Device::CPU>&>(A.Matrix()), func ); }

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void EntrywiseFill \
  ( Matrix<T,Device::CPU>& A, function<T(void)> func );  \
  EL_EXTERN template void EntrywiseFill \
  ( AbstractDistMatrix<T>& A, function<T(void)> func );

#ifdef HYDROGEN_HAVE_GPU
EL_EXTERN template void EntrywiseFill(
    Matrix<float,Device::GPU>&, function<float(void)>);
EL_EXTERN template void EntrywiseFill(
    Matrix<double,Device::GPU>&, function<double(void)>);
#endif // HYDROGEN_HAVE_GPU

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_ENTRYWISEFILL_HPP
