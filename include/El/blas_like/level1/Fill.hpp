/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_FILL_HPP
#define EL_BLAS_FILL_HPP

#ifdef HYDROGEN_HAVE_GPU
#include <hydrogen/blas/gpu/Fill.hpp>
#endif

namespace El
{

template<typename T>
void Fill( AbstractMatrix<T>& A, T alpha )
{
    EL_DEBUG_CSE
    const Int m = A.Height();
    const Int n = A.Width();
    T* ABuf = A.Buffer();
    const Int ALDim = A.LDim();
    switch (A.GetDevice())
    {
    case Device::CPU:
        // Iterate over single loop if memory is contiguous. Otherwise
        // iterate over double loop.
        if( n == 1 || ALDim == m )
        {
            EL_PARALLEL_FOR
            for( Int i=0; i<m*n; ++i )
            {
                ABuf[i] = alpha;
            }
        }
        else
        {
            EL_PARALLEL_FOR_COLLAPSE2
            for( Int j=0; j<n; ++j )
            {
                for( Int i=0; i<m; ++i )
                {
                    ABuf[i+j*ALDim] = alpha;
                }
            }
        }
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        hydrogen::Fill_GPU_impl(
            m, n, alpha, ABuf, ALDim,
            SyncInfoFromMatrix(
                static_cast<Matrix<T,Device::GPU>&>(A)).Stream());
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("Bad device type in Fill");
    }
}

template<typename T>
void Fill( AbstractDistMatrix<T>& A, T alpha )
{
    EL_DEBUG_CSE
    Fill( A.Matrix(), alpha );
}

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void Fill( AbstractMatrix<T>& A, T alpha ); \
  EL_EXTERN template void Fill( AbstractDistMatrix<T>& A, T alpha );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_FILL_HPP
