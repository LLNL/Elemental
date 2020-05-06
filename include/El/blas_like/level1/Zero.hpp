/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_ZERO_HPP
#define EL_BLAS_ZERO_HPP

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HYDROGEN_HAVE_GPU
#include <hydrogen/blas/gpu/Fill.hpp>
#endif

namespace El {

template <typename T>
void Zero_seq(AbstractMatrix<T>& A)
{
    EL_DEBUG_CSE
    const Int height = A.Height();
    const Int width = A.Width();
    const Int size = height * width;
    const Int ALDim = A.LDim();
    T* ABuf = A.Buffer();

    switch (A.GetDevice())
    {
    case Device::CPU:
        if( width == 1 || ALDim == height )
        {
            MemZero( ABuf, size );
        }
        else
        {
            for( Int j=0; j<width; ++j )
            {
                MemZero( &ABuf[j*ALDim], height );
            }
        }
        break;
    default:
        LogicError("Bad device type in Zero_seq. CPU only.");
    }
}

template <typename T>
void Zero_seq(AbstractDistMatrix<T>& A)
{
    EL_DEBUG_CSE
    Zero_seq(A.Matrix());
}

template<typename T>
void Zero( AbstractMatrix<T>& A )
{
    EL_DEBUG_CSE;
    const Int height = A.Height();
    const Int width = A.Width();
    const Int size = height * width;
    const Int ALDim = A.LDim();
    T* ABuf = A.Buffer();

    switch (A.GetDevice())
    {
    case Device::CPU:
        if( width == 1 || ALDim == height )
        {
#ifdef _OPENMP
#if defined(HYDROGEN_HAVE_OMP_TASKLOOP)
            const Int numThreads = omp_get_num_threads();
            #pragma omp taskloop default(shared)
            for(Int thread = 0; thread < numThreads; ++thread)
            {
#else
            #pragma omp parallel
            {
                const Int numThreads = omp_get_num_threads();
                const Int thread = omp_get_thread_num();
#endif
                const Int chunk = (size + numThreads - 1) / numThreads;
                const Int start = Min(chunk * thread, size);
                const Int end = Min(chunk * (thread + 1), size);
                MemZero( &ABuf[start], end - start );
            }
#else
            MemZero( ABuf, size );
#endif
        }
        else
        {
            EL_PARALLEL_FOR
            for( Int j=0; j<width; ++j )
            {
                MemZero( &ABuf[j*ALDim], height );
            }
        }
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        hydrogen::Fill_GPU_impl(
            height, width, TypeTraits<T>::Zero(), ABuf, ALDim,
            SyncInfoFromMatrix(
                static_cast<Matrix<T,Device::GPU>&>(A)));
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("Bad device type in Zero");
    }

}

template<typename T>
void Zero( AbstractDistMatrix<T>& A )
{
    EL_DEBUG_CSE
    Zero( A.Matrix() );
}


#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void Zero_seq( AbstractMatrix<T>& A ); \
  EL_EXTERN template void Zero_seq( AbstractDistMatrix<T>& A ); \
  EL_EXTERN template void Zero( AbstractMatrix<T>& A ); \
  EL_EXTERN template void Zero( AbstractDistMatrix<T>& A );

#ifdef HYDROGEN_GPU_USE_FP16
PROTO(gpu_half_type)
#endif

#define EL_ENABLE_HALF
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_ZERO_HPP
