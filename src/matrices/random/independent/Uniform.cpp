/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like/level1.hpp>
#include <El/matrices.hpp>

namespace El {

// Draw each entry from a uniform PDF over a closed ball.

template<typename T>
void MakeUniform( AbstractMatrix<T>& A, T center, Base<T> radius )
{
    EL_DEBUG_CSE
    switch (A.GetDevice())
    {
    case Device::CPU:
        MakeUniform(static_cast<Matrix<T,Device::CPU>&>(A), center, radius);
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        MakeUniform(static_cast<Matrix<T,Device::GPU>&>(A), center, radius);
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("MakeUniform: Bad device.");
    }
}

template<typename T, Device D>
void MakeUniform( Matrix<T,D>& A, T center, Base<T> radius )
{
    EL_DEBUG_CSE
    auto sampleBall = [=]() { return SampleBall(center,radius); };
    EntrywiseFill( A, function<T()>(sampleBall) );
}

template<typename T>
void Uniform( AbstractMatrix<T>& A, Int m, Int n, T center, Base<T> radius )
{
    EL_DEBUG_CSE
    A.Resize( m, n );
    MakeUniform( A, center, radius );
}

template<typename T>
void MakeUniform( AbstractDistMatrix<T>& A, T center, Base<T> radius )
{
    EL_DEBUG_CSE
    if( A.RedundantRank() == 0 )
        MakeUniform( A.Matrix(), center, radius );
    Broadcast( A, A.RedundantComm(), 0 );
}

template<typename T>
void Uniform( AbstractDistMatrix<T>& A, Int m, Int n, T center, Base<T> radius )
{
    EL_DEBUG_CSE
    A.Resize( m, n );
    MakeUniform( A, center, radius );
}

#define ABSTRACT_PROTO(T)                                               \
    template void MakeUniform(                                          \
        AbstractMatrix<T>& A, T center, Base<T> radius );               \
    template void MakeUniform(                                          \
        AbstractDistMatrix<T>& A, T center, Base<T> radius );           \
    template void Uniform(                                              \
        AbstractMatrix<T>& A, Int m, Int n, T center, Base<T> radius ); \
    template void Uniform(                                              \
        AbstractDistMatrix<T>& A,                                       \
        Int m, Int n, T center, Base<T> radius )

#define PROTO(T)                                                \
    ABSTRACT_PROTO(T);                                          \
    template void MakeUniform(                                  \
        Matrix<T,Device::CPU>& A, T center, Base<T> radius );

#ifdef HYDROGEN_HAVE_GPU
template void MakeUniform(Matrix<float,Device::GPU>&, float, Base<float>);
template void MakeUniform(Matrix<double,Device::GPU>&, double, Base<double>);

#ifdef HYDROGEN_GPU_USE_FP16
ABSTRACT_PROTO(gpu_half_type);
template void MakeUniform(Matrix<gpu_half_type,Device::GPU>&,
                          gpu_half_type, Base<gpu_half_type>);
#endif
#endif // HYDROGEN_HAVE_GPU

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#define EL_ENABLE_HALF
#include <El/macros/Instantiate.h>

} // namespace El
