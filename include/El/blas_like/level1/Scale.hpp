/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_SCALE_HPP
#define EL_BLAS_SCALE_HPP


namespace El
{

#ifdef HYDROGEN_HAVE_GPU
template <typename T, typename=EnableIf<IsComputeType<T,Device::GPU>>>
void Scale(T const& alpha, Matrix<T,Device::GPU>& A)
{
    if( alpha == TypeTraits<T>::Zero() )
    {
        Zero(A);
    }
    else if( alpha != TypeTraits<T>::One() )
    {
        gpu_blas::Scale(A.Height(), A.Width(), alpha,
                        A.Buffer(), A.LDim(),
                        SyncInfoFromMatrix(A));
    }
}

template <typename T,
          typename=DisableIf<IsComputeType<T,Device::GPU>>,
          typename=void>
void Scale(T const&, Matrix<T,Device::GPU>&)
{
    LogicError("Scale: Bad device/type combo!");
}
#endif // HYDROGEN_HAVE_GPU

template <typename T, typename S,
          typename=EnableIf<IsComputeType<T,Device::CPU>>>
void Scale(S alphaS, Matrix<T,Device::CPU>& A)
{
    EL_DEBUG_CSE;
    const T alpha = T(alphaS);

    const Int ALDim = A.LDim();
    const Int height = A.Height();
    const Int width = A.Width();
    T* ABuf = A.Buffer();

    if( alpha == TypeTraits<T>::Zero() )
    {
        Zero( A );
    }
    else if ( alpha != TypeTraits<T>::One() )
    {
        if( A.Contiguous() )
        {
            EL_PARALLEL_FOR
            for( Int i=0; i<height*width; ++i )
            {
                ABuf[i] *= alpha;
            }
        }
        else
        {
            EL_PARALLEL_FOR_COLLAPSE2
            for( Int j=0; j<width; ++j )
            {
                for( Int i=0; i<height; ++i )
                {
                    ABuf[i+j*ALDim] *= alpha;
                }
            }
        }
    }
}
template <typename T, typename S,
          typename=DisableIf<IsComputeType<T,Device::CPU>>,
          typename=void>
void Scale(S, Matrix<T,Device::CPU>&)
{
    LogicError("Scale: Bad device/type combo!");
}

template<typename T,typename S>
void Scale( S alphaS, AbstractMatrix<T>& A )
{
    EL_DEBUG_CSE;
    const T alpha = T(alphaS);

    if( alpha == TypeTraits<T>::Zero() )
    {
        Zero(A);
    }
    else if( alpha != TypeTraits<T>::One() )
    {
        switch (A.GetDevice())
        {
        case Device::CPU:
            Scale(alpha, static_cast<Matrix<T,Device::CPU>&>(A));
            break;
#ifdef HYDROGEN_HAVE_GPU
        case Device::GPU:
            Scale(alpha, static_cast<Matrix<T,Device::GPU>&>(A));
            break;
#endif // HYDROGEN_HAVE_GPU
        default:
            LogicError("Bad device type in Scale");
        }
    }

}

template<typename Real,typename S,typename>
void Scale( S alphaS, AbstractMatrix<Real>& AReal, AbstractMatrix<Real>& AImag )
{
    EL_DEBUG_CSE;
    typedef Complex<Real> C;
    const C alpha = C(alphaS);
    if( alpha != C(1) )
    {
        if( alpha == C(0) )
        {
            Zero( AReal );
            Zero( AImag );
        }
        else
        {
            const Real alphaReal=alpha.real(), alphaImag=alpha.imag();
            Matrix<Real> ARealCopy;
            Copy( AReal, ARealCopy );
            Scale( alphaReal, AReal );
            Axpy( -alphaImag, AImag, AReal );
            Scale( alphaReal, AImag );
            Axpy( alphaImag, ARealCopy, AImag );
        }
    }
}

template<typename T,typename S>
void Scale( S alpha, AbstractDistMatrix<T>& A )
{
    EL_DEBUG_CSE
    Scale( alpha, A.Matrix() );
}

template<typename Real,typename S,typename>
void Scale( S alpha, AbstractDistMatrix<Real>& AReal,
                     AbstractDistMatrix<Real>& AImag )
{
    EL_DEBUG_CSE
    Scale( alpha, AReal.Matrix(), AImag.Matrix() );
}

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void Scale \
  ( T alpha, AbstractMatrix<T>& A ); \
  EL_EXTERN template void Scale \
  ( T alpha, AbstractDistMatrix<T>& A );

#ifdef HYDROGEN_GPU_USE_FP16
PROTO(gpu_half_type)
#endif // HYDROGEN_GPU_USE_FP16

#define EL_ENABLE_HALF
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_SCALE_HPP
