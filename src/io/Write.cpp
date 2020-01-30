/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

#include "./Write/Ascii.hpp"
#include "./Write/AsciiMatlab.hpp"
#include "./Write/Binary.hpp"
#include "./Write/BinaryFlat.hpp"
#include "./Write/Image.hpp"
#include "./Write/MatrixMarket.hpp"

namespace El {

template <typename T>
void Write(AbstractMatrix<T> const& A, string basename,
           FileFormat format, string title)
{
    switch (A.GetDevice())
    {
    case Device::CPU:
        Write(static_cast<Matrix<T,Device::CPU> const&>(A),
              basename, format, title);
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
    {
        // Copy to the CPU
        Matrix<T,Device::CPU> A_CPU{static_cast<Matrix<T,Device::GPU> const&>(A)};
        Write(A_CPU, basename, format, title);
    }
    break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("Write: Bad Device type.");
    }
}

template<typename T>
void Write
( const Matrix<T>& A,
  string basename, FileFormat format, string title )
{
    EL_DEBUG_CSE
    switch( format )
    {
    case ASCII:         write::Ascii( A, basename, title );       break;
    case ASCII_MATLAB:  write::AsciiMatlab( A, basename, title ); break;
    case BINARY:        write::Binary( A, basename );             break;
    case BINARY_FLAT:   write::BinaryFlat( A, basename );         break;
    case MATRIX_MARKET: write::MatrixMarket( A, basename );       break;
    case BMP:
    case JPG:
    case JPEG:
    case PNG:
    case PPM:
    case XBM:
    case XPM:
        write::Image( A, basename, format ); break;
    default:
        LogicError("Invalid file format");
    }
}

template<typename T>
void Write
( const AbstractDistMatrix<T>& A,
  string basename, FileFormat format, string title )
{
    EL_DEBUG_CSE
    if( A.ColStride() == 1 && A.RowStride() == 1 )
    {
        if( A.CrossRank() == A.Root() && A.RedundantRank() == 0 )
            Write( A.LockedMatrix(), basename, format, title );
    }
    else
    {
        DistMatrix<T,CIRC,CIRC> A_CIRC_CIRC( A );
        if( A_CIRC_CIRC.CrossRank() == A_CIRC_CIRC.Root() )
            Write( A_CIRC_CIRC.LockedMatrix(), basename, format, title );
    }
}

#ifdef HYDROGEN_GPU_USE_FP16
template <>
void Write<gpu_half_type>(AbstractMatrix<gpu_half_type> const& A,
                          string basename,
                          FileFormat format, string title)
{
    Matrix<float> A_tmp;
    Copy(A, A_tmp);
    return Write(A_tmp, basename, format, title);
}

template <>
void Write<gpu_half_type>(
    const AbstractDistMatrix<gpu_half_type>& A,
    string basename, FileFormat format, string title)
{
    std::unique_ptr<AbstractDistMatrix<float>> A_tmp(
        AbstractDistMatrix<float>::Instantiate(A.DistData()));
    Copy(A, *A_tmp);
    Write(*A_tmp, basename, format, title);
}
#endif // HYDROGEN_GPU_USE_FP16

#define PROTO(T) \
    template void Write                         \
    ( const AbstractMatrix<T>&, string, FileFormat, string );   \
    template void Write                                 \
    ( const Matrix<T>& A,                               \
      string basename, FileFormat format, string title );       \
    template void Write                                         \
    ( const AbstractDistMatrix<T>& A,                           \
      string basename, FileFormat format, string title );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#define EL_ENABLE_HALF
#include <El/macros/Instantiate.h>

} // namespace El
