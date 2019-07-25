/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_RECV_HPP
#define EL_BLAS_RECV_HPP

namespace El {

// Recall that A must already be the correct size
template<typename T, Device D>
void Recv(Matrix<T,D>& A, mpi::Comm const& comm, int source)
{
    EL_DEBUG_CSE
    const Int height = A.Height();
    const Int width = A.Width();
    const Int size = height*width;

    SyncInfo<D> syncInfoA = SyncInfoFromMatrix(A);

    if( height == A.LDim() )
    {
        mpi::Recv( A.Buffer(), size, source, comm, syncInfoA );
    }
    else
    {
        simple_buffer<T,D> buf(size, syncInfoA);

        mpi::Recv( buf.data(), size, source, comm, syncInfoA );

        // Unpack
        copy::util::InterleaveMatrix(
            height,        width,
            buf.data(), 1, height,
            A.Buffer(), 1, A.LDim(), syncInfoA);
    }
}

template <typename T>
void Recv(AbstractMatrix<T>& A, mpi::Comm const& comm, int source)
{
    EL_DEBUG_CSE;
    switch (A.GetDevice())
    {
    case Device::CPU:
        Recv(static_cast<Matrix<T,Device::CPU>&>(A), std::move(comm), source);
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        Recv(static_cast<Matrix<T,Device::GPU>&>(A), std::move(comm), source);
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("Recv: Bad device.");
    }
}
#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T)                                                \
    EL_EXTERN template void Recv(                               \
        AbstractMatrix<T>& A, mpi::Comm const& comm, int source );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_RECV_HPP
