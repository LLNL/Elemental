/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_BROADCAST_HPP
#define EL_BLAS_BROADCAST_HPP

namespace El {

template <typename T, Device D>
void Broadcast_impl( Matrix<T,D>& A, mpi::Comm const& comm, int rank )
{
    EL_DEBUG_CSE
    const int commSize = mpi::Size( comm );
    const int commRank = mpi::Rank( comm );
    if( commSize == 1 )
        return;

    const Int height = A.Height();
    const Int width = A.Width();
    const Int size = height*width;
    SyncInfo<D> syncInfoA = SyncInfoFromMatrix(A);

    if( height == A.LDim() )
    {
        mpi::Broadcast(A.Buffer(), size, rank, comm, syncInfoA);
    }
    else
    {
        simple_buffer<T,D> buf(size, syncInfoA);

        // Pack
        if( commRank == rank )
            copy::util::InterleaveMatrix(
                height, width,
                A.LockedBuffer(), 1, A.LDim(),
                buf.data(),       1, height, syncInfoA);

        mpi::Broadcast(buf.data(), size, rank, comm, syncInfoA);

        // Unpack
        if( commRank != rank )
            copy::util::InterleaveMatrix(
                height,        width,
                buf.data(), 1, height,
                A.Buffer(), 1, A.LDim(), syncInfoA);
    }
}

template<typename T>
void Broadcast( AbstractMatrix<T>& A, mpi::Comm const& comm, int rank )
{
    switch(A.GetDevice())
    {
    case Device::CPU:
        Broadcast_impl(static_cast<Matrix<T,Device::CPU>&>(A),
                       std::move(comm), rank);
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        Broadcast_impl(static_cast<Matrix<T,Device::GPU>&>(A),
                       std::move(comm), rank);
        break;
#endif // HYROGEN_HAVE_CUDA
    default:
        LogicError("Unsupported device type.");
    }
}


template <Device D, typename T>
void Broadcast_impl( AbstractDistMatrix<T>& A, mpi::Comm const& comm, int rank )
{
    EL_DEBUG_CSE
    const int commSize = mpi::Size( comm );
    const int commRank = mpi::Rank( comm );
    if( commSize == 1 )
        return;
    if( !A.Participating() )
        return;

    const Int localHeight = A.LocalHeight();
    const Int localWidth = A.LocalWidth();
    const Int localSize = localHeight*localWidth;

    SyncInfo<D> syncInfoA = SyncInfoFromMatrix(
        static_cast<Matrix<T,D> const&>(A.LockedMatrix()));

    if( localHeight == A.LDim() )
    {
        mpi::Broadcast(A.Buffer(), localSize, rank, comm, syncInfoA);
    }
    else
    {
        simple_buffer<T,D> buf(localSize, syncInfoA);

        // Pack
        if( commRank == rank )
            copy::util::InterleaveMatrix(
                localHeight, localWidth,
                A.LockedBuffer(), 1, A.LDim(),
                buf.data(),       1, localHeight, syncInfoA);

        mpi::Broadcast(buf.data(), localSize, rank, comm, syncInfoA);

        // Unpack
        if( commRank != rank )
            copy::util::InterleaveMatrix(
                localHeight, localWidth,
                buf.data(), 1, localHeight,
                A.Buffer(), 1, A.LDim(), syncInfoA);
    }
}

template<typename T>
void Broadcast( AbstractDistMatrix<T>& A, mpi::Comm const& comm, int rank )
{
    EL_DEBUG_CSE
    switch (A.GetLocalDevice())
    {
    case Device::CPU:
        Broadcast_impl<Device::CPU>(A, std::move(comm), rank);
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        Broadcast_impl<Device::GPU>(A, std::move(comm), rank);
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("Broadcast: Bad device.");
    }
}

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void Broadcast \
  ( AbstractMatrix<T>& A, mpi::Comm const& comm, int rank ); \
  EL_EXTERN template void Broadcast \
  ( AbstractDistMatrix<T>& A, mpi::Comm const& comm, int rank );

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

#endif // ifndef EL_BLAS_BROADCAST_HPP
