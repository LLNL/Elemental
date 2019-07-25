/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_ALLREDUCE_HPP
#define EL_BLAS_ALLREDUCE_HPP

namespace El
{

template <typename T, Device D, typename>
void AllReduce(Matrix<T,D>& A, mpi::Comm const& comm, mpi::Op op)
{
    EL_DEBUG_CSE
    if(mpi::Size(comm) == 1)
        return;
    const Int height = A.Height();
    const Int width = A.Width();
    const Int size = height*width;
    if(height == A.LDim())
    {
        mpi::AllReduce(A.Buffer(), size, op, comm, SyncInfoFromMatrix(A));
    }
    else
    {
        SyncInfo<D> syncInfoA = SyncInfoFromMatrix(A);

        simple_buffer<T,D> buf(size, syncInfoA);

        // Pack
        copy::util::InterleaveMatrix(
            height, width,
            A.LockedBuffer(), 1, A.LDim(),
            buf.data(),       1, height, syncInfoA);

        mpi::AllReduce(buf.data(), size, op, comm, syncInfoA);

        // Unpack
        copy::util::InterleaveMatrix(
            height,        width,
            buf.data(), 1, height,
            A.Buffer(), 1, A.LDim(), syncInfoA);
    }
}

template <typename T, Device D,typename,typename>
void AllReduce(Matrix<T,D>& A, mpi::Comm const& comm, mpi::Op op)
{
    LogicError("AllReduce: Bad type/device combination!");
}

template <typename T>
void AllReduce(AbstractMatrix<T>& A, mpi::Comm const& comm, mpi::Op op)
{
    switch (A.GetDevice())
    {
    case Device::CPU:
        AllReduce(static_cast<Matrix<T,Device::CPU>&>(A), comm, op);
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        AllReduce(static_cast<Matrix<T,Device::GPU>&>(A), comm, op);
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("AllReduce: Bad device!");
    }
}

template<typename T>
void AllReduce(AbstractDistMatrix<T>& A, mpi::Comm const& comm, mpi::Op op)
{
    EL_DEBUG_CSE
    if(mpi::Size(comm) == 1)
        return;
    if(!A.Participating())
        return;

    AllReduce(A.Matrix(), comm, op);
}

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void AllReduce \
  (AbstractMatrix<T>& A, mpi::Comm const& comm, mpi::Op op); \
  EL_EXTERN template void AllReduce \
  (AbstractDistMatrix<T>& A, mpi::Comm const& comm, mpi::Op op);

#define EL_ENABLE_HALF
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_ALLREDUCE_HPP
