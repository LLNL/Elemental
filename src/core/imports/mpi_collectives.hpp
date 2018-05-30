/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   Copyright (c) 2013, Jeff Hammond
   All rights reserved.

   Copyright (c) 2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include "mpi_utils.hpp"
#ifdef HYDROGEN_USES_ALUMINUM
#include "Al.hpp"
#endif //HYDROGEN_USES_ALUMINUM

typedef unsigned char* UCP;

namespace El {
namespace mpi {

template<typename Real,
         typename/*=EnableIf<IsPacked<Real>>*/>
void AllReduce( const Real* sbuf, Real* rbuf, int count, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if( count != 0 )
    {
        MPI_Op opC = NativeOp<Real>( op );
        EL_CHECK_MPI
        ( MPI_Allreduce
          ( const_cast<Real*>(sbuf), rbuf, count, TypeMap<Real>(), opC,
            comm.comm ) );
    }
}

template<typename Real,
         typename/*=EnableIf<IsPacked<Real>>*/>
void AllReduce
( const Complex<Real>* sbuf, Complex<Real>* rbuf, int count, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if( count != 0 )
    {
#ifdef EL_AVOID_COMPLEX_MPI
        if( op == SUM )
        {
            MPI_Op opC = NativeOp<Real>( op );
            EL_CHECK_MPI
            ( MPI_Allreduce
                ( const_cast<Complex<Real>*>(sbuf),
                  rbuf, 2*count, TypeMap<Real>(), opC, comm.comm ) );
        }
        else
        {
            MPI_Op opC = NativeOp<Complex<Real>>( op );
            EL_CHECK_MPI
            ( MPI_Allreduce
              ( const_cast<Complex<Real>*>(sbuf),
                rbuf, count, TypeMap<Complex<Real>>(), opC, comm.comm ) );
        }
#else
        MPI_Op opC = NativeOp<Complex<Real>>( op );
        EL_CHECK_MPI
        ( MPI_Allreduce
          ( const_cast<Complex<Real>*>(sbuf),
            rbuf, count, TypeMap<Complex<Real>>(), opC, comm.comm ) );
#endif
    }
}

template<typename T,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void AllReduce
( const T* sbuf, T* rbuf, int count, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if( count == 0 )
        return;

    MPI_Op opC = NativeOp<T>( op );
    std::vector<byte> packedSend, packedRecv;
    Serialize( count, sbuf, packedSend );

    ReserveSerialized( count, rbuf, packedRecv );
    EL_CHECK_MPI
    ( MPI_Allreduce
      ( packedSend.data(), packedRecv.data(), count, TypeMap<T>(),
        opC, comm.comm ) );
    Deserialize( count, packedRecv, rbuf );
}

template<typename T>
void AllReduce( const T* sbuf, T* rbuf, int count, Comm comm )
EL_NO_RELEASE_EXCEPT
{ AllReduce( sbuf, rbuf, count, SUM, comm ); }

template<typename T>
T AllReduce( T sb, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT
{ T rb; AllReduce( &sb, &rb, 1, op, comm ); return rb; }

template<typename T>
T AllReduce( T sb, Comm comm )
EL_NO_RELEASE_EXCEPT
{ return AllReduce( sb, SUM, comm ); }

template<typename Real,
         typename/*=EnableIf<IsPacked<Real>>*/>
void AllReduce( Real* buf, int count, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if( count == 0 || Size(comm) == 1 )
        return;

    MPI_Op opC = NativeOp<Real>( op );
    EL_CHECK_MPI
    ( MPI_Allreduce
      ( MPI_IN_PLACE, buf, count, TypeMap<Real>(), opC, comm.comm ) );
}

template<typename Real,
         typename/*=EnableIf<IsPacked<Real>>*/>
void AllReduce( Complex<Real>* buf, int count, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if( count == 0 || Size(comm) == 1 )
        return;

#ifdef EL_AVOID_COMPLEX_MPI
    if( op == SUM )
    {
        MPI_Op opC = NativeOp<Real>( op );
        EL_CHECK_MPI
        ( MPI_Allreduce
          ( MPI_IN_PLACE, buf, 2*count, TypeMap<Real>(), opC, comm.comm ) );
    }
    else
    {
        MPI_Op opC = NativeOp<Complex<Real>>( op );
        EL_CHECK_MPI
        ( MPI_Allreduce
          ( MPI_IN_PLACE, buf, count, TypeMap<Complex<Real>>(),
            opC, comm.comm ) );
    }
#else
    MPI_Op opC = NativeOp<Complex<Real>>( op );
    EL_CHECK_MPI
    ( MPI_Allreduce
      ( MPI_IN_PLACE, buf, count, TypeMap<Complex<Real>>(), opC,
        comm.comm ) );
#endif
}

template<typename T,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void AllReduce( T* buf, int count, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if( count == 0 )
        return;

    MPI_Op opC = NativeOp<T>( op );
    std::vector<byte> packedSend, packedRecv;
    Serialize( count, buf, packedSend );

    ReserveSerialized( count, buf, packedRecv );
    EL_CHECK_MPI
    ( MPI_Allreduce
      ( packedSend.data(), packedRecv.data(), count, TypeMap<T>(),
        opC, comm.comm ) );
    Deserialize( count, packedRecv, buf );
}

template<typename T>
void AllReduce( T* buf, int count, Comm comm )
EL_NO_RELEASE_EXCEPT
{ AllReduce( buf, count, SUM, comm ); }


} // namespace mpi
} // namespace El

