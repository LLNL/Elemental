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
#include <El/core/Profiling.hpp>

#include "mpi_utils.hpp"
#include "mpi_collectives.hpp"

#include <El/core/imports/mpi.hpp>

typedef unsigned char* UCP;

namespace El
{
namespace mpi
{

const int ANY_SOURCE = MPI_ANY_SOURCE;
const int ANY_TAG = MPI_ANY_TAG;
const int THREAD_SINGLE = MPI_THREAD_SINGLE;
const int THREAD_FUNNELED = MPI_THREAD_FUNNELED;
const int THREAD_SERIALIZED = MPI_THREAD_SERIALIZED;
const int THREAD_MULTIPLE = MPI_THREAD_MULTIPLE;
const int UNDEFINED = MPI_UNDEFINED;

const Comm COMM_NULL;
const Comm COMM_SELF = MakeControllingComm(MPI_COMM_SELF);
const Comm COMM_WORLD = MakeControllingComm(MPI_COMM_WORLD);

const ErrorHandler ERRORS_RETURN = MPI_ERRORS_RETURN;
const ErrorHandler ERRORS_ARE_FATAL = MPI_ERRORS_ARE_FATAL;
const Group GROUP_NULL = MPI_GROUP_NULL;
const Group GROUP_EMPTY = MPI_GROUP_EMPTY;
const Op MAX = MPI_MAX;
const Op MIN = MPI_MIN;
const Op MAXLOC = MPI_MAXLOC;
const Op MINLOC = MPI_MINLOC;
const Op PROD = MPI_PROD;
const Op SUM = MPI_SUM;
const Op LOGICAL_AND = MPI_LAND;
const Op LOGICAL_OR = MPI_LOR;
const Op LOGICAL_XOR = MPI_LXOR;
const Op BINARY_AND = MPI_BAND;
const Op BINARY_OR = MPI_BOR;
const Op BINARY_XOR = MPI_BXOR;

bool CommSameSizeAsInteger() EL_NO_EXCEPT
{ return sizeof(MPI_Comm) == sizeof(int); }

bool GroupSameSizeAsInteger() EL_NO_EXCEPT
{ return sizeof(MPI_Group) == sizeof(int); }

// MPI environmental routines
// ==========================

void Initialize( int& argc, char**& argv ) EL_NO_EXCEPT
{
    AUTO_NOSYNC_PROFILE_REGION("MPI.Initialize");

    MPI_Init( &argc, &argv );
#ifdef HYDROGEN_HAVE_ALUMINUM
    Al::Initialize(argc, argv);
#endif // HYDROGEN_HAVE_ALUMINUM
}


int InitializeThread( int& argc, char**& argv, int required ) EL_NO_EXCEPT
{
    AUTO_NOSYNC_PROFILE_REGION("MPI.InitializeThread");
    int provided;

    MPI_Init_thread( &argc, &argv, required, &provided );

#ifdef HYDROGEN_HAVE_ALUMINUM
    Al::Initialize(argc, argv);
#endif // HYDROGEN_HAVE_ALUMINUM

    return provided;
}

void Finalize() EL_NO_EXCEPT
{
    AUTO_NOSYNC_PROFILE_REGION("MPI.Finalize");
#ifdef HYDROGEN_HAVE_ALUMINUM
    // Making sure finalizing Aluminum before finalizing MPI.
    Al::Finalize();
#endif // HYDROGEN_HAVE_ALUMINUM

    if (!Finalized())
        MPI_Finalize();
}

bool Initialized() EL_NO_EXCEPT
{
    int initialized;
    MPI_Initialized( &initialized );
    return initialized;
}

bool Finalized() EL_NO_EXCEPT
{
    int finalized;
    MPI_Finalized( &finalized );
    return finalized;
}

int QueryThread() EL_NO_EXCEPT
{
    int provided;
    MPI_Query_thread( &provided );
    return provided;
}

void Abort( Comm const& comm, int errCode ) EL_NO_EXCEPT
{ MPI_Abort( comm.GetMPIComm(), errCode ); }

double Time() EL_NO_EXCEPT { return MPI_Wtime(); }

void Create( UserFunction* func, bool commutes, Op& op ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_CHECK_MPI_CALL( MPI_Op_create( func, commutes, &op.op ) );
}

void Free( Op& op ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_CHECK_MPI_CALL( MPI_Op_free( &op.op ) );
}

void Free( Datatype& type ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_CHECK_MPI_CALL( MPI_Type_free( &type ) );
}

// Communicator manipulation
// =========================
int Rank( Comm const& comm ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    return comm.Rank();
}

int Size( Comm const& comm ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    return comm.Size();
}

void Create( Comm const& parentComm, Group subsetGroup, Comm& subsetComm )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    MPI_Comm tmp;
    EL_CHECK_MPI_CALL(
        MPI_Comm_create(parentComm.GetMPIComm(), subsetGroup.group, &tmp));
    subsetComm.Control(tmp);
}

void Dup( Comm const& original, Comm& duplicate ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    MPI_Comm tmp;
    EL_CHECK_MPI_CALL( MPI_Comm_dup( original.GetMPIComm(), &tmp) );
    duplicate.Control(tmp);
}


void Split( Comm const& comm, int color, int key, Comm& newComm ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    MPI_Comm tmp;
    EL_CHECK_MPI_CALL(
        MPI_Comm_split( comm.GetMPIComm(), color, key, &tmp ) );
    newComm.Control(tmp);
}

void Free( Comm& comm ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    comm.Reset();
}

namespace /* <anon> */
{
bool RawCommCongruent(MPI_Comm comm_one, MPI_Comm comm_two) EL_NO_RELEASE_EXCEPT
{
    int result;
    EL_CHECK_MPI_CALL(
        MPI_Comm_compare(comm_one, comm_two, &result));
    return ((result == MPI_IDENT) || (result == MPI_CONGRUENT));
}
}// namespace <anon>

bool Congruent( Comm const& comm1, Comm const& comm2 ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    return RawCommCongruent(comm1.GetMPIComm(), comm2.GetMPIComm());
}

bool CongruentToCommSelf( Comm const& comm ) EL_NO_RELEASE_EXCEPT
{
    return comm.Size() == 1;// RawCommCongruent(comm.GetMPIComm(), MPI_COMM_SELF);
}

bool CongruentToCommWorld( Comm const& comm ) EL_NO_RELEASE_EXCEPT
{
    int world_size;
    EL_CHECK_MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    return comm.Size() == world_size;// RawCommCongruent(comm.GetMPIcomm, MPI_COMM_WORLD);
}

Comm NewWorldComm() EL_NO_RELEASE_EXCEPT
{
    return Comm{MPI_COMM_WORLD};
}

void ErrorHandlerSet( Comm const& comm, ErrorHandler errorHandler )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_CHECK_MPI_CALL( MPI_Comm_set_errhandler( comm.GetMPIComm(), errorHandler ) );
}

// Cartesian communicator routines
// ===============================

void CartCreate
( Comm const& comm, int numDims, const int* dimensions, const int* periods,
  bool reorder, Comm& cartComm ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    MPI_Comm tmp;
    EL_CHECK_MPI_CALL(
        MPI_Cart_create(
            comm.GetMPIComm(), numDims, const_cast<int*>(dimensions),
            const_cast<int*>(periods), reorder, &tmp ) );
    cartComm.Control(tmp);
}

// FIXME
void CartSub( Comm const& comm, const int* remainingDims, Comm& subComm )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    MPI_Comm tmp;
    EL_CHECK_MPI_CALL(
        MPI_Cart_sub(
            comm.GetMPIComm(), const_cast<int*>(remainingDims), &tmp));
    subComm.Control(tmp);
}

// Group manipulation
// ==================

int Rank( Group group ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    int rank;
    EL_CHECK_MPI_CALL( MPI_Group_rank( group.group, &rank ) );
    return rank;
}

int Size( Group group ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    int size;
    EL_CHECK_MPI_CALL( MPI_Group_size( group.group, &size ) );
    return size;
}

void CommGroup( Comm const& comm, Group& group ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_CHECK_MPI_CALL( MPI_Comm_group( comm.GetMPIComm(), &group.group ) );
}

void Dup( Group group, Group& newGroup ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    // For some reason, MPI_Group_dup does not exist
    Excl( group, 0, 0, newGroup );
}

void Union( Group groupA, Group groupB, Group& newGroup ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_CHECK_MPI_CALL(
        MPI_Group_union( groupA.group, groupB.group, &newGroup.group ) );
}

void Incl( Group group, int n, const int* ranks, Group& subGroup )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_CHECK_MPI_CALL(
      MPI_Group_incl
      ( group.group, n, const_cast<int*>(ranks), &subGroup.group )
    );
}

void Excl( Group group, int n, const int* ranks, Group& subGroup )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_CHECK_MPI_CALL(
      MPI_Group_excl
      ( group.group, n, const_cast<int*>(ranks), &subGroup.group )
    );
}

void Difference( Group parent, Group subset, Group& complement )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_CHECK_MPI_CALL(
      MPI_Group_difference( parent.group, subset.group, &complement.group )
    );
}

void Free( Group& group ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_CHECK_MPI_CALL( MPI_Group_free( &group.group ) );
}

bool Congruent( Group group1, Group group2 ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    int result;
    EL_CHECK_MPI_CALL( MPI_Group_compare( group1.group, group2.group, &result ) );
    return ( result == MPI_IDENT );
}

// Rank translations
// =================

int Translate( Group origGroup, int origRank, Group newGroup )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    int newRank;
    Translate( origGroup, 1, &origRank, newGroup, &newRank );
    return newRank;
}

int Translate( Comm const& origComm, int origRank, Group newGroup )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    int newRank;
    Translate( origComm, 1, &origRank, newGroup, &newRank );
    return newRank;
}

int Translate( Group origGroup, int origRank, Comm const& newComm )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    int newRank;
    Translate( origGroup, 1, &origRank, newComm, &newRank );
    return newRank;
}

int Translate( Comm const& origComm, int origRank, Comm const& newComm )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    int newRank;
    Translate( origComm, 1, &origRank, newComm, &newRank );
    return newRank;
}

void Translate
( Group origGroup, int size, const int* origRanks,
  Group newGroup,                  int* newRanks ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_CHECK_MPI_CALL
    ( MPI_Group_translate_ranks
      ( origGroup.group, size, const_cast<int*>(origRanks),
        newGroup.group, newRanks ) );
}

void Translate(
    Comm const& origComm,  int size, const int* origRanks,
    Group newGroup, int* newRanks ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    Group origGroup;
    CommGroup( origComm, origGroup );
    Translate( origGroup, size, origRanks, newGroup, newRanks );
    Free( origGroup );
}

void Translate(
    Group origGroup, int size, const int* origRanks,
    Comm const& newComm, int* newRanks ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    Group newGroup;
    CommGroup( newComm,  newGroup  );
    Translate( origGroup, size, origRanks, newGroup, newRanks );
    Free( newGroup  );
}

void Translate(
    Comm const& origComm, int size, const int* origRanks,
    Comm const& newComm, int* newRanks ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    Group origGroup, newGroup;
    CommGroup( origComm, origGroup );
    CommGroup( newComm,  newGroup  );
    Translate( origGroup, size, origRanks, newGroup, newRanks );
    Free( origGroup );
    Free( newGroup  );
}

// Various utilities
// =================

// Wait until every process in comm reaches this statement
void Barrier( Comm const& comm ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_CHECK_MPI_CALL( MPI_Barrier( comm.GetMPIComm() ) );
}

// Test for completion
template <typename T>
bool Test( Request<T>& request ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    Status status;
    int flag;
    EL_CHECK_MPI_CALL( MPI_Test( &request.backend, &flag, &status ) );
    return flag;
}

// Ensure that the request finishes before continuing
template <typename T>
void Wait( Request<T>& request ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    Status status;
    Wait( request, status );
}

// Ensure that the request finishes before continuing
template <typename T,
         typename/*=EnableIf<IsPacked<T>>*/>
void Wait( Request<T>& request, Status& status ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_CHECK_MPI_CALL( MPI_Wait( &request.backend, &status ) );
}

// Ensure that several requests finish before continuing
template <typename T>
void WaitAll( int numRequests, Request<T>* requests ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    std::vector<Status> statuses( numRequests );
    WaitAll( numRequests, requests, statuses.data() );
}

// Ensure that several requests finish before continuing
template <typename T,
         typename/*=EnableIf<IsPacked<T>>*/>
void WaitAll( int numRequests, Request<T>* requests, Status* statuses )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifndef EL_MPI_REQUEST_IS_NOT_POINTER
    // Both MPICH and OpenMPI define MPI_Request to be a pointer to a structure,
    // which implies that the following code is legal. AFAIK, there are not
    // any popular MPI implementations which should break this logic, but
    // the alternative #ifdef logic is provided in case a breakage is observed.
    std::vector<MPI_Request> backends( numRequests );
    for( Int j=0; j<numRequests; ++j )
        backends[j] = requests[j].backend;
    EL_CHECK_MPI_CALL( MPI_Waitall( numRequests, backends.data(), statuses ) );
    // NOTE: This write back will almost always be superfluous, but it ensures
    //       that any changes to the pointer are propagated
    for( Int j=0; j<numRequests; ++j )
        requests[j].backend = backends[j];
#else
    for( Int j=0; j<numRequests; ++j )
    {
        Status status;
        MPI_Wait( &requests[j].backend, &status );
    }
#endif
}

template <typename T,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void Wait( Request<T>& request, Status& status ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_CHECK_MPI_CALL( MPI_Wait( &request.backend, &status ) );
    if( request.receivingPacked )
    {
        Deserialize
        ( request.recvCount, request.buffer.data(), request.unpackedRecvBuf );
        request.receivingPacked = false;
    }
    request.buffer.clear();
}

template <typename T,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void WaitAll( int numRequests, Request<T>* requests, Status* statuses )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifndef EL_MPI_REQUEST_IS_NOT_POINTER
    // Both MPICH and OpenMPI define MPI_Request to be a pointer to a structure,
    // which implies that the following code is legal. AFAIK, there are not
    // any popular MPI implementations which should break this logic, but
    // the alternative #ifdef logic is provided in case a breakage is observed.
    std::vector<MPI_Request> backends( numRequests );
    for( Int j=0; j<numRequests; ++j )
        backends[j] = requests[j].backend;
    EL_CHECK_MPI_CALL( MPI_Waitall( numRequests, backends.data(), statuses ) );
    // NOTE: This write back will almost always be superfluous, but it ensures
    //       that any changes to the pointer are propagated
    for( Int j=0; j<numRequests; ++j )
        requests[j].backend = backends[j];
#else
    for( Int j=0; j<numRequests; ++j )
    {
        Status status;
        MPI_Wait( &requests[j].backend, &status );
    }
#endif
    for( Int j=0; j<numRequests; ++j )
    {
        if( requests[j].receivingPacked )
        {
            Deserialize
            ( requests[j].recvCount,
              requests[j].buffer.data(),
              requests[j].unpackedRecvBuf );
            requests[j].receivingPacked = false;
        }
        requests[j].buffer.clear();
    }
}

// Nonblocking test for message completion
bool IProbe( int source, int tag, Comm const& comm, Status& status )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    int flag;
    EL_CHECK_MPI_CALL(
        MPI_Iprobe( source, tag, comm.GetMPIComm(), &flag, &status ) );
    return flag;
}
bool IProbe( int source, Comm const& comm, Status& status ) EL_NO_RELEASE_EXCEPT
{ return IProbe( source, 0, comm, status ); }

template <typename T>
int GetCount( Status& status ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    int count;
    EL_CHECK_MPI_CALL( MPI_Get_count( &status, TypeMap<T>(), &count ) );
    return count;
}

template <typename Real, Device D,
         typename/*=EnableIf<IsPacked<Real>>*/>
void TaggedSend(const Real* buf, int count, int to, int tag, Comm const& comm,
                SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

    if constexpr (IsAluminumSupported<Real, D, Collective::SEND>::value)
    {
        (void) tag; // Al doesn't use tags.
        using BE = BestBackend<Real, D, Collective::SEND>;
        Al::Send<BE>(buf, count, to, comm.template GetComm<BE>(syncInfo));
    }
    else
    {
#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
        ENSURE_HOST_SEND_BUFFER(buf, count, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

        Synchronize(syncInfo);

        EL_CHECK_MPI_CALL(
            MPI_Send(
                buf, count, TypeMap<Real>(), to, tag, comm.GetMPIComm()));
    }
}

template <typename Real, Device D,
         typename/*=EnableIf<IsPacked<Real>>*/>
void TaggedSend(
    const Complex<Real>* buf, int count, int to, int tag, Comm const& comm,
    SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    using CT = std::complex<Real>;
    if constexpr (IsAluminumSupported<CT, D, Collective::SEND>::value)
    {
        (void) tag; // Al doesn't use tags.
        using BE = BestBackend<CT, D, Collective::SEND>;
        Al::Send<BE>(buf, count, to, comm.template GetComm<BE>(syncInfo));
    }
    else if constexpr (IsAluminumSupported<Real, D, Collective::SEND>::value)
    {
        (void) tag; // Al doesn't use tags.
        using BE = BestBackend<Real, D, Collective::SEND>;
        Al::Send<BE>((Real*)buf, 2*count, to,
                 comm.template GetComm<BE>(syncInfo));
    }
    else {
#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
        ENSURE_HOST_SEND_BUFFER(buf, count, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

        Synchronize(syncInfo);
#ifdef EL_AVOID_COMPLEX_MPI
        EL_CHECK_MPI_CALL(
            MPI_Send(
                buf, 2*count, TypeMap<Real>(), to, tag, comm.GetMPIComm()));
#else
        EL_CHECK_MPI_CALL(
            MPI_Send(
                buf, count, TypeMap<Complex<Real>>(), to, tag, comm.GetMPIComm()));
#endif
    }
}

template <typename T, Device D,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void TaggedSend( const T* buf, int count, int to, int tag, Comm const& comm,
                 SyncInfo<D> const& syncInfo)
    EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_SEND_BUFFER(buf, count, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    std::vector<byte> packedBuf;
    Serialize(count, buf, packedBuf);
    EL_CHECK_MPI_CALL(
        MPI_Send(
            packedBuf.data(), count, TypeMap<T>(), to, tag, comm.GetMPIComm()));
}

template <typename T, Device D>
void Send( const T* buf, int count, int to, Comm const& comm,
           SyncInfo<D> const& syncInfo) EL_NO_RELEASE_EXCEPT
{ TaggedSend( buf, count, to, 0, comm, syncInfo ); }

template <typename T>
void TaggedSend( T b, int to, int tag, Comm const& comm )
    EL_NO_RELEASE_EXCEPT
{ TaggedSend( &b, 1, to, tag, comm, SyncInfo<Device::CPU>{}); }

template <typename T>
void Send( T b, int to, Comm const& comm )
    EL_NO_RELEASE_EXCEPT
{ TaggedSend( b, to, 0, comm ); }

template <typename Real,
         typename/*=EnableIf<IsPacked<Real>>*/>
void TaggedISend
( const Real* buf, int count, int to, int tag, Comm const& comm,
  Request<Real>& request )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_CHECK_MPI_CALL
    ( MPI_Isend
      ( const_cast<Real*>(buf), count, TypeMap<Real>(), to,
        tag, comm.GetMPIComm(), &request.backend ) );
}

template <typename Real,
         typename/*=EnableIf<IsPacked<Real>>*/>
void TaggedISend
( const Complex<Real>* buf, int count, int to, int tag, Comm const& comm,
  Request<Complex<Real>>& request ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifdef EL_AVOID_COMPLEX_MPI
    EL_CHECK_MPI_CALL
    ( MPI_Isend
      ( const_cast<Complex<Real>*>(buf), 2*count,
        TypeMap<Real>(), to, tag, comm.GetMPIComm(), &request.backend ) );
#else
    EL_CHECK_MPI_CALL
    ( MPI_Isend
      ( const_cast<Complex<Real>*>(buf), count,
        TypeMap<Complex<Real>>(), to, tag, comm.GetMPIComm(), &request.backend ) );
#endif
}

template <typename T,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void TaggedISend
( const T* buf, int count, int to, int tag, Comm const& comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    Serialize( count, buf, request.buffer );
    EL_CHECK_MPI_CALL
    ( MPI_Isend
      ( request.buffer.data(), count, TypeMap<T>(), to, tag, comm.GetMPIComm(),
        &request.backend ) );
}

template <typename T>
void ISend
( const T* buf, int count, int to, Comm const& comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT
{ TaggedISend( buf, count, to, 0, comm, request ); }

template <typename T>
void TaggedISend( T b, int to, int tag, Comm const& comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT
{ TaggedISend( &b, 1, to, tag, comm, request ); }

template <typename T>
void ISend( T b, int to, Comm const& comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT
{ TaggedISend( b, to, 0, comm, request ); }

template <typename Real,
         typename/*=EnableIf<IsPacked<Real>>*/>
void TaggedIRSend
( const Real* buf, int count, int to, int tag, Comm const& comm,
  Request<Real>& request )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_CHECK_MPI_CALL
    ( MPI_Irsend
      ( const_cast<Real*>(buf), count, TypeMap<Real>(), to,
        tag, comm.GetMPIComm(), &request.backend ) );
}

template <typename Real,
         typename/*=EnableIf<IsPacked<Real>>*/>
void TaggedIRSend
( const Complex<Real>* buf, int count, int to, int tag, Comm const& comm,
  Request<Complex<Real>>& request ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifdef EL_AVOID_COMPLEX_MPI
    EL_CHECK_MPI_CALL
    ( MPI_Irsend
      ( const_cast<Complex<Real>*>(buf), 2*count,
        TypeMap<Real>(), to, tag, comm.GetMPIComm(), &request.backend ) );
#else
    EL_CHECK_MPI_CALL
    ( MPI_Irsend
      ( const_cast<Complex<Real>*>(buf), count,
        TypeMap<Complex<Real>>(), to, tag, comm.GetMPIComm(), &request.backend ) );
#endif
}

template <typename T,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void TaggedIRSend
( const T* buf, int count, int to, int tag, Comm const& comm,
  Request<T>& request )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    Serialize( count, buf, request.buffer );
    EL_CHECK_MPI_CALL
    ( MPI_Irsend
      ( request.buffer.data(), count, TypeMap<T>(), to,
        tag, comm.GetMPIComm(), &request.backend ) );
}

template <typename T>
void IRSend
( const T* buf, int count, int to, Comm const& comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT
{ TaggedIRSend( buf, count, to, 0, comm, request ); }

template <typename T>
void TaggedIRSend( T b, int to, int tag, Comm const& comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT
{ TaggedIRSend( &b, 1, to, tag, comm, request ); }

template <typename T>
void IRSend( T b, int to, Comm const& comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT
{ TaggedIRSend( b, to, 0, comm, request ); }

template <typename Real,
         typename/*=EnableIf<IsPacked<Real>>*/>
void TaggedISSend
( const Real* buf, int count, int to, int tag, Comm const& comm,
  Request<Real>& request ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_CHECK_MPI_CALL
    ( MPI_Issend
      ( const_cast<Real*>(buf), count, TypeMap<Real>(), to,
        tag, comm.GetMPIComm(), &request.backend ) );
}

template <typename Real,
         typename/*=EnableIf<IsPacked<Real>>*/>
void TaggedISSend
( const Complex<Real>* buf, int count, int to, int tag, Comm const& comm,
  Request<Complex<Real>>& request ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifdef EL_AVOID_COMPLEX_MPI
    EL_CHECK_MPI_CALL
    ( MPI_Issend
      ( const_cast<Complex<Real>*>(buf), 2*count,
        TypeMap<Real>(), to, tag, comm.GetMPIComm(), &request.backend ) );
#else
    EL_CHECK_MPI_CALL
    ( MPI_Issend
      ( const_cast<Complex<Real>*>(buf), count,
        TypeMap<Complex<Real>>(), to, tag, comm.GetMPIComm(), &request.backend ) );
#endif
}

template <typename T,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void TaggedISSend
( const T* buf, int count, int to, int tag, Comm const& comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    Serialize( count, buf, request.buffer );
    EL_CHECK_MPI_CALL
    ( MPI_Issend
      ( request.buffer.data(), count, TypeMap<T>(), to,
        tag, comm.GetMPIComm(), &request.backend ) );
}

template <typename T>
void ISSend( const T* buf, int count, int to, Comm const& comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT
{ TaggedISSend( buf, count, to, 0, comm, request ); }

template <typename T>
void TaggedISSend( T b, int to, int tag, Comm const& comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT
{ TaggedISSend( &b, 1, to, tag, comm, request ); }

template <typename Real, Device D,
         typename/*=EnableIf<IsPacked<Real>>*/>
void TaggedRecv( Real* buf, int count, int from, int tag, Comm const& comm,
                 SyncInfo<D> const& syncInfo ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

    if constexpr (IsAluminumSupported<Real, D, Collective::RECV>::value)
    {
        (void) tag; // Al doesn't use tags.
        using BE = BestBackend<Real, D, Collective::RECV>;
        Al::Recv<BE>(buf, count, from, comm.template GetComm<BE>(syncInfo));
    }
    else
    {
#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
        ENSURE_HOST_RECV_BUFFER(buf, count, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

        Synchronize(syncInfo);
        Status status;
        EL_CHECK_MPI_CALL(
            MPI_Recv(
                buf, count, TypeMap<Real>(), from, tag, comm.GetMPIComm(), &status));
    }
}

template <typename Real, Device D,
         typename/*=EnableIf<IsPacked<Real>>*/>
void TaggedRecv( Complex<Real>* buf, int count, int from, int tag, Comm const& comm,
                 SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    using CT = std::complex<Real>;

    if constexpr (IsAluminumSupported<CT, D, Collective::RECV>::value)
    {
        (void) tag; // Al doesn't use tags.
        using BE = BestBackend<CT, D, Collective::RECV>;
        Al::Recv<BE>(buf, count, from, comm.template GetComm<BE>(syncInfo));
    }
    else if constexpr (IsAluminumSupported<Real, D, Collective::RECV>::value)
    {
        (void) tag; // Al doesn't use tags.
        using BE = BestBackend<Real, D, Collective::RECV>;
        Al::Recv<BE>((Real*) buf, 2*count, from,
                     comm.template GetComm<BE>(syncInfo));
    }
    else
    {
#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
        ENSURE_HOST_RECV_BUFFER(buf, count, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

        Synchronize(syncInfo);

        Status status;
#ifdef EL_AVOID_COMPLEX_MPI
        EL_CHECK_MPI_CALL(
            MPI_Recv(
                buf, 2*count, TypeMap<Real>(), from, tag, comm.GetMPIComm(), &status));
#else
        EL_CHECK_MPI_CALL(
            MPI_Recv(
                buf, count, TypeMap<Complex<Real>>(),
                from, tag, comm.GetMPIComm(), &status));
#endif
    }
}

template <typename T, Device D,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void TaggedRecv( T* buf, int count, int from, int tag, Comm const& comm,
                 SyncInfo<D> const& syncInfo)
    EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_RECV_BUFFER(buf, count, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    std::vector<byte> packedBuf;
    ReserveSerialized( count, buf, packedBuf );
    Status status;
    EL_CHECK_MPI_CALL(
        MPI_Recv(
            packedBuf.data(), count, TypeMap<T>(), from, tag,
            comm.GetMPIComm(), &status));
    Deserialize( count, packedBuf, buf );
}

template <typename T, Device D>
void Recv(T* buf, int count, int from, Comm const& comm, SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{ TaggedRecv( buf, count, from, ANY_TAG, comm, syncInfo ); }

template <typename T>
T TaggedRecv( int from, int tag, Comm const& comm )
EL_NO_RELEASE_EXCEPT
{
    T b;
    TaggedRecv(&b, 1, from, tag, comm, SyncInfo<Device::CPU>{});
    return b;
}

template <typename T>
T Recv(int from, Comm const& comm)
EL_NO_RELEASE_EXCEPT
{ return TaggedRecv<T>( from, ANY_TAG, comm ); }

template <typename Real,
         typename/*=EnableIf<IsPacked<Real>>*/>
void TaggedIRecv
( Real* buf, int count, int from, int tag, Comm const& comm, Request<Real>& request )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_CHECK_MPI_CALL
    ( MPI_Irecv
      ( buf, count, TypeMap<Real>(), from, tag, comm.GetMPIComm(), &request.backend ) );
}

template <typename Real,
         typename/*=EnableIf<IsPacked<Real>>*/>
void TaggedIRecv
( Complex<Real>* buf, int count, int from, int tag, Comm const& comm,
  Request<Complex<Real>>& request )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifdef EL_AVOID_COMPLEX_MPI
    EL_CHECK_MPI_CALL
    ( MPI_Irecv
      ( buf, 2*count, TypeMap<Real>(), from, tag, comm.GetMPIComm(),
        &request.backend ) );
#else
    EL_CHECK_MPI_CALL
    ( MPI_Irecv
      ( buf, count, TypeMap<Complex<Real>>(), from, tag, comm.GetMPIComm(),
        &request.backend ) );
#endif
}

template <typename T,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void TaggedIRecv
( T* buf, int count, int from, int tag, Comm const& comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    request.receivingPacked = true;
    request.recvCount = count;
    request.unpackedRecvBuf = buf;
    ReserveSerialized( count, buf, request.buffer );
    EL_CHECK_MPI_CALL
    ( MPI_Irecv
      ( request.buffer.data(), count, TypeMap<T>(), from, tag, comm.GetMPIComm(),
        &request.backend ) );
}

template <typename T>
void IRecv( T* buf, int count, int from, Comm const& comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT
{ TaggedIRecv( buf, count, from, ANY_TAG, comm, request ); }

template <typename T>
T TaggedIRecv( int from, int tag, Comm const& comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT
{ T b; TaggedIRecv( &b, 1, from, tag, comm, request ); return b; }

template <typename T>
T IRecv( int from, Comm const& comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT
{ return TaggedIRecv<T>( from, ANY_TAG, comm, request ); }

template <typename Real, Device D,
         typename/*=EnableIf<IsPacked<Real>>*/>
void TaggedSendRecv(
    const Real* sbuf, int sc, int to,   int stag,
    Real* rbuf, int rc, int from, int rtag, Comm const& comm,
    SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_SEND_BUFFER(sbuf, sc, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, rc, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    Status status;
    EL_CHECK_MPI_CALL(
        MPI_Sendrecv(
            sbuf, sc, TypeMap<Real>(), to,   stag,
            rbuf, rc, TypeMap<Real>(), from, rtag,
            comm.GetMPIComm(), &status));
}

template <typename Real, Device D,
         typename/*=EnableIf<IsPacked<Real>>*/>
void TaggedSendRecv(
    const Complex<Real>* sbuf, int sc, int to, int stag,
    Complex<Real>* rbuf, int rc, int from, int rtag, Comm const& comm,
    SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_SEND_BUFFER(sbuf, sc, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, rc, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    Status status;
#ifdef EL_AVOID_COMPLEX_MPI
    EL_CHECK_MPI_CALL(
        MPI_Sendrecv(
            sbuf, 2*sc, TypeMap<Real>(), to,   stag,
            rbuf, 2*rc, TypeMap<Real>(), from, rtag,
            comm.GetMPIComm(), &status));
#else
    EL_CHECK_MPI_CALL(
        MPI_Sendrecv(
          sbuf, sc, TypeMap<Complex<Real>>(), to,   stag,
          rbuf, rc, TypeMap<Complex<Real>>(), from, rtag,
          comm.GetMPIComm(), &status));
#endif
}

template <typename T, Device D,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void TaggedSendRecv(
    const T* sbuf, int sc, int to,   int stag,
    T* rbuf, int rc, int from, int rtag, Comm const& comm,
    SyncInfo<D> const& syncInfo)
    EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_SEND_BUFFER(sbuf, sc, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, rc, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    Status status;
    std::vector<byte> packedSend, packedRecv;
    Serialize( sc, sbuf, packedSend );
    ReserveSerialized( rc, rbuf, packedRecv );
    EL_CHECK_MPI_CALL
    ( MPI_Sendrecv
      ( packedSend.data(), sc, TypeMap<T>(), to,   stag,
        packedRecv.data(), rc, TypeMap<T>(), from, rtag,
        comm.GetMPIComm(), &status ) );
    Deserialize( rc, packedRecv, rbuf );
}

template <typename T, Device D>
void SendRecv(
    const T* sbuf, int sc, int to,
    T* rbuf, int rc, int from, Comm const& comm,
    SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{ TaggedSendRecv( sbuf, sc, to, 0, rbuf, rc, from, ANY_TAG, comm, syncInfo ); }

template <typename T, Device D>
T TaggedSendRecv( T sb, int to, int stag, int from, int rtag, Comm const& comm,
                  SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{
    T rb;
    TaggedSendRecv( &sb, 1, to, stag, &rb, 1, from, rtag, comm, syncInfo );
    return rb;
}

template <typename Real, Device D,
         typename/*=EnableIf<IsPacked<Real>>*/>
void TaggedSendRecv(
    Real* buf, int count, int to, int stag, int from, int rtag, Comm const& comm,
    SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_INPLACE_BUFFER(buf, count, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    Status status;
    EL_CHECK_MPI_CALL(
        MPI_Sendrecv_replace(
            buf, count, TypeMap<Real>(), to, stag, from, rtag, comm.GetMPIComm(),
            &status));
}

template <typename Real, Device D,
         typename/*=EnableIf<IsPacked<Real>>*/>
void TaggedSendRecv(
    Complex<Real>* buf, int count, int to, int stag, int from, int rtag,
    Comm const& comm, SyncInfo<D> const& syncInfo )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_INPLACE_BUFFER(buf, count, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    Status status;
#ifdef EL_AVOID_COMPLEX_MPI
    EL_CHECK_MPI_CALL(
        MPI_Sendrecv_replace(
            buf, 2*count, TypeMap<Real>(), to, stag, from, rtag, comm.GetMPIComm(),
            &status));
#else
    EL_CHECK_MPI_CALL(
        MPI_Sendrecv_replace(
            buf, count, TypeMap<Complex<Real>>(),
            to, stag, from, rtag, comm.GetMPIComm(), &status));
#endif
}

template <typename T, Device D,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void TaggedSendRecv(
    T* buf, int count, int to, int stag, int from, int rtag, Comm const& comm,
    SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_INPLACE_BUFFER(buf, count, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    std::vector<byte> packedBuf;
    ReserveSerialized( count, buf, packedBuf );
    Serialize( count, buf, packedBuf );
    Status status;
    EL_CHECK_MPI_CALL(
        MPI_Sendrecv_replace(
            packedBuf.data(), count, TypeMap<T>(), to, stag, from, rtag,
            comm.GetMPIComm(), &status));
    Deserialize( count, packedBuf, buf );
}

template <typename T, Device D>
void SendRecv( T* buf, int count, int to, int from, Comm const& comm,
               SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{ TaggedSendRecv(buf, count, to, 0, from, ANY_TAG, comm, syncInfo); }

template <typename Real,
         typename/*=EnableIf<IsPacked<Real>>*/>
void IBroadcast
( Real* buf, int count, int root, Comm const& comm, Request<Real>& request )
{
    EL_DEBUG_CSE;
    EL_CHECK_MPI_CALL
    ( MPI_Ibcast
      ( buf, count, TypeMap<Real>(), root, comm.GetMPIComm(), &request.backend ) );
}

template <typename Real,
         typename/*=EnableIf<IsPacked<Real>>*/>
void IBroadcast
( Complex<Real>* buf, int count, int root, Comm const& comm,
  Request<Complex<Real>>& request )
{
    EL_DEBUG_CSE;
#ifdef EL_AVOID_COMPLEX_MPI
    EL_CHECK_MPI_CALL
    ( MPI_Ibcast
      ( buf, 2*count, TypeMap<Real>(), root, comm.GetMPIComm(), &request.backend ) );
#else
    EL_CHECK_MPI_CALL
    ( MPI_Ibcast
      ( buf, count, TypeMap<Complex<Real>>(), root, comm.GetMPIComm(),
        &request.backend ) );
#endif
}

template <typename T,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void IBroadcast
( T* buf, int count, int root, Comm const& comm, Request<T>& request )
{
    EL_DEBUG_CSE;
    request.receivingPacked = true;
    request.recvCount = count;
    request.unpackedRecvBuf = buf;
    ReserveSerialized( count, buf, request.buffer );
    EL_CHECK_MPI_CALL
    ( MPI_Ibcast
      ( request.buffer.data(), count, TypeMap<T>(), root, comm.GetMPIComm(),
        &request.backend ) );
}

template <typename T>
void IBroadcast( T& b, int root, Comm const& comm, Request<T>& request )
{ IBroadcast( &b, 1, root, comm, request ); }

template <typename Real,
         typename/*=EnableIf<IsPacked<Real>>*/>
void IGather
( const Real* sbuf, int sc,
        Real* rbuf, int rc,
  int root, Comm const& comm,
  Request<Real>& request )
{
    EL_DEBUG_CSE;
    EL_CHECK_MPI_CALL
    ( MPI_Igather
      ( const_cast<Real*>(sbuf), sc, TypeMap<Real>(),
        rbuf,                    rc, TypeMap<Real>(), root, comm.GetMPIComm(),
        &request.backend ) );
}

template <typename Real,
         typename/*=EnableIf<IsPacked<Real>>*/>
void IGather
( const Complex<Real>* sbuf, int sc,
        Complex<Real>* rbuf, int rc,
  int root, Comm const& comm,
  Request<Complex<Real>>& request )
{
    EL_DEBUG_CSE;
#ifdef EL_AVOID_COMPLEX_MPI
    EL_CHECK_MPI_CALL
    ( MPI_Igather
      ( const_cast<Complex<Real>*>(sbuf), 2*sc, TypeMap<Real>(),
        rbuf,                             2*rc, TypeMap<Real>(),
        root, comm.GetMPIComm(), &request.backend ) );
#else
    EL_CHECK_MPI_CALL
    ( MPI_Igather
      ( const_cast<Complex<Real>*>(sbuf), sc, TypeMap<Complex<Real>>(),
        rbuf,                             rc, TypeMap<Complex<Real>>(),
        root, comm.GetMPIComm(), &request.backend ) );
#endif
}

template <typename T,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void IGather
( const T* sbuf, int sc,
        T* rbuf, int rc,
  int root, Comm const& comm,
  Request<T>& request )
{
    EL_DEBUG_CSE;
    if( mpi::Rank(comm) == root )
    {
        const int commSize = mpi::Size(comm);
        request.receivingPacked = true;
        request.recvCount = rc*commSize;
        request.unpackedRecvBuf = rbuf;
        ReserveSerialized( rc*commSize, rbuf, request.buffer );
    }
    EL_CHECK_MPI_CALL
    ( MPI_Igather
      ( request.buffer.data(), sc, TypeMap<T>(),
        rbuf,                  rc, TypeMap<T>(), root, comm.GetMPIComm(),
        &request.backend ) );
}

template <typename Real, Device D,
          typename/*=EnableIf<IsPacked<Real>>*/>
void Gather(
    const Real* sbuf, int sc,
    Real* rbuf, const int* rcs, const int* rds,
    int root, Comm const& comm, SyncInfo<D> const& syncInfo )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto const commRank = Rank(comm);
    auto const commSize = Size(comm);
    // FIXME (trb): This is technically just an upper bound on the size
    auto const recvSize =
        (commRank == root ?
         static_cast<size_t>(rds[commSize-1]+rcs[commSize-1]) : 0UL);
    ENSURE_HOST_SEND_BUFFER(sbuf, sc, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, recvSize, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    EL_CHECK_MPI_CALL(
        MPI_Gatherv(
            sbuf, sc, TypeMap<Real>(),
            rbuf, rcs, rds, TypeMap<Real>(),
            root, comm.GetMPIComm()));
}

template <typename Real, Device D,
         typename/*=EnableIf<IsPacked<Real>>*/>
void Gather(
    const Complex<Real>* sbuf, int sc,
    Complex<Real>* rbuf, const int* rcs, const int* rds, int root,
    Comm const& comm, SyncInfo<D> const& syncInfo )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto const commRank = Rank(comm);
    auto const commSize = Size(comm);
    // FIXME (trb): This is technically just an upper bound on the size
    auto const recvSize =
        (commRank == root ?
         static_cast<size_t>(rds[commSize-1]+rcs[commSize-1]) : 0UL);
    ENSURE_HOST_SEND_BUFFER(sbuf, sc, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, recvSize, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

#ifdef EL_AVOID_COMPLEX_MPI
    const int commRank = Rank( comm );
    const int commSize = Size( comm );
    std::vector<int> rcsDouble, rdsDouble;
    if( commRank == root )
    {
        rcsDouble.resize( commSize );
        rdsDouble.resize( commSize );
        for( int i=0; i<commSize; ++i )
        {
            rcsDouble[i] = 2*rcs[i];
            rdsDouble[i] = 2*rds[i];
        }
    }
    EL_CHECK_MPI_CALL(
        MPI_Gatherv(
            sbuf, 2*sc, TypeMap<Real>(),
            rbuf, rcsDouble.data(), rdsDouble.data(), TypeMap<Real>(),
            root, comm.GetMPIComm() ) );
#else
    EL_CHECK_MPI_CALL(
        MPI_Gatherv(
            sbuf, sc, TypeMap<Complex<Real>>(),
            rbuf, rcs, rds, TypeMap<Complex<Real>>(),
            root, comm.GetMPIComm()));
#endif
}

template <typename T, Device D,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void Gather(
    const T* sbuf, int sc,
    T* rbuf, const int* rcs, const int* rds,
    int root, Comm const& comm, SyncInfo<D> const& syncInfo )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

    Synchronize(syncInfo);

    auto const commRank = Rank(comm);
    auto const commSize = Size(comm);
    // FIXME (trb): This is technically just an upper bound on the size
    auto const totalRecv =
        (commRank == root ?
         static_cast<size_t>(rds[commSize-1]+rcs[commSize-1]) : 0UL);

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_SEND_BUFFER(sbuf, sc, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, totalRecv, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    std::vector<byte> packedSend, packedRecv;
    Serialize( sc, sbuf, packedSend );

    if( commRank == root )
        ReserveSerialized( totalRecv, rbuf, packedRecv );
    EL_CHECK_MPI_CALL(
        MPI_Gatherv(
             packedSend.data(), sc, TypeMap<T>(),
             packedRecv.data(), rcs, rds, TypeMap<T>(),
             root, comm.GetMPIComm()));
    if( commRank == root )
        Deserialize( totalRecv, packedRecv, rbuf );
}

template <typename Real, Device D,
         typename/*=EnableIf<IsPacked<Real>>*/>
void AllGather(
    const Real* sbuf, int sc,
    Real* rbuf, const int* rcs, const int* rds, Comm const& comm,
    SyncInfo<D> const& syncInfo)
    EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto const commSize = Size(comm);
    // FIXME (trb): This is technically just an upper bound on the size
    auto const recvSize =
        static_cast<size_t>(rds[commSize-1]+rcs[commSize-1]);
    ENSURE_HOST_SEND_BUFFER(sbuf, sc, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, recvSize, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

#ifdef EL_USE_BYTE_ALLGATHERS
#ifndef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    const int commSize = Size( comm );
#endif
    std::vector<int> byteRcs( commSize ), byteRds( commSize );
    for( int i=0; i<commSize; ++i )
    {
        byteRcs[i] = sizeof(Real)*rcs[i];
        byteRds[i] = sizeof(Real)*rds[i];
    }
    EL_CHECK_MPI_CALL(
        MPI_Allgatherv(
            reinterpret_cast<UCP>(const_cast<Real*>(sbuf)),
            sizeof(Real)*sc, MPI_UNSIGNED_CHAR,
            reinterpret_cast<UCP>(rbuf),
            byteRcs.data(), byteRds.data(), MPI_UNSIGNED_CHAR,
            comm.GetMPIComm() ) );
#else
    EL_CHECK_MPI_CALL(
        MPI_Allgatherv(
            sbuf, sc, TypeMap<Real>(),
            rbuf, rcs, rds, TypeMap<Real>(), comm.GetMPIComm()));
#endif
}

template <typename Real, Device D,
         typename/*=EnableIf<IsPacked<Real>>*/>
void AllGather(
    const Complex<Real>* sbuf, int sc,
    Complex<Real>* rbuf, const int* rcs, const int* rds, Comm const& comm,
    SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto const commSize = Size(comm);
    // FIXME (trb): This is technically just an upper bound on the size
    auto const recvSize =
        static_cast<size_t>(rds[commSize-1]+rcs[commSize-1]);
    ENSURE_HOST_SEND_BUFFER(sbuf, sc, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, recvSize, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

#ifdef EL_USE_BYTE_ALLGATHERS
#ifndef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    const int commSize = Size( comm );
#endif

    std::vector<int> byteRcs( commSize ), byteRds( commSize );
    for( int i=0; i<commSize; ++i )
    {
        byteRcs[i] = 2*sizeof(Real)*rcs[i];
        byteRds[i] = 2*sizeof(Real)*rds[i];
    }
    EL_CHECK_MPI_CALL(
        MPI_Allgatherv(
            reinterpret_cast<UCP>(const_cast<Complex<Real>*>(sbuf)),
            2*sizeof(Real)*sc, MPI_UNSIGNED_CHAR,
            reinterpret_cast<UCP>(rbuf),
            byteRcs.data(), byteRds.data(), MPI_UNSIGNED_CHAR,
            comm.GetMPIComm()));
#else
#ifdef EL_AVOID_COMPLEX_MPI
    const int commSize = Size( comm );
    std::vector<int> realRcs( commSize ), realRds( commSize );
    for( int i=0; i<commSize; ++i )
    {
        realRcs[i] = 2*rcs[i];
        realRds[i] = 2*rds[i];
    }
    EL_CHECK_MPI_CALL(
        MPI_Allgatherv(
            sbuf, 2*sc, TypeMap<Real>(),
            rbuf, realRcs.data(), realRds.data(),
            TypeMap<Real>(), comm.GetMPIComm()));
#else
    EL_CHECK_MPI_CALL(
        MPI_Allgatherv(
            sbuf, sc, TypeMap<Complex<Real>>(),
            rbuf, rcs, rds, TypeMap<Complex<Real>>(), comm.GetMPIComm()));
#endif
#endif
}

template <typename T, Device D,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void AllGather(
    const T* sbuf, int sc,
    T* rbuf, const int* rcs, const int* rds, Comm const& comm,
    SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

    const int commSize = mpi::Size(comm);
    const int totalRecv = rcs[commSize-1]+rds[commSize-1];

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_SEND_BUFFER(sbuf, sc, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, totalRecv, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    std::vector<byte> packedSend, packedRecv;
    Serialize( sc, sbuf, packedSend );

    ReserveSerialized( totalRecv, rbuf, packedRecv );
    EL_CHECK_MPI_CALL(
        MPI_Allgatherv(
            packedSend.data(), sc, TypeMap<T>(),
        packedRecv.data(), rcs, rds, TypeMap<T>(), comm.GetMPIComm()));
    Deserialize( totalRecv, packedRecv, rbuf );
}


template <typename Real, Device D,
         typename/*=EnableIf<IsPacked<Real>>*/>
void Scatter(
    Real* buf, int sc, int rc, int root, Comm const& comm,
    SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

    auto const commRank = Rank( comm );

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto const commSize = Size(comm);
    auto const totalSend =
        (commRank == root ? static_cast<size_t>(sc*commSize) : 0UL);
    auto const totalRecv =
        (commRank == root ? 0UL : static_cast<size_t>(rc));
    auto const bufSize = std::max(totalSend, totalRecv);
    ENSURE_HOST_BUFFER_PREPOST_XFER(
        buf, bufSize, 0, totalSend, 0, totalRecv, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    if( commRank == root )
    {
        EL_CHECK_MPI_CALL(
            MPI_Scatter(
                buf,          sc, TypeMap<Real>(),
                MPI_IN_PLACE, rc, TypeMap<Real>(), root, comm.GetMPIComm()));
    }
    else
    {
        EL_CHECK_MPI_CALL(
            MPI_Scatter(
                0,   sc, TypeMap<Real>(),
                buf, rc, TypeMap<Real>(), root, comm.GetMPIComm()));
    }
}

template <typename Real, Device D,
         typename/*=EnableIf<IsPacked<Real>>*/>
void Scatter(
    Complex<Real>* buf, int sc, int rc, int root, Comm const& comm,
    SyncInfo<D> const& syncInfo )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

    auto const commRank = Rank( comm );

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto const commSize = Size(comm);
    auto const totalSend =
        (commRank == root ? static_cast<size_t>(sc*commSize) : 0UL);
    auto const totalRecv =
        (commRank == root ? 0UL : static_cast<size_t>(rc));
    auto const bufSize = std::max(totalSend, totalRecv);
    ENSURE_HOST_BUFFER_PREPOST_XFER(
        buf, bufSize, 0, totalSend, 0, totalRecv, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    if( commRank == root )
    {
#ifdef EL_AVOID_COMPLEX_MPI
        EL_CHECK_MPI_CALL(
            MPI_Scatter(
                buf,          2*sc, TypeMap<Real>(),
                MPI_IN_PLACE, 2*rc, TypeMap<Real>(), root, comm.GetMPIComm() ) );
#else
        EL_CHECK_MPI_CALL(
            MPI_Scatter(
                buf,          sc, TypeMap<Complex<Real>>(),
                MPI_IN_PLACE, rc, TypeMap<Complex<Real>>(), root, comm.GetMPIComm() ) );
#endif
    }
    else
    {
#ifdef EL_AVOID_COMPLEX_MPI
        EL_CHECK_MPI_CALL(
            MPI_Scatter(
                0,   2*sc, TypeMap<Real>(),
                buf, 2*rc, TypeMap<Real>(), root, comm.GetMPIComm() ) );
#else
        EL_CHECK_MPI_CALL(
            MPI_Scatter(
                0,   sc, TypeMap<Complex<Real>>(),
                buf, rc, TypeMap<Complex<Real>>(), root, comm.GetMPIComm() ) );
#endif
    }
}

template <typename T, Device D,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void Scatter(T* buf, int sc, int rc, int root, Comm const& comm,
             SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    auto const commSize = mpi::Size(comm);
    auto const commRank = Rank( comm );
    auto const totalSend =
        (commRank == root ? static_cast<size_t>(sc*commSize) : 0UL);

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto const totalRecv =
        (commRank == root ? 0UL : static_cast<size_t>(rc));
    auto const bufSize = std::max(totalSend, totalRecv);
    ENSURE_HOST_BUFFER_PREPOST_XFER(
        buf, bufSize, 0, totalSend, 0, totalRecv, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    // TODO(poulson): Use in-place option?

    std::vector<byte> packedSend, packedRecv;
    if( commRank == root )
        Serialize( totalSend, buf, packedSend );

    ReserveSerialized( rc, buf, packedRecv );
    EL_CHECK_MPI_CALL(
        MPI_Scatter(
            packedSend.data(), sc, TypeMap<T>(),
            packedRecv.data(), rc, TypeMap<T>(), root, comm.GetMPIComm() ) );
    Deserialize( rc, packedRecv, buf );
}


template <typename Real, Device D,
         typename/*=EnableIf<IsPacked<Real>>*/>
void AllToAll(
    const Real* sbuf, const int* scs, const int* sds,
    Real* rbuf, const int* rcs, const int* rds, Comm const& comm, SyncInfo<D> const& syncInfo )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto const commSize = Size(comm);
    auto const totalSend =
        static_cast<size_t>(sds[commSize-1] + scs[commSize-1]);
    auto const totalRecv =
        static_cast<size_t>(rds[commSize-1] + rcs[commSize-1]);
    ENSURE_HOST_SEND_BUFFER(sbuf, totalSend, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, totalRecv, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    EL_CHECK_MPI_CALL(
        MPI_Alltoallv(
            sbuf, scs, sds, TypeMap<Real>(),
            rbuf, rcs, rds, TypeMap<Real>(),
            comm.GetMPIComm()));
}

template <typename Real, Device D,
          typename/*=EnableIf<IsPacked<Real>>*/>
void AllToAll(
    const Complex<Real>* sbuf, const int* scs, const int* sds,
    Complex<Real>* rbuf, const int* rcs, const int* rds, Comm const& comm,
    SyncInfo<D> const& syncInfo )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto const commSize = Size(comm);
    auto const totalSend =
        static_cast<size_t>(sds[commSize-1] + scs[commSize-1]);
    auto const totalRecv =
        static_cast<size_t>(rds[commSize-1] + rcs[commSize-1]);
    ENSURE_HOST_SEND_BUFFER(sbuf, totalSend, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, totalRecv, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

#ifdef EL_AVOID_COMPLEX_MPI
    int p;
    MPI_Comm_size( comm.GetMPIComm(), &p );
    std::vector<int> scsDoubled(p), sdsDoubled(p),
                rcsDoubled(p), rdsDoubled(p);
    for( int i=0; i<p; ++i )
    {
        scsDoubled[i] = 2*scs[i];
        sdsDoubled[i] = 2*sds[i];
        rcsDoubled[i] = 2*rcs[i];
        rdsDoubled[i] = 2*rds[i];
    }
    EL_CHECK_MPI_CALL(
    MPI_Alltoallv(
        const_cast<Complex<Real>*>(sbuf),
        scsDoubled.data(), sdsDoubled.data(), TypeMap<Real>(),
        rbuf, rcsDoubled.data(), rdsDoubled.data(), TypeMap<Real>(), comm.GetMPIComm() ) );
#else
    EL_CHECK_MPI_CALL(
    MPI_Alltoallv(
        sbuf, scs, sds, TypeMap<Complex<Real>>(),
        rbuf, rcs, rds, TypeMap<Complex<Real>>(),
        comm.GetMPIComm()));
#endif
}

template <typename T, Device D,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void AllToAll(
    const T* sbuf, const int* scs, const int* sds,
    T* rbuf, const int* rcs, const int* rds, Comm const& comm,
    SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

    auto const commSize = Size(comm);
    auto const totalSend =
        static_cast<size_t>(sds[commSize-1] + scs[commSize-1]);
    auto const totalRecv =
        static_cast<size_t>(rds[commSize-1] + rcs[commSize-1]);
#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_SEND_BUFFER(sbuf, totalSend, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, totalRecv, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    std::vector<byte> packedSend, packedRecv;
    Serialize( totalSend, sbuf, packedSend );
    ReserveSerialized( totalRecv, rbuf, packedRecv );
    EL_CHECK_MPI_CALL(
    MPI_Alltoallv(
        packedSend.data(),
        scs, sds, TypeMap<T>(),
        packedRecv.data(),
        rcs, rds, TypeMap<T>(),
        comm.GetMPIComm() ) );
    Deserialize( totalRecv, packedRecv, rbuf );
}

template <typename T>
std::vector<T> AllToAll(
    const std::vector<T>& sendBuf,
    const std::vector<int>& sendCounts,
    const std::vector<int>& sendOffs,
    Comm const& comm )
EL_NO_RELEASE_EXCEPT
{
    // LogicError("AllToAll: Is this used? Tell Tom if so.");
    /* Tom: this is used when the Dot test is run with 2 processes.  Debian OS with mpich. --BTNG */

    SyncInfo<Device::CPU> syncInfo;
    const int commSize = Size( comm );
    std::vector<int> recvCounts(commSize);
    AllToAll( sendCounts.data(), 1, recvCounts.data(), 1, comm, syncInfo );
    std::vector<int> recvOffs;
    const int totalRecv = El::Scan( recvCounts, recvOffs );
    std::vector<T> recvBuf(totalRecv);
    AllToAll(
        sendBuf.data(), sendCounts.data(), sendOffs.data(),
        recvBuf.data(), recvCounts.data(), recvOffs.data(), comm,
        syncInfo);
    return recvBuf;
}


template <typename Real, Device D,
         typename/*=EnableIf<IsPacked<Real>>*/>
void ReduceScatter(
    const Real* sbuf, Real* rbuf, const int* rcs, Op op, Comm const& comm,
    SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto const commRank = mpi::Rank(comm);
    auto const commSize = mpi::Size(comm);
    auto const totalSend = std::accumulate(rcs,rcs+commSize,0UL);
    auto const totalRecv = static_cast<size_t>(rcs[commRank]);
    ENSURE_HOST_SEND_BUFFER(sbuf, totalSend, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, totalRecv, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    MPI_Op opC = NativeOp<Real>( op );
    EL_CHECK_MPI_CALL(
        MPI_Reduce_scatter(
            sbuf, rbuf, rcs, TypeMap<Real>(), opC, comm.GetMPIComm()));
}

template <typename Real, Device D,
         typename/*=EnableIf<IsPacked<Real>>*/>
void ReduceScatter(
    const Complex<Real>* sbuf, Complex<Real>* rbuf, const int* rcs,
    Op op, Comm const& comm, SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto const commRank = mpi::Rank(comm);
    auto const commSize = mpi::Size(comm);
    auto const totalSend = std::accumulate(rcs,rcs+commSize,0UL);
    auto const totalRecv = static_cast<size_t>(rcs[commRank]);
    ENSURE_HOST_SEND_BUFFER(sbuf, totalSend, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, totalRecv, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

#ifdef EL_AVOID_COMPLEX_MPI
    if( op == SUM )
    {
        MPI_Op opC = NativeOp<Real>( op );
        int p;
        MPI_Comm_size( comm.GetMPIComm(), &p );
        std::vector<int> rcsDoubled(p);
        for( int i=0; i<p; ++i )
            rcsDoubled[i] = 2*rcs[i];
        EL_CHECK_MPI_CALL(
            MPI_Reduce_scatter(
                sbuf, rbuf, rcsDoubled.data(), TypeMap<Real>(), opC,
                comm.GetMPIComm()));
    }
    else
    {
        MPI_Op opC = NativeOp<Complex<Real>>( op );
        EL_CHECK_MPI_CALL(
        MPI_Reduce_scatter(
            sbuf, rbuf, rcs, TypeMap<Complex<Real>>(), opC, comm.GetMPIComm()));
    }
#else
    MPI_Op opC = NativeOp<Complex<Real>>( op );
    EL_CHECK_MPI_CALL(
        MPI_Reduce_scatter(
            sbuf, rbuf, rcs, TypeMap<Complex<Real>>(), opC, comm.GetMPIComm()));
#endif
}

template <typename T, Device D,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void ReduceScatter(
    const T* sbuf, T* rbuf, const int* rcs, Op op, Comm const& comm,
    SyncInfo<D> const& syncInfo )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

    Synchronize(syncInfo);

    auto const commRank = mpi::Rank(comm);
    auto const commSize = mpi::Size(comm);
    auto const totalSend = std::accumulate(rcs,rcs+commSize,0UL);
    auto const totalRecv = static_cast<size_t>(rcs[commRank]);
#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_SEND_BUFFER(sbuf, totalSend, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, totalRecv, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    MPI_Op opC = NativeOp<T>( op );
    std::vector<byte> packedSend, packedRecv;
    Serialize( totalSend, sbuf, packedSend );
    ReserveSerialized( totalRecv, rbuf, packedRecv );
    EL_CHECK_MPI_CALL(
        MPI_Reduce_scatter(
            packedSend.data(), packedRecv.data(), rcs,
            TypeMap<T>(), opC, comm.GetMPIComm() ) );
    Deserialize( totalRecv, packedRecv, rbuf );
}

template <typename T, Device D>
void ReduceScatter(
    const T* sbuf, T* rbuf, const int* rcs, Comm const& comm,
    SyncInfo<D> const& syncInfo )
EL_NO_RELEASE_EXCEPT
{ ReduceScatter( sbuf, rbuf, rcs, SUM, comm, syncInfo ); }

void VerifySendsAndRecvs(
    const std::vector<int>& sendCounts,
    const std::vector<int>& recvCounts, Comm const& comm )
{
    EL_DEBUG_CSE;

    LogicError("VerifySendsAndRecvs: Is this used? Tell Tom if so.");
    const int commSize = Size( comm );
    std::vector<int> actualRecvCounts(commSize);
    AllToAll(
        sendCounts.data(),       1,
        actualRecvCounts.data(), 1, comm, SyncInfo<Device::CPU>{} );
    for( int q=0; q<commSize; ++q )
        if( actualRecvCounts[q] != recvCounts[q] )
            LogicError(
                "Expected recv count of ",recvCounts[q],
                " but recv'd ",actualRecvCounts[q]," from process ",q);
}

template <typename Real, Device D,
         typename/*=EnableIf<IsPacked<Real>>*/>
void Scan(const Real* sbuf, Real* rbuf, int count, Op op, Comm const& comm,
          SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

    if (count == 0)
        return;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_SEND_BUFFER(sbuf, count, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, count, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    MPI_Op opC = NativeOp<Real>( op );
    EL_CHECK_MPI_CALL(
        MPI_Scan(
            sbuf, rbuf, count, TypeMap<Real>(), opC, comm.GetMPIComm()));
}

template <typename Real, Device D,
         typename/*=EnableIf<IsPacked<Real>>*/>
void Scan(
    const Complex<Real>* sbuf, Complex<Real>* rbuf, int count, Op op,
    Comm const& comm, SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

    if (count == 0)
        return;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_SEND_BUFFER(sbuf, count, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, count, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

#ifdef EL_AVOID_COMPLEX_MPI
    if( op == SUM )
    {
        MPI_Op opC = NativeOp<Real>( op );
        EL_CHECK_MPI_CALL(
            MPI_Scan(
                sbuf, rbuf, 2*count, TypeMap<Real>(), opC, comm.GetMPIComm()));
    }
    else
    {
        MPI_Op opC = NativeOp<Complex<Real>>( op );
        EL_CHECK_MPI_CALL(
            MPI_Scan(
                sbuf, rbuf, count, TypeMap<Complex<Real>>(), opC,
                comm.GetMPIComm()));
    }
#else
    MPI_Op opC = NativeOp<Complex<Real>>( op );
    EL_CHECK_MPI_CALL(
        MPI_Scan(
            sbuf, rbuf, count, TypeMap<Complex<Real>>(), opC, comm.GetMPIComm()));
#endif
}

template <typename T, Device D,
          typename/*=DisableIf<IsPacked<T>>*/,
          typename/*=void*/>
void Scan(
    const T* sbuf, T* rbuf, int count, Op op, Comm const& comm,
    SyncInfo<D> const& syncInfo )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

    if (count == 0)
        return;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_SEND_BUFFER(sbuf, count, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, count, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    MPI_Op opC = NativeOp<T>( op );
    std::vector<byte> packedSend, packedRecv;
    Serialize( count, sbuf, packedSend );
    ReserveSerialized( count, rbuf, packedRecv );
    EL_CHECK_MPI_CALL(
        MPI_Scan(
            packedSend.data(), packedRecv.data(), count, TypeMap<T>(),
            opC, comm.GetMPIComm()));
    Deserialize( count, packedRecv, rbuf );
}

template <typename T, Device D>
void Scan(
    const T* sbuf, T* rbuf, int count, Comm const& comm, SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{ Scan(sbuf, rbuf, count, SUM, comm, syncInfo); }

template <typename T>
T Scan( T sb, Op op, Comm const& comm )
EL_NO_RELEASE_EXCEPT
{
    T rb;
    Scan( &sb, &rb, 1, op, comm, SyncInfo<Device::CPU>{} );
    return rb;
}

template <typename T>
T Scan(T sb, Comm const& comm)
EL_NO_RELEASE_EXCEPT
{
    T rb;
    Scan( &sb, &rb, 1, SUM, comm, SyncInfo<Device::CPU>{} );
    return rb;
}

template <typename Real, Device D,
         typename/*=EnableIf<IsPacked<Real>>*/>
void Scan( Real* buf, int count, Op op, Comm const& comm,
           SyncInfo<D> const& syncInfo )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

    if (count == 0)
        return;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_INPLACE_BUFFER(buf, count, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    MPI_Op opC = NativeOp<Real>( op );
    EL_CHECK_MPI_CALL(
        MPI_Scan(
            MPI_IN_PLACE, buf, count, TypeMap<Real>(), opC, comm.GetMPIComm()));
}

template <typename Real, Device D,
         typename/*=EnableIf<IsPacked<Real>>*/>
void Scan( Complex<Real>* buf, int count, Op op, Comm const& comm,
           SyncInfo<D> const& syncInfo )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

    if( count == 0 )
        return;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_INPLACE_BUFFER(buf, count, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

#ifdef EL_AVOID_COMPLEX_MPI
    if( op == SUM )
    {
        MPI_Op opC = NativeOp<Real>( op );
        EL_CHECK_MPI_CALL(
            MPI_Scan(
                MPI_IN_PLACE, buf, 2*count, TypeMap<Real>(), opC,
                comm.GetMPIComm()));
    }
    else
    {
        MPI_Op opC = NativeOp<Complex<Real>>( op );
        EL_CHECK_MPI_CALL(
            MPI_Scan(
                MPI_IN_PLACE, buf, count, TypeMap<Complex<Real>>(), opC,
                comm.GetMPIComm()));
    }
#else
    MPI_Op opC = NativeOp<Complex<Real>>( op );
    EL_CHECK_MPI_CALL(
        MPI_Scan(
            MPI_IN_PLACE, buf, count, TypeMap<Complex<Real>>(), opC,
            comm.GetMPIComm()));
#endif
}

template <typename T, Device D,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void Scan( T* buf, int count, Op op, Comm const& comm, SyncInfo<D> const& syncInfo )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;

    if( count == 0 )
        return;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_INPLACE_BUFFER(buf, count, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    MPI_Op opC = NativeOp<T>( op );
    std::vector<byte> packedSend, packedRecv;
    Serialize( count, buf, packedSend );
    ReserveSerialized( count, buf, packedRecv );
    EL_CHECK_MPI_CALL(
        MPI_Scan(
            packedSend.data(), packedRecv.data(), count, TypeMap<T>(),
            opC, comm.GetMPIComm() ) );
    Deserialize( count, packedRecv, buf );
}

template <typename T, Device D>
void Scan( T* buf, int count, Comm const& comm, SyncInfo<D> const& syncInfo )
EL_NO_RELEASE_EXCEPT
{ Scan( buf, count, SUM, comm, syncInfo ); }

template <typename T>
void SparseAllToAll
( const std::vector<T>& sendBuffer,
  const std::vector<int>& sendCounts,
  const std::vector<int>& sendDispls,
        std::vector<T>& recvBuffer,
  const std::vector<int>& recvCounts,
  const std::vector<int>& recvDispls,
        Comm const& comm )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(VerifySendsAndRecvs( sendCounts, recvCounts, comm ))
#ifdef EL_USE_CUSTOM_ALLTOALLV
    const int commSize = Size( comm );
    int numSends=0,numRecvs=0;
    for( int q=0; q<commSize; ++q )
    {
        if( sendCounts[q] != 0 )
            ++numSends;
        if( recvCounts[q] != 0 )
            ++numRecvs;
    }
    std::vector<Status> statuses(numSends+numRecvs);
    std::vector<Request<T>> requests(numSends+numRecvs);
    int rCount=0;
    for( int q=0; q<commSize; ++q )
    {
        int count = recvCounts[q];
        int displ = recvDispls[q];
        if( count != 0 )
            IRecv( &recvBuffer[displ], count, q, comm, requests[rCount++] );
    }

    // Ensure that recvs are posted before the sends
    // (Invalid MPI_Irecv's have been observed otherwise)
    Barrier( comm );

    for( int q=0; q<commSize; ++q )
    {
        int count = sendCounts[q];
        int displ = sendDispls[q];
        if( count != 0 )
            IRSend( &sendBuffer[displ], count, q, comm, requests[rCount++] );
    }
    WaitAll( numSends+numRecvs, requests.data(), statuses.data() );
#else
    AllToAll(
        sendBuffer.data(), sendCounts.data(), sendDispls.data(),
        recvBuffer.data(), recvCounts.data(), recvDispls.data(), comm,
        SyncInfo<Device::CPU>{});
#endif
}

#define MPI_PROTO_DEVICELESS_COMMON(T)                                  \
    template bool Test(Request<T>& request) EL_NO_RELEASE_EXCEPT;       \
    template void Wait(Request<T>& request) EL_NO_RELEASE_EXCEPT;       \
    template void Wait(Request<T>& request, Status& status)             \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void WaitAll(int numRequests, Request<T>* requests)        \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void WaitAll(                                              \
        int numRequests, Request<T>* requests, Status* statuses)        \
        EL_NO_RELEASE_EXCEPT;                                           \
    template int GetCount<T>(Status& status) EL_NO_RELEASE_EXCEPT;      \
    template std::vector<T> AllToAll(                                        \
        const std::vector<T>& sendBuf,                                       \
        const std::vector<int>& sendCounts,                                  \
        const std::vector<int>& sendOffs,                                    \
        Comm const& comm)                                                      \
        EL_NO_RELEASE_EXCEPT;                                           \
    template T Scan(T sb, Op op, Comm const& comm)                             \
        EL_NO_RELEASE_EXCEPT;                                           \
    template T Scan(T sb, Comm const& comm)                                    \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void TaggedSend(T b, int to, int tag, Comm const& comm)           \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void Send(T b, int to, Comm const& comm)                          \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void ISend(                                                \
        const T* buf, int count, int to, Comm const& comm,                     \
        Request<T>& request)                                            \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void TaggedISend(                                          \
        T buf, int to, int tag, Comm const& comm, Request<T>& request)         \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void ISend(T buf, int to, Comm const& comm, Request<T>& request)  \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void ISSend(                                               \
        const T* buf, int count, int to, Comm const& comm,                     \
        Request<T>& request)                                            \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void TaggedISSend(                                         \
        T b, int to, int tag, Comm const& comm, Request<T>& request)           \
        EL_NO_RELEASE_EXCEPT;                                           \
    template T TaggedRecv<T>(                                           \
        int from, int tag, Comm const& comm)                                   \
        EL_NO_RELEASE_EXCEPT;                                           \
    template T Recv(int from, Comm const& comm)                                \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void IRecv(                                                \
        T* buf, int count, int from, Comm const& comm, Request<T>& request)    \
        EL_NO_RELEASE_EXCEPT;                                           \
    template T TaggedIRecv<T>(                                          \
        int from, int tag, Comm const& comm, Request<T>& request)              \
        EL_NO_RELEASE_EXCEPT;                                           \
    template T IRecv<T>(int from, Comm const& comm, Request<T>& request)       \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void IBroadcast(                                           \
        T& b, int root, Comm const& comm, Request<T>& request);

#define MPI_PROTO_DEVICELESS(T)                                         \
    template void TaggedISend(                                          \
        const T* buf, int count, int to, int tag, Comm const& comm,            \
        Request<T>& request)                                            \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void TaggedISSend(                                         \
        const T* buf, int count, int to, int tag, Comm const& comm,            \
        Request<T>& request)                                            \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void TaggedIRecv(                                          \
        T* buf, int count, int from, int tag, Comm const& comm,                \
        Request<T>& request)                                            \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void IBroadcast(                                           \
        T* buf, int count, int root, Comm const& comm, Request<T>& request);   \
    template void IGather(                                              \
        const T* sbuf, int sc,                                          \
        T* rbuf, int rc,                                                \
        int root, Comm const& comm, Request<T>& request);                      \
    MPI_PROTO_DEVICELESS_COMMON(T)

#define MPI_PROTO_DEVICELESS_COMPLEX(T)                                 \
    template void TaggedISend<T>(                                       \
        const Complex<T>* buf, int count, int to, int tag, Comm const& comm,   \
        Request<Complex<T>>& request)                                   \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void TaggedISSend<T>(                                      \
        const Complex<T>* buf, int count, int to, int tag, Comm const& comm,   \
        Request<Complex<T>>& request)                                   \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void TaggedIRecv<T>(                                       \
        Complex<T>* buf, int count, int from, int tag, Comm const& comm,       \
        Request<Complex<T>>& request)                                   \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void IBroadcast<T>(                                        \
        Complex<T>* buf, int count, int root, Comm const& comm,                \
        Request<Complex<T>>& request);                                  \
    template void IGather<T>(                                           \
        const Complex<T>* sbuf, int sc,                                 \
        Complex<T>* rbuf, int rc,                                       \
        int root, Comm const& comm, Request<Complex<T>>& request);             \
    MPI_PROTO_DEVICELESS_COMMON(Complex<T>)

#define MPI_PROTO_COMMON_DEV(T,D)               \
    template void Send(                                                 \
        const T* buf, int count, int to, Comm const& comm, SyncInfo<D> const&) \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void Recv(                                                 \
        T* buf, int count, int from, Comm const& comm, SyncInfo<D> const&)     \
        EL_NO_RELEASE_EXCEPT;                                           \
    template T TaggedSendRecv(                                          \
        T sb, int to, int stag, int from, int rtag, Comm const& comm,          \
        SyncInfo<D> const&)                                             \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void ReduceScatter(                                        \
        const T* sbuf, T* rbuf, const int* rcs, Comm const& comm,              \
        SyncInfo<D> const&)                                             \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void Scan(                                                 \
        const T* sbuf, T* rbuf, int count, Comm const& comm,                   \
        SyncInfo<D> const&)                                             \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void Scan(                                                 \
        T* buf, int count, Comm const& comm, SyncInfo<D> const&)               \
        EL_NO_RELEASE_EXCEPT;

#define MPI_PROTO_DEV(T,D)                                              \
    template void TaggedSend(                                           \
        const T* buf, int count, int to, int tag, Comm const& comm,            \
        SyncInfo<D> const&)                                             \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void TaggedRecv(                                           \
        T* buf, int count, int from, int tag, Comm const& comm,                \
        SyncInfo<D> const&)                                             \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void TaggedSendRecv(                                       \
        const T* sbuf, int sc, int to,   int stag,                      \
        T* rbuf, int rc, int from, int rtag, Comm const& comm,                 \
        SyncInfo<D> const&)                                             \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void TaggedSendRecv(                                       \
        T* buf, int count, int to, int stag, int from, int rtag,        \
        Comm const& comm,                                                      \
        SyncInfo<D> const&)                                             \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void Gather(                                               \
        const T* sbuf, int sc,                                          \
        T* rbuf, const int* rcs, const int* rds, int root, Comm const& comm,   \
        SyncInfo<D> const&)                                             \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void AllGather(                                            \
        const T* sbuf, int sc,                                          \
        T* rbuf, const int* rcs, const int* rds, Comm const& comm,             \
        SyncInfo<D> const&)                                             \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void Scatter(                                              \
        T* buf, int sc, int rc, int root, Comm const& comm,                    \
        SyncInfo<D> const&)                                             \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void AllToAll(                                             \
        const T* sbuf, const int* scs, const int* sds,                  \
        T* rbuf, const int* rcs, const int* rds, Comm const& comm,             \
        SyncInfo<D> const&)                                             \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void ReduceScatter(                                        \
        const T* sbuf, T* rbuf, const int* rcs, Op op, Comm const& comm,       \
        SyncInfo<D> const&)                                             \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void Scan(                                                 \
        const T* sbuf, T* rbuf, int count, Op op, Comm const& comm,            \
        SyncInfo<D> const&)                                             \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void Scan(                                                 \
        T* buf, int count, Op op, Comm const& comm, SyncInfo<D> const&)        \
        EL_NO_RELEASE_EXCEPT;                                           \
    MPI_PROTO_COMMON_DEV(T,D)

#define MPI_PROTO_COMPLEX_DEV(T,D)                                      \
    template void TaggedSend<T>(                                        \
        const Complex<T>* buf, int count, int to, int tag, Comm const& comm,   \
        SyncInfo<D> const&)                                             \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void TaggedRecv<T>(                                        \
        Complex<T>* buf, int count, int from, int tag, Comm const& comm,       \
        SyncInfo<D> const&)                                             \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void TaggedSendRecv<T>(                                    \
        const Complex<T>* sbuf, int sc, int to,   int stag,             \
        Complex<T>* rbuf, int rc, int from, int rtag, Comm const& comm,        \
        SyncInfo<D> const&)                                             \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void TaggedSendRecv<T>(                                    \
        Complex<T>* buf, int count, int to, int stag, int from, int rtag, \
        Comm const& comm, SyncInfo<D> const&)                                  \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void Gather<T>(                                            \
        const Complex<T>* sbuf, int sc,                                 \
        Complex<T>* rbuf, const int* rcs, const int* rds, int root,     \
        Comm const& comm, SyncInfo<D> const&)                                  \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void AllGather<T>(                                         \
        const Complex<T>* sbuf, int sc,                                 \
        Complex<T>* rbuf, const int* rcs, const int* rds, Comm const& comm,    \
        SyncInfo<D> const&)                                             \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void Scatter<T>(                                           \
        Complex<T>* buf, int sc, int rc, int root, Comm const& comm,           \
        SyncInfo<D> const&)                                             \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void AllToAll<T>(                                          \
        const Complex<T>* sbuf, const int* scs, const int* sds,         \
        Complex<T>* rbuf, const int* rcs, const int* rds, Comm const& comm,    \
        SyncInfo<D> const&)                                             \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void ReduceScatter<T>(                                     \
        const Complex<T>* sbuf, Complex<T>* rbuf, const int* rcs, Op op, \
        Comm const& comm, SyncInfo<D> const&)                                  \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void Scan<T>(                                              \
        const Complex<T>* sbuf, Complex<T>* rbuf, int count, Op op,     \
        Comm const& comm, SyncInfo<D> const&)                                  \
        EL_NO_RELEASE_EXCEPT;                                           \
    template void Scan<T>(                                              \
        Complex<T>* buf, int count, Op op, Comm const& comm,                   \
        SyncInfo<D> const&)                                             \
        EL_NO_RELEASE_EXCEPT;                                           \
    MPI_PROTO_COMMON_DEV(Complex<T>,D)

#ifdef HYDROGEN_HAVE_GPU
#define MPI_PROTO(T) \
    MPI_PROTO_DEVICELESS(T) \
    MPI_PROTO_DEV(T, Device::CPU) \
    MPI_PROTO_DEV(T, Device::GPU)
#define MPI_PROTO_COMPLEX(T) \
    MPI_PROTO_DEVICELESS_COMPLEX(T) \
    MPI_PROTO_COMPLEX_DEV(T, Device::CPU) \
    MPI_PROTO_COMPLEX_DEV(T, Device::GPU)

#ifdef HYDROGEN_GPU_USE_FP16
MPI_PROTO(gpu_half_type)
MPI_PROTO(Entry<gpu_half_type>)
#endif

#else
#define MPI_PROTO(T) \
    MPI_PROTO_DEVICELESS(T) \
    MPI_PROTO_DEV(T, Device::CPU)
#define MPI_PROTO_COMPLEX(T) \
    MPI_PROTO_DEVICELESS_COMPLEX(T)                    \
    MPI_PROTO_COMPLEX_DEV(T, Device::CPU)
#endif // HYDROGEN_HAVE_GPU

MPI_PROTO(byte)
MPI_PROTO(int)
MPI_PROTO(unsigned)
MPI_PROTO(long int)
MPI_PROTO(unsigned long)
MPI_PROTO(long long int)
MPI_PROTO(unsigned long long)
MPI_PROTO(ValueInt<Int>)
MPI_PROTO(Entry<Int>)
MPI_PROTO(float)
MPI_PROTO_COMPLEX(float)
MPI_PROTO(ValueInt<float>)
MPI_PROTO(ValueInt<Complex<float>>)
MPI_PROTO(Entry<float>)
MPI_PROTO(Entry<Complex<float>>)
MPI_PROTO(double)
MPI_PROTO_COMPLEX(double)
MPI_PROTO(ValueInt<double>)
MPI_PROTO(ValueInt<Complex<double>>)
MPI_PROTO(Entry<double>)
MPI_PROTO(Entry<Complex<double>>)
#ifdef HYDROGEN_HAVE_HALF
MPI_PROTO(cpu_half_type)
MPI_PROTO(ValueInt<cpu_half_type>)
MPI_PROTO(Entry<cpu_half_type>)
#endif
#ifdef HYDROGEN_HAVE_QD
MPI_PROTO(DoubleDouble)
MPI_PROTO(QuadDouble)
MPI_PROTO_COMPLEX(DoubleDouble)
MPI_PROTO_COMPLEX(QuadDouble)
MPI_PROTO(ValueInt<DoubleDouble>)
MPI_PROTO(ValueInt<QuadDouble>)
MPI_PROTO(ValueInt<Complex<DoubleDouble>>)
MPI_PROTO(ValueInt<Complex<QuadDouble>>)
MPI_PROTO(Entry<DoubleDouble>)
MPI_PROTO(Entry<QuadDouble>)
MPI_PROTO(Entry<Complex<DoubleDouble>>)
MPI_PROTO(Entry<Complex<QuadDouble>>)
#endif
#ifdef HYDROGEN_HAVE_QUADMATH
MPI_PROTO(Quad)
MPI_PROTO_COMPLEX(Complex<Quad>)
MPI_PROTO(ValueInt<Quad>)
MPI_PROTO(ValueInt<Complex<Quad>>)
MPI_PROTO(Entry<Quad>)
MPI_PROTO(Entry<Complex<Quad>>)
#endif
#ifdef HYDROGEN_HAVE_MPC
MPI_PROTO(BigInt)
MPI_PROTO(BigFloat)
MPI_PROTO_COMPLEX(Complex<BigFloat>)
MPI_PROTO(ValueInt<BigInt>)
MPI_PROTO(ValueInt<BigFloat>)
MPI_PROTO(ValueInt<Complex<BigFloat>>)
MPI_PROTO(Entry<BigInt>)
MPI_PROTO(Entry<BigFloat>)
MPI_PROTO(Entry<Complex<BigFloat>>)
#endif

#define PROTO(T)                                \
    template void SparseAllToAll(               \
        const std::vector<T>& sendBuffer,            \
        const std::vector<int>& sendCounts,          \
        const std::vector<int>& sendDispls,          \
        std::vector<T>& recvBuffer,                  \
        const std::vector<int>& recvCounts,          \
        const std::vector<int>& recvDispls,          \
        Comm const& comm)                              \
        EL_NO_RELEASE_EXCEPT;

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#define EL_ENABLE_HALF
#include <El/macros/Instantiate.h>

} // namespace mpi
} // namespace El
