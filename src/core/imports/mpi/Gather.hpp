// Gather

namespace El
{
namespace mpi
{

#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D,
          typename/*=EnableIf<IsAluminumSupported<T,D,COLL>>*/>
void Gather(
    const T* sbuf, int sc,
    T* rbuf, int rc, int root, Comm const& comm, SyncInfo<D> const& syncInfo )
{
    EL_DEBUG_CSE

    using Backend = BestBackend<T,D,Collective::GATHER>;
    Al::Gather<Backend>(
        sbuf, rbuf, sc, root, comm.template GetComm<Backend>(syncInfo));
}
#endif // HYDROGEN_HAVE_ALUMINUM

template <typename T, Device D, typename, typename, typename, typename, typename>
void Gather(
    const T* sbuf, int sc,
    T* rbuf, int rc, int root, Comm const& comm,
    SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto const rank = mpi::Rank(comm);
    auto const size_c = mpi::Size(comm);
    auto const recvCount
        = (rank == root ? static_cast<size_t>(rc*size_c) : 0UL);
    ENSURE_HOST_SEND_BUFFER(sbuf, sc, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, recvCount, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    EL_CHECK_MPI_CALL(
        MPI_Gather(
            sbuf, sc, TypeMap<T>(),
            rbuf, rc, TypeMap<T>(), root, comm.GetMPIComm()));
}

template <typename T, Device D, typename, typename, typename, typename>
void Gather(
    const Complex<T>* sbuf, int sc,
    Complex<T>* rbuf, int rc, int root, Comm const& comm,
    SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto const rank = mpi::Rank(comm);
    auto const size_c = mpi::Size(comm);
    auto const recvCount
        = (rank == root ? static_cast<size_t>(rc*size_c) : 0UL);
    ENSURE_HOST_SEND_BUFFER(sbuf, sc, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, recvCount, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

#ifdef EL_AVOID_COMPLEX_MPI
    EL_CHECK_MPI_CALL(
        MPI_Gather(
            sbuf, 2*sc, TypeMap<T>(),
            rbuf, 2*rc, TypeMap<T>(),
            root, comm.GetMPIComm()));
#else
    EL_CHECK_MPI_CALL(
        MPI_Gather(
            sbuf, sc, TypeMap<Complex<T>>(),
            rbuf, rc, TypeMap<Complex<T>>(),
            root, comm.GetMPIComm()));
#endif
}

template <typename T, Device D, typename, typename, typename>
void Gather(
    const T* sbuf, int sc,
    T* rbuf, int rc, int root, Comm const& comm, SyncInfo<D> const& syncInfo )
{
    EL_DEBUG_CSE

    const int commSize = mpi::Size(comm);
    const int commRank = mpi::Rank(comm);
    const int totalRecv = rc*commSize;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto const recvCount
        = (commRank == root ? static_cast<size_t>(rc*commSize) : 0UL);
    ENSURE_HOST_SEND_BUFFER(sbuf, sc, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, recvCount, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    std::vector<byte> packedSend, packedRecv;
    Serialize( sc, sbuf, packedSend );

    if( commRank == root )
        ReserveSerialized( totalRecv, rbuf, packedRecv );
    EL_CHECK_MPI_CALL(
        MPI_Gather(
            packedSend.data(), sc, TypeMap<T>(),
            packedRecv.data(), rc, TypeMap<T>(), root, comm.GetMPIComm() ) );
    if( commRank == root )
        Deserialize( totalRecv, packedRecv, rbuf );
}

template <typename T, Device D, typename, typename>
void Gather(const T*, int, T*, int, int, Comm const&, SyncInfo<D> const&)
{
    LogicError("Gather: Bad device/type combination.");
}

#define MPI_COLLECTIVE_PROTO_DEV(T,D) \
    template void Gather(const T* sbuf, int sc, T* rbuf, int rc, int root, \
                         Comm const& comm, SyncInfo<D> const&);

#define MPI_COLLECTIVE_COMPLEX_PROTO_DEV(T,D) \
    template void Gather(const Complex<T>* sbuf, int sc, Complex<T>* rbuf, \
                         int rc, int root, Comm const& comm, SyncInfo<D> const&);

#ifndef HYDROGEN_HAVE_GPU
#define MPI_COLLECTIVE_PROTO(T) \
    MPI_COLLECTIVE_PROTO_DEV(T,Device::CPU)
#define MPI_COLLECTIVE_COMPLEX_PROTO(T) \
    MPI_COLLECTIVE_COMPLEX_PROTO_DEV(T,Device::CPU)
#else
#define MPI_COLLECTIVE_PROTO(T) \
    MPI_COLLECTIVE_PROTO_DEV(T,Device::CPU) \
    MPI_COLLECTIVE_PROTO_DEV(T,Device::GPU)
#define MPI_COLLECTIVE_COMPLEX_PROTO(T) \
    MPI_COLLECTIVE_COMPLEX_PROTO_DEV(T,Device::CPU) \
    MPI_COLLECTIVE_COMPLEX_PROTO_DEV(T,Device::GPU)
#endif

MPI_COLLECTIVE_PROTO(byte)
MPI_COLLECTIVE_PROTO(int)
MPI_COLLECTIVE_PROTO(unsigned)
MPI_COLLECTIVE_PROTO(long int)
MPI_COLLECTIVE_PROTO(unsigned long)
MPI_COLLECTIVE_PROTO(float)
MPI_COLLECTIVE_PROTO(double)
MPI_COLLECTIVE_PROTO(long long int)
MPI_COLLECTIVE_PROTO(unsigned long long)
MPI_COLLECTIVE_PROTO(ValueInt<Int>)
MPI_COLLECTIVE_PROTO(Entry<Int>)
MPI_COLLECTIVE_COMPLEX_PROTO(float)
MPI_COLLECTIVE_PROTO(ValueInt<float>)
MPI_COLLECTIVE_PROTO(ValueInt<Complex<float>>)
MPI_COLLECTIVE_PROTO(Entry<float>)
MPI_COLLECTIVE_PROTO(Entry<Complex<float>>)
MPI_COLLECTIVE_COMPLEX_PROTO(double)
MPI_COLLECTIVE_PROTO(ValueInt<double>)
MPI_COLLECTIVE_PROTO(ValueInt<Complex<double>>)
MPI_COLLECTIVE_PROTO(Entry<double>)
MPI_COLLECTIVE_PROTO(Entry<Complex<double>>)
#ifdef HYDROGEN_HAVE_HALF
MPI_COLLECTIVE_PROTO(cpu_half_type)
MPI_COLLECTIVE_PROTO(Entry<cpu_half_type>)
#endif
#ifdef HYDROGEN_GPU_USE_FP16
MPI_COLLECTIVE_PROTO(gpu_half_type)
#endif // HYDROGEN_GPU_USE_FP16
#ifdef HYDROGEN_HAVE_QD
MPI_COLLECTIVE_PROTO(DoubleDouble)
MPI_COLLECTIVE_PROTO(QuadDouble)
MPI_COLLECTIVE_COMPLEX_PROTO(DoubleDouble)
MPI_COLLECTIVE_COMPLEX_PROTO(QuadDouble)
MPI_COLLECTIVE_PROTO(ValueInt<DoubleDouble>)
MPI_COLLECTIVE_PROTO(ValueInt<QuadDouble>)
MPI_COLLECTIVE_PROTO(ValueInt<Complex<DoubleDouble>>)
MPI_COLLECTIVE_PROTO(ValueInt<Complex<QuadDouble>>)
MPI_COLLECTIVE_PROTO(Entry<DoubleDouble>)
MPI_COLLECTIVE_PROTO(Entry<QuadDouble>)
MPI_COLLECTIVE_PROTO(Entry<Complex<DoubleDouble>>)
MPI_COLLECTIVE_PROTO(Entry<Complex<QuadDouble>>)
#endif
#ifdef HYDROGEN_HAVE_QUADMATH
MPI_COLLECTIVE_PROTO(Quad)
MPI_COLLECTIVE_COMPLEX_PROTO(Quad)
MPI_COLLECTIVE_PROTO(ValueInt<Quad>)
MPI_COLLECTIVE_PROTO(ValueInt<Complex<Quad>>)
MPI_COLLECTIVE_PROTO(Entry<Quad>)
MPI_COLLECTIVE_PROTO(Entry<Complex<Quad>>)
#endif
#ifdef HYDROGEN_HAVE_MPC
MPI_COLLECTIVE_PROTO(BigInt)
MPI_COLLECTIVE_PROTO(BigFloat)
MPI_COLLECTIVE_COMPLEX_PROTO(BigFloat)
MPI_COLLECTIVE_PROTO(ValueInt<BigInt>)
MPI_COLLECTIVE_PROTO(ValueInt<BigFloat>)
MPI_COLLECTIVE_PROTO(ValueInt<Complex<BigFloat>>)
MPI_COLLECTIVE_PROTO(Entry<BigInt>)
MPI_COLLECTIVE_PROTO(Entry<BigFloat>)
MPI_COLLECTIVE_PROTO(Entry<Complex<BigFloat>>)
#endif

#undef MPI_COLLECTIVE_PROTO
#undef MPI_COLLECTIVE_PROTO_DEV
#undef MPI_COLLECTIVE_COMPLEX_PROTO
#undef MPI_COLLECTIVE_COMPLEX_PROTO_DEV
} // namespace mpi
} // namespace El
