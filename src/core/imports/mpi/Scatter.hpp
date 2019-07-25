// Scatter

namespace El
{
namespace mpi
{

#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D,
          typename/*=EnableIf<IsAluminumSupported<T,D,COLL>>*/>
void Scatter(
    const T* sbuf, int sc,
    T* rbuf, int rc, int root, Comm const& comm, SyncInfo<D> const& syncInfo )
{
    EL_DEBUG_CSE

    using Backend = BestBackend<T,D,Collective::GATHER>;
    Al::Scatter<Backend>(sbuf, rbuf, sc, root,
                         comm.template GetComm<Backend>(syncInfo));
}
#endif // HYDROGEN_HAVE_ALUMINUM

template <typename T, Device D,
          typename, typename, typename, typename, typename>
void Scatter(
    const T* sbuf, int sc,
    T* rbuf, int rc, int root, Comm const& comm,
    SyncInfo<D> const& syncInfo)
{
#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto const commSize = Size(comm);
    auto const commRank = Rank(comm);
    auto const totalSend =
        (commRank == root ? static_cast<size_t>(sc*commSize) : 0UL);
    ENSURE_HOST_SEND_BUFFER(sbuf, totalSend, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, rc, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);
    EL_CHECK_MPI_CALL(
        MPI_Scatter(
            sbuf, sc, TypeMap<T>(),
            rbuf, rc, TypeMap<T>(), root, comm.GetMPIComm()));
}

template <typename T, Device D, typename, typename, typename, typename>
void Scatter(
    Complex<T> const* sbuf, int sc,
    Complex<T>* rbuf, int rc, int root, Comm const& comm,
    SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto const commSize = Size(comm);
    auto const commRank = Rank(comm);
    auto const totalSend =
        (commRank == root ? static_cast<size_t>(sc*commSize) : 0UL);
    ENSURE_HOST_SEND_BUFFER(sbuf, totalSend, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, rc, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

#ifdef EL_AVOID_COMPLEX_MPI
    EL_CHECK_MPI_CALL(
        MPI_Scatter(
            sbuf, 2*sc, TypeMap<T>(),
            rbuf, 2*rc, TypeMap<T>(), root, comm.GetMPIComm()));
#else
    EL_CHECK_MPI_CALL(
        MPI_Scatter(
            sbuf, sc, TypeMap<Complex<T>>(),
            rbuf,rc, TypeMap<Complex<T>>(),
            root, comm.GetMPIComm()));
#endif
}

template <typename T, Device D, typename, typename, typename>
void Scatter(
    const T* sbuf, int sc,
    T* rbuf, int rc, int root, Comm const& comm, SyncInfo<D> const& syncInfo )
{
    EL_DEBUG_CSE

    auto const commSize = Size(comm);
    auto const commRank = Rank(comm);
    auto const totalSend =
        (commRank == root ? static_cast<size_t>(sc*commSize) : 0UL);

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_SEND_BUFFER(sbuf, totalSend, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, rc, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    std::vector<byte> packedSend, packedRecv;
    if( commRank == root )
        Serialize( totalSend, sbuf, packedSend );

    ReserveSerialized( rc, rbuf, packedRecv );
    EL_CHECK_MPI_CALL(
        MPI_Scatter(
            packedSend.data(), sc, TypeMap<T>(),
            packedRecv.data(), rc, TypeMap<T>(), root, comm.GetMPIComm()));
    Deserialize( rc, packedRecv, rbuf );
}

template <typename T, Device D, typename, typename>
void Scatter(const T*, int, T*, int, int, Comm const&, SyncInfo<D> const&)
{
    LogicError("Scatter: Bad device/type combination.");
}

#define MPI_COLLECTIVE_PROTO_DEV(T,D) \
    template void Scatter(const T* sbuf, int sc, T* rbuf, int rc, int root, \
                          Comm const& comm, SyncInfo<D> const&)

#ifdef HYDROGEN_HAVE_GPU
#define MPI_COLLECTIVE_PROTO(T) \
    MPI_COLLECTIVE_PROTO_DEV(T,Device::CPU);    \
    MPI_COLLECTIVE_PROTO_DEV(T,Device::GPU)
#else
#define MPI_COLLECTIVE_PROTO(T) \
    MPI_COLLECTIVE_PROTO_DEV(T,Device::CPU)
#endif

MPI_COLLECTIVE_PROTO(byte);
MPI_COLLECTIVE_PROTO(int);
MPI_COLLECTIVE_PROTO(unsigned);
MPI_COLLECTIVE_PROTO(long int);
MPI_COLLECTIVE_PROTO(unsigned long);
MPI_COLLECTIVE_PROTO(float);
MPI_COLLECTIVE_PROTO(double);
MPI_COLLECTIVE_PROTO(long long int);
MPI_COLLECTIVE_PROTO(unsigned long long);
MPI_COLLECTIVE_PROTO(ValueInt<Int>);
MPI_COLLECTIVE_PROTO(Entry<Int>);
MPI_COLLECTIVE_PROTO(Complex<float>);
MPI_COLLECTIVE_PROTO(ValueInt<float>);
MPI_COLLECTIVE_PROTO(ValueInt<Complex<float>>);
MPI_COLLECTIVE_PROTO(Entry<float>);
MPI_COLLECTIVE_PROTO(Entry<Complex<float>>);
MPI_COLLECTIVE_PROTO(Complex<double>);
MPI_COLLECTIVE_PROTO(ValueInt<double>);
MPI_COLLECTIVE_PROTO(ValueInt<Complex<double>>);
MPI_COLLECTIVE_PROTO(Entry<double>);
MPI_COLLECTIVE_PROTO(Entry<Complex<double>>);
#ifdef HYDROGEN_HAVE_HALF
MPI_COLLECTIVE_PROTO(cpu_half_type);
MPI_COLLECTIVE_PROTO(Entry<cpu_half_type>);
#endif
#ifdef HYDROGEN_GPU_USE_FP16
MPI_COLLECTIVE_PROTO(gpu_half_type);
#endif
#ifdef HYDROGEN_HAVE_QD
MPI_COLLECTIVE_PROTO(DoubleDouble);
MPI_COLLECTIVE_PROTO(QuadDouble);
MPI_COLLECTIVE_PROTO(Complex<DoubleDouble>);
MPI_COLLECTIVE_PROTO(Complex<QuadDouble>);
MPI_COLLECTIVE_PROTO(ValueInt<DoubleDouble>);
MPI_COLLECTIVE_PROTO(ValueInt<QuadDouble>);
MPI_COLLECTIVE_PROTO(ValueInt<Complex<DoubleDouble>>);
MPI_COLLECTIVE_PROTO(ValueInt<Complex<QuadDouble>>);
MPI_COLLECTIVE_PROTO(Entry<DoubleDouble>);
MPI_COLLECTIVE_PROTO(Entry<QuadDouble>);
MPI_COLLECTIVE_PROTO(Entry<Complex<DoubleDouble>>);
MPI_COLLECTIVE_PROTO(Entry<Complex<QuadDouble>>);
#endif
#ifdef HYDROGEN_HAVE_QUADMATH
MPI_COLLECTIVE_PROTO(Quad);
MPI_COLLECTIVE_PROTO(Complex<Quad>);
MPI_COLLECTIVE_PROTO(ValueInt<Quad>);
MPI_COLLECTIVE_PROTO(ValueInt<Complex<Quad>>);
MPI_COLLECTIVE_PROTO(Entry<Quad>);
MPI_COLLECTIVE_PROTO(Entry<Complex<Quad>>);
#endif
#ifdef HYDROGEN_HAVE_MPC
MPI_COLLECTIVE_PROTO(BigInt);
MPI_COLLECTIVE_PROTO(BigFloat);
MPI_COLLECTIVE_PROTO(Complex<BigFloat>);
MPI_COLLECTIVE_PROTO(ValueInt<BigInt>);
MPI_COLLECTIVE_PROTO(ValueInt<BigFloat>);
MPI_COLLECTIVE_PROTO(ValueInt<Complex<BigFloat>>);
MPI_COLLECTIVE_PROTO(Entry<BigInt>);
MPI_COLLECTIVE_PROTO(Entry<BigFloat>);
MPI_COLLECTIVE_PROTO(Entry<Complex<BigFloat>>);
#endif

#undef MPI_COLLECTIVE_PROTO
#undef MPI_COLLECTIVE_PROTO_DEV
#undef MPI_COLLECTIVE_COMPLEX_PROTO
#undef MPI_COLLECTIVE_COMPLEX_PROTO_DEV
} // namespace mpi
} // namespace El
