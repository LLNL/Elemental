// ReduceScatter

namespace El
{
namespace mpi
{

// IsValidAluminumDeviceType should mean both that the device/type
// combo is valid and that the backend supports this collective.
//
#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D,
          typename/*=EnableIf<IsAluminumSupported<T,D,
                                  Collectives::REDUCESCATTER>>*/>
void ReduceScatter(T const* sbuf, T* rbuf, int count, Op op, Comm const& comm,
                   SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    using Backend = BestBackend<T,D,Collective::REDUCESCATTER>;
    Al::Reduce_scatter<Backend>(
        sbuf, rbuf, count, MPI_Op2ReductionOperator(AlNativeOp<T>(op)),
        comm.template GetComm<Backend>(syncInfo));
}
#endif // HYDROGEN_HAVE_ALUMINUM

template <typename T, Device D, typename, typename, typename, typename, typename>
void ReduceScatter(T const* sbuf, T* rbuf, int count, Op op, Comm const& comm,
                    SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto size_c = Size(comm);

    ENSURE_HOST_SEND_BUFFER(sbuf, count*size_c, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, count, syncInfo);
#endif

    Synchronize(syncInfo);

    EL_CHECK_MPI_CALL(
        MPI_Reduce_scatter_block(
            sbuf, rbuf, count, TypeMap<T>(), NativeOp<T>(op), comm.GetMPIComm()));
}

template <typename T, Device D, typename, typename, typename, typename>
void ReduceScatter(Complex<T> const* sbuf, Complex<T>* rbuf,
                   int count, Op op, Comm const& comm, SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto size_c = Size(comm);

    ENSURE_HOST_SEND_BUFFER(sbuf, count*size_c, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, count, syncInfo);
#endif

    Synchronize(syncInfo);

# ifdef EL_AVOID_COMPLEX_MPI
    EL_CHECK_MPI_CALL(
        MPI_Reduce_scatter_block(
            sbuf, rbuf, 2*count, TypeMap<T>(), NativeOp<T>(op), comm.GetMPIComm()));
# else
    EL_CHECK_MPI_CALL(
        MPI_Reduce_scatter_block(
            sbuf, rbuf, count,
            TypeMap<Complex<T>>(), NativeOp<Complex<T>>(op),
            comm.GetMPIComm()));
# endif
}

template <typename T, Device D, typename, typename, typename>
void ReduceScatter(T const* sbuf, T* rbuf, int count, Op op, Comm const& comm,
                   SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    const int commSize = mpi::Size(comm);
    const int totalSend = count*commSize;
    const int totalRecv = count;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_SEND_BUFFER(sbuf, totalSend, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, totalRecv, syncInfo);
#endif

    Synchronize(syncInfo);

    std::vector<byte> packedSend, packedRecv;
    Serialize(totalSend, sbuf, packedSend);

    ReserveSerialized(totalRecv, rbuf, packedRecv);
    EL_CHECK_MPI_CALL(
        MPI_Reduce_scatter_block(
            packedSend.data(), packedRecv.data(), count, TypeMap<T>(),
            NativeOp<T>(op), comm.GetMPIComm()));

    Deserialize(totalRecv, packedRecv, rbuf);
}

template <typename T, Device D, typename, typename>
void ReduceScatter(T const*, T*, int, Op, Comm const&, SyncInfo<D> const&)
{
    LogicError("ReduceScatter: Bad device/type combination.");
}

//
// The "IN_PLACE" Reduce_scatter
//

#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D,
          typename/*=EnableIf<IsAluminumSupported<T,D,
                                  Collectives::REDUCESCATTER>>*/>
void ReduceScatter(T* buf, int count, Op op, Comm const& comm,
                   SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    using Backend = BestBackend<T,D,Collective::REDUCESCATTER>;
    Al::Reduce_scatter<Backend>(
        buf, count, MPI_Op2ReductionOperator(AlNativeOp<T>(op)),
        comm.template GetComm<Backend>(syncInfo));
}
#endif // HYDROGEN_HAVE_ALUMINUM

template <typename T, Device D, typename, typename, typename, typename, typename>
void ReduceScatter(T* buf, int count, Op op, Comm const& comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0 || Size(comm) == 1)
        return;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto const size_c = Size(comm);
    ENSURE_HOST_BUFFER_PREPOST_XFER(
        buf, count*size_c, 0, count*size_c, 0, count, syncInfo);
#endif

    Synchronize(syncInfo);

    EL_CHECK_MPI_CALL(
        MPI_Reduce_scatter_block(
            MPI_IN_PLACE, buf, count,
            TypeMap<T>(), NativeOp<T>(op), comm.GetMPIComm()));
}

template <typename T, Device D, typename, typename, typename, typename>
void ReduceScatter(Complex<T>* buf, int count, Op op, Comm const& comm,
                   SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0 || Size(comm) == 1)
        return;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto const size_c = Size(comm);
    ENSURE_HOST_BUFFER_PREPOST_XFER(
        buf, count*size_c, 0, count*size_c, 0, count, syncInfo);
#endif

    Synchronize(syncInfo);

#ifdef EL_AVOID_COMPLEX_MPI
    EL_CHECK_MPI_CALL(
        MPI_Reduce_scatter_block(
            MPI_IN_PLACE, buf, 2*count,
            TypeMap<T>(), NativeOp<T>(op), comm.GetMPIComm()));
#else
    EL_CHECK_MPI_CALL(
        MPI_Reduce_scatter_block(
            MPI_IN_PLACE, buf, count,
            TypeMap<Complex<T>>(),
            NativeOp<Complex<T>>(op), comm.GetMPIComm()));
#endif
}

template <typename T, Device D, typename, typename, typename>
void ReduceScatter(T* buf, int count, Op op, Comm const& comm,
                   SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;
    const int commSize = mpi::Size(comm);
    const int totalSend = count*commSize;
    const int totalRecv = count;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_BUFFER_PREPOST_XFER(
        buf, totalSend, 0, totalSend, 0, totalRecv, syncInfo);
#endif

    Synchronize(syncInfo);

    std::vector<byte> packedSend, packedRecv;
    Serialize(totalSend, buf, packedSend);

    ReserveSerialized(totalRecv, buf, packedRecv);
    EL_CHECK_MPI_CALL(
        MPI_Reduce_scatter_block(
            packedSend.data(), packedRecv.data(), count, TypeMap<T>(),
            NativeOp<T>(op), comm.GetMPIComm()));

    Deserialize(totalRecv, packedRecv, buf);
}

template <typename T, Device D, typename, typename>
void ReduceScatter(T*, int, Op, Comm const&, SyncInfo<D> const&)
{
    LogicError("ReduceScatter: Bad device/type combination.");
}

//
// The other stuff
//

template <typename T, Device D>
void ReduceScatter(T const* sbuf, T* rbuf, int rc, Comm const& comm,
                   SyncInfo<D> const& syncInfo)
{
    ReduceScatter(sbuf, rbuf, rc, SUM, comm, syncInfo);
}

template <typename T, Device D>
T ReduceScatter(T sb, Op op, Comm const& comm, SyncInfo<D> const& syncInfo)
{
    T rb; ReduceScatter(&sb, &rb, 1, op, comm, syncInfo); return rb;
}

template <typename T, Device D>
T ReduceScatter(T sb, Comm const& comm, SyncInfo<D> const& syncInfo)
{
    return ReduceScatter(sb, SUM, comm, syncInfo);
}

template <typename T, Device D>
void ReduceScatter(T* buf, int rc, Comm const& comm, SyncInfo<D> const& syncInfo)
{
    ReduceScatter(buf, rc, SUM, comm, syncInfo);
}

#define MPI_REDUCESCATTER_PROTO_DEV(T,D)                                \
    template void ReduceScatter(T const*, T*, int rc, Op op, Comm const&,      \
                                SyncInfo<D> const&);                    \
    template void ReduceScatter(T*, int, Op, Comm const&, SyncInfo<D> const&); \
    template void ReduceScatter(T const*, T*, int rc, Comm const&,             \
                                SyncInfo<D> const&);                    \
    template T ReduceScatter(T, Op, Comm const&, SyncInfo<D> const&);          \
    template T ReduceScatter(T, Comm const&, SyncInfo<D> const&);              \
    template void ReduceScatter(T*, int, Comm const&, SyncInfo<D> const&)

#ifndef HYDROGEN_HAVE_GPU
#define MPI_REDUCESCATTER_PROTO(T)             \
    MPI_REDUCESCATTER_PROTO_DEV(T,Device::CPU)
#else
#define MPI_REDUCESCATTER_PROTO(T)             \
    MPI_REDUCESCATTER_PROTO_DEV(T,Device::CPU); \
    MPI_REDUCESCATTER_PROTO_DEV(T,Device::GPU)
#endif // HYDROGEN_HAVE_GPU

MPI_REDUCESCATTER_PROTO(byte);
MPI_REDUCESCATTER_PROTO(int);
MPI_REDUCESCATTER_PROTO(unsigned);
MPI_REDUCESCATTER_PROTO(long int);
MPI_REDUCESCATTER_PROTO(unsigned long);
MPI_REDUCESCATTER_PROTO(float);
MPI_REDUCESCATTER_PROTO(double);
MPI_REDUCESCATTER_PROTO(long long int);
MPI_REDUCESCATTER_PROTO(unsigned long long);
MPI_REDUCESCATTER_PROTO(ValueInt<Int>);
MPI_REDUCESCATTER_PROTO(Entry<Int>);
MPI_REDUCESCATTER_PROTO(Complex<float>);
MPI_REDUCESCATTER_PROTO(ValueInt<float>);
MPI_REDUCESCATTER_PROTO(ValueInt<Complex<float>>);
MPI_REDUCESCATTER_PROTO(Entry<float>);
MPI_REDUCESCATTER_PROTO(Entry<Complex<float>>);
MPI_REDUCESCATTER_PROTO(Complex<double>);
MPI_REDUCESCATTER_PROTO(ValueInt<double>);
MPI_REDUCESCATTER_PROTO(ValueInt<Complex<double>>);
MPI_REDUCESCATTER_PROTO(Entry<double>);
MPI_REDUCESCATTER_PROTO(Entry<Complex<double>>);
#ifdef HYDROGEN_HAVE_HALF
MPI_REDUCESCATTER_PROTO(cpu_half_type);
MPI_REDUCESCATTER_PROTO(Entry<cpu_half_type>);
#endif
#ifdef HYDROGEN_GPU_USE_FP16
MPI_REDUCESCATTER_PROTO(gpu_half_type);
#endif
#ifdef HYDROGEN_HAVE_QD
MPI_REDUCESCATTER_PROTO(DoubleDouble);
MPI_REDUCESCATTER_PROTO(QuadDouble);
MPI_REDUCESCATTER_PROTO(Complex<DoubleDouble>);
MPI_REDUCESCATTER_PROTO(Complex<QuadDouble>);
MPI_REDUCESCATTER_PROTO(ValueInt<DoubleDouble>);
MPI_REDUCESCATTER_PROTO(ValueInt<QuadDouble>);
MPI_REDUCESCATTER_PROTO(ValueInt<Complex<DoubleDouble>>);
MPI_REDUCESCATTER_PROTO(ValueInt<Complex<QuadDouble>>);
MPI_REDUCESCATTER_PROTO(Entry<DoubleDouble>);
MPI_REDUCESCATTER_PROTO(Entry<QuadDouble>);
MPI_REDUCESCATTER_PROTO(Entry<Complex<DoubleDouble>>);
MPI_REDUCESCATTER_PROTO(Entry<Complex<QuadDouble>>);
#endif
#ifdef HYDROGEN_HAVE_QUADMATH
MPI_REDUCESCATTER_PROTO(Quad);
MPI_REDUCESCATTER_PROTO(Complex<Quad>);
MPI_REDUCESCATTER_PROTO(ValueInt<Quad>);
MPI_REDUCESCATTER_PROTO(ValueInt<Complex<Quad>>);
MPI_REDUCESCATTER_PROTO(Entry<Quad>);
MPI_REDUCESCATTER_PROTO(Entry<Complex<Quad>>);
#endif
#ifdef HYDROGEN_HAVE_MPC
MPI_REDUCESCATTER_PROTO(BigInt);
MPI_REDUCESCATTER_PROTO(BigFloat);
MPI_REDUCESCATTER_PROTO(Complex<BigFloat>);
MPI_REDUCESCATTER_PROTO(ValueInt<BigInt>);
MPI_REDUCESCATTER_PROTO(ValueInt<BigFloat>);
MPI_REDUCESCATTER_PROTO(ValueInt<Complex<BigFloat>>);
MPI_REDUCESCATTER_PROTO(Entry<BigInt>);
MPI_REDUCESCATTER_PROTO(Entry<BigFloat>);
MPI_REDUCESCATTER_PROTO(Entry<Complex<BigFloat>>);
#endif

}// namespace mpi
}// namespace El
