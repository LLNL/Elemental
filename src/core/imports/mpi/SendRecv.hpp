// Collective

namespace El
{
namespace mpi
{

#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D,
          typename/*=EnableIf<IsAluminumSupported<T,D,COLL>>*/>
void SendRecv(const T* sbuf, int sc, int to,
              T* rbuf, int rc, int from, Comm const& comm,
              SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE;

    using Backend = BestBackend<T,D,Collective::SENDRECV>;
    Al::SendRecv<Backend>(
        sbuf, sc, to, rbuf, rc, from,
        comm.template GetComm<Backend>(syncInfo));
}

namespace {

template <typename B> struct BackendTag {};

template <typename T>
void do_copy_n(T const* src, int count, T* dst, SyncInfo<Device::CPU> const&)
{
    std::copy_n(src, count, dst);
}

#ifdef HYDROGEN_HAVE_GPU
template <typename T>
void do_copy_n(T const* src, int count, T* dst, SyncInfo<Device::GPU> const& si)
{
    gpu::Copy1DIntraDevice(src, dst, count, si);
}
#endif // HYDROGEN_HAVE_GPU

template <typename T, Device D, typename Backend>
void SafeInPlaceSendRecv(T* buf, int count, int to, int from, Comm const& comm,
                         SyncInfo<D> const& syncInfo, BackendTag<Backend>)
{
    hydrogen::simple_buffer<T, D> tmp_recv_buf(count, syncInfo);
    Al::SendRecv<Backend>(
        buf, count, to, tmp_recv_buf.data(), count, from,
        comm.template GetComm<Backend>(syncInfo));
    do_copy_n(tmp_recv_buf.data(), count, buf, syncInfo);
}

#ifdef HYDROGEN_HAVE_AL_HOST_XFER
template <typename T, Device D>
void SafeInPlaceSendRecv(T* buf, int count, int to, int from, Comm const& comm,
                         SyncInfo<D> const& syncInfo,
                         BackendTag<Al::HostTransferBackend>)
{
    // This is ok due to implementation details of how the
    // HostTransfer backend works.
    using Backend = Al::HostTransferBackend;
    Al::SendRecv<Backend>(
        buf, count, to, buf, count, from,
        comm.template GetComm<Backend>(syncInfo));
}
#endif
} // namespace

template <typename T, Device D,
          typename/*=EnableIf<IsAluminumSupported<T,D,COLL>>*/>
void SendRecv(T* buf, int count, int to, int from, Comm const& comm,
              SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE;
    using Backend = BestBackend<T,D,Collective::SENDRECV>;

#ifdef HYDROGEN_AL_SUPPORTS_INPLACE_SENDRECV
    Al::SendRecv<Backend>(
        buf, count, to, from,
        comm.template GetComm<Backend>(syncInfo));
#else
    SafeInPlaceSendRecv(buf, count, to, from, comm, syncInfo,
                        BackendTag<Backend>{});
#endif
}
#endif // HYDROGEN_HAVE_ALUMINUM

template <typename T, Device D, typename, typename, typename>
void SendRecv(const T* sbuf, int sc, int to,
              T* rbuf, int rc, int from, Comm const& comm,
              SyncInfo<D> const& syncInfo)
{
    TaggedSendRecv(sbuf, sc, to, 0, rbuf, rc, from, ANY_TAG,
                   std::move(comm), syncInfo);
}

template <typename T, Device D, typename, typename>
void SendRecv(const T* sbuf, int sc, int to,
              T* rbuf, int rc, int from, Comm const& comm,
              SyncInfo<D> const& syncInfo)
{
    LogicError("SendRecv: Bad type/device combination.");
}


template <typename T, Device D, typename, typename, typename>
void SendRecv( T* buf, int count, int to, int from, Comm const& comm,
               SyncInfo<D> const& syncInfo)
{ TaggedSendRecv(buf, count, to, 0, from, ANY_TAG, comm, syncInfo); }

template <typename T, Device D, typename, typename>
void SendRecv( T* buf, int count, int to, int from, Comm const& comm,
               SyncInfo<D> const& syncInfo)
{
    LogicError("SendRecv: Bad type/device combination.");
}

template <typename T, Device D>
T SendRecv( T sb, int to, int from, Comm const& comm, SyncInfo<D> const& syncInfo )
{ return TaggedSendRecv( sb, to, 0, from, ANY_TAG, comm, syncInfo ); }

#define MPI_COLLECTIVE_PROTO_DEV(T,D) \
    template void SendRecv(                                             \
        const T* sbuf, int sc, int to,                                  \
        T* rbuf, int rc, int from, Comm const& comm, SyncInfo<D> const&);      \
    template T SendRecv(                                                \
        T sb, int to, int from, Comm const& comm, SyncInfo<D> const&);         \
    template void SendRecv(                                             \
        T* buf, int count, int to, int from, Comm const& comm,                 \
        SyncInfo<D> const&)

#ifndef HYDROGEN_HAVE_GPU
#define MPI_COLLECTIVE_PROTO(T)             \
    MPI_COLLECTIVE_PROTO_DEV(T,Device::CPU)
#else
#define MPI_COLLECTIVE_PROTO(T)             \
    MPI_COLLECTIVE_PROTO_DEV(T,Device::CPU);    \
    MPI_COLLECTIVE_PROTO_DEV(T,Device::GPU)
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
#endif // HYDROGEN_GPU_USE_FP16
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
} // namespace mpi
} // namespace El
