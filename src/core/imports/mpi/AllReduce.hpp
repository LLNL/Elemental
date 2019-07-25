namespace El
{
namespace mpi
{

//
// The "normal" allreduce (not "IN_PLACE").
//

#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D, typename>
void AllReduce(T const* sbuf, T* rbuf, int count, Op op, Comm const& comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    using Backend = BestBackend<T,D,Collective::ALLREDUCE>;

    if (count == 0)
        return;

    Al::Allreduce<Backend>(
        sbuf, rbuf, count, MPI_Op2ReductionOperator(AlNativeOp<T>(op)),
        comm.template GetComm<Backend>(syncInfo));
}
#endif // HYDROGEN_HAVE_ALUMINUM

template <typename T, Device D, typename, typename, typename, typename, typename>
void AllReduce(T const* sbuf, T* rbuf, int count, Op op, Comm const& comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_SEND_BUFFER(sbuf, count, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, count, syncInfo);
#endif

    Synchronize(syncInfo);

    EL_CHECK_MPI_CALL(
        MPI_Allreduce(
            const_cast<T*>(sbuf), rbuf,
            count, TypeMap<T>(), NativeOp<T>(op), comm.GetMPIComm()));
}

template <typename T, Device D, typename, typename, typename, typename>
void AllReduce(Complex<T> const* sbuf, Complex<T>* rbuf, int count, Op op,
               Comm const& comm, SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE

    if (count == 0)
        return;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_SEND_BUFFER(sbuf, count, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, count, syncInfo);
#endif

    Synchronize(syncInfo);

#ifdef EL_AVOID_COMPLEX_MPI
    if (op == SUM)
    {
        EL_CHECK_MPI_CALL(
            MPI_Allreduce(
                const_cast<Complex<T>*>(sbuf), rbuf, 2*count,
                TypeMap<T>(), NativeOp<T>(op), comm.GetMPIComm()));
    }
    else
    {
        EL_CHECK_MPI_CALL(
            MPI_Allreduce(
                const_cast<Complex<T>*>(sbuf), rbuf, count,
                TypeMap<Complex<T>>(), NativeOp<Complex<T>>(op), comm.GetMPIComm()));
    }
#else
    EL_CHECK_MPI_CALL(
        MPI_Allreduce(
            const_cast<Complex<T>*>(sbuf), rbuf, count,
            TypeMap<Complex<T>>(), NativeOp<Complex<T>>(op), comm.GetMPIComm()));
#endif
}

template <typename T, Device D, typename, typename, typename>
void AllReduce(T const* sbuf, T* rbuf, int count, Op op, Comm const& comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    MPI_Op opC = NativeOp<T>(op);
    std::vector<byte> packedSend, packedRecv;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_SEND_BUFFER(sbuf, count, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, count, syncInfo);
#endif

    Serialize(count, sbuf, packedSend);

    ReserveSerialized(count, rbuf, packedRecv);
    EL_CHECK_MPI_CALL(
        MPI_Allreduce(
            packedSend.data(), packedRecv.data(),
            count, TypeMap<T>(), opC, comm.GetMPIComm()));
    Deserialize(count, packedRecv, rbuf);
}

template <typename T, Device D, typename, typename>
void AllReduce(T const*, T*, int, Op, Comm const&, SyncInfo<D> const&)
{
    LogicError("AllReduce: Bad device/type combination.");
}

//
// The "IN_PLACE" allreduce
//

#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D,
          typename/*=EnableIf<IsAluminumDeviceType<T,D>>*/>
void AllReduce(T* buf, int count, Op op, Comm const& comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    using Backend = BestBackend<T,D,Collective::ALLREDUCE>;

    if (count == 0)
        return;

    Al::Allreduce<Backend>(
        buf, count, MPI_Op2ReductionOperator(AlNativeOp<T>(op)),
        comm.template GetComm<Backend>(syncInfo));
}
#endif // HYDROGEN_HAVE_ALUMINUM

template <typename T, Device D, typename, typename, typename, typename, typename>
void AllReduce(T* buf, int count, Op op, Comm const& comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0 || Size(comm) == 1)
        return;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_INPLACE_BUFFER(buf, count, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    EL_CHECK_MPI_CALL(
        MPI_Allreduce(
            MPI_IN_PLACE, buf,
            count, TypeMap<T>(), NativeOp<T>(op), comm.GetMPIComm()));
}

template <typename T, Device D, typename, typename, typename, typename>
void AllReduce(Complex<T>* buf, int count, Op op, Comm const& comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0 || Size(comm) == 1)
        return;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_INPLACE_BUFFER(buf, count, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

#ifdef EL_AVOID_COMPLEX_MPI
    if (op == SUM)
    {
        EL_CHECK_MPI_CALL(
            MPI_Allreduce(
                MPI_IN_PLACE, buf, 2*count,
                TypeMap<T>(), NativeOp<T>(op), comm.GetMPIComm()));
    }
    else
    {
        EL_CHECK_MPI_CALL(
            MPI_Allreduce(
                MPI_IN_PLACE, buf, count, TypeMap<Complex<T>>(),
                NativeOp<Complex<T>>(op), comm.GetMPIComm()));
    }
#else
    EL_CHECK_MPI_CALL(
        MPI_Allreduce(
            MPI_IN_PLACE, buf, count,
            TypeMap<Complex<T>>(), NativeOp<Complex<T>>(op),
            comm.GetMPIComm()));
#endif
}

template <typename T, Device D, typename, typename, typename>
void AllReduce(T* buf, int count, Op op, Comm const& comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_INPLACE_BUFFER(buf, count, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    MPI_Op opC = NativeOp<T>(op);
    std::vector<byte> packedSend, packedRecv;
    Serialize(count, buf, packedSend);

    ReserveSerialized(count, buf, packedRecv);
    EL_CHECK_MPI_CALL(
        MPI_Allreduce(
            packedSend.data(), packedRecv.data(),
            count, TypeMap<T>(), opC, comm.GetMPIComm()));
    Deserialize(count, packedRecv, buf);
}

template <typename T, Device D, typename, typename>
void AllReduce(T*, int, Op, Comm const&, SyncInfo<D> const&)
{
    LogicError("AllReduce: Bad device/type combination.");
}

template<typename T, Device D>
void AllReduce(const T* sbuf, T* rbuf, int count, Comm const& comm, SyncInfo<D> const& syncInfo)
{ AllReduce(sbuf, rbuf, count, SUM, std::move(comm), syncInfo); }

template<typename T, Device D>
T AllReduce(T sb, Op op, Comm const& comm, SyncInfo<D> const& syncInfo)
{ T rb; AllReduce(&sb, &rb, 1, op, std::move(comm), syncInfo); return rb; }

template<typename T, Device D>
T AllReduce(T sb, Comm const& comm, SyncInfo<D> const& syncInfo)
{ return AllReduce(sb, SUM, std::move(comm), syncInfo); }

template<typename T, Device D>
void AllReduce(T* buf, int count, Comm const& comm, SyncInfo<D> const& syncInfo)
{ AllReduce(buf, count, SUM, std::move(comm), syncInfo); }

#define MPI_ALLREDUCE_PROTO_DEV(T,D)                                    \
    template void AllReduce(                                            \
        const T*, T*, int, Op, Comm const&, SyncInfo<D> const&);               \
    template void AllReduce(T*, int, Op, Comm const&, SyncInfo<D> const&);     \
    template void AllReduce(                                            \
        const T*, T*, int, Comm const&, SyncInfo<D> const&);                   \
    template T AllReduce(T, Op, Comm const&, SyncInfo<D> const&);              \
    template T AllReduce(T, Comm const&, SyncInfo<D> const&);                  \
    template void AllReduce(T*, int, Comm const&, SyncInfo<D> const&)

#ifndef HYDROGEN_HAVE_GPU
#define MPI_ALLREDUCE_PROTO(T)             \
    MPI_ALLREDUCE_PROTO_DEV(T,Device::CPU)
#else
#define MPI_ALLREDUCE_PROTO(T)             \
    MPI_ALLREDUCE_PROTO_DEV(T,Device::CPU);     \
    MPI_ALLREDUCE_PROTO_DEV(T,Device::GPU)
#endif // HYDROGEN_HAVE_GPU

MPI_ALLREDUCE_PROTO(byte);
MPI_ALLREDUCE_PROTO(int);
MPI_ALLREDUCE_PROTO(unsigned);
MPI_ALLREDUCE_PROTO(long int);
MPI_ALLREDUCE_PROTO(unsigned long);
MPI_ALLREDUCE_PROTO(float);
MPI_ALLREDUCE_PROTO(double);
MPI_ALLREDUCE_PROTO(long long int);
MPI_ALLREDUCE_PROTO(unsigned long long);
MPI_ALLREDUCE_PROTO(ValueInt<Int>);
MPI_ALLREDUCE_PROTO(Entry<Int>);
MPI_ALLREDUCE_PROTO(Complex<float>);
MPI_ALLREDUCE_PROTO(ValueInt<float>);
MPI_ALLREDUCE_PROTO(ValueInt<Complex<float>>);
MPI_ALLREDUCE_PROTO(Entry<float>);
MPI_ALLREDUCE_PROTO(Entry<Complex<float>>);
MPI_ALLREDUCE_PROTO(Complex<double>);
MPI_ALLREDUCE_PROTO(ValueInt<double>);
MPI_ALLREDUCE_PROTO(ValueInt<Complex<double>>);
MPI_ALLREDUCE_PROTO(Entry<double>);
MPI_ALLREDUCE_PROTO(Entry<Complex<double>>);
#ifdef HYDROGEN_HAVE_HALF
MPI_ALLREDUCE_PROTO(cpu_half_type);
MPI_ALLREDUCE_PROTO(ValueInt<cpu_half_type>);
MPI_ALLREDUCE_PROTO(Entry<cpu_half_type>);
#endif
#ifdef HYDROGEN_GPU_USE_FP16
MPI_ALLREDUCE_PROTO(gpu_half_type);
#endif
#ifdef HYDROGEN_HAVE_QD
MPI_ALLREDUCE_PROTO(DoubleDouble);
MPI_ALLREDUCE_PROTO(QuadDouble);
MPI_ALLREDUCE_PROTO(Complex<DoubleDouble>);
MPI_ALLREDUCE_PROTO(Complex<QuadDouble>);
MPI_ALLREDUCE_PROTO(ValueInt<DoubleDouble>);
MPI_ALLREDUCE_PROTO(ValueInt<QuadDouble>);
MPI_ALLREDUCE_PROTO(ValueInt<Complex<DoubleDouble>>);
MPI_ALLREDUCE_PROTO(ValueInt<Complex<QuadDouble>>);
MPI_ALLREDUCE_PROTO(Entry<DoubleDouble>);
MPI_ALLREDUCE_PROTO(Entry<QuadDouble>);
MPI_ALLREDUCE_PROTO(Entry<Complex<DoubleDouble>>);
MPI_ALLREDUCE_PROTO(Entry<Complex<QuadDouble>>);
#endif
#ifdef HYDROGEN_HAVE_QUADMATH
MPI_ALLREDUCE_PROTO(Quad);
MPI_ALLREDUCE_PROTO(Complex<Quad>);
MPI_ALLREDUCE_PROTO(ValueInt<Quad>);
MPI_ALLREDUCE_PROTO(ValueInt<Complex<Quad>>);
MPI_ALLREDUCE_PROTO(Entry<Quad>);
MPI_ALLREDUCE_PROTO(Entry<Complex<Quad>>);
#endif
#ifdef HYDROGEN_HAVE_MPC
MPI_ALLREDUCE_PROTO(BigInt);
MPI_ALLREDUCE_PROTO(BigFloat);
MPI_ALLREDUCE_PROTO(Complex<BigFloat>);
MPI_ALLREDUCE_PROTO(ValueInt<BigInt>);
MPI_ALLREDUCE_PROTO(ValueInt<BigFloat>);
MPI_ALLREDUCE_PROTO(ValueInt<Complex<BigFloat>>);
MPI_ALLREDUCE_PROTO(Entry<BigInt>);
MPI_ALLREDUCE_PROTO(Entry<BigFloat>);
MPI_ALLREDUCE_PROTO(Entry<Complex<BigFloat>>);
#endif

}// namespace mpi
}// namespace El
