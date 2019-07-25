// Broadcast

#include <El/core/imports/aluminum.hpp>

namespace El
{
namespace mpi
{

#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D,
          typename/*=EnableIf<IsAluminumSupported<T,D,COLL>>*/>
void Broadcast(T* buffer, int count, int root, Comm const& comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE

    using Backend = BestBackend<T,D,Collective::BROADCAST>;
    Al::Bcast<Backend>(
        buffer, count, root, comm.template GetComm<Backend>(syncInfo));
}
#endif // HYDROGEN_HAVE_ALUMINUM

template <typename T, Device D, typename, typename, typename, typename, typename>
void Broadcast(T* buffer, int count, int root, Comm const& comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (Size(comm) == 1 || count == 0)
        return;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    // I want to pre-transfer if root, I want to post-transfer if not root
    auto const rank = Rank(comm);
    auto const pre_xfer_size = (rank == root ? static_cast<size_t>(count) : 0);
    auto const post_xfer_size = (rank != root ? static_cast<size_t>(count) : 0);

    ENSURE_HOST_BUFFER_PREPOST_XFER(
        buffer, count, 0UL, pre_xfer_size, 0UL, post_xfer_size, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);// NOOP on CPU,
                          // cudaStreamSynchronize on GPU.
    EL_CHECK_MPI_CALL(MPI_Bcast(buffer, count, TypeMap<T>(), root, comm.GetMPIComm()));
}

template <typename T, Device D, typename, typename, typename, typename>
void Broadcast(Complex<T>* buffer, int count, int root, Comm const& comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (Size(comm) == 1 || count == 0)
        return;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    // I want to pre-transfer if root, I want to post-transfer if not root
    auto const rank = Rank(comm);
    auto const pre_xfer_size = (rank == root ? static_cast<size_t>(count) : 0);
    auto const post_xfer_size = (rank != root ? static_cast<size_t>(count) : 0);

    ENSURE_HOST_BUFFER_PREPOST_XFER(
        buffer, count, 0UL, pre_xfer_size, 0UL, post_xfer_size, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

#ifdef EL_AVOID_COMPLEX_MPI
    EL_CHECK_MPI_CALL(MPI_Bcast(buffer, 2*count, TypeMap<T>(), root, comm.GetMPIComm()));
#else
    EL_CHECK_MPI_CALL(MPI_Bcast(buffer, count, TypeMap<Complex<T>>(), root, comm.GetMPIComm()));
#endif
}

template <typename T, Device D, typename, typename, typename>
void Broadcast(T* buffer, int count, int root, Comm const& comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (Size(comm) == 1 || count == 0)
        return;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    // I want to pre-transfer if root, I want to post-transfer if not root
    auto const rank = Rank(comm);
    auto const pre_xfer_size = (rank == root ? static_cast<size_t>(count) : 0);
    auto const post_xfer_size = (rank != root ? static_cast<size_t>(count) : 0);

    ENSURE_HOST_BUFFER_PREPOST_XFER(
        buffer, count, 0UL, pre_xfer_size, 0UL, post_xfer_size, syncInfo);
#endif // HYDROGEN_ENSURE_HOST_MPI_BUFFERS

    Synchronize(syncInfo);

    std::vector<byte> packedBuf;
    Serialize(count, buffer, packedBuf);
    EL_CHECK_MPI_CALL(MPI_Bcast(packedBuf.data(), count, TypeMap<T>(), root, comm.GetMPIComm()));
    Deserialize(count, packedBuf, buffer);
}

template <typename T, Device D, typename, typename>
void Broadcast(T*, int, int, Comm const&, SyncInfo<D> const&)
{
    LogicError("Broadcast: Bad device/type combination.");
}

template <typename T, Device D>
void Broadcast( T& b, int root, Comm const& comm, SyncInfo<D> const& syncInfo )
{ Broadcast( &b, 1, root, std::move(comm), syncInfo ); }

#define MPI_BROADCAST_PROTO_DEV(T,D)                                    \
    template void Broadcast(T*, int, int, Comm const&, SyncInfo<D> const&);       \
    template void Broadcast(T&, int, Comm const&, SyncInfo<D> const&)

#ifndef HYDROGEN_HAVE_GPU
#define MPI_BROADCAST_PROTO(T)                  \
    MPI_BROADCAST_PROTO_DEV(T,Device::CPU)
#else
#define MPI_BROADCAST_PROTO(T)                  \
    MPI_BROADCAST_PROTO_DEV(T,Device::CPU);     \
    MPI_BROADCAST_PROTO_DEV(T,Device::GPU)
#endif // HYDROGEN_HAVE_GPU

MPI_BROADCAST_PROTO(byte);
MPI_BROADCAST_PROTO(int);
MPI_BROADCAST_PROTO(unsigned);
MPI_BROADCAST_PROTO(long int);
MPI_BROADCAST_PROTO(unsigned long);
MPI_BROADCAST_PROTO(float);
MPI_BROADCAST_PROTO(double);
MPI_BROADCAST_PROTO(long long int);
MPI_BROADCAST_PROTO(unsigned long long);
MPI_BROADCAST_PROTO(ValueInt<Int>);
MPI_BROADCAST_PROTO(Entry<Int>);
MPI_BROADCAST_PROTO(Complex<float>);
MPI_BROADCAST_PROTO(ValueInt<float>);
MPI_BROADCAST_PROTO(ValueInt<Complex<float>>);
MPI_BROADCAST_PROTO(Entry<float>);
MPI_BROADCAST_PROTO(Entry<Complex<float>>);
MPI_BROADCAST_PROTO(Complex<double>);
MPI_BROADCAST_PROTO(ValueInt<double>);
MPI_BROADCAST_PROTO(ValueInt<Complex<double>>);
MPI_BROADCAST_PROTO(Entry<double>);
MPI_BROADCAST_PROTO(Entry<Complex<double>>);
#ifdef HYDROGEN_HAVE_HALF
MPI_BROADCAST_PROTO(cpu_half_type);
MPI_BROADCAST_PROTO(ValueInt<cpu_half_type>);
MPI_BROADCAST_PROTO(Entry<cpu_half_type>);
#endif
#ifdef HYDROGEN_GPU_USE_FP16
MPI_BROADCAST_PROTO(gpu_half_type);
MPI_BROADCAST_PROTO(Entry<gpu_half_type>);
#endif
#ifdef HYDROGEN_HAVE_QD
MPI_BROADCAST_PROTO(DoubleDouble);
MPI_BROADCAST_PROTO(QuadDouble);
MPI_BROADCAST_PROTO(Complex<DoubleDouble>);
MPI_BROADCAST_PROTO(Complex<QuadDouble>);
MPI_BROADCAST_PROTO(ValueInt<DoubleDouble>);
MPI_BROADCAST_PROTO(ValueInt<QuadDouble>);
MPI_BROADCAST_PROTO(ValueInt<Complex<DoubleDouble>>);
MPI_BROADCAST_PROTO(ValueInt<Complex<QuadDouble>>);
MPI_BROADCAST_PROTO(Entry<DoubleDouble>);
MPI_BROADCAST_PROTO(Entry<QuadDouble>);
MPI_BROADCAST_PROTO(Entry<Complex<DoubleDouble>>);
MPI_BROADCAST_PROTO(Entry<Complex<QuadDouble>>);
#endif
#ifdef HYDROGEN_HAVE_QUADMATH
MPI_BROADCAST_PROTO(Quad);
MPI_BROADCAST_PROTO(Complex<Quad>);
MPI_BROADCAST_PROTO(ValueInt<Quad>);
MPI_BROADCAST_PROTO(ValueInt<Complex<Quad>>);
MPI_BROADCAST_PROTO(Entry<Quad>);
MPI_BROADCAST_PROTO(Entry<Complex<Quad>>);
#endif
#ifdef HYDROGEN_HAVE_MPC
MPI_BROADCAST_PROTO(BigInt);
MPI_BROADCAST_PROTO(BigFloat);
MPI_BROADCAST_PROTO(Complex<BigFloat>);
MPI_BROADCAST_PROTO(ValueInt<BigInt>);
MPI_BROADCAST_PROTO(ValueInt<BigFloat>);
MPI_BROADCAST_PROTO(ValueInt<Complex<BigFloat>>);
MPI_BROADCAST_PROTO(Entry<BigInt>);
MPI_BROADCAST_PROTO(Entry<BigFloat>);
MPI_BROADCAST_PROTO(Entry<Complex<BigFloat>>);
#endif

}// namespace mpi
}// namespace El
