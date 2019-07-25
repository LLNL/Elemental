/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_AXPYCONTRACT_HPP
#define EL_BLAS_AXPYCONTRACT_HPP

namespace El {

namespace axpy_contract {

// (Partial(U),V) -> (U,V)
template<typename T, Device D>
void PartialColScatter
( T alpha,
  const ElementalMatrix<T>& A,
        ElementalMatrix<T>& B )
{
    EL_DEBUG_CSE
    AssertSameGrids( A, B );
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        LogicError("A and B must be the same size");

#ifdef EL_CACHE_WARNINGS
    if( A.Width() != 1 && A.Grid().Rank() == 0 )
    {
        cerr <<
          "axpy_contract::PartialColScatterUpdate potentially causes a large "
          "amount of cache-thrashing. If possible, avoid it by forming the "
          "(conjugate-)transpose of the [UGath,* ] matrix instead."
          << endl;
    }
#endif
    if( B.ColAlign() % A.ColStride() == A.ColAlign() )
    {

        SyncInfo<D>
            syncInfoA = SyncInfoFromMatrix(static_cast<Matrix<T,D> const&>(A.LockedMatrix())),
            syncInfoB = SyncInfoFromMatrix(static_cast<Matrix<T,D> const&>(B.LockedMatrix()));

        auto syncHelper = MakeMultiSync(syncInfoB, syncInfoA);

        const Int colStride = B.ColStride();
        const Int colStridePart = B.PartialColStride();
        const Int colStrideUnion = B.PartialUnionColStride();
        const Int colRankPart = B.PartialColRank();
        const Int colAlign = B.ColAlign();

        const Int height = B.Height();
        const Int width = B.Width();
        const Int localHeight = B.LocalHeight();
        const Int maxLocalHeight = MaxLength( height, colStride );
        const Int recvSize = mpi::Pad( maxLocalHeight*width );
        const Int sendSize = colStrideUnion*recvSize;

        // We explicitly zero-initialize rather than calling FastResize to avoid
        // inadvertently causing a floating-point exception in the reduction of
        // the padding entries.
        simple_buffer<T,D> buffer(sendSize, TypeTraits<T>::Zero(), syncInfoB);

        // Pack
        copy::util::PartialColStridedPack(
            height, width,
            colAlign, colStride,
            colStrideUnion, colStridePart, colRankPart,
            A.ColShift(),
            A.LockedBuffer(), A.LDim(),
            buffer.data(), recvSize, syncInfoB);

        // Communicate
        mpi::ReduceScatter(buffer.data(), recvSize, B.PartialUnionColComm(),
                           syncInfoB);

        // Unpack our received data
        axpy::util::InterleaveMatrixUpdate(
            alpha, localHeight, width,
            buffer.data(), 1, localHeight,
            B.Buffer(),    1, B.LDim(), syncInfoB);
    }
    else
        LogicError("Unaligned PartialColScatter not implemented");
}

// (U,Partial(V)) -> (U,V)
template<typename T, Device D>
void PartialRowScatter(
    T alpha,
    ElementalMatrix<T> const& A,
    ElementalMatrix<T>& B )
{
    EL_DEBUG_CSE
    AssertSameGrids( A, B );
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        LogicError("Matrix sizes did not match");
    if( !B.Participating() )
        return;

    if( B.RowAlign() % A.RowStride() == A.RowAlign() )
    {
        SyncInfo<D>
            syncInfoA = SyncInfoFromMatrix(static_cast<Matrix<T,D> const&>(A.LockedMatrix())),
            syncInfoB = SyncInfoFromMatrix(static_cast<Matrix<T,D> const&>(B.LockedMatrix()));

        auto syncHelper = MakeMultiSync(syncInfoB, syncInfoA);

        const Int rowStride = B.RowStride();
        const Int rowStridePart = B.PartialRowStride();
        const Int rowStrideUnion = B.PartialUnionRowStride();
        const Int rowRankPart = B.PartialRowRank();

        const Int height = B.Height();
        const Int width = B.Width();
        const Int maxLocalWidth = MaxLength( width, rowStride );
        const Int recvSize = mpi::Pad( height*maxLocalWidth );
        const Int sendSize = rowStrideUnion*recvSize;

        simple_buffer<T,D> buffer(sendSize, TypeTraits<T>::Zero(), syncInfoB);

        // Pack
        copy::util::PartialRowStridedPack(
            height, width,
            B.RowAlign(), rowStride,
            rowStrideUnion, rowStridePart, rowRankPart,
            A.RowShift(),
            A.LockedBuffer(), A.LDim(),
            buffer.data(),    recvSize, syncInfoB);

        // Communicate
        mpi::ReduceScatter(buffer.data(), recvSize, B.PartialUnionRowComm(),
                           syncInfoB);

        // Unpack our received data
        axpy::util::InterleaveMatrixUpdate(
            alpha, height, B.LocalWidth(),
            buffer.data(), 1, height,
            B.Buffer(),    1, B.LDim(), syncInfoB);
    }
    else
        LogicError("Unaligned PartialRowScatter not implemented");
}

// (Collect(U),V) -> (U,V)
template <typename T, Device D>
void ColScatter
( T alpha,
  const ElementalMatrix<T>& A,
        ElementalMatrix<T>& B )
{
    EL_DEBUG_CSE
    AssertSameGrids( A, B );
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        LogicError("A and B must be the same size");
#ifdef EL_VECTOR_WARNINGS
    if( A.Width() == 1 && B.Grid().Rank() == 0 )
    {
        cerr <<
          "The vector version of ColScatter does not"
          " yet have a vector version implemented, but it would only "
          "require a modification of the vector version of RowScatter"
          << endl;
    }
#endif
#ifdef EL_CACHE_WARNINGS
    if( A.Width() != 1 && B.Grid().Rank() == 0 )
    {
        cerr <<
          "axpy_contract::ColScatter potentially causes a large "
          "amount of cache-thrashing. If possible, avoid it by forming the "
          "(conjugate-)transpose of the [* ,V] matrix instead." << endl;
    }
#endif
    if( !B.Participating() )
        return;
    const Int height = B.Height();
    const Int localHeight = B.LocalHeight();
    const Int localWidth = B.LocalWidth();

    const Int colAlign = B.ColAlign();
    const Int colStride = B.ColStride();

    const Int rowDiff = B.RowAlign()-A.RowAlign();

    SyncInfo<D>
        syncInfoA = SyncInfoFromMatrix(static_cast<Matrix<T,D> const&>(A.LockedMatrix())),
        syncInfoB = SyncInfoFromMatrix(static_cast<Matrix<T,D> const&>(B.LockedMatrix()));

    auto syncHelper = MakeMultiSync(syncInfoB, syncInfoA);

    // TODO: Allow for modular equivalence if possible
    if( rowDiff == 0 )
    {
        const Int maxLocalHeight = MaxLength(height,colStride);

        const Int recvSize = mpi::Pad( maxLocalHeight*localWidth );
        const Int sendSize = colStride*recvSize;
        simple_buffer<T,D> buffer(sendSize, TypeTraits<T>::Zero(), syncInfoB);

        // Pack
        copy::util::ColStridedPack(
            height, localWidth,
            colAlign, colStride,
            A.LockedBuffer(), A.LDim(),
            buffer.data(),    recvSize, syncInfoB);

        // Communicate
        mpi::ReduceScatter(buffer.data(), recvSize, B.ColComm(),
                           syncInfoB);

        // Update with our received data
        axpy::util::InterleaveMatrixUpdate(
            alpha, localHeight, localWidth,
            buffer.data(), 1, localHeight,
            B.Buffer(),    1, B.LDim(), syncInfoB);
    }
    else
    {
#ifdef EL_UNALIGNED_WARNINGS
        if( B.Grid().Rank() == 0 )
            cerr << "Unaligned ColScatter" << endl;
#endif
        const Int localWidthA = A.LocalWidth();
        const Int maxLocalHeight = MaxLength(height,colStride);

        const Int recvSize_RS = mpi::Pad( maxLocalHeight*localWidthA );
        const Int sendSize_RS = colStride*recvSize_RS;
        const Int recvSize_SR = localHeight*localWidth;

        simple_buffer<T,D> buffer(
            recvSize_RS + Max(sendSize_RS,recvSize_SR), TypeTraits<T>::Zero(), syncInfoB);
        T* firstBuf = buffer.data();
        T* secondBuf = buffer.data() + recvSize_RS;

        // Pack
        copy::util::ColStridedPack(
            height, localWidth,
            colAlign, colStride,
            A.LockedBuffer(), A.LDim(),
            secondBuf,        recvSize_RS, syncInfoB);

        // Reduce-scatter over each col
        mpi::ReduceScatter(secondBuf, firstBuf, recvSize_RS, B.ColComm(),
                           syncInfoB);

        // Trade reduced data with the appropriate col
        const Int sendCol = Mod( B.RowRank()+rowDiff, B.RowStride() );
        const Int recvCol = Mod( B.RowRank()-rowDiff, B.RowStride() );
        mpi::SendRecv(
            firstBuf,  localHeight*localWidthA, sendCol,
            secondBuf, localHeight*localWidth,  recvCol, B.RowComm(),
            syncInfoB);

        // Update with our received data
        axpy::util::InterleaveMatrixUpdate(
            alpha, localHeight, localWidth,
            secondBuf,  1, localHeight,
            B.Buffer(), 1, B.LDim(), syncInfoB);
    }
}

// (U,Collect(V)) -> (U,V)
template <typename T, Device D>
void RowScatter
( T alpha,
  const ElementalMatrix<T>& A,
        ElementalMatrix<T>& B )
{
    EL_DEBUG_CSE
    AssertSameGrids( A, B );
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        LogicError("Matrix sizes did not match");
    if( !B.Participating() )
        return;

    const Int width = B.Width();
    const Int colDiff = B.ColAlign()-A.ColAlign();

    SyncInfo<D>
        syncInfoA = SyncInfoFromMatrix(static_cast<Matrix<T,D> const&>(A.LockedMatrix())),
        syncInfoB = SyncInfoFromMatrix(static_cast<Matrix<T,D> const&>(B.LockedMatrix()));

    auto syncHelper = MakeMultiSync(syncInfoB, syncInfoA);

    if( colDiff == 0 )
    {
        if( width == 1 )
        {
            const Int localHeight = B.LocalHeight();
            const Int portionSize = mpi::Pad( localHeight );
            simple_buffer<T,D> buffer(portionSize, TypeTraits<T>::Zero(), syncInfoB);

            // Reduce to rowAlign
            const Int rowAlign = B.RowAlign();
            mpi::Reduce(
                A.LockedBuffer(), buffer.data(), portionSize,
                rowAlign, B.RowComm(), syncInfoB);

            if( B.RowRank() == rowAlign )
            {
                axpy::util::InterleaveMatrixUpdate(
                    alpha, localHeight, 1,
                    buffer.data(), 1, localHeight,
                    B.Buffer(),    1, B.LDim(), syncInfoB);
            }
        }
        else
        {
            const Int rowStride = B.RowStride();
            const Int rowAlign = B.RowAlign();

            const Int localHeight = B.LocalHeight();
            const Int localWidth = B.LocalWidth();
            const Int maxLocalWidth = MaxLength(width,rowStride);

            const Int portionSize = mpi::Pad( localHeight*maxLocalWidth );
            const Int sendSize = rowStride*portionSize;

            // Pack
            simple_buffer<T,D> buffer(sendSize, TypeTraits<T>::Zero(), syncInfoB);
            copy::util::RowStridedPack(
                localHeight, width,
                rowAlign, rowStride,
                A.LockedBuffer(), A.LDim(),
                buffer.data(), portionSize, syncInfoB);

            // Communicate
            mpi::ReduceScatter(buffer.data(), portionSize, B.RowComm(),
                               syncInfoB);

            // Update with our received data
            axpy::util::InterleaveMatrixUpdate(
                alpha, localHeight, localWidth,
                buffer.data(), 1, localHeight,
                B.Buffer(),    1, B.LDim(), syncInfoB);
        }
    }
    else
    {
#ifdef EL_UNALIGNED_WARNINGS
        if( B.Grid().Rank() == 0 )
            cerr << "Unaligned RowScatter" << endl;
#endif
        const Int colRank = B.ColRank();
        const Int colStride = B.ColStride();

        const Int sendRow = Mod( colRank+colDiff, colStride );
        const Int recvRow = Mod( colRank-colDiff, colStride );

        const Int localHeight = B.LocalHeight();
        const Int localHeightA = A.LocalHeight();

        if( width == 1 )
        {
            simple_buffer<T,D> buffer(
                localHeight + localHeightA, TypeTraits<T>::Zero(), syncInfoB);
            T* sendBuf = buffer.data();
            T* recvBuf = buffer.data() + localHeightA;

            // Reduce to rowAlign
            const Int rowAlign = B.RowAlign();
            mpi::Reduce(
                A.LockedBuffer(), sendBuf, localHeightA, rowAlign, B.RowComm(),
                syncInfoB);

            if( B.RowRank() == rowAlign )
            {
                // Perform the realignment
                mpi::SendRecv(
                    sendBuf, localHeightA, sendRow,
                    recvBuf, localHeight,  recvRow, B.ColComm(),
                    syncInfoB);

                axpy::util::InterleaveMatrixUpdate(
                    alpha, localHeight, 1,
                    recvBuf,    1, localHeight,
                    B.Buffer(), 1, B.LDim(), syncInfoB);
            }
        }
        else
        {
            const Int rowStride = B.RowStride();
            const Int rowAlign = B.RowAlign();

            const Int localWidth = B.LocalWidth();
            const Int maxLocalWidth = MaxLength(width,rowStride);

            const Int recvSize_RS = mpi::Pad( localHeightA*maxLocalWidth );
            const Int sendSize_RS = rowStride * recvSize_RS;
            const Int recvSize_SR = localHeight * localWidth;

            simple_buffer<T,D> buffer(
                recvSize_RS + Max(sendSize_RS,recvSize_SR), TypeTraits<T>::Zero(), syncInfoB);
            T* firstBuf = buffer.data();
            T* secondBuf = buffer.data() + recvSize_RS;

            // Pack
            copy::util::RowStridedPack(
                localHeightA, width,
                rowAlign, rowStride,
                A.LockedBuffer(), A.LDim(),
                secondBuf,        recvSize_RS, syncInfoB);

            // Reduce-scatter over each process row
            mpi::ReduceScatter(secondBuf, firstBuf, recvSize_RS, B.RowComm(),
                               syncInfoB);

            // Trade reduced data with the appropriate process row
            mpi::SendRecv(
                firstBuf,  localHeightA*localWidth, sendRow,
                secondBuf, localHeight*localWidth,  recvRow, B.ColComm(),
                syncInfoB);

            // Update with our received data
            axpy::util::InterleaveMatrixUpdate(
                alpha, localHeight, localWidth,
                secondBuf,  1, localHeight,
                B.Buffer(), 1, B.LDim(), syncInfoB);
        }
    }
}

// (Collect(U),Collect(V)) -> (U,V)
template <typename T, Device D>
void Scatter
( T alpha,
  const ElementalMatrix<T>& A,
        ElementalMatrix<T>& B )
{
    EL_DEBUG_CSE
    AssertSameGrids( A, B );
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        LogicError("Sizes of A and B must match");
    if( !B.Participating() )
        return;

    const Int colStride = B.ColStride();
    const Int rowStride = B.RowStride();
    const Int colAlign = B.ColAlign();
    const Int rowAlign = B.RowAlign();

    const Int height = B.Height();
    const Int width = B.Width();
    const Int localHeight = B.LocalHeight();
    const Int localWidth = B.LocalWidth();
    const Int maxLocalHeight = MaxLength(height,colStride);
    const Int maxLocalWidth = MaxLength(width,rowStride);

    const Int recvSize = mpi::Pad( maxLocalHeight*maxLocalWidth );
    const Int sendSize = colStride*rowStride*recvSize;

    SyncInfo<D>
        syncInfoA = SyncInfoFromMatrix(static_cast<Matrix<T,D> const&>(A.LockedMatrix())),
        syncInfoB = SyncInfoFromMatrix(static_cast<Matrix<T,D> const&>(B.LockedMatrix()));

    auto syncHelper = MakeMultiSync(syncInfoB, syncInfoA);

    simple_buffer<T,D> buffer(sendSize, TypeTraits<T>::Zero(), syncInfoB);

    // Pack
    copy::util::StridedPack(
        height, width,
        colAlign, colStride,
        rowAlign, rowStride,
        A.LockedBuffer(), A.LDim(),
        buffer.data(),    recvSize, syncInfoB);

    // Communicate
    mpi::ReduceScatter(buffer.data(), recvSize, B.DistComm(),
                       syncInfoB);

    // Unpack our received data
    axpy::util::InterleaveMatrixUpdate(
        alpha, localHeight, localWidth,
        buffer.data(), 1, localHeight,
        B.Buffer(),    1, B.LDim(), syncInfoB);
}

} // namespace axpy_contract

template <Device D, typename T, typename=EnableIf<IsDeviceValidType<T,D>>>
void AxpyContract_impl
( T alpha,
  const ElementalMatrix<T>& A,
        ElementalMatrix<T>& B )
{
    EL_DEBUG_CSE
    if ((A.GetLocalDevice() != D) || (B.GetLocalDevice() != D))
        LogicError("AxpyContract: Bad device.");

    const Dist U = B.ColDist();
    const Dist V = B.RowDist();
    if( A.ColDist() == U && A.RowDist() == V )
        Axpy( alpha, A, B );// FIXME
    else if( A.ColDist() == Partial(U) && A.RowDist() == V )
        axpy_contract::PartialColScatter<T,D>( alpha, A, B );
    else if( A.ColDist() == U && A.RowDist() == Partial(V) )
        axpy_contract::PartialRowScatter<T,D>( alpha, A, B );
    else if( A.ColDist() == Collect(U) && A.RowDist() == V )
        axpy_contract::ColScatter<T,D>( alpha, A, B );
    else if( A.ColDist() == U && A.RowDist() == Collect(V) )
        axpy_contract::RowScatter<T,D>( alpha, A, B );
    else if( A.ColDist() == Collect(U) && A.RowDist() == Collect(V) )
        axpy_contract::Scatter<T,D>( alpha, A, B );
    else
        LogicError("Incompatible distributions");
}

template <Device D, typename T,
          typename=DisableIf<IsDeviceValidType<T,D>>, typename=void>
void AxpyContract_impl
( T alpha,
  const ElementalMatrix<T>& A,
        ElementalMatrix<T>& B )
{
    LogicError("AxpyContract: Bad device/type combination.");
}

template <typename T>
void AxpyContract
( T alpha,
  const ElementalMatrix<T>& A,
        ElementalMatrix<T>& B )
{
    EL_DEBUG_CSE
    if (A.GetLocalDevice() != B.GetLocalDevice())
        LogicError("AxpyContract: Bad device.");

    switch (A.GetLocalDevice())
    {
    case Device::CPU:
        AxpyContract_impl<Device::CPU>(alpha,A,B);
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        AxpyContract_impl<Device::GPU>(alpha,A,B);
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("AxpyContract: Bad device type.");
    }
}
template<typename T>
void AxpyContract
( T alpha,
  const BlockMatrix<T>& A,
        BlockMatrix<T>& B )
{
    EL_DEBUG_CSE
    AssertSameGrids( A, B );
    LogicError("This routine is not yet written");
}

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void AxpyContract \
  ( T alpha, \
    const ElementalMatrix<T>& A, \
          ElementalMatrix<T>& B ); \
  EL_EXTERN template void AxpyContract \
  ( T alpha, \
    const BlockMatrix<T>& A, \
          BlockMatrix<T>& B );

#ifdef HYDROGEN_GPU_USE_FP16
EL_EXTERN template void AxpyContract(
    gpu_half_type alpha,
    const ElementalMatrix<gpu_half_type>& A,
    ElementalMatrix<gpu_half_type>& B );
#endif // HYDROGEN_GPU_USE_FP16
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#define EL_ENABLE_HALF
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_AXPYCONTRACT_HPP
