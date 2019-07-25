/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_COPY_ROWFILTER_HPP
#define EL_BLAS_COPY_ROWFILTER_HPP

namespace El {
namespace copy {

// (U,Collect(V)) |-> (U,V)
template <Device D, typename T>
void RowFilter_impl
( const ElementalMatrix<T>& A, ElementalMatrix<T>& B )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if( A.ColDist() != B.ColDist() ||
          A.RowDist() != Collect(B.RowDist()) )
          LogicError("Incompatible distributions");
    )
    AssertSameGrids( A, B );

    B.AlignColsAndResize
    ( A.ColAlign(), A.Height(), A.Width(), false, false );
    if( !B.Participating() )
        return;

    const Int rowStride = B.RowStride();
    const Int rowShift = B.RowShift();

    const Int localHeight = B.LocalHeight();
    const Int localWidth = B.LocalWidth();

    SyncInfo<D> syncInfoA = SyncInfoFromMatrix(static_cast<Matrix<T,D> const&>(A.LockedMatrix())),
        syncInfoB = SyncInfoFromMatrix(static_cast<Matrix<T,D> const&>(B.LockedMatrix()));

    auto syncHelper = MakeMultiSync(syncInfoB, syncInfoA);

    const Int colDiff = B.ColAlign() - A.ColAlign();
    if( colDiff == 0 )
    {
        util::InterleaveMatrix(
            localHeight, localWidth,
            A.LockedBuffer(0,rowShift), 1, rowStride*A.LDim(),
            B.Buffer(),                 1, B.LDim(), syncInfoB);
    }
    else
    {
#ifdef EL_UNALIGNED_WARNINGS
        if( B.Grid().Rank() == 0 )
            Output("Unaligned RowFilter");
#endif
        const Int colStride = B.ColStride();
        const Int sendColRank = Mod( B.ColRank()+colDiff, colStride );
        const Int recvColRank = Mod( B.ColRank()-colDiff, colStride );
        const Int localHeightA = A.LocalHeight();
        const Int sendSize = localHeightA*localWidth;
        const Int recvSize = localHeight *localWidth;

        simple_buffer<T,D> buffer(sendSize+recvSize, syncInfoB);
        T* sendBuf = buffer.data();
        T* recvBuf = buffer.data() + sendSize;

        // Pack
        util::InterleaveMatrix(
            localHeightA, localWidth,
            A.LockedBuffer(0,rowShift), 1, rowStride*A.LDim(),
            sendBuf,                    1, localHeightA, syncInfoB);

        // Realign
        mpi::SendRecv(
            sendBuf, sendSize, sendColRank,
            recvBuf, recvSize, recvColRank, B.ColComm(), syncInfoB);

        // Unpack
        util::InterleaveMatrix(
            localHeight, localWidth,
            recvBuf,    1, localHeight,
            B.Buffer(), 1, B.LDim(), syncInfoB);
    }
}

template <typename T>
void RowFilter
( const ElementalMatrix<T>& A, ElementalMatrix<T>& B )
{
    EL_DEBUG_CSE
    if (A.GetLocalDevice() != B.GetLocalDevice())
        LogicError("Interdevice row filter not supported yet.");

    switch (A.GetLocalDevice())
    {
    case Device::CPU:
        RowFilter_impl<Device::CPU>(A,B);
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        RowFilter_impl<Device::GPU>(A,B);
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("RowFilter: Bad device.");
    }
}

template<typename T>
void RowFilter
( const BlockMatrix<T>& A, BlockMatrix<T>& B )
{
    EL_DEBUG_CSE
    AssertSameGrids( A, B );
    EL_DEBUG_ONLY(
      if( A.ColDist() != B.ColDist() ||
          A.RowDist() != Collect(B.RowDist()) )
          LogicError("Incompatible distributions");
    )

    const Int height = A.Height();
    const Int width = A.Width();
    const Int colCut = A.ColCut();
    const Int blockHeight = A.BlockHeight();
    const Int blockWidth = A.BlockWidth();

    B.AlignAndResize
    ( blockHeight, blockWidth, A.ColAlign(), 0, colCut, 0,
      height, width, false, false );
    // TODO(poulson): Realign if the cuts are different
    if( A.BlockHeight() != B.BlockHeight() || A.ColCut() != B.ColCut() )
    {
        EL_DEBUG_ONLY(
          Output("Performing expensive GeneralPurpose RowFilter");
        )
        GeneralPurpose( A, B );
        return;
    }
    if( !B.Participating() )
        return;

    const Int rowStride = B.RowStride();
    const Int rowShift = B.RowShift();

    const Int localHeight = B.LocalHeight();
    const Int localWidth = B.LocalWidth();

    const Int colDiff = B.ColAlign() - A.ColAlign();
    if( colDiff == 0 )
    {
        util::BlockedRowFilter
        ( localHeight, width,
          rowShift, rowStride, B.BlockWidth(), B.RowCut(),
          A.LockedBuffer(), A.LDim(),
          B.Buffer(), B.LDim() );
    }
    else
    {
#ifdef EL_UNALIGNED_WARNINGS
        if( B.Grid().Rank() == 0 )
            Output("Unaligned RowFilter");
#endif
        const Int colStride = B.ColStride();
        const Int sendColRank = Mod( B.ColRank()+colDiff, colStride );
        const Int recvColRank = Mod( B.ColRank()-colDiff, colStride );
        const Int localHeightA = A.LocalHeight();
        const Int sendSize = localHeightA*localWidth;
        const Int recvSize = localHeight *localWidth;

        vector<T> buffer;
        FastResize( buffer, sendSize+recvSize );
        T* sendBuf = &buffer[0];
        T* recvBuf = &buffer[sendSize];

        // Pack
        util::BlockedRowFilter
        ( localHeightA, width,
          rowShift, rowStride, B.BlockWidth(), B.RowCut(),
          A.LockedBuffer(), A.LDim(),
          sendBuf, localHeightA );

        // Realign
        mpi::SendRecv
        ( sendBuf, sendSize, sendColRank,
          recvBuf, recvSize, recvColRank, B.ColComm(),
          SyncInfo<Device::CPU>{} );

        // Unpack
        util::InterleaveMatrix
        ( localHeight, localWidth,
          recvBuf,    1, localHeight,
          B.Buffer(), 1, B.LDim(), SyncInfo<Device::CPU>{} );
    }
}

} // namespace copy
} // namespace El

#endif // ifndef EL_BLAS_COPY_ROWFILTER_HPP
