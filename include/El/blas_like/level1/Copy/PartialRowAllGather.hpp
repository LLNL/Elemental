/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_COPY_PARTIALROWALLGATHER_HPP
#define EL_BLAS_COPY_PARTIALROWALLGATHER_HPP

namespace El {
namespace copy {

// (U,V) |-> (U,Partial(V))
template<Device D, typename T>
void PartialRowAllGather_impl
( const ElementalMatrix<T>& A, ElementalMatrix<T>& B )
{
    const Int height = A.Height();
    const Int width = A.Width();
    B.AlignRowsAndResize
    ( Mod(A.RowAlign(),B.RowStride()), height, width, false, false );

    const Int rowStride = A.RowStride();
    const Int rowStrideUnion = A.PartialUnionRowStride();
    const Int rowStridePart = A.PartialRowStride();
    const Int rowRankPart = A.PartialRowRank();
    const Int rowDiff = B.RowAlign() - Mod(A.RowAlign(),rowStridePart);

    const Int maxLocalWidth = MaxLength(width,rowStride);
    const Int portionSize = mpi::Pad( height*maxLocalWidth );

    SyncInfo<D> syncInfoA = SyncInfoFromMatrix(static_cast<Matrix<T,D> const&>(A.LockedMatrix())),
        syncInfoB = SyncInfoFromMatrix(static_cast<Matrix<T,D> const&>(B.LockedMatrix()));

    auto syncHelper = MakeMultiSync(syncInfoB, syncInfoA);

    if( rowDiff == 0 )
    {
        if( A.PartialUnionRowStride() == 1 )
        {
            Copy( A.LockedMatrix(), B.Matrix() );
        }
        else
        {
            simple_buffer<T,D> buffer((rowStrideUnion+1)*portionSize,
                                      syncInfoB);
            T* firstBuf = buffer.data();
            T* secondBuf = buffer.data() + portionSize;

            // Pack
            util::InterleaveMatrix(
                height, A.LocalWidth(),
                A.LockedBuffer(), 1, A.LDim(),
                firstBuf,         1, height, syncInfoB );

            // Communicate
            mpi::AllGather(
                firstBuf, portionSize, secondBuf, portionSize,
                A.PartialUnionRowComm(), syncInfoB);

            // Unpack
            util::PartialRowStridedUnpack(
                height, width,
                A.RowAlign(), rowStride,
                rowStrideUnion, rowStridePart, rowRankPart,
                B.RowShift(),
                secondBuf, portionSize,
                B.Buffer(), B.LDim(), syncInfoB);
        }
    }
    else
    {
#ifdef EL_UNALIGNED_WARNINGS
        if( A.Grid().Rank() == 0 )
            cerr << "Unaligned PartialRowAllGather" << endl;
#endif
        simple_buffer<T,D> buffer((rowStrideUnion+1)*portionSize,
                                  syncInfoB);
        T* firstBuf = buffer.data();
        T* secondBuf = buffer.data() + portionSize;

        // Perform a SendRecv to match the row alignments
        util::InterleaveMatrix(
            height, A.LocalWidth(),
            A.LockedBuffer(), 1, A.LDim(),
            secondBuf,        1, height, syncInfoB);

        const Int sendRowRank = Mod( A.RowRank()+rowDiff, rowStride );
        const Int recvRowRank = Mod( A.RowRank()-rowDiff, rowStride );

        mpi::SendRecv(
            secondBuf, portionSize, sendRowRank,
            firstBuf,  portionSize, recvRowRank, A.RowComm(), syncInfoB);

        // Use the SendRecv as an input to the partial union AllGather
        mpi::AllGather(
            firstBuf,  portionSize,
            secondBuf, portionSize, A.PartialUnionRowComm(), syncInfoB);

        // Unpack
        util::PartialRowStridedUnpack(
            height, width,
            A.RowAlign()+rowDiff, rowStride,
            rowStrideUnion, rowStridePart, rowRankPart,
            B.RowShift(),
            secondBuf, portionSize,
            B.Buffer(), B.LDim(), syncInfoB);
    }
}

template<typename T>
void PartialRowAllGather
( const ElementalMatrix<T>& A, ElementalMatrix<T>& B )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if( B.ColDist() != A.ColDist() ||
          B.RowDist() != Partial(A.RowDist()) )
          LogicError("Incompatible distributions");
    )
    AssertSameGrids( A, B );

    if( !A.Participating() )
        return;

    EL_DEBUG_ONLY(
        if( A.LocalHeight() != A.Height() )
          LogicError("This routine assumes columns are not distributed");
    )

    switch (A.GetLocalDevice())
    {
    case Device::CPU:
        PartialRowAllGather_impl<Device::CPU>(A,B);
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        PartialRowAllGather_impl<Device::GPU>(A,B);
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("PartialRowAllGather: Bad device.");
    }
}

template<typename T>
void PartialRowAllGather
( const BlockMatrix<T>& A,
        BlockMatrix<T>& B )
{
    EL_DEBUG_CSE
    AssertSameGrids( A, B );
    // TODO(poulson): More efficient implementation
    GeneralPurpose( A, B );
}

} // namespace copy
} // namespace El

#endif // ifndef EL_BLAS_COPY_PARTIALROWALLGATHER_HPP
