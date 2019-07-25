/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_COPY_EXCHANGE_HPP
#define EL_BLAS_COPY_EXCHANGE_HPP

namespace El {
namespace copy {

template<typename T, Device D, typename=EnableIf<IsDeviceValidType<T,D>>>
void Exchange_impl
( const ElementalMatrix<T>& A,
        ElementalMatrix<T>& B,
  int sendRank, int recvRank, mpi::Comm const& comm )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(AssertSameGrids( A, B ))

    const int myRank = mpi::Rank( comm );
    EL_DEBUG_ONLY(
      if( myRank == sendRank && myRank != recvRank )
          LogicError("Sending to self but receiving from someone else");
      if( myRank != sendRank && myRank == recvRank )
          LogicError("Receiving from self but sending to someone else");
    )
    B.Resize( A.Height(), A.Width() );

    SyncInfo<D>
        syncInfoA = SyncInfoFromMatrix(
            static_cast<Matrix<T,D> const&>(A.LockedMatrix())),
        syncInfoB = SyncInfoFromMatrix(
            static_cast<Matrix<T,D> const&>(B.LockedMatrix()));

    if( myRank == sendRank )
    {
        mpi::EnsureComm<T,Collective::SENDRECV>(comm, syncInfoB);
        Copy( A.LockedMatrix(), B.Matrix() );
        return;
    }

    const Int localHeightA = A.LocalHeight();
    const Int localHeightB = B.LocalHeight();
    const Int localWidthA = A.LocalWidth();
    const Int localWidthB = B.LocalWidth();
    const Int sendSize = localHeightA*localWidthA;
    const Int recvSize = localHeightB*localWidthB;

    const bool contigA = ( A.LocalHeight() == A.LDim() );
    const bool contigB = ( B.LocalHeight() == B.LDim() );


    auto syncHelper = MakeMultiSync(syncInfoB, syncInfoA);

    if( contigA && contigB )
    {
        mpi::SendRecv(
            A.LockedBuffer(), sendSize, sendRank,
            B.Buffer(),       recvSize, recvRank, comm, syncInfoB);
    }
    else if( contigB )
    {
        // Pack A's data
        simple_buffer<T,D> buf(sendSize, syncInfoB);
        copy::util::InterleaveMatrix(
            localHeightA, localWidthA,
            A.LockedBuffer(), 1, A.LDim(),
            buf.data(),       1, localHeightA, syncInfoB);

        // Exchange with the partner
        mpi::SendRecv(
            buf.data(), sendSize, sendRank,
            B.Buffer(), recvSize, recvRank, comm, syncInfoB);
    }
    else if( contigA )
    {
        // Exchange with the partner
        simple_buffer<T,D> buf(recvSize, syncInfoB);

        mpi::SendRecv(
            A.LockedBuffer(), sendSize, sendRank,
            buf.data(),       recvSize, recvRank, comm,
            syncInfoB);

        // Unpack
        copy::util::InterleaveMatrix(
            localHeightB, localWidthB,
            buf.data(), 1, localHeightB,
            B.Buffer(), 1, B.LDim(), syncInfoB);
    }
    else
    {
        // Pack A's data
        simple_buffer<T,D> sendBuf(sendSize, syncInfoB);
        copy::util::InterleaveMatrix(
            localHeightA, localWidthA,
            A.LockedBuffer(), 1, A.LDim(),
            sendBuf.data(),   1, localHeightA, syncInfoB);

        // Exchange with the partner
        simple_buffer<T,D> recvBuf(recvSize, syncInfoB);

        mpi::SendRecv(
            sendBuf.data(), sendSize, sendRank,
            recvBuf.data(), recvSize, recvRank, comm, syncInfoB);

        // Unpack
        copy::util::InterleaveMatrix(
            localHeightB, localWidthB,
            recvBuf.data(), 1, localHeightB,
            B.Buffer(),     1, B.LDim(), syncInfoB);
    }
}

template<typename T,Device D,
         typename=DisableIf<IsDeviceValidType<T,D>>,typename=void>
void Exchange_impl
( const ElementalMatrix<T>& A,
        ElementalMatrix<T>& B,
  int sendRank, int recvRank, mpi::Comm const& comm )
{
    LogicError("Exchange: Bad Device/type combo.");
}

template<typename T>
void Exchange
( const ElementalMatrix<T>& A,
        ElementalMatrix<T>& B,
  int sendRank, int recvRank, mpi::Comm const& comm )
{
    if (A.GetLocalDevice() != B.GetLocalDevice())
        LogicError("Exchange: Device error.");
    switch (A.GetLocalDevice())
    {
    case Device::CPU:
        Exchange_impl<T,Device::CPU>(A,B,sendRank,recvRank,comm);
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        Exchange_impl<T,Device::GPU>(A,B,sendRank,recvRank,comm);
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("Exchange: Bad device.");
    }
}

template<typename T,Dist U,Dist V,Device D>
void ColwiseVectorExchange
( DistMatrix<T,ProductDist<U,V>(),STAR,ELEMENT,D> const& A,
  DistMatrix<T,ProductDist<V,U>(),STAR,ELEMENT,D>& B )
{
    EL_DEBUG_CSE
    AssertSameGrids( A, B );

    if( !B.Participating() )
        return;

    const Int distSize = A.DistSize();
    const Int colDiff = A.ColShift() - B.ColShift();
    const Int sendRankB = Mod( B.DistRank()+colDiff, distSize );
    const Int recvRankA = Mod( A.DistRank()-colDiff, distSize );
    const Int recvRankB =
      (recvRankA/A.PartialColStride())+
      (recvRankA%A.PartialColStride())*A.PartialUnionColStride();
    copy::Exchange_impl<T,D>( A, B, sendRankB, recvRankB, B.DistComm() );
}

template<typename T,Dist U,Dist V,Device D>
void RowwiseVectorExchange
( DistMatrix<T,STAR,ProductDist<U,V>(),ELEMENT,D> const& A,
  DistMatrix<T,STAR,ProductDist<V,U>(),ELEMENT,D>& B )
{
    EL_DEBUG_CSE
    AssertSameGrids( A, B );

    if( !B.Participating() )
        return;

    const Int distSize = A.DistSize();
    const Int rowDiff = A.RowShift() - B.RowShift();
    const Int sendRankB = Mod( B.DistRank()+rowDiff, distSize );
    const Int recvRankA = Mod( A.DistRank()-rowDiff, distSize );
    const Int recvRankB =
      (recvRankA/A.PartialRowStride())+
      (recvRankA%A.PartialRowStride())*A.PartialUnionRowStride();
    copy::Exchange_impl<T,D>( A, B, sendRankB, recvRankB, B.DistComm() );
}

} // namespace copy
} // namespace El

#endif // ifndef EL_BLAS_COPY_EXCHANGE_HPP
