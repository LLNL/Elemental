/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_COPY_TRANSLATEBETWEENGRIDS_HPP
#define EL_BLAS_COPY_TRANSLATEBETWEENGRIDS_HPP

#include "core/environment/decl.hpp"
#include <optional>

namespace El
{
namespace copy
{

template<typename T,Dist U,Dist V,Device D1,Device D2>
void TranslateBetweenGrids(
  DistMatrix<T,U,V,ELEMENT,D1> const& A,
  DistMatrix<T,U,V,ELEMENT,D2>& B)
{
    EL_DEBUG_CSE
    DistMatrix<T,CIRC,CIRC,ELEMENT,D2> ACirc(A.Grid(), A.Root()), BCirc(B.Grid(), B.Root());
    ACirc.Resize(A.Height(), A.Width());
    BCirc.Resize(A.Height(), A.Width());
    B.Resize(A.Height(), A.Width());
    if (A.Participating()) {
      El::Copy(A, ACirc);
    }
    TranslateBetweenGrids(ACirc, BCirc);
    if (B.Participating()) {
      El::Copy(BCirc, B);
    }
}

template<typename T, Device D>
void TranslateBetweenGrids(
  DistMatrix<T,CIRC,CIRC,ELEMENT,D> const& A,
  DistMatrix<T,CIRC,CIRC,ELEMENT,D>& B)
{

  // Matrix dimensions
  const Int m = A.Height();
  const Int n = A.Width();
  B.Resize(m, n);
  if (m <= 0 || n <= 0) {
    return;
  }

  const bool amRootA = A.IsLocalCol(0);
  const bool amRootB = B.IsLocalCol(0);
  mpi::Comm const& viewingCommB = B.Grid().ViewingComm();
  if (amRootA && amRootB) {
    El::Copy(A.LockedMatrix(), B.Matrix());
  }
  else if (amRootA) {
    const Int recvViewingRank = B.Grid().VCToViewing(B.Root());
    El::Send(A.LockedMatrix(), viewingCommB, recvViewingRank);
  }
  else if (amRootB) {
    mpi::Group owningGroupA = A.Grid().OwningGroup();
    const Int sendViewingRank = mpi::Translate(
      owningGroupA,
      A.Root(),
      viewingCommB);
    if (sendViewingRank < 0 || sendViewingRank >= viewingCommB.Size()) {
      LogicError(
        "TranslateBetweenGrids: Owning group for matrix A "
        "is not a subset of viewing communicator for matrix B");
    }
    El::Recv(B.Matrix(), viewingCommB, sendViewingRank);
  }

}

template<typename T, Device D>
void TranslateBetweenGrids(
  DistMatrix<T,MC,MR,ELEMENT,D> const& A,
  DistMatrix<T,MC,MR,ELEMENT,D>& B)
{
  EL_DEBUG_CSE;

  // See TranslateBetweenGrids implementation for [STAR,VC] matrices
  // for a description of the algorithm. The [MC,MR] matrix
  // implementation is similar, except it is 2D.

  // Matrix dimensions
  const Int m = A.Height();
  const Int n = A.Width();
  B.Resize(m, n);
  const Int mLocA = A.LocalHeight();
  const Int nLocA = A.LocalWidth();
  const Int mLocB = B.LocalHeight();
  const Int nLocB = B.LocalWidth();

  // Return immediately if there is no local data
  const bool inAGrid = A.Participating();
  const bool inBGrid = B.Participating();
  if (!inAGrid && !inBGrid) {
    return;
  }

  // Compute the number of messages to send/recv
  const Int colStrideA = A.ColStride();
  const Int rowStrideA = A.RowStride();
  const Int colStrideB = B.ColStride();
  const Int rowStrideB = B.RowStride();
  const Int colStrideGCD = GCD(colStrideA, colStrideB);
  const Int rowStrideGCD = GCD(rowStrideA, rowStrideB);
  const Int numColSends = Min(colStrideB/colStrideGCD, mLocA);
  const Int numRowSends = Min(rowStrideB/rowStrideGCD, nLocA);
  const Int numColRecvs = Min(colStrideA/colStrideGCD, mLocB);
  const Int numRowRecvs = Min(rowStrideA/rowStrideGCD, nLocB);

  // Synchronize compute streams
  SyncInfo<D> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
  SyncInfo<D> syncInfoB = SyncInfoFromMatrix(B.Matrix());
  auto syncHelper = MakeMultiSync(syncInfoB, syncInfoA);
  const SyncInfo<D>& syncInfo = syncHelper;

  // Translate the ranks from A's VC communicator to B's viewing so
  // that we can match send/recv communicators. Since A's VC
  // communicator is not necessarily defined on every process, we
  // instead work with A's owning group.
  mpi::Comm const& viewingCommB = B.Grid().ViewingComm();
  mpi::Group owningGroupA = A.Grid().OwningGroup();
  const int sizeA = A.Grid().Size();
  vector<int> viewingRanksA(sizeA), owningRanksA(sizeA);
  if(A.Grid().Order() == COLUMN_MAJOR) {
    std::iota(owningRanksA.begin(), owningRanksA.end(), 0);
  }
  else {
    // The (i,j) = i + j*colStrideA rank in the column-major ordering is
    // equal to the j + i*rowStrideA rank in a row-major ordering.
    // Since we desire rankMap[i+j*colStrideA] to correspond to process
    // (i,j) in A's grid's rank in this viewing group, ranks[i+j*colStrideA]
    // should correspond to process (i,j) in A's owning group. Since the
    // owning group is ordered row-major in this case, its rank is
    // j+i*rowStrideA. Note that setting
    // ranks[j+i*rowStrideA] = i+j*colStrideA is *NOT* valid.
    for (int i=0; i<colStrideA; ++i) {
      for (int j=0; j<rowStrideA; ++j) {
        owningRanksA[i+j*colStrideA] = j+i*rowStrideA;
      }
    }
  }
  mpi::Translate(
    owningGroupA, sizeA, owningRanksA.data(),
    viewingCommB, viewingRanksA.data());
  const Int colRankA = A.ColRank();
  const Int rowRankA = A.RowRank();
  const Int colRankB = B.ColRank();
  const Int rowRankB = B.RowRank();
  const Int viewingRank = viewingCommB.Rank();
  if (viewingRank < 0 || viewingRank >= viewingCommB.Size()) {
    LogicError(
      "TranslateBetweenGrids: Owning group for matrix A "
      "is not a subset of viewing communicator for matrix B");
  }

  // Workspace buffer to pack/unpack data
  Int messageBufSize = 0;
  if (inAGrid && numColSends > 0 && numRowSends > 0) {
    messageBufSize = Max(
      MaxLength(mLocA,numColSends)*MaxLength(nLocA,numRowSends),
      messageBufSize);
  }
  if (inBGrid && numColRecvs > 0 && numRowRecvs > 0) {
    messageBufSize = Max(
      MaxLength(mLocB,numColRecvs)*MaxLength(nLocB,numRowRecvs),
      messageBufSize);
  }
  simple_buffer<T,D> messageBuf(messageBufSize, syncInfo);

  // Figure out which ranks to send/recv messages
  std::map<Int,std::pair<Int,Int>> messageColRanks, messageRowRanks;
  for (Int iLocA=0; iLocA<numColSends; ++iLocA) {
    const Int i = A.GlobalRow(iLocA);
    const Int recvColRank = B.RowOwner(i);
    messageColRanks[i] = std::pair<Int,Int>(colRankA, recvColRank);
  }
  for (Int jLocA=0; jLocA<numRowSends; ++jLocA) {
    const Int j = A.GlobalCol(jLocA);
    const Int recvRowRank = B.ColOwner(j);
    messageRowRanks[j] = std::pair<Int,Int>(rowRankA, recvRowRank);
  }
  for (Int iLocB=0; iLocB<numColRecvs; ++iLocB) {
    const Int i = B.GlobalRow(iLocB);
    const Int sendColRank = A.RowOwner(i);
    messageColRanks[i] = std::pair<Int,Int>(sendColRank, colRankB);
  }
  for (Int jLocB=0; jLocB<numRowRecvs; ++jLocB) {
    const Int j = B.GlobalCol(jLocB);
    const Int sendRowRank = A.ColOwner(j);
    messageRowRanks[j] = std::pair<Int,Int>(sendRowRank, rowRankB);
  }

  // Send/recv messages
  for (auto const& rowInfo : messageRowRanks) {
    for (auto const& colInfo : messageColRanks) {
      const Int i = colInfo.first;
      const Int j = rowInfo.first;
      const Int sendColRank = colInfo.second.first;
      const Int sendRowRank = rowInfo.second.first;
      const Int recvColRank = colInfo.second.second;
      const Int recvRowRank = rowInfo.second.second;

      // Figure out source and target ranks
      const Int sendVCRank = sendColRank + sendRowRank*colStrideA;
      const Int recvVCRank = recvColRank + recvRowRank*colStrideB;
      const Int sendViewingRank = viewingRanksA[sendVCRank];
      const Int recvViewingRank = B.Grid().VCToViewing(recvVCRank);

      // Figure out message size and offset
      Int iLocA = -1;
      Int jLocA = -1;
      Int iLocB = -1;
      Int jLocB = -1;
      Int messageHeight = 0;
      Int messageWidth = 0;
      if (viewingRank == sendViewingRank) {
        iLocA = A.LocalRow(i);
        jLocA = A.LocalCol(j);
        messageHeight = Length(mLocA, iLocA, numColSends);
        messageWidth = Length(nLocA, jLocA, numRowSends);
      }
      if (viewingRank == recvViewingRank) {
        iLocB = B.LocalRow(i);
        jLocB = B.LocalCol(j);
        messageHeight = Length(mLocB, iLocB, numColRecvs);
        messageWidth = Length(nLocB, jLocB, numRowRecvs);
      }

      if (viewingRank == sendViewingRank && viewingRank == recvViewingRank) {
        // Copy data locally
        copy::util::InterleaveMatrix(
          messageHeight, messageWidth,
          A.LockedBuffer(iLocA,jLocA), numColSends, numRowSends*A.LDim(),
          B.Buffer(iLocB,jLocB), numColRecvs, numRowRecvs*B.LDim(),
          syncInfo);
      }
      else if (viewingRank == sendViewingRank) {
        // Send data to other rank
        copy::util::InterleaveMatrix(
          messageHeight, messageWidth,
          A.LockedBuffer(iLocA,jLocA), numColSends, numRowSends*A.LDim(),
          messageBuf.data(), 1, messageHeight,
          syncInfo);
        mpi::Send(
          messageBuf.data(), messageHeight*messageWidth,
          recvViewingRank, viewingCommB, syncInfo);
      }
      else if (viewingRank == recvViewingRank) {
        // Receive data from other rank
        mpi::Recv(
          messageBuf.data(), messageHeight*messageWidth,
          sendViewingRank, viewingCommB, syncInfo);
        copy::util::InterleaveMatrix(
          messageHeight, messageWidth,
          messageBuf.data(), 1, messageHeight,
          B.Buffer(iLocB,jLocB), numColRecvs, numRowRecvs*B.LDim(),
          syncInfo);
      }

    }
  }

}

template<typename T, Device D1, Device D2>
void TranslateBetweenGridsAllreduceBasic
(DistMatrix<T,STAR,VC,ELEMENT,D1> & A,
    std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector)
{
    //<T,STAR,VC,ELEMENT,D2>
    /*
    Logically, the values in B_vector are summed together and copied to A.
    This function is specific to the LBANN with implementation for specific cases.

    Subgrids in B_vector are assumed to evenly divide the grid in A.
    This is a basic allreduce implementation with no overlapped communication.
    */
    EL_DEBUG_CSE

    Int indexB = -1;
    const Int numSubGrids = int(B_Vector.size());

    for(Int i = 0; i<numSubGrids; ++i)
    {
        if((*B_Vector[i]).Participating())
        {
            if(indexB!=-1)
            {
                LogicError("TranslateBetweenGridsAllreduceBasic: ",
                   "Error: rank is in multiple subgrids");
            }
            indexB = i;
        }
    }

    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>(&(*B_Vector[indexB]));

    const Int rowStrideB = B->RowStride();
    const Int sizeB = B->Grid().VCSize();
    const Int m = B->Height();
    const Int n = B->Width();
    const Int mLocB = B->LocalHeight();
    const Int nLocB = B->LocalWidth();

    const Int posInSubGrid = B->Grid().VCRank();
    const Int myLocalRankB = posInSubGrid;
    const Int posInGrid = A.Grid().VCRank();
    A.Resize(m,n);

    mpi::Comm const& viewingCommA = A.Grid().ViewingComm();

    Int rowStrideA = A.RowStride();
    const Int sizeA = A.Grid().VCSize();
    const Int rowGCD = GCD(rowStrideB, rowStrideA);
    const Int rowLCM = rowStrideB*rowStrideA / rowGCD;

    //Asserting parent grid evenly divides number of columns
    if(n%sizeA!=0)
    {
        LogicError("TranslateBetweenGridsAllreduceBasic: ",
                   "Number of columns should be evenly divided by the size of the grid A (parent grid)");
    }

    // Parent Subgrid Size: 4 Child Subgrid Size: 3
    // Parent 0 1 2 3 0 1 2 3 0 1 2 3
    // Child  0 1 2 0 1 2 0 1 2 0 1 2

    std::vector<bool> require_data(sizeB,false);
    std::vector<int> index_to_put(sizeB,-1);
    std::vector<int> index_from(sizeB,-1);
    Int temp_require_data = Mod(posInGrid, sizeB) ;
    double total_iter = posInGrid;

    for(Int i=0; i< int(rowLCM/sizeA); ++i)
    {

        require_data[temp_require_data] = true;
        index_to_put[temp_require_data] = i;
        index_from[temp_require_data] = int(std::floor(total_iter/sizeB));

        total_iter += sizeA;
        temp_require_data = Mod(temp_require_data+sizeA, sizeB );
    }

    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
    SyncInfo<D2> syncInfoB = SyncInfoFromMatrix(B->LockedMatrix());
    SyncInfo<D1> syncGeneral = SyncInfo<D1>();

    //

    const bool inAGrid = A.Participating();
    const bool inBGrid = indexB >=0 ? true:false;

    const Int maxSendSize = mLocB * nLocB;

    simple_buffer<T,D1> send_buf(maxSendSize , syncInfoB);
    simple_buffer<T,D2> recv_buf(maxSendSize , syncInfoA);

    T* sendBuf = send_buf.data();
    T* recvBuf = recv_buf.data();

    for(Int localRankB = 0; localRankB < sizeB; localRankB++)
    {
        if(myLocalRankB==localRankB && inBGrid)
        {
            copy::util::InterleaveMatrix(
                mLocB, nLocB,
                B->LockedBuffer(0,0),
                1, B->LDim(),
                sendBuf, 1, mLocB, syncInfoB);

        }
        else
        {
            T val = El::To<T>(0);
            hydrogen::details::setBufferToValue(sendBuf, maxSendSize, val,syncGeneral);
        }

        Synchronize(syncGeneral);
        mpi::AllReduce( sendBuf, recvBuf, maxSendSize, mpi::SUM, viewingCommA,syncInfoB);


        if(inAGrid)
        {
            if(require_data[localRankB])
            {
                int sendWidth = int(n / rowLCM);
                copy::util::InterleaveMatrix(
                                m, sendWidth,
                                recvBuf + (index_from[localRankB] * m ) , 1, m*(rowLCM/sizeB),
                                A.Buffer(0,index_to_put[localRankB]),
                                1, (rowLCM/sizeA)*A.LDim(),
                                syncInfoA);

            }
            Synchronize(syncInfoA);
        }



    }
    Synchronize(syncInfoA);
    Synchronize(syncInfoB);


}

template<typename T, Device D1, Device D2>
void TranslateBetweenGridsAllreduceOptComm
(DistMatrix<T,STAR,VC,ELEMENT,D1> & A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, mpi::Comm const& allreduceComm, SyncInfo<D1> & syncGeneral)
{
    //<T,STAR,VC,ELEMENT,D2>
    /*
    Logically, the values in B_vector are summed together and copied to A.
    This function is specific to LBANN sub-graph parallelism with implementations for
    specific cases.

    Subgrids in B_vector are assumed to evenly divide the grid in A.
    This is a segmented allreduce implementation that requires a
    a communicator over corresponding processes in the subgrids.
    */
    EL_DEBUG_CSE

    Int indexB = -1;
    const Int numSubGrids = int(B_Vector.size());

    for(Int i = 0; i<numSubGrids; ++i)
    {
        if((*B_Vector[i]).Participating())
        {
            if(indexB!=-1)
            {
                LogicError("TranslateBetweenGridsAllreduceOptComm: ",
                   "Error: rank is in multiple subgrids");
            }
            indexB = i;
        }
    }

    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>(&(*B_Vector[indexB]));

    const Int rowStrideB = B->RowStride();
    const Int sizeB = B->Grid().VCSize();

    const Int m = B->Height();
    const Int n = B->Width();
    const Int mLocB = B->LocalHeight();
    const Int nLocB = B->LocalWidth();


    const Int posInGrid = A.Grid().VCRank();
    A.Resize(m,n);

    Int rowStrideA = A.RowStride();
    const Int sizeA = A.Grid().VCSize();
    const Int rowGCD = GCD(rowStrideB, rowStrideA);
    const Int rowLCM = rowStrideB*rowStrideA / rowGCD;

    //Asserting parent grid evenly divides number of columns
    if(n%sizeA!=0)
    {
        LogicError("TranslateBetweenGridsAllreduceOptComm: ",
                   "Number of columns should be evenly divided by the size of the grid A (parent grid)");
    }


    // Parent Subgrid Size: 4 Child Subgrid Size: 3
    // Parent 0 1 2 3 0 1 2 3 0 1 2 3
    // Child  0 1 2 0 1 2 0 1 2 0 1 2



    const Int index_from = int(std::floor(posInGrid/sizeB));
    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
    SyncInfo<D2> syncInfoB = SyncInfoFromMatrix(B->LockedMatrix());


    const bool inAGrid = A.Participating();
    const bool inBGrid = indexB >=0 ? true:false;

    const Int maxSendSize = mLocB * nLocB;

    simple_buffer<T,D1> send_buf(maxSendSize, syncInfoB);
    simple_buffer<T,D2> recv_buf(maxSendSize, syncInfoA);

    T* sendBuf = send_buf.data();
    T* recvBuf = recv_buf.data();

    if(inBGrid)
    {
        copy::util::InterleaveMatrix(
                mLocB, nLocB,
                B->LockedBuffer(0,0),
                1, B->LDim(),
                sendBuf, 1, mLocB, syncGeneral);

    }
    //Synchronize(syncGeneral);
    mpi::AllReduce( sendBuf, recvBuf, maxSendSize, mpi::SUM, allreduceComm,syncGeneral);

    if(inAGrid)
    {
        int sendWidth = int(n / rowLCM);
        copy::util::InterleaveMatrix(
                m, sendWidth,
                recvBuf + (index_from * m ) , 1, m*(rowLCM/sizeB),
                A.Buffer(0,0),
                1, (rowLCM/sizeA)*A.LDim(),
                syncGeneral);

    }

}



template<typename T, Device D1, Device D2>
void TranslateBetweenGridsAllreduceOpt
(DistMatrix<T,STAR,VC,ELEMENT,D1> & A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector)
{
    //<T,STAR,VC,ELEMENT,D2>
    /*
    Logically, the values in B_vector are summed together and copied to A.
    This function is specific to LBANN sub-graph parallelism with implementations for
    specific cases.

    Subgrids in B_vector are assumed to evenly divide the grid in A.
    This is a segmented allreduce implementation that internally splits a communicator.
    Note that communicator splitting is expensive on GPU.
    */
    EL_DEBUG_CSE

    Int indexB = -1;
    const Int numSubGrids = int(B_Vector.size());

    for(Int i = 0; i<numSubGrids; ++i)
    {
        if((*B_Vector[i]).Participating())
        {
            if(indexB!=-1)
            {
                LogicError("TranslateBetweenGridsAllreduceOpt: ",
                   "Error: rank is in multiple subgrids");
            }
            indexB = i;
        }
    }

    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));
    const Int rowStrideB = B->RowStride();
    const Int sizeB = B->Grid().VCSize();


    const Int m = B->Height();
    const Int n = B->Width();
    const Int mLocB = B->LocalHeight();
    const Int nLocB = B->LocalWidth();

    const Int posInSubGrid = B->Grid().VCRank();

    const Int posInGrid = A.Grid().ViewingRank();

    A.Resize(m,n);



    mpi::Comm const& viewingCommA = A.Grid().ViewingComm();
    Int rowStrideA = A.RowStride();
    const Int sizeA = A.Grid().VCSize();
    const Int rowGCD = GCD(rowStrideB, rowStrideA);
    const Int rowLCM = rowStrideB*rowStrideA / rowGCD;

    //Asserting parent grid evenly divides number of columns
    if(n%sizeA!=0)
    {
        LogicError("TranslateBetweenGridsAllreduceOpt: ",
                   "Number of columns should be evenly divided by the size of the grid A (parent grid)");
    }

    // Parent Subgrid Size: 4 Child Subgrid Size: 3
    // Parent 0 1 2 3 0 1 2 3 0 1 2 3
    // Child  0 1 2 0 1 2 0 1 2 0 1 2



    const Int index_from = int(std::floor(posInGrid/sizeB));

    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
    SyncInfo<D2> syncInfoB = SyncInfoFromMatrix(B->LockedMatrix());
    SyncInfo<D1> syncGeneral = SyncInfo<D1>();

    const bool inAGrid = A.Participating();
    const bool inBGrid = indexB >=0 ? true:false;

    const Int maxSendSize = mLocB * nLocB;

    simple_buffer<T,D1> send_buf(maxSendSize, syncInfoB);
    simple_buffer<T,D2> recv_buf(maxSendSize, syncInfoA);

    T* sendBuf = send_buf.data();
    T* recvBuf = recv_buf.data();

    mpi::Comm allreduceComm;

    mpi::Split(viewingCommA, posInSubGrid, posInGrid, allreduceComm);

    if(inBGrid)
    {
        copy::util::InterleaveMatrix(
                mLocB, nLocB,
                B->LockedBuffer(0,0),
                1, B->LDim(),
                sendBuf, 1, mLocB, syncGeneral);

    }
    //Synchronize(syncGeneral);
    mpi::AllReduce( sendBuf, recvBuf, maxSendSize, mpi::SUM, allreduceComm,syncGeneral);

    if(inAGrid)
    {
        int sendWidth = int(n / rowLCM);
        copy::util::InterleaveMatrix(
                m, sendWidth,
                recvBuf + (index_from * m ) , 1, m*(rowLCM/sizeB),
                A.Buffer(0,0),
                1, (rowLCM/sizeA)*A.LDim(),
                syncGeneral);

    }

}

template<typename T, Device D1, Device D2>
void TranslateBetweenGridsAllreduce
(DistMatrix<T,STAR,VC,ELEMENT,D1> & A,
    std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, int version)

{
    //Some better logic can be written here
    // 0: Best Algo
    // 1: Basic Allreduce without communication overlap
    // 2: Opt Allreduce but have large overhead for GPU (spliting comm)
    //


    if(version==0 || version == 1)
    {
        TranslateBetweenGridsAllreduceBasic<T, D1, D2>(A,
                                                B_Vector);
    }
    else if (version == 2)
    {
        TranslateBetweenGridsAllreduceOpt<T, D1, D2>(A,
                                            B_Vector);
    }
    else
    {
        LogicError("TranslateBetweenGridsAllreduce: ",
                   "Invalid version, it has to be [0,1,2], 0: Default");
    }
}

template<typename T, Device D1, Device D2>
void TranslateBetweenGridsAllreduce
(DistMatrix<T,STAR,VC,ELEMENT,D1> & A,
    std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, mpi::Comm const& allreduceComm, SyncInfo<D1> & syncGeneral, int version)

{
    //Some better logic can be written here
    // 0: Best Algo
    // 1: if spliting comm is given then we have only one algo for now
    //


    if(version==0 || version == 1)
    {
        TranslateBetweenGridsAllreduceOptComm<T, D1, D2>(A,
                                                B_Vector,
                                                allreduceComm,
                                                syncGeneral);
    }

    else
    {
        LogicError("TranslateBetweenGridsAllreduce: ",
                   "Invalid version, it has to be [0,1], 0: Default");
    }
}

template<typename T, Device D1, Device D2>
void TranslateBetweenGridsScatterComm
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, int splitDim,  mpi::Comm const& ScatterComm, SyncInfo<D1> & syncGeneral)
{
    /*
    Scatter data in Column-Major ordering along the last dimension
    Last dimension should be divisible number of child layers
    Size of B_vector is equal to the number of child layers

    Subgrids in B_vector are assumed to be subset of resources in A grid


    Resources are assumed to be distribted equally among different subgrids


    */
    EL_DEBUG_CSE
    const Int m = A.Height();
    const Int n = A.Width();
    const Int mLocA = A.LocalHeight();
    const Int nLocA = A.LocalWidth();



    Int rowStrideA = A.RowStride();


    const Int numChildLayers = int(B_Vector.size());
    const Int sizeA = A.Grid().VCSize();

    for(Int i=0; i<numChildLayers;++i)
    {
        B_Vector[i]->Resize(int(m/numChildLayers),n);
    }


    if(m%splitDim != 0)
    {
        LogicError("TranslateBetweenGridsScatterComm: ",
                   "feature dimension should be divisible by split dimension");
    }
    if(splitDim%numChildLayers!=0)
    {
        LogicError("TranslateBetweenGridsScatterComm: ",
                   "Split dimension must be divisible by number of children layers or number of splits");
    }

    //Asserting parent grid evenly divides number of columns
    if(n%sizeA!=0)
    {
        LogicError("TranslateBetweenGridsScatterComm: ",
                   "Number of columns should be evenly divided by the size of the grid A (parent grid)");
    }


    const Int scatterCommSize = mpi::Size( ScatterComm );
    const int sendCounts = int((mLocA*nLocA)/scatterCommSize);
    const int numMatricesInSubGrid  = int(numChildLayers / scatterCommSize);

    std::vector<Int> indexBVec;

    for(Int i = 0; i<numChildLayers; ++i)
    {
        if(B_Vector[i]->Participating())
        {
            indexBVec.push_back(i);

        }
    }
    const Int indexB = indexBVec[0];
    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));
    Matrix<T,D2> conversionMatrix(mLocA,nLocA), transposedMatrix(int((mLocA*nLocA)/splitDim),splitDim), recvTransposedMatrix(int((mLocA*nLocA)/splitDim),int(splitDim/scatterCommSize));

    Copy(A.LockedMatrix(), conversionMatrix);

    conversionMatrix.Resize(splitDim, int((mLocA*nLocA)/splitDim));

    Transpose(conversionMatrix,transposedMatrix);

    conversionMatrix.Resize(int(mLocA/scatterCommSize),nLocA);

    const Int rowStrideB = B->RowStride();
    const Int sizeB = B->Grid().VCSize();
    const Int rowGCD = GCD(rowStrideB, rowStrideA);
    const Int rowLCM = rowStrideB*rowStrideA / rowGCD;



    // Parent Subgrid Size: 4 Child Subgrid Size: 3
    // Parent 0 1 2 3 0 1 2 3 0 1 2 3
    // Child  0 1 2 0 1 2 0 1 2 0 1 2
    std::vector<int> index_to_put(sizeA,-1);

    for(Int i = 0; i < int(rowLCM/sizeB); ++i)
    {
        index_to_put[i] = i;
    }


    SyncInfo<D2> syncInfoB = SyncInfoFromMatrix(recvTransposedMatrix);


    int partialHeight = int(splitDim/ scatterCommSize);
    int partialChildHeight = int(partialHeight / numMatricesInSubGrid);

    conversionMatrix.Resize(partialChildHeight,nLocA);

    for(Int localDataRankA = 0; localDataRankA < int(rowLCM/sizeB); localDataRankA++)
    {

        //comm is useless parameter in this function
        //Aluminum infer comm from sync object



        mpi::Scatter((T *)transposedMatrix.Buffer(), sendCounts, (T *)recvTransposedMatrix.Buffer(), sendCounts, localDataRankA, ScatterComm,
               syncGeneral);

        Synchronize(syncGeneral);

        int sendWidth = int(n / rowLCM);

        for(Int childLayerSubGrid = 0; childLayerSubGrid < numMatricesInSubGrid; ++childLayerSubGrid)
        {
            Transpose(recvTransposedMatrix( Range<Int>(0,sendWidth*(mLocA/splitDim)),Range<Int>(partialChildHeight*childLayerSubGrid, partialChildHeight*(childLayerSubGrid+1))),conversionMatrix);
            copy::util::InterleaveMatrix(
                        partialChildHeight*(mLocA/splitDim), sendWidth,
                        conversionMatrix.Buffer()  , 1, partialChildHeight*(mLocA/splitDim),
                        dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->Buffer(0,localDataRankA),
                        1, dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->LDim() * int(rowLCM/sizeB),
                        syncInfoB);
        }


        Synchronize(syncInfoB);



    }

}



// FIX ME using interleave operation for transpose
template<typename T, Device D1, Device D2>
void TranslateBetweenGridsScatterOptComm
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, int splitDim,  mpi::Comm const& ScatterComm, SyncInfo<D1> & syncGeneral)
{
    /*
    Scatter data in Column-Major ordering along the last dimension
    Last dimension should be divisible number of child layers
    Size of B_vector is equal to the number of child layers

    Subgrids in B_vector are assumed to be subset of resources in A grid


    Resources are assumed to be distribted equally among different subgrids


    */
    EL_DEBUG_CSE
    const Int m = A.Height();
    const Int n = A.Width();
    const Int mLocA = A.LocalHeight();
    const Int nLocA = A.LocalWidth();



    Int rowStrideA = A.RowStride();



    const Int numChildLayers = int(B_Vector.size());
    const Int sizeA = A.Grid().VCSize();

    for(Int i=0; i<numChildLayers;++i)
    {
        B_Vector[i]->Resize(int(m/numChildLayers),n);
    }


    if(m%splitDim != 0)
    {
        LogicError("TranslateBetweenGridsScatterOptComm: ",
                   "feature dimension should be divisible by split dimension");
    }
    if(splitDim%numChildLayers!=0)
    {
        LogicError("TranslateBetweenGridsScatterOptComm: ",
                   "Split dimension must be divisible by number of children layers or number of splits");
    }

    //Asserting parent grid evenly divides number of columns
    if(n%sizeA!=0)
    {
        LogicError("TranslateBetweenGridsScatterOptComm: ",
                   "Number of columns should be evenly divided by the size of the grid A (parent grid)");
    }


    //const Int myRankViewing = mpi::Rank(viewingCommA);
    const Int scatterCommSize = mpi::Size( ScatterComm );
    const int sendCounts = int((mLocA*nLocA)/scatterCommSize);
    const int numMatricesInSubGrid  = int(numChildLayers / scatterCommSize);

    std::vector<Int> indexBVec;
    std::vector<SyncInfo<D2>> syncInfoBVector;

    for(Int i = 0; i<numChildLayers; ++i)
    {
        if(B_Vector[i]->Participating())
        {
            indexBVec.push_back(i);
            syncInfoBVector.push_back(SyncInfoFromMatrix(dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[i]))->LockedMatrix()) );

        }
    }
    const Int indexB = indexBVec[0];
    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));
    Matrix<T,D2>  transposedMatrix(int((mLocA*nLocA)/splitDim),splitDim), recvTransposedMatrix(int((mLocA*nLocA)/splitDim),int(splitDim/scatterCommSize));

    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());

    const Int maxSendSize = mLocA*nLocA;



    simple_buffer<T,D2> send_buf(maxSendSize, syncInfoA);


    T* sendBuf = send_buf.data();




    copy::util::InterleaveMatrix(
                        splitDim, int((mLocA*nLocA)/splitDim),
                        A.LockedBuffer()  , 1, splitDim,
                        sendBuf,
                        int((mLocA*nLocA)/splitDim),1,
                        syncInfoA);

    const Int rowStrideB = B->RowStride();
    const Int sizeB = B->Grid().VCSize();
    const Int rowGCD = GCD(rowStrideB, rowStrideA);
    const Int rowLCM = rowStrideB*rowStrideA / rowGCD;



    // Parent Subgrid Size: 4 Child Subgrid Size: 3
    // Parent 0 1 2 3 0 1 2 3 0 1 2 3
    // Child  0 1 2 0 1 2 0 1 2 0 1 2


    std::vector<int> index_to_put(sizeA,-1);

    for(Int i = 0; i < int(rowLCM/sizeB); ++i)
    {
        index_to_put[i] = i;
    }


    int partialHeight = int(splitDim/ scatterCommSize);
    int partialChildHeight = int(partialHeight / numMatricesInSubGrid);
    const Int localSubgridHeight = int(splitDim/ scatterCommSize);
    const Int recvLocalHeight =  (mLocA/splitDim) * (localSubgridHeight/numMatricesInSubGrid);
    int sendWidth = int(n / rowLCM);

    std::vector<Matrix<T,D2>> conversionMatrixVector;
    for(Int childLayerSubGrid = 0; childLayerSubGrid < numMatricesInSubGrid; ++childLayerSubGrid)
    {
        conversionMatrixVector.push_back(Matrix<T,D2>(recvLocalHeight, sendWidth));
    }




    for(Int localDataRankA = 0; localDataRankA < int(rowLCM/sizeB); localDataRankA++)
    {

        mpi::Scatter(
          sendBuf, sendCounts,
          recvTransposedMatrix.Buffer(), sendCounts,
          localDataRankA, ScatterComm, syncGeneral);

        for(Int childLayerSubGrid = 0; childLayerSubGrid < numMatricesInSubGrid; ++childLayerSubGrid)
        {

            Transpose(recvTransposedMatrix( Range<Int>(0,sendWidth*(mLocA/splitDim)),Range<Int>(partialChildHeight*childLayerSubGrid, partialChildHeight*(childLayerSubGrid+1))),conversionMatrixVector[childLayerSubGrid]);

            copy::util::InterleaveMatrix(
                        partialChildHeight*(mLocA/splitDim), sendWidth,
                        conversionMatrixVector[childLayerSubGrid].Buffer()  , 1, partialChildHeight*(mLocA/splitDim),
                        dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->Buffer(0,localDataRankA),
                        1, dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->LDim() * int(rowLCM/sizeB),
                        syncInfoBVector[childLayerSubGrid]);

        }




    }


}




template<typename T, Device D1, Device D2>
void TranslateBetweenGridsSliceGatherOptComm
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, int splitDim,  mpi::Comm const& gatherComm, SyncInfo<D1> & syncGeneral)
{
    /*
    Scatter data in Column-Major ordering along the last dimension
    Last dimension should be divisible number of child layers
    Size of B_vector is equal to the number of child layers

    Subgrids in B_vector are assumed to be subset of resources in A grid
    Resources are assumed to be distribted equally among different subgrids

    Uses Allgather to perform slice layer functionality as AllGather has better implementation
    */
    EL_DEBUG_CSE
    const Int m = A.Height();
    const Int n = A.Width();
    const Int mLocA = A.LocalHeight();
    const Int nLocA = A.LocalWidth();



    Int rowStrideA = A.RowStride();

    const Int numChildLayers = int(B_Vector.size());
    const Int sizeA = A.Grid().VCSize();

    for(Int i=0; i<numChildLayers;++i)
    {
        B_Vector[i]->Resize(int(m/numChildLayers),n);
    }


    if(m%splitDim != 0)
    {
        LogicError("TranslateBetweenGridsSliceGatherOptComm: ",
                   "feature dimension should be divisible by split dimension");
    }
    if(splitDim%numChildLayers!=0)
    {
        LogicError("TranslateBetweenGridsSliceGatherOptComm: ",
                   "Split dimension must be divisible by number of children layers or number of splits");
    }
    //Asserting parent grid evenly divides number of columns
    if(n%sizeA!=0)
    {
        LogicError("TranslateBetweenGridsSliceGatherOptComm: ",
                   "Number of columns should be evenly divided by the size of the grid A (parent grid)");
    }

    const Int gatherCommSize = mpi::Size( gatherComm );

    const int numMatricesInSubGrid  = int(numChildLayers / gatherCommSize);

    std::vector<Int> indexBVec;
    std::vector<SyncInfo<D2>> syncInfoBVector;

    for(Int i = 0; i<numChildLayers; ++i)
    {
        if(B_Vector[i]->Participating())
        {
            indexBVec.push_back(i);
            syncInfoBVector.push_back(SyncInfoFromMatrix(dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[i]))->LockedMatrix()) );

        }
    }
    const Int indexB = indexBVec[0];
    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));
    Matrix<T,D2>  transposedMatrix(int((mLocA*nLocA)/splitDim),splitDim), recvTransposedMatrix(int((mLocA*nLocA)/splitDim),int(splitDim/gatherCommSize)), sendTransposedMatrix(nLocA,mLocA);



    const Int maxRecvSize = int(mLocA*nLocA*gatherCommSize);
    simple_buffer<T,D2> recv_buf(maxRecvSize, syncGeneral);
    T* recvBuf = recv_buf.data();




    Transpose(A.LockedMatrix(),sendTransposedMatrix);

    const Int rowStrideB = B->RowStride();
    const Int sizeB = B->Grid().VCSize();


    const Int rowGCD = GCD(rowStrideB, rowStrideA);
    const Int rowLCM = rowStrideB*rowStrideA / rowGCD;

    const Int posInGrid = A.Grid().VCRank();
    const Int subgridNumber = int(std::floor(posInGrid/sizeB));



    // Parent Subgrid Size: 4 Child Subgrid Size: 3
    // Parent 0 1 2 3 0 1 2 3 0 1 2 3
    // Child  0 1 2 0 1 2 0 1 2 0 1 2

    std::vector<int> index_to_put(sizeA,-1);




    for(Int i = 0; i < int(rowLCM/sizeB); ++i)
    {
        index_to_put[i] = i;
    }



    const Int perSubgridSplitHeight = splitDim / gatherCommSize;
    const Int childLayerSplitHeight = perSubgridSplitHeight / numMatricesInSubGrid;

    std::vector<Matrix<T,D2>> conversionMatrixVector;
    for(Int childLayerSubGrid = 0; childLayerSubGrid < numMatricesInSubGrid; ++childLayerSubGrid)
    {
        conversionMatrixVector.push_back(Matrix<T,D2>(childLayerSplitHeight * nLocA * (mLocA/splitDim), gatherCommSize));
    }


    mpi::AllGather(sendTransposedMatrix.Buffer(), mLocA*nLocA,
                    recvBuf, mLocA*nLocA,
                    gatherComm, syncGeneral);



    Matrix<T,D2>  tempMatrix(gatherCommSize , childLayerSplitHeight * nLocA * (mLocA/splitDim));

    for(Int childLayerSubGrid = 0; childLayerSubGrid < numMatricesInSubGrid; ++childLayerSubGrid)
    {
        const Int colNumberChildLayer = childLayerSubGrid + (numMatricesInSubGrid * subgridNumber);

        //strieded copy to get appropriate rows (scattering the data among child layers)
        copy::util::InterleaveMatrix(
            childLayerSplitHeight * nLocA, (mLocA/splitDim)*gatherCommSize,
            recvBuf + colNumberChildLayer * childLayerSplitHeight * nLocA  , 1, childLayerSplitHeight * nLocA * numChildLayers,
            conversionMatrixVector[childLayerSubGrid].Buffer(),
            1, childLayerSplitHeight * nLocA,
            syncGeneral);
        Transpose(conversionMatrixVector[childLayerSubGrid], tempMatrix);


        tempMatrix.Resize(nLocA * gatherCommSize , childLayerSplitHeight * (mLocA/splitDim));

        Transpose(tempMatrix,dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->Matrix());


    }

}



template<typename T, Device D1, Device D2>
void TranslateBetweenGridsScatterCommParentSmall
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, int splitDim,  mpi::Comm const& ScatterComm, SyncInfo<D1> & syncGeneral)
{
    /*
    Scatter data in Column-Major ordering along the last dimension
    Last dimension should be divisible number of child layers
    Size of B_vector is equal to the number of child layers

    A : 0 1 2 3
    B1: 0 1
    B2: 2 3
    B3: 4 5
    B4: 6 7


    */
    EL_DEBUG_CSE
    Int m = A.Height();
    Int n = A.Width();
    Int mLocA = A.LocalHeight();
    Int nLocA = A.LocalWidth();

    mpi::Comm const& viewingCommA = A.Grid().ViewingComm();



    Int rowStrideA = A.RowStride();


    const Int numChildLayers = int(B_Vector.size());
    const Int sizeA = A.Grid().VCSize();



    SyncInfo<D1> syncGeneralMetaData = SyncInfo<D1>();
    const bool inAGrid = A.Participating();

    Int recvMetaData[4];

    Int metaData[4];
    if(inAGrid)
    {

        metaData[0] = m;
        metaData[1] = n;
        metaData[2] = mLocA;
        metaData[3] = nLocA;
    }
    else
    {
        metaData[0] = 0;
        metaData[1] = 0;
        metaData[2] = 0;
        metaData[3] = 0;
    }

    const std::vector<Int> sendMetaData (metaData,metaData + 4 );

    Synchronize(syncGeneral);

    mpi::AllReduce( sendMetaData.data(), recvMetaData, 4, mpi::MAX, viewingCommA,syncGeneralMetaData);
    Synchronize(syncGeneralMetaData);

    m = recvMetaData[0];
    n = recvMetaData[1];
    mLocA = recvMetaData[2];
    nLocA = recvMetaData[3];


    for(Int i=0; i<numChildLayers;++i)
    {
        B_Vector[i]->Resize(int(m/numChildLayers),n);
    }


    if(m%splitDim != 0)
    {
        LogicError("TranslateBetweenGridsScatterCommParentSmall: ",
                   "feature dimension should be divisible by split dimension");
    }
    if(splitDim%numChildLayers!=0)
    {
        LogicError("TranslateBetweenGridsScatterCommParentSmall: ",
                   "Split dimension must be divisible by number of children layers or number of splits");
    }
    //Asserting parent grid evenly divides number of columns
    if(n%sizeA!=0)
    {
        LogicError("TranslateBetweenGridsScatterCommParentSmall: ",
                   "Number of columns should be evenly divided by the size of the grid A (parent grid)");
    }


    const Int scatterCommSize = mpi::Size( ScatterComm );
    const int sendCounts = int((mLocA*nLocA)/scatterCommSize);
    const int numMatricesInSubGrid  = int(numChildLayers / scatterCommSize);

    std::vector<Int> indexBVec;

    for(Int i = 0; i<numChildLayers; ++i)

    {
        if(B_Vector[i]->Participating())
        {
            indexBVec.push_back(i);

        }
    }
    const Int indexB = indexBVec[0];

    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));
    Matrix<T,D2> conversionMatrix(mLocA,nLocA), transposedMatrix(int((mLocA*nLocA)/splitDim),splitDim), recvTransposedMatrix(int((mLocA*nLocA)/splitDim),int(splitDim/scatterCommSize));

    if(inAGrid)
    {
        Copy(A.LockedMatrix(), conversionMatrix);

        conversionMatrix.Resize(splitDim, int((mLocA*nLocA)/splitDim));

        Transpose(conversionMatrix,transposedMatrix);
    }


    conversionMatrix.Resize(int(mLocA/scatterCommSize),nLocA);

    const Int rowStrideB = B->RowStride();
    const Int sizeB = B->Grid().VCSize();
    const Int rowGCD = GCD(rowStrideB, rowStrideA);
    const Int rowLCM = rowStrideB*rowStrideA / rowGCD;




    if(sizeA%sizeB!= 0)
    {
        LogicError("TranslateBetweenGridsScatterCommParentSmall: ",
                   "Number of processes in A grid should be divided by subgrid size");
    }


    // Parent Subgrid Size: 4 Child Subgrid Size: 3
    // Parent 0 1 2 3 0 1 2 3 0 1 2 3
    // Child  0 1 2 0 1 2 0 1 2 0 1 2
    std::vector<int> index_to_put(sizeA,-1);

    for(Int i = 0; i < int(rowLCM/sizeB); ++i)
    {
        index_to_put[i] = i;
    }


    SyncInfo<D2> syncInfoB = SyncInfoFromMatrix(recvTransposedMatrix);


    int partialHeight = int(splitDim/ scatterCommSize);
    int partialChildHeight = int(partialHeight / numMatricesInSubGrid);

    conversionMatrix.Resize(partialChildHeight,nLocA);

    for(Int localDataRankA = 0; localDataRankA < int(rowLCM/sizeB); localDataRankA++)
    {

        //comm is useless parameter in this function
        //Aluminum infer comm from sync object



        mpi::Scatter((T *)transposedMatrix.Buffer(), sendCounts, (T *)recvTransposedMatrix.Buffer(), sendCounts, localDataRankA, ScatterComm,
               syncGeneral);

        Synchronize(syncGeneral);

        int sendWidth = int(n / rowLCM);

        for(Int childLayerSubGrid = 0; childLayerSubGrid < numMatricesInSubGrid; ++childLayerSubGrid)
        {
            Transpose(recvTransposedMatrix( Range<Int>(0,sendWidth*(mLocA/splitDim)),Range<Int>(partialChildHeight*childLayerSubGrid, partialChildHeight*(childLayerSubGrid+1))),conversionMatrix);
            copy::util::InterleaveMatrix(
                        partialChildHeight*(mLocA/splitDim), sendWidth,
                        conversionMatrix.Buffer()  , 1, partialChildHeight*(mLocA/splitDim),
                        dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->Buffer(0,localDataRankA),
                        1, dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->LDim() * int(rowLCM/sizeB),
                        syncInfoB);
        }


        Synchronize(syncInfoB);



    }

}



template<typename T, Device D1, Device D2>
void TranslateBetweenGridsSliceGatherParentSmall
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, int splitDim,  mpi::Comm const& gatherComm, SyncInfo<D1> & syncGeneral)
{
    /*
    Scatter data in Column-Major ordering along the last dimension
    Last dimension should be divisible number of child layers
    Size of B_vector is equal to the number of child layers

    A : 0 1 2 3
    B1: 0 1
    B2: 2 3
    B3: 4 5
    B4: 6 7

    Uses Allgather to perform slice layer functionality as AllGather has better implementation
    */
    EL_DEBUG_CSE
    Int m = A.Height();
    Int n = A.Width();
    Int mLocA = A.LocalHeight();
    Int nLocA = A.LocalWidth();

    mpi::Comm const& viewingCommA = A.Grid().ViewingComm();




    const Int numChildLayers = int(B_Vector.size());
    const Int sizeA = A.Grid().VCSize();




    if(m%splitDim != 0)
    {
        LogicError("TranslateBetweenGridsSliceGatherParentSmall: ",
                   "feature dimension should be divisible by split dimension");
    }
    if(splitDim%numChildLayers!=0)
    {
        LogicError("TranslateBetweenGridsSliceGatherParentSmall: ",
                   "Split dimension must be divisible by number of children layers or number of splits");
    }

    SyncInfo<D1> syncGeneralMetaData = SyncInfo<D1>();
    const bool inAGrid = A.Participating();

    Int recvMetaData[6];

    Int metaData[6];
    if(inAGrid)
    {

        metaData[0] = m;
        metaData[1] = n;
        metaData[2] = mLocA;
        metaData[3] = nLocA;
    }
    else
    {
        metaData[0] = 0;
        metaData[1] = 0;
        metaData[2] = 0;
        metaData[3] = 0;
    }



    const std::vector<Int> sendMetaData (metaData,metaData + 4 );



    Synchronize(syncGeneral);


    mpi::AllReduce( sendMetaData.data(), recvMetaData, 4, mpi::MAX, viewingCommA,syncGeneralMetaData);
    Synchronize(syncGeneralMetaData);

    m = recvMetaData[0];
    n = recvMetaData[1];
    mLocA = recvMetaData[2];
    nLocA = recvMetaData[3];


    //Asserting parent grid evenly divides number of columns
    if(n%sizeA!=0)
    {
        LogicError("TranslateBetweenGridsSliceGatherParentSmall: ",
                   "Number of columns should be evenly divided by the size of the grid A (parent grid)");
    }


    for(Int i=0; i<numChildLayers;++i)
    {
        B_Vector[i]->Resize(int(m/numChildLayers),n);
    }

    const Int gatherCommSize = mpi::Size( gatherComm );
    //number of relevant ranks from which data should be received

    const int numMatricesInSubGrid  = int(numChildLayers / gatherCommSize);

    std::vector<Int> indexBVec;
    std::vector<SyncInfo<D2>> syncInfoBVector;

    for(Int i = 0; i<numChildLayers; ++i)
    {
        if(B_Vector[i]->Participating())
        {
            indexBVec.push_back(i);
            syncInfoBVector.push_back(SyncInfoFromMatrix(dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[i]))->LockedMatrix()) );

        }
    }
    const Int indexB = indexBVec[0];
    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));
    Matrix<T,D2>  transposedMatrix(int((mLocA*nLocA)/splitDim),splitDim), recvTransposedMatrix(int((mLocA*nLocA)/splitDim),int(splitDim/gatherCommSize)), sendTransposedMatrix(nLocA,mLocA);



    const Int maxRecvSize = int(mLocA*nLocA*gatherCommSize);

    simple_buffer<T,D2> recv_buf(maxRecvSize, syncGeneral);



    T* recvBuf = recv_buf.data();



    if(inAGrid)
    {
        Transpose(A.LockedMatrix(),sendTransposedMatrix);
    }





    const Int sizeB = B->Grid().VCSize();
    const Int numRanksToRecv = sizeA / sizeB;





    // const Int posInGrid = A.Grid().VCRank();
    // const Int subgridNumber = int(std::floor(posInGrid/sizeB));
    const Int subgridNumber = El::mpi::Rank(gatherComm);


    if(sizeA%sizeB!= 0)
    {
        LogicError("TranslateBetweenGridsSliceGatherParentSmall: ",
                   "Number of processes in A grid should be divided by subgrid size");
    }



    // Parent Subgrid Size: 4 Child Subgrid Size: 3
    // Parent 0 1 2 3 0 1 2 3 0 1 2 3
    // Child  0 1 2 0 1 2 0 1 2 0 1 2






    const Int perSubgridSplitHeight = splitDim / gatherCommSize;
    const Int childLayerSplitHeight = perSubgridSplitHeight / numMatricesInSubGrid;

    std::vector<Matrix<T,D2>> conversionMatrixVector;
    for(Int childLayerSubGrid = 0; childLayerSubGrid < numMatricesInSubGrid; ++childLayerSubGrid)
    {
        conversionMatrixVector.push_back(Matrix<T,D2>(childLayerSplitHeight * nLocA * (mLocA/splitDim), numRanksToRecv));
    }


    mpi::AllGather(sendTransposedMatrix.Buffer(), mLocA*nLocA,
                    recvBuf, mLocA*nLocA,
                    gatherComm, syncGeneral);

    Matrix<T,D2>  tempMatrix(numRanksToRecv , childLayerSplitHeight * nLocA * (mLocA/splitDim));

    for(Int childLayerSubGrid = 0; childLayerSubGrid < numMatricesInSubGrid; ++childLayerSubGrid)
    {
        const Int colNumberChildLayer = childLayerSubGrid + (numMatricesInSubGrid * subgridNumber);

        //strieded copy to get appropriate rows (scattering the data among child layers)
        copy::util::InterleaveMatrix(
            childLayerSplitHeight * nLocA, (mLocA/splitDim)*numRanksToRecv,
            recvBuf + colNumberChildLayer * childLayerSplitHeight * nLocA  , 1, childLayerSplitHeight * nLocA * numChildLayers,
            conversionMatrixVector[childLayerSubGrid].Buffer(),
            1, childLayerSplitHeight * nLocA,
            syncGeneral);


        Transpose(conversionMatrixVector[childLayerSubGrid], tempMatrix);

        tempMatrix.Resize(nLocA * numRanksToRecv , childLayerSplitHeight * (mLocA/splitDim));

        Transpose(tempMatrix,dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->Matrix());


    }

}

template<typename T, Device D1, Device D2>
void TranslateBetweenGridsScatterCommSameSizeSubGrids
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, int splitDim,  mpi::Comm const& ScatterComm, SyncInfo<D1> & syncGeneral)
{
    /*
    Scatter data in Column-Major ordering along the last dimension
    Last dimension should be divisible number of child layers
    Size of B_vector is equal to the number of child layers

    Subgrids in B_vector are assumed to be subset of resources in A grid

    Parent Grid: 1 2 3 4 (must match the ranks of subgrid1)
    Subgrid1:  1  2  3  4
    Subgrid2:  5  6  7  8
    Subgrid3:  9 10 11 12
    Subgrid4: 13 14 15 16
    */
    EL_DEBUG_CSE
    Int m = A.Height();
    Int n = A.Width();
    Int mLocA = A.LocalHeight();
    Int nLocA = A.LocalWidth();
    const Int sizeA = A.Grid().VCSize();

    mpi::Comm const& viewingCommA = A.Grid().ViewingComm();






    const Int numChildLayers = int(B_Vector.size());



    SyncInfo<D1> syncGeneralMetaData = SyncInfo<D1>();
    const bool inAGrid = A.Participating();


    Int recvMetaData[6];

    Int metaData[6];
    if(inAGrid)
    {

        metaData[0] = m;
        metaData[1] = n;
        metaData[2] = mLocA;
        metaData[3] = nLocA;
    }
    else
    {
        metaData[0] = 0;
        metaData[1] = 0;
        metaData[2] = 0;
        metaData[3] = 0;
    }

    const std::vector<Int> sendMetaData (metaData,metaData + 4 );



    Synchronize(syncGeneral);


    mpi::AllReduce( sendMetaData.data(), recvMetaData, 4, mpi::MAX, viewingCommA,syncGeneralMetaData);
    Synchronize(syncGeneralMetaData);

    m = recvMetaData[0];
    n = recvMetaData[1];
    mLocA = recvMetaData[2];
    nLocA = recvMetaData[3];



    for(Int i=0; i<numChildLayers;++i)
    {
        B_Vector[i]->Resize(int(m/numChildLayers),n);
    }


    if( (inAGrid ==true && B_Vector[0]->Participating()==false) ||
        (inAGrid ==false && B_Vector[0]->Participating()==true) )
    {
        //A grid should have same ranks as of subgrid1
        LogicError("TranslateBetweenGridsScatteSameSizeSubGrids: ",
                   "Owning ranks in Grid A should be same as of Owning Ranks in Subgrid 1");

    }

    if(m%splitDim != 0)
    {
        LogicError("TranslateBetweenGridsScatteSameSizeSubGrids: ",
                   "feature dimension should be divisible by split dimension");
    }
    if(splitDim%numChildLayers!=0)
    {
        LogicError("TranslateBetweenGridsScatteSameSizeSubGrids: ",
                   "Split dimension must be divisible by number of children layers or number of splits");
    }

    //Asserting parent grid evenly divides number of columns
    if(n%sizeA!=0)
    {
        LogicError("TranslateBetweenGridsScatteSameSizeSubGrids: ",
                   "Number of columns should be evenly divided by the size of the grid A (parent grid)");
    }


    const Int scatterCommSize = mpi::Size( ScatterComm );
    const int sendCounts = int((mLocA*nLocA)/scatterCommSize);
    const int numMatricesInSubGrid  = int(numChildLayers / scatterCommSize);

    std::vector<Int> indexBVec;

    for(Int i = 0; i<numChildLayers; ++i)
    {
        if(B_Vector[i]->Participating())
        {
            indexBVec.push_back(i);

        }
    }


    Matrix<T,D2> conversionMatrix(mLocA,nLocA), transposedMatrix(int((mLocA*nLocA)/splitDim),splitDim), tempMatrix((mLocA/splitDim) * nLocA, splitDim/numChildLayers);


    simple_buffer<T,D2> recv_buf(sendCounts, syncGeneral);
    T* recvBuf = recv_buf.data();


    if(inAGrid)
    {
        Copy(A.LockedMatrix(), conversionMatrix);

        conversionMatrix.Resize(splitDim, int((mLocA*nLocA)/splitDim));

        Transpose(conversionMatrix,transposedMatrix);



    }




    conversionMatrix.Resize((mLocA/splitDim) * nLocA, splitDim/numChildLayers);



    mpi::Scatter(   (T *)transposedMatrix.Buffer(), sendCounts,
                    recvBuf, sendCounts,
                    0, ScatterComm,
                    syncGeneral);

    Synchronize(syncGeneral);
    int perChildLayerSize = (mLocA/splitDim) * nLocA * (splitDim/numChildLayers);

    for(Int childLayerSubGrid = 0; childLayerSubGrid < numMatricesInSubGrid; ++childLayerSubGrid)
    {


        copy::util::InterleaveMatrix(
            (mLocA/splitDim) * nLocA, splitDim/numChildLayers,
            recvBuf + perChildLayerSize *childLayerSubGrid , 1, (mLocA/splitDim) * nLocA,
            conversionMatrix.Buffer(),
            1, (mLocA/splitDim) * nLocA,
            syncGeneral);



        Transpose(conversionMatrix, tempMatrix);


        tempMatrix.Resize( mLocA / (numChildLayers) , nLocA);

        copy::util::InterleaveMatrix(
            mLocA / (numChildLayers), nLocA,
            tempMatrix.Buffer() , 1, mLocA / (numChildLayers),
            dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->Buffer(),
            1, mLocA / (numChildLayers),
            syncGeneral);
    }
}


template<typename T, Device D1, Device D2>
void TranslateBetweenGridsSliceBroadcastCommSameSizeSubGrids
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, int splitDim,  mpi::Comm const& ScatterComm, SyncInfo<D1> & syncGeneral)
{
    /*
    Scatter data in Column-Major ordering along the last dimension
    Last dimension should be divisible number of child layers
    Size of B_vector is equal to the number of child layers

    Subgrids in B_vector are assumed to be subset of resources in A grid

    Parent Grid: 1 2 3 4 (must match the ranks of subgrid1)
    Subgrid1:  1  2  3  4
    Subgrid2:  5  6  7  8
    Subgrid3:  9 10 11 12
    Subgrid4: 13 14 15 16
    */
    EL_DEBUG_CSE
    Int m = A.Height();
    Int n = A.Width();
    Int mLocA = A.LocalHeight();
    Int nLocA = A.LocalWidth();
    const Int sizeA = A.Grid().VCSize();

    mpi::Comm const& viewingCommA = A.Grid().ViewingComm();






    const Int numChildLayers = int(B_Vector.size());



    SyncInfo<D1> syncGeneralMetaData = SyncInfo<D1>();
    const bool inAGrid = A.Participating();


    Int recvMetaData[6];

    Int metaData[6];
    if(inAGrid)
    {

        metaData[0] = m;
        metaData[1] = n;
        metaData[2] = mLocA;
        metaData[3] = nLocA;
    }
    else
    {
        metaData[0] = 0;
        metaData[1] = 0;
        metaData[2] = 0;
        metaData[3] = 0;
    }

    const std::vector<Int> sendMetaData (metaData,metaData + 4 );



    Synchronize(syncGeneral);


    mpi::AllReduce( sendMetaData.data(), recvMetaData, 4, mpi::MAX, viewingCommA,syncGeneralMetaData);
    Synchronize(syncGeneralMetaData);

    m = recvMetaData[0];
    n = recvMetaData[1];
    mLocA = recvMetaData[2];
    nLocA = recvMetaData[3];



    for(Int i=0; i<numChildLayers;++i)
    {
        B_Vector[i]->Resize(int(m/numChildLayers),n);
    }


    if( (inAGrid ==true && B_Vector[0]->Participating()==false) ||
        (inAGrid ==false && B_Vector[0]->Participating()==true) )
    {
        //A gird should have same ranks as of subgrid1
        LogicError("TranslateBetweenGridsScatteSameSizeSubGrids: ",
                   "Owning ranks in Grid A should be same as of Owning Ranks in Subgrid 1");

    }

    if(m%splitDim != 0)
    {
        LogicError("TranslateBetweenGridsScatteSameSizeSubGrids: ",
                   "feature dimension should be divisible by split dimension");
    }
    if(splitDim%numChildLayers!=0)
    {
        LogicError("TranslateBetweenGridsScatteSameSizeSubGrids: ",
                   "Split dimension must be divisible by number of children layers or number of splits");
    }
    //Asserting parent grid evenly divides number of columns
    if(n%sizeA!=0)
    {
        LogicError("TranslateBetweenGridsScatteSameSizeSubGrids: ",
                   "Number of columns should be evenly divided by the size of the grid A (parent grid)");
    }


    const Int scatterCommSize = mpi::Size( ScatterComm );
    const int sendCounts = mLocA*nLocA;
    const int numMatricesInSubGrid  = int(numChildLayers / scatterCommSize);

    std::vector<Int> indexBVec;

    for(Int i = 0; i<numChildLayers; ++i)
    {
        if(B_Vector[i]->Participating())
        {
            indexBVec.push_back(i);

        }
    }

    Matrix<T,D2> conversionMatrix(mLocA,nLocA), transposedMatrix(int((mLocA*nLocA)/splitDim),splitDim), tempMatrix((mLocA/splitDim) * nLocA, splitDim/numChildLayers);

    if(inAGrid)
    {
        Copy(A.LockedMatrix(), conversionMatrix);

        conversionMatrix.Resize(splitDim, int((mLocA*nLocA)/splitDim));

        Transpose(conversionMatrix,transposedMatrix);
    }





    const Int posInScatter = mpi::Rank(ScatterComm);



    conversionMatrix.Resize((mLocA/splitDim) * nLocA, splitDim/numChildLayers);





    mpi::Broadcast((T *)transposedMatrix.Buffer(), sendCounts, 0, ScatterComm,
               syncGeneral);




    Synchronize(syncGeneral);
    int perChildLayerSize = (mLocA/splitDim) * nLocA * (splitDim/numChildLayers);



    for(Int childLayerSubGrid = 0; childLayerSubGrid < numMatricesInSubGrid; ++childLayerSubGrid)
    {
        int bufferOffset = (perChildLayerSize *childLayerSubGrid) + (numMatricesInSubGrid * posInScatter * perChildLayerSize );


        copy::util::InterleaveMatrix(
            (mLocA/splitDim) * nLocA, splitDim/numChildLayers,
            transposedMatrix.Buffer() + bufferOffset, 1, (mLocA/splitDim) * nLocA,
            conversionMatrix.Buffer(),
            1, (mLocA/splitDim) * nLocA,
            syncGeneral);



        Transpose(conversionMatrix, tempMatrix);


        tempMatrix.Resize( mLocA / (numChildLayers) , nLocA);

        copy::util::InterleaveMatrix(
            mLocA / (numChildLayers), nLocA,
            tempMatrix.Buffer() , 1, mLocA / (numChildLayers),
            dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->Buffer(),
            1, mLocA / (numChildLayers),
            syncGeneral);
    }
}

template<typename T, Device D1, Device D2>
void TranslateBetweenGridsSliceConcatAlongFirstDim
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, int splitDim,  mpi::Comm const& gatherComm, SyncInfo<D1> & syncGeneral)
{
    /*
    Scatter data in Column-Major ordering along the last dimension
    when given data is sliced along the first dimension
    Last dimension should be divisible number of child layers
    Size of B_vector is equal to the number of child layers

    Subgrids in B_vector are assumed to be subset of resources in A grid
    Resources are assumed to be distribted equally among different subgrids

    Uses Allgather to perform slice layer functionality as AllGather has better implementation
    */
    EL_DEBUG_CSE

    // In this case, m!=mlocA and n==nlocA
    const Int m = A.Height();
    const Int n = A.Width();
    const Int mLocA = A.LocalHeight();
    const Int nLocA = A.LocalWidth();






    const Int numChildLayers = int(B_Vector.size());
    const Int sizeA = A.Grid().VCSize();

    for(Int i=0; i<numChildLayers;++i)
    {
        B_Vector[i]->Resize(int(m/numChildLayers),n);
    }


    if(m%splitDim != 0)
    {
        LogicError("TranslateBetweenGridsSliceConcatAlongFirstDim: ",
                   "feature dimension should be divisible by split dimension");
    }
    if(splitDim%numChildLayers!=0)
    {
        LogicError("TranslateBetweenGridsSliceConcatAlongFirstDim: ",
                   "Split dimension must be divisible by number of children layers or number of splits");
    }
    //Asserting parent grid evenly divides number of columns
    if(n%sizeA!=0)
    {
        LogicError("TranslateBetweenGridsSliceConcatAlongFirstDim: ",
                   "Number of columns should be evenly divided by the size of the grid A (parent grid)");
    }

    const Int gatherCommSize = mpi::Size( gatherComm );

    const int numMatricesInSubGrid  = int(numChildLayers / gatherCommSize);

    std::vector<Int> indexBVec;
    std::vector<SyncInfo<D2>> syncInfoBVector;

    for(Int i = 0; i<numChildLayers; ++i)
    {
        if(B_Vector[i]->Participating())
        {
            indexBVec.push_back(i);
            syncInfoBVector.push_back(SyncInfoFromMatrix(dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[i]))->LockedMatrix()) );

        }
    }
    const Int indexB = indexBVec[0];
    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));
    Matrix<T,D2>  transposedMatrix(int((mLocA*nLocA)/splitDim),splitDim), recvTransposedMatrix(nLocA, mLocA*gatherCommSize), sendTransposedMatrix(nLocA,mLocA);

    Transpose(A.LockedMatrix(),sendTransposedMatrix);

    const Int sizeB = B->Grid().VCSize();
    const Int posInGrid = A.Grid().VCRank();
    const Int subgridNumber = int(std::floor(posInGrid/sizeB));


    std::vector<int> index_to_put(sizeA,-1);



    const Int childLayerSplitHeight = splitDim / numChildLayers;



    std::vector<Matrix<T,D2>> conversionMatrixVector;
    for(Int childLayerSubGrid = 0; childLayerSubGrid < numMatricesInSubGrid; ++childLayerSubGrid)
    {

        // conversionMatrixVector.push_back(Matrix<T,D2>(nLocA, (mLocA*gatherCommSize)/numChildLayers));
        conversionMatrixVector.push_back(Matrix<T,D2>(nLocA, (mLocA)/numChildLayers));
    }


    mpi::AllGather(sendTransposedMatrix.Buffer(), mLocA*nLocA,
                    recvTransposedMatrix.Buffer(), mLocA*nLocA,
                    gatherComm, syncGeneral);





    for(Int childLayerSubGrid = 0; childLayerSubGrid < numMatricesInSubGrid; ++childLayerSubGrid)
    {
        const Int colNumberChildLayer = childLayerSubGrid + (numMatricesInSubGrid * subgridNumber);

        //strieded copy to get appropriate rows (scattering the data among child layers)
        copy::util::InterleaveMatrix(
            // childLayerSplitHeight * nLocA, (mLocA/splitDim)*gatherCommSize,
            childLayerSplitHeight * nLocA, (mLocA/splitDim),
            recvTransposedMatrix.Buffer() + colNumberChildLayer * childLayerSplitHeight * nLocA  , 1, childLayerSplitHeight * nLocA * numChildLayers,
            conversionMatrixVector[childLayerSubGrid].Buffer(),
            1, childLayerSplitHeight * nLocA,
            syncGeneral);

        // Matrix<T,D2>  tempMatrix( (mLocA*gatherCommSize)/numChildLayers, nLocA);
        Matrix<T,D2>  tempMatrix( (mLocA)/numChildLayers, nLocA);
        Transpose(conversionMatrixVector[childLayerSubGrid], tempMatrix);

        std::cout << "Temp matrix Height " << tempMatrix.Height()
                  << " width " << tempMatrix.Width()
                  << std::endl;

        copy::util::InterleaveMatrix(
            // childLayerSplitHeight * nLocA, (mLocA/splitDim)*gatherCommSize,
            mLocA/numChildLayers, nLocA,
            tempMatrix.Buffer()  , 1, mLocA/numChildLayers,
            dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->Buffer(),
            1, mLocA/numChildLayers,
            syncGeneral);

        // Copy(tempMatrix,dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->Matrix());

    }

}

template<typename T, Device D1, Device D2>
void TranslateBetweenGridsScatter
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, int splitDim,  mpi::Comm const& ScatterComm,  SyncInfo<D1> & syncGeneral , int version)

{
    //Some better logic can be written here
    // 0: Best Algo
    // 1: Basic Scatter
    // 2: Broadcast
    //

    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[0]));

    const Int sizeA = A.Grid().VCSize();
    const Int sizeB = B->Grid().VCSize();
    const Int commSize = El::mpi::Size(ScatterComm);

    if(sizeA == sizeB)
    {
        //SubGrid VCSizc is equal to Parent Grid
        if(version==0 || version == 3)
        {
            TranslateBetweenGridsScatterCommSameSizeSubGrids<T, D1, D2>(A,
                                                    B_Vector,
                                                    splitDim,
                                                    ScatterComm,
                                                    syncGeneral);
        }
        else if (version == 2 || version == 1 )
        {
            TranslateBetweenGridsSliceBroadcastCommSameSizeSubGrids<T, D1, D2>(A,
                                                B_Vector,
                                                splitDim,
                                                ScatterComm,
                                                syncGeneral);
        }
        else
        {
            LogicError("TranslateBetweenGridsScatter: ",
                       "Invalid version, it has to be [0,1,2,3], 0: Default");
        }
    }

    else if (commSize > sizeA/sizeB)
    {

        if(version==0 || version == 3)
        {
            TranslateBetweenGridsScatterCommParentSmall<T, D1, D2>(A,
                                                    B_Vector,
                                                    splitDim,
                                                    ScatterComm,
                                                    syncGeneral);
        }
        else if (version == 2 || version == 1 )
        {
            TranslateBetweenGridsSliceGatherParentSmall<T, D1, D2>(A,
                                                B_Vector,
                                                splitDim,
                                                ScatterComm,
                                                syncGeneral);
        }
        else
        {
            LogicError("TranslateBetweenGridsScatter: ",
                       "Invalid version, it has to be [0,1,2,3], 0: Default");
        }
    }
    // 0: Best Algo
    // 1: Basic Scatter
    // 2: Scater with Opt (interleave for transpose)
    // 3: Gather
    else
    {
        if(version==0 || version == 3)
        {
            TranslateBetweenGridsSliceGatherOptComm<T, D1, D2>(A,
                                                    B_Vector,
                                                    splitDim,
                                                    ScatterComm,
                                                    syncGeneral);
        }
        else if (version == 2)
        {
            TranslateBetweenGridsScatterOptComm<T, D1, D2>(A,
                                                B_Vector,
                                                splitDim,
                                                ScatterComm,
                                                syncGeneral);
        }
        else if(version == 1)
        {
            TranslateBetweenGridsScatterOptComm<T, D1, D2>(A,
                                                B_Vector,
                                                splitDim,
                                                ScatterComm,
                                                syncGeneral);
        }
        else
        {
            LogicError("TranslateBetweenGridsScatter: ",
                       "Invalid version, it has to be [0,1,2,3], 0: Default");
        }
    }

}

template<typename T>
void TranslateBetweenGridsSliceCol
( AbstractDistMatrix<T> const& A,
  AbstractDistMatrix<T> & B)
{
    /*
    Scatter data in Column-Major ordering along the Columns of Elemental matrix

    Used to scatter data from input layer to subgrids in Topology aware design

    Size of B_vector is equal to the Number of subgraph subgrids (not number of branches in subgrpah)

    Subgrids in B_vector are assumed to be subset of resources in A grid

    Resources are assumed to be distributed equally among different subgrids

    It is a local operation. No Communication needed.
    */
    EL_DEBUG_CSE
    const Int numProcesses = A.Grid().VCSize();
    const Int mLocA = A.LocalHeight();
    const Int nLocA = A.LocalWidth();
    const Int m = A.Height();
    const Int n = A.Width();

    const Int numSubGrids = numProcesses / B.Grid().VCSize();
    B.Resize(m,Int(n/numSubGrids));


    if(A.GetLocalDevice()==Device::CPU)
    {
        SyncInfo<Device::CPU> syncInfoB;

        copy::util::InterleaveMatrix(
                mLocA, nLocA,
                A.LockedBuffer()  , 1, mLocA,
                B.Buffer(),
                1, mLocA,
                syncInfoB);
    }
    else
    {
#ifdef HYDROGEN_HAVE_GPU
        auto const& syncInfoB =
            SyncInfoFromMatrix(
                dynamic_cast<Matrix<T, El::Device::GPU> const&>(
                    B.LockedMatrix()));

        copy::util::InterleaveMatrix(
                mLocA, nLocA,
                A.LockedBuffer()  , 1, mLocA,
                B.Buffer(),
                1, mLocA,
                syncInfoB);
#else
        LogicError("Invalid device");
#endif
    }


}

template<typename T, Device D1, Device D2>
void TranslateBetweenGridsSliceColVector
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector)
{
    /*
    Scatter data in Column-Major ordering along the Columns of Elemental matrix

    Used to scatter data from input layer to subgrids in Topology aware design

    Size of B_vector is equal to the Number of subgraph subgrids (not number of branches in subgrpah)

    Subgrids in B_vector are assumed to be subset of resources in A grid

    Resources are assumed to be distribted equally among different subgrids

    It is a local operation. No Communication needed.
    */
    EL_DEBUG_CSE

    const Int numSubGrids = B_Vector.size();

    const Int mLocA = A.LocalHeight();
    const Int nLocA = A.LocalWidth();
    const Int m = A.Height();
    const Int n = A.Width();

    Int bIndex=-1;

    for(Int i=0; i<numSubGrids;++i)
    {
        B_Vector[i]->Resize(m,Int(n/numSubGrids));

        if(B_Vector[i]->Participating())
        {
            if(bIndex==-1)
            {
                bIndex = i;
            }
            else
            {
                LogicError("TranslateBetweenGridsSliceCol: ",
                   "Subgrids should be mutually exclusive");
            }
        }
    }


    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());

    copy::util::InterleaveMatrix(
            mLocA, nLocA,
            A.LockedBuffer()  , 1, mLocA,
            dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[bIndex]))->Buffer(),
            1, mLocA,
            syncInfoA);

}

template<typename T, Device D1, Device D2>
void TranslateBetweenGridsGatherComm
(DistMatrix<T,STAR,VC,ELEMENT,D1> & A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, int splitDim,  mpi::Comm const& gatherComm, SyncInfo<D1> & syncGeneral)
{
    /*
    Gather data in Column-Major ordering along the last dimension

    Size of B_vector is equal to the number of parent layers

    Subgrids in B_vector are assumed to be subset of resources in A grid

    Resources are assumed to be distribted equally among different subgrids

    or

    A : 0 1 2 3
    B1: 0 1
    B2: 2 3
    B3: 4 5
    B4: 6 7


    */
    EL_DEBUG_CSE

    std::vector<Int> indexBVec;
    const Int numParentLayers = int(B_Vector.size());

    for(Int i = 0; i<numParentLayers; ++i)
    {
        if(B_Vector[i]->Participating())
        {
            indexBVec.push_back(i);

        }
    }
    const Int indexB = indexBVec[0];

    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));

    const Int sizeB = B->Grid().VCSize();

    const Int m = B->Height();
    const Int n = B->Width();
    const Int mLocB = B->LocalHeight();
    const Int nLocB = B->LocalWidth();
    const Int posInGrid = A.Grid().VCRank();
    const bool inAGrid = A.Participating();




    const Int sizeA = A.Grid().VCSize();

     const Int index_from = int(std::floor(posInGrid/sizeB));

    A.Resize(m*numParentLayers,n);

    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());

    const Int gatherCommSize = mpi::Size( gatherComm );
    const int numMatricesInSubGrid  = int(numParentLayers / gatherCommSize);

    //mlocB and m should be equal
    if(mLocB!=m)
    {
        LogicError("TranslateBetweenGridsGatherComm: ",
                   "mLocB and m should be same");

    }
    if(n%sizeA!=0)
    {
        LogicError("TranslateBetweenGridsGatherComm: ",
                   "Width must be divisible by number of resources in A matrix");
    }
    if(mLocB%splitDim!=0)
    {
        LogicError("TranslateBetweenGridsGatherComm: ",
                   "Height in B matrix must be divisible by splitDim");

    }
    //Asserting parent grid evenly divides number of columns
    if(n%sizeA!=0)
    {
        LogicError("TranslateBetweenGridsGatherComm: ",
                   "Number of columns should be evenly divided by the size of the grid A (parent grid)");
    }
    const int totalSizeComm = mLocB * numParentLayers * int(n/sizeB);

    const int maxSendSize = mLocB*nLocB * numMatricesInSubGrid;

    simple_buffer<T,D2> send_buf(maxSendSize, syncInfoA);
    T* sendBuf = send_buf.data();


    Matrix<T,D1> conversionMatrix(mLocB,nLocB), transposedMatrix(int((mLocB*nLocB)/splitDim),splitDim), recvTransposedMatrix(int(totalSizeComm/(splitDim*numParentLayers)),splitDim*numParentLayers);

    //SyncInfo<D2> syncInfoConversionMatrix= SyncInfoFromMatrix(conversionMatrix);
    for(Int parentLayerSubGrid = 0; parentLayerSubGrid < numMatricesInSubGrid; ++parentLayerSubGrid)
    {
        Copy(dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[parentLayerSubGrid]]))->LockedMatrix(), conversionMatrix);
        conversionMatrix.Resize(splitDim, int((mLocB*nLocB)/splitDim));
        Transpose(conversionMatrix,transposedMatrix);
        copy::util::InterleaveMatrix(
            int((mLocB*nLocB)/splitDim),splitDim,
            transposedMatrix.Buffer()  , 1, int((mLocB*nLocB)/splitDim),
            sendBuf + parentLayerSubGrid*mLocB*nLocB,
            1, int((mLocB*nLocB)/splitDim),
            syncGeneral);

    }

    mpi::AllGather(sendBuf, mLocB*nLocB*numMatricesInSubGrid,
                    recvTransposedMatrix.Buffer(), mLocB*nLocB*numMatricesInSubGrid,
                    gatherComm, syncGeneral);


    if(inAGrid)
    {
        Matrix<T,D1> resizedRecvMatrix(m*numParentLayers,int(n/sizeB));

        Transpose(recvTransposedMatrix, resizedRecvMatrix);

        copy::util::InterleaveMatrix(
                            m*numParentLayers, int(n/sizeA),
                            resizedRecvMatrix.Buffer() + index_from *m*numParentLayers , 1, m*numParentLayers*(sizeA/sizeB),
                            A.Buffer(0,0),
                            1, m*numParentLayers,
                            syncInfoA);
    }




}

// FIX ME
template<typename T, Device D1, Device D2>
void TranslateBetweenGridsGatherOptComm
(DistMatrix<T,STAR,VC,ELEMENT,D1> & A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, int splitDim,  mpi::Comm const& gatherComm, SyncInfo<D1> & syncGeneral)
{
    /*
    Gather data in Column-Major ordering along the last dimension

    Size of B_vector is equal to the number of parent layers

    Subgrids in B_vector are assumed to be subset of resources in A grid

    Resources are assumed to be distribted equally among different subgrids


    */
    // This Function has some bugs in Interleave function
    EL_DEBUG_CSE

    std::vector<Int> indexBVec;
    std::vector<SyncInfo<D2>> syncInfoBVector;
    const Int numParentLayers = int(B_Vector.size());



    for(Int i = 0; i<numParentLayers; ++i)
    {
        if(B_Vector[i]->Participating())
        {
            indexBVec.push_back(i);
            syncInfoBVector.push_back(SyncInfoFromMatrix(dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[i]))->LockedMatrix()) );


        }
    }
    const Int indexB = indexBVec[0];

    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));

    const Int sizeB = B->Grid().VCSize();

    const Int m = B->Height();
    const Int n = B->Width();
    const Int mLocB = B->LocalHeight();
    const Int nLocB = B->LocalWidth();

    const Int posInGrid = A.Grid().VCRank();


    const Int sizeA = A.Grid().VCSize();

     const Int index_from = int(std::floor(posInGrid/sizeB));

    A.Resize(m*numParentLayers,n);

    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());

    const Int gatherCommSize = mpi::Size( gatherComm );
    const int numMatricesInSubGrid  = int(numParentLayers / gatherCommSize);

    //mlocB and m should be equal
    if(mLocB!=m)
    {
        LogicError("TranslateBetweenGridsGatherOptComm: ",
                   "mLocB and m should be same");


    }
    if(n%sizeA!=0)
    {
        LogicError("TranslateBetweenGridsGatherOptComm: ",
                   "Width must be divisible by number of resources in A matrix");
    }
    if(mLocB%splitDim!=0)
    {
        LogicError("TranslateBetweenGridsGatherOptComm: ",
                   "Height in B matrix must be divisible by splitDim");

    }
    const int totalSizeComm = mLocB * numParentLayers * int(n/sizeB);

    const int maxSendSize = mLocB*nLocB * numMatricesInSubGrid;


    simple_buffer<T,D2> send_buf(maxSendSize, syncGeneral);

    T* sendBuf = send_buf.data();


    Matrix<T,D1> conversionMatrix(mLocB,nLocB), transposedMatrix(int((mLocB*nLocB)/splitDim),splitDim), recvTransposedMatrix(int(totalSizeComm/(splitDim*numParentLayers)),splitDim*numParentLayers);


    for(Int parentLayerSubGrid = 0; parentLayerSubGrid < numMatricesInSubGrid; ++parentLayerSubGrid)
    {

        copy::util::InterleaveMatrix(
            splitDim, int((mLocB*nLocB)/splitDim),
            dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[parentLayerSubGrid]]))->LockedBuffer()  , 1, splitDim,
            sendBuf + parentLayerSubGrid*mLocB*nLocB,
            int((mLocB*nLocB)/splitDim), 1,
            syncInfoBVector[parentLayerSubGrid]);

    }
    Synchronize(syncGeneral);


    mpi::AllGather(sendBuf, mLocB*nLocB*numMatricesInSubGrid,
                    recvTransposedMatrix.Buffer(), mLocB*nLocB*numMatricesInSubGrid,
                    gatherComm, syncGeneral);



    Matrix<T,D1> resizedRecvMatrix(m*numParentLayers,int(n/sizeB));

    Transpose(recvTransposedMatrix, resizedRecvMatrix);

    copy::util::InterleaveMatrix(
                        m*numParentLayers, int(n/sizeA),
                        resizedRecvMatrix.Buffer() + index_from *m*numParentLayers , 1, m*numParentLayers*(sizeA/sizeB),
                        A.Buffer(0,0),
                        1, m*numParentLayers,
                        syncInfoA);



}


template<typename T, Device D1, Device D2>
void TranslateBetweenGridsGatherCommSameSizeSubGrids
(DistMatrix<T,STAR,VC,ELEMENT,D1> & A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, int splitDim,  mpi::Comm const& gatherComm, SyncInfo<D1> & syncGeneral)
{
    /*
    Gather data in Column-Major ordering along the last dimension

    Size of B_vector is equal to the number of parent layers

    Parent Grid: 1 2 3 4 (must match the ranks of subgrid1)
    Subgrid1:  1  2  3  4
    Subgrid2:  5  6  7  8
    Subgrid3:  9 10 11 12
    Subgrid4: 13 14 15 16



    */
    EL_DEBUG_CSE

    std::vector<Int> indexBVec;
    const Int numParentLayers = int(B_Vector.size());

    for(Int i = 0; i<numParentLayers; ++i)
    {
        if(B_Vector[i]->Participating())
        {
            indexBVec.push_back(i);

        }
    }
    const Int indexB = indexBVec[0];

    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));

    const Int sizeB = B->Grid().VCSize();

    const Int m = B->Height();
    const Int n = B->Width();
    const Int mLocB = B->LocalHeight();
    const Int nLocB = B->LocalWidth();

    const bool inAGrid = A.Participating();




    const Int sizeA = A.Grid().VCSize();



    A.Resize(m*numParentLayers,n);

    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());

    const Int gatherCommSize = mpi::Size( gatherComm );
    const int numMatricesInSubGrid  = int(numParentLayers / gatherCommSize);


    if( (inAGrid ==true && B_Vector[0]->Participating()==false) ||
        (inAGrid ==false && B_Vector[0]->Participating()==true) )
    {
        //A gird should have same ranks as of subgrid1
        LogicError("TranslateBetweenGridsGatherCommSameSizeSubGrids: ",
                   "Owning ranks in Grid A/Parent Grid should be same as of Owning Ranks in Subgrid 1");

    }

    //mlocB and m should be equal
    if(mLocB!=m)
    {
        LogicError("TranslateBetweenGridsGatherCommSameSizeSubGrids: ",
                   "mLocB and m should be same");

    }
    if(n%sizeA!=0)
    {
        LogicError("TranslateBetweenGridsGatherCommSameSizeSubGrids: ",
                   "Width must be divisible by number of resources in A matrix");
    }
    if(mLocB%splitDim!=0)
    {
        LogicError("TranslateBetweenGridsGatherCommSameSizeSubGrids: ",
                   "Height in B matrix must be divisible by splitDim");

    }
    const int totalSizeComm = mLocB * numParentLayers * int(n/sizeB);

    const int maxSendSize = mLocB*nLocB * numMatricesInSubGrid;

    simple_buffer<T,D2> send_buf(maxSendSize, syncInfoA);
    T* sendBuf = send_buf.data();


    Matrix<T,D1> conversionMatrix(mLocB,nLocB), transposedMatrix(int((mLocB*nLocB)/splitDim),splitDim), recvTransposedMatrix(int(totalSizeComm/(splitDim*numParentLayers)),splitDim*numParentLayers);

    for(Int parentLayerSubGrid = 0; parentLayerSubGrid < numMatricesInSubGrid; ++parentLayerSubGrid)
    {
        Copy(dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[parentLayerSubGrid]]))->LockedMatrix(), conversionMatrix);
        conversionMatrix.Resize(splitDim, int((mLocB*nLocB)/splitDim));
        Transpose(conversionMatrix,transposedMatrix);
        copy::util::InterleaveMatrix(
            int((mLocB*nLocB)/splitDim),splitDim,
            transposedMatrix.Buffer()  , 1, int((mLocB*nLocB)/splitDim),
            sendBuf + parentLayerSubGrid*mLocB*nLocB,
            1, int((mLocB*nLocB)/splitDim),
            syncGeneral);

    }

    mpi::Gather(sendBuf, mLocB*nLocB*numMatricesInSubGrid,
                    recvTransposedMatrix.Buffer(), mLocB*nLocB*numMatricesInSubGrid,
                    0, gatherComm, syncGeneral);


    if(inAGrid)
    {
        Matrix<T,D1> resizedRecvMatrix(m*numParentLayers,int(n/sizeB));

        Transpose(recvTransposedMatrix, resizedRecvMatrix);

        copy::util::InterleaveMatrix(
                            m*numParentLayers, int(n/sizeB),
                            resizedRecvMatrix.Buffer()  , 1, m*numParentLayers,
                            A.Buffer(0,0),
                            1, m*numParentLayers,
                            syncInfoA);
    }
}





template<typename T, Device D1, Device D2>
void TranslateBetweenGridsAllGatherCommSameSizeSubGrids

(DistMatrix<T,STAR,VC,ELEMENT,D1> & A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, int splitDim,  mpi::Comm const& gatherComm, SyncInfo<D1> & syncGeneral)
{
    /*
    Gather data in Column-Major ordering along the last dimension

    Size of B_vector is equal to the number of parent layers

    Parent Grid: 1 2 3 4 (must match the ranks of subgrid1)
    Subgrid1:  1  2  3  4
    Subgrid2:  5  6  7  8
    Subgrid3:  9 10 11 12
    Subgrid4: 13 14 15 16



    */
    EL_DEBUG_CSE

    std::vector<Int> indexBVec;
    const Int numParentLayers = int(B_Vector.size());

    for(Int i = 0; i<numParentLayers; ++i)
    {
        if(B_Vector[i]->Participating())
        {
            indexBVec.push_back(i);

        }
    }
    const Int indexB = indexBVec[0];

    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));

    const Int sizeB = B->Grid().VCSize();

    const Int m = B->Height();
    const Int n = B->Width();
    const Int mLocB = B->LocalHeight();
    const Int nLocB = B->LocalWidth();

    const bool inAGrid = A.Participating();




    const Int sizeA = A.Grid().VCSize();



    A.Resize(m*numParentLayers,n);

    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());

    const Int gatherCommSize = mpi::Size( gatherComm );
    const int numMatricesInSubGrid  = int(numParentLayers / gatherCommSize);


    if( (inAGrid ==true && B_Vector[0]->Participating()==false) ||
        (inAGrid ==false && B_Vector[0]->Participating()==true) )
    {
        //A gird should have same ranks as of subgrid1
        LogicError("TranslateBetweenGridsAllGatherCommSameSizeSubGrids: ",
                   "Owning ranks in Grid A/Parent Grid should be same as of Owning Ranks in Subgrid 1");

    }

    //mlocB and m should be equal
    if(mLocB!=m)
    {
        LogicError("TranslateBetweenGridsAllGatherCommSameSizeSubGrids: ",
                   "mLocB and m should be same");

    }
    if(n%sizeA!=0)
    {
        LogicError("TranslateBetweenGridsAllGatherCommSameSizeSubGrids: ",
                   "Width must be divisible by number of resources in A matrix");
    }
    if(mLocB%splitDim!=0)
    {
        LogicError("TranslateBetweenGridsAllGatherCommSameSizeSubGrids: ",
                   "Height in B matrix must be divisible by splitDim");

    }
    const int totalSizeComm = mLocB * numParentLayers * int(n/sizeB);

    const int maxSendSize = mLocB*nLocB * numMatricesInSubGrid;

    simple_buffer<T,D2> send_buf(maxSendSize, syncInfoA);
    T* sendBuf = send_buf.data();


    Matrix<T,D1> conversionMatrix(mLocB,nLocB), transposedMatrix(int((mLocB*nLocB)/splitDim),splitDim), recvTransposedMatrix(int(totalSizeComm/(splitDim*numParentLayers)),splitDim*numParentLayers);

    for(Int parentLayerSubGrid = 0; parentLayerSubGrid < numMatricesInSubGrid; ++parentLayerSubGrid)
    {
        Copy(dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[parentLayerSubGrid]]))->LockedMatrix(), conversionMatrix);
        conversionMatrix.Resize(splitDim, int((mLocB*nLocB)/splitDim));
        Transpose(conversionMatrix,transposedMatrix);
        copy::util::InterleaveMatrix(
            int((mLocB*nLocB)/splitDim),splitDim,
            transposedMatrix.Buffer()  , 1, int((mLocB*nLocB)/splitDim),
            sendBuf + parentLayerSubGrid*mLocB*nLocB,
            1, int((mLocB*nLocB)/splitDim),
            syncGeneral);

    }

    mpi::AllGather(sendBuf, mLocB*nLocB*numMatricesInSubGrid,
                    recvTransposedMatrix.Buffer(), mLocB*nLocB*numMatricesInSubGrid,
                    gatherComm, syncGeneral);


    if(inAGrid)
    {
        Matrix<T,D1> resizedRecvMatrix(m*numParentLayers,int(n/sizeB));

        Transpose(recvTransposedMatrix, resizedRecvMatrix);

        copy::util::InterleaveMatrix(
                            m*numParentLayers, int(n/sizeB),
                            resizedRecvMatrix.Buffer()  , 1, m*numParentLayers,
                            A.Buffer(0,0),
                            1, m*numParentLayers,
                            syncInfoA);
    }

}


template<typename T, Device D1, Device D2>
void TranslateBetweenConatSliceFirstChannel
(DistMatrix<T,STAR,VC,ELEMENT,D1> & A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector,
  int splitDim,
  mpi::Comm const& gatherComm,
  SyncInfo<D1> & syncGeneral)
{
    // it Slice the data along first channel of features
    // given the slices alog the last channel
    // Concat layer for D3 design
    //
    EL_DEBUG_CSE
    std::vector<Int> indexBVec;
    const Int numParentLayers = int(B_Vector.size());


    for(Int i = 0; i<numParentLayers; ++i)
    {
        if(B_Vector[i]->Participating())
        {
            indexBVec.push_back(i);


        }
    }
    const Int indexB = indexBVec[0];

    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));

    const Int sizeB = B->Grid().VCSize();

    const Int m = B->Height();
    const Int n = B->Width();
    const Int mLocB = B->LocalHeight();
    const Int nLocB = B->LocalWidth();


    const bool inAGrid = A.Participating();

    // const Int index_from = int(std::floor(posInGrid/sizeB));
    const Int index_from = mpi::Rank(gatherComm);



    A.Resize(m*numParentLayers,n);

    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());

    const Int gatherCommSize = mpi::Size( gatherComm );
    const int numMatricesInSubGrid  = int(numParentLayers / gatherCommSize);

    //mlocB and m should be equal
    if(mLocB!=m)
    {

        LogicError("TranslateBetweenConatSliceFirstChannel: ",
                   "mLocB and m should be same");

    }
    if(mLocB%splitDim!=0)
    {
        LogicError("TranslateBetweenConatSliceFirstChannel: ",

                   "Height in B matrix must be divisible by splitDim");

    }
    const int totalSizeComm = mLocB * numParentLayers * int(n/sizeB);

    const int maxSendSize = mLocB*nLocB * numMatricesInSubGrid;


    simple_buffer<T,D2> send_buf(maxSendSize, syncInfoA);

    T* sendBuf = send_buf.data();


    Matrix<T,D1> conversionMatrix(mLocB,nLocB), transposedMatrix(int((mLocB*nLocB)/splitDim),splitDim), recvTransposedMatrix(int(totalSizeComm/(splitDim*numParentLayers)),splitDim*numParentLayers);


    //SyncInfo<D2> syncInfoConversionMatrix= SyncInfoFromMatrix(conversionMatrix);
    for(Int parentLayerSubGrid = 0; parentLayerSubGrid < numMatricesInSubGrid; ++parentLayerSubGrid)
    {
        Copy(dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[parentLayerSubGrid]]))->LockedMatrix(), conversionMatrix);
        conversionMatrix.Resize(splitDim, int((mLocB*nLocB)/splitDim));
        Transpose(conversionMatrix,transposedMatrix);
        copy::util::InterleaveMatrix(
            int((mLocB*nLocB)/splitDim),splitDim,
            transposedMatrix.Buffer()  , 1, int((mLocB*nLocB)/splitDim),
            sendBuf + parentLayerSubGrid*mLocB*nLocB,
            1, int((mLocB*nLocB)/splitDim),
            syncGeneral);

    }


    mpi::AllGather(sendBuf, mLocB*nLocB*numMatricesInSubGrid,
                    recvTransposedMatrix.Buffer(), mLocB*nLocB*numMatricesInSubGrid,
                    gatherComm, syncGeneral);




    if(inAGrid)
    {
        Matrix<T,D1> resizedRecvMatrix(m*numParentLayers, nLocB);

        Transpose(recvTransposedMatrix, resizedRecvMatrix);


        copy::util::InterleaveMatrix(
                            int((m*numParentLayers)/gatherCommSize), nLocB,
                            resizedRecvMatrix.Buffer() + index_from * int((m*numParentLayers)/gatherCommSize) , 1, m*numParentLayers,
                            A.Buffer(0,0),
                            1, int((m*numParentLayers)/gatherCommSize),
                            syncInfoA);
    }


}


template<typename T, Device D1, Device D2>
void TranslateBetweenGridsGather
(DistMatrix<T,STAR,VC,ELEMENT,D1> & A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, int splitDim,  mpi::Comm const& gatherComm, SyncInfo<D1> & syncGeneral, int version)
{
    //Some better logic can be written here


    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[0]));


    // 0: Best Algo
    // 1: Basic Gather
    // 2: AllGather implementation (NCCL does not have Gather)

    const Int sizeA = A.Grid().VCSize();
    const Int sizeB = B->Grid().VCSize();
    const Int commSize = El::mpi::Size(gatherComm);
    if(sizeA == sizeB)
    {
        if(version==0 || version == 1)
        {
            TranslateBetweenGridsGatherCommSameSizeSubGrids<T, D1, D2>(A,
                                                    B_Vector,
                                                    splitDim,
                                                    gatherComm,
                                                    syncGeneral);
        }
        else if (version == 2)
        {
            TranslateBetweenGridsAllGatherCommSameSizeSubGrids<T, D1, D2>(A,

                                                B_Vector,
                                                splitDim,
                                                gatherComm,
                                                syncGeneral);

        }
        else
        {
            LogicError("TranslateBetweenGridsGather: ",
                       "Invalid version, it has to be [0,1,2], 0: Default");
        }
    }
    // 0: Best Algo
    // 1: Basic Gather
    // 2: Gather with Opt (interleave for transpose) (FIX IT)
    //
    else if (commSize > sizeA/sizeB)
    {
       TranslateBetweenGridsGatherComm<T, D1, D2>(A,
                                                    B_Vector,
                                                    splitDim,
                                                    gatherComm,
                                                    syncGeneral);
    }
    else
    {
        if(version==0 || version == 1)
        {
            TranslateBetweenGridsGatherComm<T, D1, D2>(A,
                                                    B_Vector,
                                                    splitDim,
                                                    gatherComm,
                                                    syncGeneral);
        }
        else if (version == 2)
        {
            TranslateBetweenGridsGatherOptComm<T, D1, D2>(A,
                                                B_Vector,
                                                splitDim,
                                                gatherComm,
                                                syncGeneral);
        }
        else
        {
            LogicError("TranslateBetweenGridsGather: ",
                       "Invalid version, it has to be [0,1,2], 0: Default");
        }
    }





}



template<typename T, Device D1, Device D2>
void TranslateBetweenGridsBroadcastOptComm
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector,  mpi::Comm const& broadcastComm, SyncInfo<D1> & syncGeneral)
{
    //<T,STAR,VC,ELEMENT,D2>
    /*
    This function is specific to the LBANN with implementation for specific cases
    Subgrids in B_vector are assumed to be subset of resources in A grid
    Same terminology as Allreduce functions
    */


    EL_DEBUG_CSE

    Int m = A.Height();
    Int n = A.Width();
    Int mLocA = A.LocalHeight();
    Int nLocA = A.LocalWidth();


    mpi::Comm const& viewingCommA = A.Grid().ViewingComm();
    Int rowStrideA = A.RowStride();

    const Int numSubGrids = int(B_Vector.size());
    const Int sizeA = A.Grid().VCSize();

    //Asserting parent grid evenly divides number of columns
    if(n%sizeA!=0)
    {
        LogicError("TranslateBetweenGridsBroadcastOptComm: ",
                   "Number of columns should be evenly divided by the size of the grid A (parent grid)");
    }





    Int indexB = -1;

    for(Int i = 0; i<numSubGrids; ++i)
    {
        if(B_Vector[i]->Participating())
        {
            if(indexB!=-1)
            {
                LogicError("TranslateBetweenGridsBroadcast: ",
                   "Error: rank is in multiple subgrids");

            }
            indexB = i;
        }
    }
    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));




    const Int rowStrideB = B->RowStride();
    const Int sizeB = B->Grid().VCSize();
    const Int rowGCD = GCD(rowStrideB, rowStrideA);
    const Int rowLCM = rowStrideB*rowStrideA / rowGCD;
    const Int posInGrid = A.Grid().VCRank();


    SyncInfo<D1> syncGeneralMetaData = SyncInfo<D1>();
    const bool inAGrid = A.Participating();


    Int recvMetaData[6];

    Int metaData[6];
    if(inAGrid)
    {

        metaData[0] = m;
        metaData[1] = n;
        metaData[2] = mLocA;
        metaData[3] = nLocA;
    }
    else
    {
        metaData[0] = 0;
        metaData[1] = 0;
        metaData[2] = 0;
        metaData[3] = 0;
    }

    const std::vector<Int> sendMetaData (metaData,metaData + 4 );


    Synchronize(syncGeneral);

    if(static_cast<size_t>(sizeA) != sizeB*B_Vector.size())
    {
        mpi::AllReduce( sendMetaData.data(), recvMetaData, 4, mpi::MAX, viewingCommA,syncGeneralMetaData);
        Synchronize(syncGeneralMetaData);

        m = recvMetaData[0];
        n = recvMetaData[1];
        mLocA = recvMetaData[2];
        nLocA = recvMetaData[3];
    }

    for(Int i=0; i<numSubGrids;++i)
    {
        B_Vector[i]->Resize(m,n);
    }

    // Parent Subgrid Size: 4 Child Subgrid Size: 3
    // Parent 0 1 2 3 0 1 2 3 0 1 2 3
    // Child  0 1 2 0 1 2 0 1 2 0 1 2

    std::vector<int> index_to_put(sizeA,-1);

    for(Int i = 0; i < int(rowLCM/sizeB); ++i)
    {
        index_to_put[i] = i;
    }

    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
    SyncInfo<D2> syncInfoB = SyncInfoFromMatrix(B->LockedMatrix());

    // const bool inAGrid = A.Participating();
    const bool inBGrid = indexB >=0 ? true:false;

    const Int maxSendSize = mLocA * nLocA;
    simple_buffer<T,D1> send_buf(maxSendSize , syncInfoA);

    simple_buffer<T,D2> recv_buf(inBGrid ? maxSendSize : 0, syncInfoB);
    T* sendBuf = send_buf.data();


    const int myLocalDataRankA = int(std::floor(posInGrid/sizeB));

    for(Int localDataRankA = 0; localDataRankA < int(rowLCM/sizeB); localDataRankA++)
    {

        if(inAGrid)
        {
           if(myLocalDataRankA==localDataRankA )
            {
                copy::util::InterleaveMatrix(
                    mLocA, nLocA,
                    A.LockedBuffer(0,0),
                    1, A.LDim(),
                    sendBuf, 1, mLocA, syncInfoA);

            }
        }


        //comm is useless parameter in this function
        //Aluminum infer comm from sunc object
        Broadcast(sendBuf, mLocA*nLocA, localDataRankA, broadcastComm,
               syncGeneral);

        Synchronize(syncGeneral);



        int sendWidth = int(n / rowLCM);
        copy::util::InterleaveMatrix(
                        m, sendWidth,
                        sendBuf  , 1, m*(rowLCM/sizeA),
                        B->Buffer(0,index_to_put[localDataRankA]),
                        1, (rowLCM/sizeB)*B->LDim(),
                        syncInfoB);


        Synchronize(syncInfoB);

    }

}


template<typename T, Device D1, Device D2>
void TranslateBetweenGridsBroadcastBasic
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector)
{
    //<T,STAR,VC,ELEMENT,D2>
    /*
    This function is specific to the LBANN with implementation for specific cases
    Subgrids in B_vector are assumed to be subset of resources in A grid
    Same terminology as Allreduce functions
    */


    EL_DEBUG_CSE

    Int m = A.Height();
    Int n = A.Width();
    Int mLocA = A.LocalHeight();
    Int nLocA = A.LocalWidth();
    const bool inAGrid = A.Participating();


    mpi::Comm const& viewingCommA = A.Grid().ViewingComm();
    Int rowStrideA = A.RowStride();

    const Int numSubGrids = int(B_Vector.size());
    const Int sizeA = A.Grid().VCSize();

    //Asserting parent grid evenly divides number of columns
    if(n%sizeA!=0)
    {
        LogicError("TranslateBetweenGridsBroadcastBasic: ",
                   "Number of columns should be evenly divided by the size of the grid A (parent grid)");
    }



    Int indexB = -1;

    for(Int i = 0; i<numSubGrids; ++i)
    {
        if(B_Vector[i]->Participating())
        {
            if(indexB!=-1)
            {
                LogicError("TranslateBetweenGridsBroadcast: ",
                   "Error: rank is in multiple subgrids");
            }
            indexB = i;
        }
    }
    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));

    const Int posInSubGrid = B->Grid().VCRank();


    const Int rowStrideB = B->RowStride();
    const Int sizeB = B->Grid().VCSize();
    const Int rowGCD = GCD(rowStrideB, rowStrideA);
    const Int rowLCM = rowStrideB*rowStrideA / rowGCD;
    const Int myLocalRankA = A.Grid().VCRank();


    SyncInfo<D1> syncGeneralMetaData = SyncInfo<D1>();


    Int recvMetaData[6];

    Int metaData[6];
    if(inAGrid)
    {

        metaData[0] = m;
        metaData[1] = n;
        metaData[2] = mLocA;
        metaData[3] = nLocA;
    }
    else
    {
        metaData[0] = 0;
        metaData[1] = 0;
        metaData[2] = 0;
        metaData[3] = 0;
    }

    const std::vector<Int> sendMetaData (metaData,metaData + 4 );




    if(static_cast<size_t>(sizeA) != sizeB*B_Vector.size())
    {
        mpi::AllReduce( sendMetaData.data(), recvMetaData, 4, mpi::MAX, viewingCommA,syncGeneralMetaData);
        Synchronize(syncGeneralMetaData);

        m = recvMetaData[0];
        n = recvMetaData[1];
        mLocA = recvMetaData[2];
        nLocA = recvMetaData[3];
    }

    for(Int i=0; i<numSubGrids;++i)
    {
        B_Vector[i]->Resize(m,n);
    }



    // Parent Subgrid Size: 4 Child Subgrid Size: 3
    // Parent 0 1 2 3 0 1 2 3 0 1 2 3
    // Child  0 1 2 0 1 2 0 1 2 0 1 2

    std::vector<bool> require_data(sizeA,false);
    std::vector<int> index_to_put(sizeA,-1);
    std::vector<int> index_from(sizeA,-1);
    Int temp_require_data = posInSubGrid;
    double total_iter = posInSubGrid;



    for(Int i = 0; i < int(rowLCM/sizeB); ++i)
    {
        if(require_data[temp_require_data]==true)
        {
            LogicError("TranslateBetweenGridsBroadcast: ",
                   "Cannot receive input from  same rank twice");
        }
        require_data[temp_require_data] = true;
        index_to_put[temp_require_data] = i;
        index_from[temp_require_data] = int(std::floor(total_iter/sizeA));
        total_iter = total_iter + sizeB;
        temp_require_data =  Mod(temp_require_data + sizeB, sizeA);


    }

    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
    SyncInfo<D2> syncInfoB = SyncInfoFromMatrix(B->LockedMatrix());
    SyncInfo<D1> syncGeneral = SyncInfo<D1>();

    //


    const Int maxSendSize = mLocA * nLocA;
    simple_buffer<T,D1> send_buf(maxSendSize, syncInfoA);

    T* sendBuf = send_buf.data();
    //T* recvBuf = recv_buf.data();




    for(Int localRankA = 0; localRankA < sizeA; localRankA++)
    {
        if(myLocalRankA==localRankA && inAGrid)
        {
            copy::util::InterleaveMatrix(
                mLocA, nLocA,
                A.LockedBuffer(0,0),
                1, A.LDim(),
                sendBuf, 1, mLocA, syncInfoA);

        }
        //comm is useless parameter in this function
        //Aluminum infer comm from sunc object
        Broadcast(sendBuf, mLocA*nLocA, localRankA, viewingCommA,
               syncInfoA);

        Synchronize(syncGeneral);

        if(require_data[localRankA])
        {
            int sendWidth = int(n / rowLCM);
            copy::util::InterleaveMatrix(
                            m, sendWidth,
                            sendBuf + (index_from[localRankA] * m ) , 1, m*(rowLCM/sizeA),
                            B->Buffer(0,index_to_put[localRankA]),
                            1, (rowLCM/sizeB)*B->LDim(),
                            syncInfoB);

        }
        Synchronize(syncInfoB);



    }

}


template<typename T, Device D1, Device D2>
void TranslateBetweenGridsBroadcast
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, int version)
{
    //Some better logic can be written here
    // 0: Best Algo
    // 1: Basic Broadcast without communication overlap
    //
    //


    if(version==0 || version == 1)
    {
        TranslateBetweenGridsBroadcastBasic<T, D1, D2>(A,
                                                B_Vector);
    }
    else
    {
        LogicError("TranslateBetweenGridsBroadcast: ",
                   "Invalid version, it has to be [0,1], 0: Default");
    }

}

template<typename T, Device D1, Device D2>
void TranslateBetweenGridsBroadcast
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
    std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, mpi::Comm const& broadcastComm, SyncInfo<D1> & syncGeneral, int version)

{
    //Some better logic can be written here
    // 0: Best Algo
    // 1: if spliting comm is given then we have only one algo for now
    //


    if(version==0 || version == 1)
    {
        TranslateBetweenGridsBroadcastOptComm<T, D1, D2>(A,
                                                B_Vector,
                                                broadcastComm,
                                                syncGeneral);
    }

    else
    {
        LogicError("TranslateBetweenGridsAllreduce: ",
                   "Invalid version, it has to be [0,1], 0: Default");
    }
}



#define PROTO_DIFF_BASE(T, D) \
  template void TranslateBetweenGridsAllreduceBasic<T,D,D>(DistMatrix<T,STAR,VC,ELEMENT,D> & , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& ); \
  template void TranslateBetweenGridsAllreduceOpt<T,D,D>(DistMatrix<T,STAR,VC,ELEMENT,D> & , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& ); \
  template void TranslateBetweenGridsAllreduce <T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> & , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& , int ); \
  template void TranslateBetweenGridsAllreduce <T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> & , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& , mpi::Comm const& , SyncInfo<D> &, int ); \
  template void TranslateBetweenGridsAllreduceOptComm <T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> & , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>&, mpi::Comm const& , SyncInfo<D> & ); \
  \
  template void TranslateBetweenGridsBroadcast <T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> const& , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& , int );\
  template void TranslateBetweenGridsBroadcast <T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> const& , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& , mpi::Comm const& , SyncInfo<D> &, int );\
  template void TranslateBetweenGridsBroadcastBasic <T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> const& , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& ); \
  template void TranslateBetweenGridsBroadcastOptComm<T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> const& , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& ,  mpi::Comm const& , SyncInfo<D> & ); \
  \
  template void TranslateBetweenGridsScatter<T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> const& , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& , int ,  mpi::Comm const& , SyncInfo<D> &, int );\
  template void TranslateBetweenGridsScatterComm<T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> const& , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& , int ,  mpi::Comm const& , SyncInfo<D> & );\
  template void TranslateBetweenGridsScatterOptComm<T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> const& , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& , int ,  mpi::Comm const& , SyncInfo<D> & );\
  \
  template void TranslateBetweenGridsGather<T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> & , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& , int ,  mpi::Comm const& , SyncInfo<D> &, int );\
  template void TranslateBetweenGridsGatherComm<T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> & , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& , int ,  mpi::Comm const& , SyncInfo<D> & );\
  template void TranslateBetweenGridsGatherOptComm<T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> & , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& , int ,  mpi::Comm const& , SyncInfo<D> & );\
  \
  template void TranslateBetweenGridsSliceGatherOptComm<T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> const& , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& , int ,  mpi::Comm const& , SyncInfo<D> & );\
  \
  template void TranslateBetweenGridsSliceGatherParentSmall<T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> const& , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& , int ,  mpi::Comm const& , SyncInfo<D> & );\
  template void TranslateBetweenGridsScatterCommParentSmall<T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> const& , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& , int ,  mpi::Comm const& , SyncInfo<D> & );\
  \
  template void TranslateBetweenGridsScatterCommSameSizeSubGrids<T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> const& , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& , int ,  mpi::Comm const& , SyncInfo<D> & );\
  template void TranslateBetweenGridsSliceBroadcastCommSameSizeSubGrids<T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> const& , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& , int ,  mpi::Comm const& , SyncInfo<D> & );\
  template void TranslateBetweenGridsGatherCommSameSizeSubGrids<T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> & , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& , int ,  mpi::Comm const& , SyncInfo<D> & );\
  template void TranslateBetweenGridsAllGatherCommSameSizeSubGrids<T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> & , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& , int ,  mpi::Comm const& , SyncInfo<D> & );\
  \
  template void TranslateBetweenGridsSliceColVector<T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> const& , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& );\
  \
  template void TranslateBetweenGridsSliceConcatAlongFirstDim<T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> const& , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& , int ,  mpi::Comm const& , SyncInfo<D> & );\
  template void TranslateBetweenConatSliceFirstChannel<T,D,D> (DistMatrix<T,STAR,VC,ELEMENT,D> & , std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& , int ,  mpi::Comm const& , SyncInfo<D> & );





#define PROTO_DIFF_CPU(T) PROTO_DIFF_BASE(T,El::Device::CPU)
#ifdef HYDROGEN_HAVE_GPU
#define PROTO_DIFF_GPU(T) PROTO_DIFF_BASE(T,El::Device::GPU)
#else
#define PROTO_DIFF_GPU(T)
#endif
#define PROTO_DIFF(T) \
  PROTO_DIFF_CPU(T)  \
  PROTO_DIFF_GPU(T)

PROTO_DIFF(float)
PROTO_DIFF(double)

#ifdef HYDROGEN_HAVE_HALF
PROTO_DIFF_CPU(cpu_half_type);
#ifdef HYDROGEN_GPU_USE_FP16
PROTO_DIFF_GPU(gpu_half_type);
#endif
#endif


template void TranslateBetweenGridsSliceCol<double> (AbstractDistMatrix<double> const& , AbstractDistMatrix<double> &);
// template void TranslateBetweenGridsSliceCol<double, Device::GPU,Device::GPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::GPU> const& , DistMatrix<double,STAR,VC,ELEMENT,Device::GPU> &);
template void TranslateBetweenGridsSliceCol<float> (AbstractDistMatrix<float> const& , AbstractDistMatrix<float> &);
// template void TranslateBetweenGridsSliceCol<float, Device::GPU,Device::GPU> (DistMatrix<float,STAR,VC,ELEMENT,Device::GPU> const& , DistMatrix<float,STAR,VC,ELEMENT,Device::GPU> &);



// FIX ME Memory leak in LBANN
template<typename T, Device D1, Device D2>
void TranslateBetweenGridsAsync
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
  DistMatrix<T,STAR,VC,ELEMENT,D2>& B)
{
    EL_DEBUG_CSE;
    Int m = A.Height();
    Int n = A.Width();
    const Int mLocA = A.LocalHeight();
    const Int nLocA = A.LocalWidth();

    mpi::Comm const& viewingCommB = B.Grid().ViewingComm();
    mpi::Group owningGroupA = A.Grid().OwningGroup();

    // Just need to ensure that each viewing comm contains the other team's
    // owning comm. Congruence is too strong.

    // Compute the number of process rows and columns that each process
    // needs to send to.


    Int colStrideA = A.ColStride();
    Int rowStrideA = A.RowStride();
    Int colAlignA = A.ColAlign();
    Int rowAlignA = A.RowAlign();
    SyncInfo<D1> syncGeneral = SyncInfo<D1>();


    const bool inAGrid = A.Participating();


    Int recvMetaData[6];

    Int metaData[6];
    if(inAGrid)
    {

        metaData[0] = m;
        metaData[1] = n;
        metaData[2] = colStrideA;
        metaData[3] = rowStrideA;
        metaData[4] = colAlignA;
        metaData[5] = rowAlignA;


    }
    else
    {
        metaData[0] = 0;
        metaData[1] = 0;
        metaData[2] = 0;
        metaData[3] = 0;
        metaData[4] = 0;
        metaData[5] = 0;
    }

    const std::vector<Int> sendMetaData (metaData,metaData + 6 );


    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
    Synchronize(syncGeneral);


    mpi::AllReduce( sendMetaData.data(), recvMetaData, 6, mpi::MAX, viewingCommB,syncGeneral);
    Synchronize(syncGeneral);

    m = recvMetaData[0];
    n = recvMetaData[1];
    colStrideA = recvMetaData[2];
    rowStrideA = recvMetaData[3];
    colAlignA = recvMetaData[4];
    rowAlignA = recvMetaData[5];

    B.Resize(m, n);
    const Int colStrideB = B.ColStride();
    const Int rowStrideB = B.RowStride();

    const Int colRankB = B.ColRank();
    const Int rowRankB = B.RowRank();
    const Int colAlignB = B.ColAlign();
    const Int rowAlignB = B.RowAlign();
    const bool inBGrid = B.Participating();


    SyncInfo<D2> syncInfoB = SyncInfoFromMatrix(B.LockedMatrix());


    const Int rowGCD = GCD(rowStrideB, rowStrideA);
    const Int rowLCM = rowStrideB*rowStrideA / rowGCD;
    const Int numRowSends = rowLCM / rowStrideA ;
    const Int numRowRecvs = rowLCM / rowStrideB;
    const Int myRankViewing = mpi::Rank(viewingCommB);

    const Int rankBRecv = Mod(B.Grid().Rank(), rowStrideA);


    //Setup for receiving data in B
    const Int sendColOffset = colAlignA;
    const Int recvColOffset =
      Mod(colAlignB,colStrideB);
    const Int sendRowOffset = rowAlignA;
    const Int recvRowOffset =
      Mod(0*rowStrideA+rowAlignB,rowStrideB);

    const Int colShift = Mod(colRankB-recvColOffset, colStrideB);
    const Int rowShift = Mod(rowRankB-recvRowOffset, rowStrideB);


    const Int numInB = B.Grid().Rank();

    const Int firstSendRow = Mod(colShift+sendColOffset,colStrideA);
    const Int firstSendCol = Mod(rowShift+sendRowOffset,rowStrideA);


    Int sendCol = firstSendCol;

    Int sendRow = firstSendRow;

    if(!inBGrid && !inAGrid)
        return;

    const Int maxSendSize =
      (n/(rowStrideA*numRowSends)+1) * (m);


    // Translate the ranks from A's VC communicator to B's viewing so that
    // we can match send/recv communicators. Since A's VC communicator is not
    // necessarily defined on every process, we instead work with A's owning
    // group and account for row-major ordering if necessary.
    const int sizeA = A.Grid().Size();
    vector<int> rankMap(sizeA), ranks(sizeA);
    if(A.Grid().Order() == COLUMN_MAJOR)
    {
        for(int j=0; j<sizeA; ++j)
            ranks[j] = j;
    }
    else
    {
        // The (i,j) = i + j*colStrideA rank in the column-major ordering is
        // equal to the j + i*rowStrideA rank in a row-major ordering.
        // Since we desire rankMap[i+j*colStrideA] to correspond to process
        // (i,j) in A's grid's rank in this viewing group, ranks[i+j*colStrideA]
        // should correspond to process (i,j) in A's owning group. Since the
        // owning group is ordered row-major in this case, its rank is
        // j+i*rowStrideA. Note that setting
        // ranks[j+i*rowStrideA] = i+j*colStrideA is *NOT* valid.
        for(int i=0; i<colStrideA; ++i)
            for(int j=0; j<rowStrideA; ++j)
                ranks[i+j*colStrideA] = j+i*rowStrideA;
    }
    mpi::Translate(
        owningGroupA, sizeA, ranks.data(), viewingCommB, rankMap.data());

    std::vector<simple_buffer<T,D1>> sendBufVector(numRowSends);

    for(Int i=0; i<numRowSends; ++i)
    {
        sendBufVector[i].allocate(inAGrid ? maxSendSize : 0);

    }


    simple_buffer<T,D2> recv_buf(inBGrid ? maxSendSize : 0, syncInfoB);

    T* recvBuf = recv_buf.data();



    //Checking if process are in both A and B grids
    // Just transfer the data directly
    for (Int rowSend = 0; rowSend < numRowSends; rowSend++)
    {
        const Int recvVCRank = Mod(A.Grid().Rank() + rowSend*rowStrideA, rowStrideB);
        const Int recvViewingRank = B.Grid().VCToViewing(recvVCRank);

        if(recvViewingRank==myRankViewing)
        {
            Int sendWidth = Length(nLocA,rowSend,numRowSends);

            Int rowRecv = 0;

            for(rowRecv = 0; rowRecv<numRowRecvs; ++rowRecv)
            {
                const Int sendVCRank = Mod((sendRow + rankBRecv),rowStrideA);
                sendRow = Mod(sendRow+rowStrideB,rowStrideA);
                if(rankMap[sendVCRank]==myRankViewing) break;
            }


            copy::util::InterleaveMatrix(
                mLocA, sendWidth,
                A.LockedBuffer(0,rowSend),
                1, numRowSends*A.LDim(),
                B.Buffer(0,rowRecv),
                1, (numRowRecvs)*B.LDim(),
                syncInfoB);
            Synchronize(syncInfoA);
            Synchronize(syncInfoB);


        }

    }




    std::vector<mpi::Request<T>> sendRequests(numRowSends);
    std::vector<bool> sendRequestsUsed(numRowSends,false);
    for(Int rowSend=0; rowSend<numRowSends; ++rowSend)
    {

        mpi::Request<T> sendRequest;

        const Int recvVCRank = Mod(A.Grid().Rank() + rowSend*rowStrideA, rowStrideB);
        const Int recvViewingRank = B.Grid().VCToViewing(recvVCRank);

        if(inAGrid && recvViewingRank!=myRankViewing)
        {
            //Pack Data
            Int sendWidth = Length(nLocA,rowSend,numRowSends);
            //std::printf("sendWidth from send %d\n", sendWidth);
            copy::util::InterleaveMatrix(
                    mLocA, sendWidth,
                    A.LockedBuffer(0,rowSend),
                    1, numRowSends*A.LDim(),
                    sendBufVector[rowSend].data(), 1, mLocA, syncInfoA);


            Synchronize(syncInfoA);
            sendRequestsUsed[rowSend] = true;


            mpi::ISend
            (sendBufVector[rowSend].data(), mLocA*sendWidth, recvViewingRank,
              viewingCommB, sendRequests[rowSend]);

        }

    }


    //start receiving data from other processes
    sendRow = firstSendRow;



    for(Int rowRecv=0; rowRecv<numRowRecvs; ++rowRecv)
    {

        const Int sendVCRank = Mod((sendRow + rankBRecv),rowStrideA);

        if(inBGrid && rankMap[sendVCRank]!=myRankViewing)
        {



            const Int sendWidth = ((rowRecv*rowStrideB + numInB)>= Mod(n,rowLCM)) ? floor(n/rowLCM) : floor(n/rowLCM)+1;



            mpi::Recv(
                recvBuf, m*sendWidth, rankMap[sendVCRank],
                viewingCommB, syncInfoB);

            // Unpack the data
            copy::util::InterleaveMatrix(
                m, sendWidth,
                recvBuf, 1, m,
                B.Buffer(0,rowRecv),
                1, (numRowRecvs)*B.LDim(),
                syncInfoB);

        }
        // Set up the next send col
        sendCol = Mod(sendCol+rowStrideB,rowStrideA);
        sendRow = Mod(sendRow+rowStrideB,rowStrideA);




    }
    if(inAGrid)
        for (Int i=0;i<numRowSends;++i)
        {
            if(sendRequestsUsed[i])
            {
                mpi::Wait(sendRequests[i]);
            }

        }

    sendBufVector.clear();

}

template<typename T, Device D>
void TranslateBetweenGrids(
  DistMatrix<T,STAR,VC,ELEMENT,D> const& A,
  DistMatrix<T,STAR,VC,ELEMENT,D>& B)
{
  EL_DEBUG_CSE;

  /* Overview
     We broadcast the size of A to all the ranks in B to make sure that
     all ranks in B subgrid has the correct size of A.
     Since we are using blocking communication, some care is required
     to avoid deadlocks. Let's start with a naive algorithm for
     [STAR,VC] matrices and optimize it in steps:

     1. Given an m x n matrix, iterate over the n global columns on
     every rank. If a rank owns the column in both A and B, just copy
     it locally. If it owns it only in A, perform a send. If it owns
     it only in B, perform a recv.

     2. Instead of iterating through all of the global columns on
     every rank, just iterate through the local columns. Go through
     the local columns in A and in B to figure out where to send/recv,
     sort by the global column indices, and then perform the
     sends/recvs. The resulting communication is identical to 1.

     3. Consolidate columns with the same source and target ranks, so
     we only have to send a single message. These columns have a
     regular stride (left as an exercise to the reader), so it is
     straightforward to pack them from A into a temporary buffer,
     communicate, and unpack to B. We use the same communication order
     as 1 and 2, but only with the first columns for each
     source/target rank pair.

  */

  // Matrix dimensions
  Int m = A.Height();
  Int n = A.Width();
  Int strideA = A.RowStride();
  Int ALDim = A.LDim();

  mpi::Comm const& viewingCommB = B.Grid().ViewingComm();
  bool const inAGrid = A.Participating();
  bool const inBGrid = B.Participating();

  Int recvMetaData[4];
  Int metaData[4];
  if(inAGrid)
  {
    metaData[0] = m;
    metaData[1] = n;
    metaData[2] = strideA;
    metaData[3] = ALDim;
  }
  else
  {
    metaData[0] = 0;
    metaData[1] = 0;
    metaData[2] = 0;
    metaData[3] = 0;
  }
  const std::vector<Int> sendMetaData (metaData, metaData + 4);
  mpi::AllReduce(sendMetaData.data(),
                 recvMetaData,
                 4,
                 mpi::MAX,
                 viewingCommB,
                 SyncInfo<El::Device::CPU>{});

  m = recvMetaData[0];
  n = recvMetaData[1];
  strideA = recvMetaData[2];
  ALDim =recvMetaData[3];

  B.Resize(m, n);
  const Int nLocA = A.LocalWidth();
  const Int nLocB = B.LocalWidth();

  // Return immediately if there is no local data
  if (!inAGrid && !inBGrid) {
    return;
  }

  // Compute the number of messages to send/recv
  const Int strideB = B.RowStride();
  const Int strideGCD = GCD(strideA, strideB);
  const Int numSends = Min(strideB/strideGCD, nLocA);
  const Int numRecvs = Min(strideA/strideGCD, nLocB);

  // Synchronize compute streams
  SyncInfo<D> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
  SyncInfo<D> syncInfoB = SyncInfoFromMatrix(B.Matrix());

  std::optional<MultiSync<D, D>> maybeMultiSync;
  if (inAGrid && inBGrid)
      maybeMultiSync.emplace(syncInfoB, syncInfoA);

  SyncInfo<D> const syncInfo =
      (maybeMultiSync.has_value()
       ? *maybeMultiSync
       : (inAGrid ? syncInfoA : syncInfoB));

  // Translate the ranks from A's VC communicator to B's viewing so
  // that we can match send/recv communicators. Since A's VC
  // communicator is not necessarily defined on every process, we
  // instead work with A's owning group.
  mpi::Group owningGroupA = A.Grid().OwningGroup();
  const int sizeA = A.Grid().Size();
  vector<int> viewingRanksA(sizeA), owningRanksA(sizeA);
  std::iota(owningRanksA.begin(), owningRanksA.end(), 0);
  mpi::Translate(
    owningGroupA, sizeA, owningRanksA.data(),
    viewingCommB, viewingRanksA.data());
  const Int viewingRank = viewingCommB.Rank();
  if (viewingRank < 0 || viewingRank >= viewingCommB.Size()) {
    LogicError(
      "TranslateBetweenGrids: Owning group for matrix A "
      "is not a subset of viewing communicator for matrix B");
  }

  // Workspace buffer to pack/unpack data
  Int messageBufSize = 0;
  if (inAGrid && numSends > 0) {
    messageBufSize = Max(m*MaxLength(nLocA,numSends), messageBufSize);
  }
  if (inBGrid && numRecvs > 0) {
    messageBufSize = Max(m*MaxLength(nLocB,numRecvs), messageBufSize);
  }
  simple_buffer<T,D> messageBuf(messageBufSize, syncInfo);

  // Figure out which ranks to send/recv messages
  std::map<Int,std::pair<Int,Int>> messageRanks;
  for (Int jLocA=0; jLocA<numSends; ++jLocA) {
    const Int j = A.GlobalCol(jLocA);
    const Int recvVCRank = B.ColOwner(j);
    const Int recvViewingRank = B.Grid().VCToViewing(recvVCRank);
    messageRanks[j] = std::pair<Int,Int>(viewingRank, recvViewingRank);
  }
  for (Int jLocB=0; jLocB<numRecvs; ++jLocB) {
    const Int j = B.GlobalCol(jLocB);
    const Int sendVCRank = A.ColOwner(j);
    const Int sendViewingRank = viewingRanksA[sendVCRank];
    messageRanks[j] = std::pair<Int,Int>(sendViewingRank, viewingRank);
  }

  // Send/recv messages
  for (auto const& message : messageRanks) {
    const Int j = message.first;
    const Int sendViewingRank = message.second.first;
    const Int recvViewingRank = message.second.second;

    // Figure out message size
    Int jLocA = -1;
    Int jLocB = -1;
    Int messageWidth = 0;
    if (viewingRank == sendViewingRank) {
      jLocA = A.LocalCol(j);
      messageWidth = Length(nLocA, jLocA, numSends);
    }
    if (viewingRank == recvViewingRank) {
      jLocB = B.LocalCol(j);
      messageWidth = Length(nLocB, jLocB, numRecvs);
    }

    if (viewingRank == sendViewingRank && viewingRank == recvViewingRank) {
      // Copy data locally
      copy::util::InterleaveMatrix(
        m, messageWidth,
        A.LockedBuffer(0,jLocA), 1, numSends*ALDim,
        B.Buffer(0,jLocB), 1, numRecvs*B.LDim(),
        syncInfo);
    }
    else if (viewingRank == sendViewingRank) {
      // Send data to other rank
      copy::util::InterleaveMatrix(
        m, messageWidth,
        A.LockedBuffer(0,jLocA), 1, numSends*ALDim,
        messageBuf.data(), 1, m,
        syncInfo);
      mpi::Send(
        messageBuf.data(), m*messageWidth,
        recvViewingRank, viewingCommB, syncInfo);
    }
    else if (viewingRank == recvViewingRank) {
      // Receive data from other rank
      mpi::Recv(
        messageBuf.data(), m*messageWidth,
        sendViewingRank, viewingCommB, syncInfo);
      copy::util::InterleaveMatrix(
        m, messageWidth,
        messageBuf.data(), 1, m,
        B.Buffer(0,jLocB), 1, numRecvs*B.LDim(),
        syncInfo);
    }

  }

}

template void TranslateBetweenGridsAsync<double, Device::CPU, Device::CPU>(
    DistMatrix<double, STAR, VC, ELEMENT, Device::CPU> const&,
    DistMatrix<double, STAR, VC, ELEMENT, Device::CPU>&);
#ifdef HYDROGEN_HAVE_GPU
template void TranslateBetweenGridsAsync<double, Device::GPU,Device::GPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::GPU> const& ,DistMatrix<double,STAR,VC,ELEMENT,Device::GPU>& );
#endif // HYDROGEN_HAVE_GPU

} // namespace copy
} // namespace El

#endif // ifndef EL_BLAS_COPY_TRANSLATEBETWEENGRIDS_HPP
