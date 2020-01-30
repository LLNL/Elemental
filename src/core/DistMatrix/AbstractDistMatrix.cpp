/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/core/DistMatrix.hpp>
#include <El/blas_like/level1/Copy.hpp>
#include <El/blas_like/level1/Scale.hpp>

namespace El
{

// Public section
// ##############

// Constructors and destructors
// ============================

template<typename T>
AbstractDistMatrix<T>::AbstractDistMatrix(const El::Grid& grid, int root)
: root_(root), grid_(&grid)
{ }

template<typename T>
AbstractDistMatrix<T>::AbstractDistMatrix(AbstractDistMatrix<T>&& A)
EL_NO_EXCEPT
: viewType_(A.viewType_),
  height_(A.height_),
  width_(A.width_),
  colConstrained_(A.colConstrained_),
  rowConstrained_(A.rowConstrained_),
  rootConstrained_(A.rootConstrained_),
  colAlign_(A.colAlign_),
  rowAlign_(A.rowAlign_),
  colShift_(A.colShift_),
  rowShift_(A.rowShift_),
  root_(A.root_),
  grid_(A.grid_)
{
//    Matrix().ShallowSwap(A.Matrix());
}

template<typename T>
AbstractDistMatrix<T>::~AbstractDistMatrix() { }

// Assignment and reconfiguration
// ==============================
template<typename T>
void
AbstractDistMatrix<T>::Empty(bool freeMemory)
{
    this->EmptyData(freeMemory);

    colAlign_ = 0;
    rowAlign_ = 0;
    colConstrained_ = false;
    rowConstrained_ = false;
    rootConstrained_ = false;
    SetShifts();
}

template<typename T>
void
AbstractDistMatrix<T>::EmptyData(bool freeMemory)
{
    Matrix().Empty_(freeMemory);
    viewType_ = OWNER;
    height_ = 0;
    width_ = 0;

    do_empty_data_();

//    SwapClear(remoteUpdates_);
}

template<typename T>
void
AbstractDistMatrix<T>::SetGrid(const El::Grid& grid)
{
    if(grid_ != &grid)
    {
        grid_ = &grid;
        Empty(false);
    }
}

template<typename T>
void
AbstractDistMatrix<T>::FreeAlignments()
{
    if(!Viewing())
    {
        colConstrained_ = false;
        rowConstrained_ = false;
        rootConstrained_ = false;
    }
    else
        LogicError("Cannot free alignments of views");
}

template<typename T>
void
AbstractDistMatrix<T>::MakeSizeConsistent(bool includingViewers)
{
    EL_DEBUG_CSE

    const Int msgSize = 2;
    Int message[msgSize];
    if(CrossRank() == Root())
    {
        message[0] = height_;
        message[1] = width_;
    }

    const auto& grid = *grid_;
    if(!grid.InGrid() && !includingViewers)
        LogicError("Non-participating process called MakeSizeConsistent");
    if(grid.InGrid())
    {
        if (this->GetLocalDevice() == Device::CPU)
            mpi::Broadcast(message, msgSize, Root(), CrossComm(),
                           SyncInfo<Device::CPU>{});
#ifdef HYDROGEN_HAVE_GPU
        else if (this->GetLocalDevice() == Device::GPU)
        {
            auto syncInfo = SyncInfoFromMatrix(
                static_cast<::El::Matrix<T,Device::GPU> const&>(
                    this->LockedMatrix()));
            mpi::Broadcast(message, msgSize, Root(), CrossComm(),
                           syncInfo);
        }
#endif // HYDROGEN_HAVE_GPU
        else
            LogicError("AbstractMatrix: Bad Device!");
    }
    if(includingViewers)
    {
        const Int vcRoot = grid.VCToViewing(0);
        if (this->GetLocalDevice() == Device::CPU)
            mpi::Broadcast(message, msgSize, vcRoot, grid.ViewingComm(),
                           SyncInfo<Device::CPU>{});
#ifdef HYDROGEN_HAVE_GPU
        else if (this->GetLocalDevice() == Device::GPU)
        {
            auto syncInfo = SyncInfoFromMatrix(
                static_cast<::El::Matrix<T,Device::GPU> const&>(
                    this->LockedMatrix()));
            mpi::Broadcast(message, msgSize, vcRoot, grid.ViewingComm(),
                           syncInfo);
        }
#endif // HYDROGEN_HAVE_GPU
        else
            LogicError("AbstractMatrix: Bad Device!");

    }
    const Int newHeight = message[0];
    const Int newWidth  = message[1];
    Resize(newHeight, newWidth);
}

// Realignment
// -----------

template<typename T>
void
AbstractDistMatrix<T>::SetRoot(int root, bool constrain)
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if(root < 0 || root >= CrossSize())
          LogicError("Invalid root");
   )
    if(root != root_)
        EmptyData(false);
    root_ = root;
    if(constrain)
        rootConstrained_ = true;
    SetShifts();
}

// Operator overloading
// ====================

// Move assignment
// ---------------
template<typename T>
AbstractDistMatrix<T>&
AbstractDistMatrix<T>::operator=(AbstractDistMatrix<T>&& A)
{
    EL_DEBUG_CSE
    if(Viewing() || A.Viewing())
    {
        El::Copy(A, *this);
    }
    else
    {
        Matrix().ShallowSwap(A.Matrix());
        viewType_ = A.viewType_;
        height_ = A.height_;
        width_ = A.width_;
        colConstrained_ = A.colConstrained_;
        rowConstrained_ = A.rowConstrained_;
        rootConstrained_ = A.rootConstrained_;
        colAlign_ = A.colAlign_;
        rowAlign_ = A.rowAlign_;
        colShift_ = A.colShift_;
        rowShift_ = A.rowShift_;
        root_ = A.root_;
        grid_ = A.grid_;
    }
    return *this;
}

// Rescaling
// ---------
template<typename T>
const AbstractDistMatrix<T>&
AbstractDistMatrix<T>::operator*=(T alpha)
{
    EL_DEBUG_CSE
    Scale(alpha, *this);
    return *this;
}

// Basic queries
// =============

// Global matrix information
// -------------------------

template<typename T>
Int AbstractDistMatrix<T>::Height() const EL_NO_EXCEPT { return height_; }
template<typename T>
Int AbstractDistMatrix<T>::Width() const EL_NO_EXCEPT { return width_; }

template<typename T>
Int AbstractDistMatrix<T>::DiagonalLength(Int offset) const EL_NO_EXCEPT
{ return El::DiagonalLength(height_,width_,offset); }

template<typename T>
bool AbstractDistMatrix<T>::Viewing() const EL_NO_EXCEPT
{ return IsViewing(viewType_); }
template<typename T>
bool AbstractDistMatrix<T>::Locked() const EL_NO_EXCEPT
{ return IsLocked(viewType_); }
template<typename T>
bool AbstractDistMatrix<T>::IsEmpty() const EL_NO_EXCEPT
{ return height_ < 1 || width_ < 1; }

// Local matrix information
// ------------------------

template<typename T>
Int AbstractDistMatrix<T>::LocalHeight() const EL_NO_EXCEPT
{ return LockedMatrix().Height(); }
template<typename T>
Int AbstractDistMatrix<T>::LocalWidth() const EL_NO_EXCEPT
{ return LockedMatrix().Width(); }
template<typename T>
Int AbstractDistMatrix<T>::LDim() const EL_NO_EXCEPT
{ return LockedMatrix().LDim(); }
template<typename T>
bool AbstractDistMatrix<T>::Contiguous() const EL_NO_EXCEPT
{ return LockedMatrix().Contiguous(); }

template<typename T>
size_t
AbstractDistMatrix<T>::AllocatedMemory() const EL_NO_EXCEPT
{ return LockedMatrix().MemorySize(); }

template<typename T>
T*
AbstractDistMatrix<T>::Buffer() EL_NO_RELEASE_EXCEPT
{ return Matrix().Buffer(); }

template<typename T>
T*
AbstractDistMatrix<T>::Buffer(Int iLoc, Int jLoc) EL_NO_RELEASE_EXCEPT
{ return Matrix().Buffer(iLoc,jLoc); }

template<typename T>
const T*
AbstractDistMatrix<T>::LockedBuffer() const EL_NO_EXCEPT
{ return LockedMatrix().LockedBuffer(); }

template<typename T>
const T*
AbstractDistMatrix<T>::LockedBuffer(Int iLoc, Int jLoc) const EL_NO_EXCEPT
{ return LockedMatrix().LockedBuffer(iLoc,jLoc); }

// Distribution information
// ------------------------

template<typename T>
const El::Grid& AbstractDistMatrix<T>::Grid() const EL_NO_EXCEPT
{ return *grid_; }

template<typename T>
int AbstractDistMatrix<T>::ColAlign() const EL_NO_EXCEPT { return colAlign_; }
template<typename T>
int AbstractDistMatrix<T>::RowAlign() const EL_NO_EXCEPT { return rowAlign_; }

template<typename T>
int AbstractDistMatrix<T>::ColShift() const EL_NO_EXCEPT { return colShift_; }
template<typename T>
int AbstractDistMatrix<T>::RowShift() const EL_NO_EXCEPT { return rowShift_; }

template<typename T>
bool AbstractDistMatrix<T>::ColConstrained() const EL_NO_EXCEPT
{ return colConstrained_; }
template<typename T>
bool AbstractDistMatrix<T>::RowConstrained() const EL_NO_EXCEPT
{ return rowConstrained_; }
template<typename T>
bool AbstractDistMatrix<T>::RootConstrained() const EL_NO_EXCEPT
{ return rootConstrained_; }

template<typename T>
bool AbstractDistMatrix<T>::Participating() const EL_NO_RELEASE_EXCEPT
{ return grid_->InGrid() && (CrossRank()==root_); }

template<typename T>
int AbstractDistMatrix<T>::Owner(Int i, Int j) const EL_NO_EXCEPT
{ return RowOwner(i)+ColOwner(j)*ColStride(); }

template<typename T>
Int AbstractDistMatrix<T>::LocalRow(Int i) const EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if(!IsLocalRow(i))
          LogicError
          ("Row ",i," is owned by ",RowOwner(i),", not ",ColRank());
   )
    return LocalRowOffset(i);
}

template<typename T>
Int AbstractDistMatrix<T>::LocalCol(Int j) const EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if(!IsLocalCol(j))
          LogicError
          ("Column ",j," is owned by ",ColOwner(j),", not ",RowRank());
   )
    return LocalColOffset(j);
}

template<typename T>
Int AbstractDistMatrix<T>::LocalRow(Int i, int rowOwner) const
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if(RowOwner(i) != rowOwner)
          LogicError
          ("Row ",i,"is owned by ",RowOwner(i)," not ",rowOwner);
   )
    return LocalRowOffset(i,rowOwner);
}

template<typename T>
Int AbstractDistMatrix<T>::LocalCol(Int j, int colOwner) const
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if(ColOwner(j) != colOwner)
          LogicError
          ("Column ",j,"is owned by ",ColOwner(j),", not ",colOwner);
   )
    return LocalColOffset(j,colOwner);
}

template<typename T>
bool AbstractDistMatrix<T>::IsLocalRow(Int i) const EL_NO_RELEASE_EXCEPT
{ return Participating() && RowOwner(i) == ColRank(); }
template<typename T>
bool AbstractDistMatrix<T>::IsLocalCol(Int j) const EL_NO_RELEASE_EXCEPT
{ return Participating() && ColOwner(j) == RowRank(); }
template<typename T>
bool AbstractDistMatrix<T>::IsLocal(Int i, Int j) const EL_NO_RELEASE_EXCEPT
{ return IsLocalRow(i) && IsLocalCol(j); }

template<typename T>
int AbstractDistMatrix<T>::Root() const EL_NO_EXCEPT { return root_; }

template<typename T>
El::DistData AbstractDistMatrix<T>::DistData() const EL_NO_EXCEPT
{ return El::DistData(*this); }

template<typename T>
void
AbstractDistMatrix<T>::SetShifts()
{
    if(Participating())
    {
        colShift_ = Shift(ColRank(),colAlign_,ColStride());
        rowShift_ = Shift(RowRank(),rowAlign_,RowStride());
    }
    else
    {
        colShift_ = 0;
        rowShift_ = 0;
    }
}

template<typename T>
void
AbstractDistMatrix<T>::SetColShift()
{
    if(Participating())
        colShift_ = Shift(ColRank(),colAlign_,ColStride());
    else
        colShift_ = 0;
}

template<typename T>
void
AbstractDistMatrix<T>::SetRowShift()
{
    if(Participating())
        rowShift_ = Shift(RowRank(),rowAlign_,RowStride());
    else
        rowShift_ = 0;
}

// Assertions
// ==========

template<typename T>
void
AbstractDistMatrix<T>::AssertNotLocked() const
{
    if(Locked())
        LogicError("Assertion that matrix not be a locked view failed");
}

template<typename T>
void
AbstractDistMatrix<T>::AssertNotStoringData() const
{
    if(LockedMatrix().MemorySize() > 0)
        LogicError("Assertion that matrix not be storing data failed");
}

template<typename T>
void
AbstractDistMatrix<T>::AssertValidEntry(Int i, Int j) const
{
    if(i == END) i = height_ - 1;
    if(j == END) j = width_ - 1;
    if(i < 0 || i >= Height() || j < 0 || j >= Width())
        LogicError
        ("Entry (",i,",",j,") is out of bounds of ",Height(),
         " x ",Width()," matrix");
}

template<typename T>
void
AbstractDistMatrix<T>::AssertValidSubmatrix
(Int i, Int j, Int height, Int width) const
{
    if(i == END) i = height_ - 1;
    if(j == END) j = width_ - 1;
    if(i < 0 || j < 0)
        LogicError("Indices of submatrix were negative");
    if(height < 0 || width < 0)
        LogicError("Dimensions of submatrix were negative");
    if((i+height) > Height() || (j+width) > Width())
        LogicError
        ("Submatrix is out of bounds: accessing up to (",i+height-1,
         ",",j+width-1,") of ",Height()," x ",Width()," matrix");
}

template<typename T>
void
AbstractDistMatrix<T>::AssertSameSize(Int height, Int width) const
{
    if(Height() != height || Width() != width)
        LogicError("Assertion that matrices be the same size failed");
}

// Static functions
// ================

namespace
{

template<typename T>
AbstractDistMatrix<T>*
Instantiate_
(const El::Grid& grid, int root,
 Dist colDist, Dist rowDist, DistWrap wrap, Device dev)
{
#define GUARD(CDIST,RDIST,WRAP,DEVICE)                          \
    (colDist == CDIST) && (rowDist == RDIST) && (wrap == WRAP)  \
        && (dev == DEVICE)
#define PAYLOAD(CDIST,RDIST,WRAP,DEVICE)                                \
    return new DistMatrix<T,CDIST,RDIST,WRAP,DEVICE>(grid, root);
#include "El/macros/DeviceGuardAndPayload.h"
    LogicError
    ("Invalid template arguments for DistMatrix "
     "(colDist=",Int(colDist),", rowDist=",Int(rowDist),", "
     "wrap=",Int(wrap),", dev=",Int(dev),")");
    return nullptr;
}

} // namespace <anon>

template<typename T>
AbstractDistMatrix<T>*
AbstractDistMatrix<T>::Instantiate
(const El::Grid& grid, int root,
 Dist colDist, Dist rowDist, DistWrap wrap, Device dev)
{
    LogicError("Invalid type for DistMatrix");
    return nullptr;
}

template<>
AbstractDistMatrix<float>*
AbstractDistMatrix<float>::Instantiate
(const El::Grid& grid, int root,
 Dist colDist, Dist rowDist, DistWrap wrap, Device dev)
{
    return Instantiate_<float>(grid, root, colDist, rowDist, wrap, dev);
}

template<>
AbstractDistMatrix<double>*
AbstractDistMatrix<double>::Instantiate
(const El::Grid& grid, int root,
 Dist colDist, Dist rowDist, DistWrap wrap, Device dev)
{
    return Instantiate_<double>(grid, root, colDist, rowDist, wrap, dev);
}

#ifdef HYDROGEN_HAVE_HALF


template<>
AbstractDistMatrix<cpu_half_type>*
AbstractDistMatrix<cpu_half_type>::Instantiate
(const El::Grid& grid, int root,
 Dist colDist, Dist rowDist, DistWrap wrap, Device dev)
{
    if (dev != Device::CPU)
    {
        LogicError("cpu_half_type only supported on CPU");
    }

#define GUARD(CDIST,RDIST,WRAP)                                         \
    (colDist == CDIST) && (rowDist == RDIST) && (wrap == WRAP)
#define PAYLOAD(CDIST,RDIST,WRAP)                                       \
    return new DistMatrix<cpu_half_type,CDIST,RDIST,WRAP,Device::CPU>(grid, root);
#include "El/macros/GuardAndPayload.h"
    LogicError
        ("Invalid template arguments for DistMatrix "
         "(colDist=",Int(colDist),", rowDist=",Int(rowDist),", "
         "wrap=",Int(wrap),", dev=",Int(dev),")");
    return nullptr;
}

#ifdef HYDROGEN_GPU_USE_FP16
template<>
AbstractDistMatrix<gpu_half_type>*
AbstractDistMatrix<gpu_half_type>::Instantiate
(const El::Grid& grid, int root,
 Dist colDist, Dist rowDist, DistWrap wrap, Device dev)
{
    if (dev != Device::GPU)
    {
        LogicError("gpu_half_type only supported on GPU");
    }
    if (wrap != ELEMENT)
    {
        LogicError("GPU matrices only supported with ELEMENT distributions");
    }
#define GUARD(CDIST,RDIST,WRAP)                                         \
    (colDist == CDIST) && (rowDist == RDIST) && (wrap == WRAP)
#define PAYLOAD(CDIST,RDIST,WRAP)                                       \
    return new DistMatrix<gpu_half_type,CDIST,RDIST,ELEMENT,Device::GPU>(grid, root);
#include "El/macros/GuardAndPayload.h"
    LogicError
        ("Invalid template arguments for DistMatrix "
         "(colDist=",Int(colDist),", rowDist=",Int(rowDist),", "
         "wrap=",Int(wrap),", dev=",Int(dev),")");
    return nullptr;
}
#endif // HYDROGEN_GPU_USE_FP16
#endif // HYDROGEN_HAVE_HALF

template<typename T>
AbstractDistMatrix<T>*
AbstractDistMatrix<T>::Instantiate(const El::DistData& data)
{
    const DistWrap wrap = ((data.blockHeight == 1
                            && data.blockWidth == 1) ?
                           ELEMENT : BLOCK);
    return AbstractDistMatrix<T>::Instantiate
           (*data.grid, data.root,
            data.colDist, data.rowDist, wrap, data.device);
}

// Private section
// ###############

// Exchange metadata with another matrix
// =====================================

template<typename T>
void
AbstractDistMatrix<T>::ShallowSwap(AbstractDistMatrix<T>& A)
{
    Matrix().ShallowSwap(A.Matrix());
    std::swap(viewType_, A.viewType_);
    std::swap(height_ , A.height_);
    std::swap(width_, A.width_);
    std::swap(colConstrained_, A.colConstrained_);
    std::swap(rowConstrained_, A.rowConstrained_);
    std::swap(rootConstrained_, A.rootConstrained_);
    std::swap(colAlign_, A.colAlign_);
    std::swap(rowAlign_, A.rowAlign_);
    std::swap(colShift_, A.colShift_);
    std::swap(rowShift_, A.rowShift_);
    std::swap(root_, A.root_);
    std::swap(grid_, A.grid_);
}

// Instantiations for {Int,Real,Complex<Real>} for each Real in {float,double,half}
// ###########################################################################

#define PROTO(T) template class AbstractDistMatrix<T>;

#ifdef HYDROGEN_GPU_USE_FP16
PROTO(gpu_half_type)
#endif

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#define EL_ENABLE_HALF
#include <El/macros/Instantiate.h>

} // namespace El
