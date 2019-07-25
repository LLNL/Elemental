/*
  Copyright (c) 2009-2016, Jack Poulson
  All rights reserved.

  This file is part of Elemental and is under the BSD 2-Clause License,
  which can be found in the LICENSE file in the root directory, or at
  http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_VIEW_IMPL_HPP
#define EL_VIEW_IMPL_HPP

namespace El
{

// View an entire matrix
// =====================

// (Sequential) matrix
// -------------------

template<typename T, Device D>
void View(Matrix<T,D>& A, Matrix<T,D>& B)
{
    EL_DEBUG_CSE
        if (B.Locked())
            A.LockedAttach
                (B.Height(), B.Width(), B.LockedBuffer(), B.LDim());
        else
            A.Attach(B.Height(), B.Width(), B.Buffer(), B.LDim());
}

template<typename T, Device D>
void LockedView(Matrix<T,D>& A, const Matrix<T,D>& B)
{
    EL_DEBUG_CSE
        A.LockedAttach(B.Height(), B.Width(), B.LockedBuffer(), B.LDim());
}

template<typename T, Device D>
Matrix<T,D> View(Matrix<T,D>& B)
{
    Matrix<T,D> A;
    View(A, B);
    return A;
}

template<typename T, Device D>
Matrix<T,D> LockedView(const Matrix<T,D>& B)
{
    Matrix<T,D> A;
    LockedView(A, B);
    return A;
}

// Abstract Sequential Matrix
// --------------------------

template<typename T>
void View(AbstractMatrix<T>& A, AbstractMatrix<T>& B)
{
    if (A.GetDevice() != B.GetDevice())
        LogicError("View requires matching device types.");

    switch(A.GetDevice()) {
    case Device::CPU:
        View(static_cast<Matrix<T,Device::CPU>&>(A),
             static_cast<Matrix<T,Device::CPU>&>(B));
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        View(static_cast<Matrix<T,Device::GPU>&>(A),
             static_cast<Matrix<T,Device::GPU>&>(B));
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("Unsupported device type.");
    }
}

template<typename T>
void LockedView(AbstractMatrix<T>& A, const AbstractMatrix<T>& B)
{
    if (A.GetDevice() != B.GetDevice())
        LogicError("View requires matching device types.");

    switch(A.GetDevice()) {
    case Device::CPU:
        LockedView(static_cast<Matrix<T,Device::CPU>&>(A),
                   static_cast<const Matrix<T,Device::CPU>&>(B));
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        LockedView(static_cast<Matrix<T,Device::GPU>&>(A),
                   static_cast<const Matrix<T,Device::GPU>&>(B));
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("Unsupported device type.");
    }
}

// ElementalMatrix
// ---------------

template<typename T>
void View(ElementalMatrix<T>& A, ElementalMatrix<T>& B)
{
    EL_DEBUG_CSE
        EL_DEBUG_ONLY(AssertSameDist(A.DistData(), B.DistData()))
        if (B.Locked())
            A.LockedAttach
                (B.Height(), B.Width(), B.Grid(), B.ColAlign(), B.RowAlign(),
                 B.LockedBuffer(), B.LDim(), B.Root());
        else
            A.Attach
                (B.Height(), B.Width(), B.Grid(), B.ColAlign(), B.RowAlign(),
                 B.Buffer(), B.LDim(), B.Root());
}

template<typename T>
void LockedView
(ElementalMatrix<T>& A, const ElementalMatrix<T>& B)
{
    EL_DEBUG_CSE
        EL_DEBUG_ONLY(AssertSameDist(A.DistData(), B.DistData()))
        A.LockedAttach
        (B.Height(), B.Width(), B.Grid(), B.ColAlign(), B.RowAlign(),
         B.LockedBuffer(), B.LDim(), B.Root());
}

// Return by value
// ^^^^^^^^^^^^^^^
template<typename T,Dist U,Dist V,DistWrap wrapType,Device D>
DistMatrix<T,U,V,wrapType,D> View(DistMatrix<T,U,V,wrapType,D>& B)
{
    DistMatrix<T,U,V,wrapType,D> A(B.Grid());
    View(A, B);
    return A;
}

template<typename T,Dist U,Dist V,DistWrap wrapType,Device D>
DistMatrix<T,U,V,wrapType,D> LockedView(const DistMatrix<T,U,V,wrapType,D>& B)
{
    DistMatrix<T,U,V,wrapType,D> A(B.Grid());
    LockedView(A, B);
    return A;
}

// BlockMatrix
// -----------
template<typename T>
void View(BlockMatrix<T>& A, BlockMatrix<T>& B)
{
    EL_DEBUG_CSE
        EL_DEBUG_ONLY(AssertSameDist(A.DistData(), B.DistData()))
        if (B.Locked())
            A.LockedAttach
                (B.Height(), B.Width(), B.Grid(), B.BlockHeight(), B.BlockWidth(),
                 B.ColAlign(), B.RowAlign(), B.ColCut(), B.RowCut(),
                 B.LockedBuffer(), B.LDim(), B.Root());
        else
            A.Attach
                (B.Height(), B.Width(), B.Grid(), B.BlockHeight(), B.BlockWidth(),
                 B.ColAlign(), B.RowAlign(), B.ColCut(), B.RowCut(),
                 B.Buffer(), B.LDim(), B.Root());
}

template<typename T>
void LockedView
(BlockMatrix<T>& A, const BlockMatrix<T>& B)
{
    EL_DEBUG_CSE
        EL_DEBUG_ONLY(AssertSameDist(A.DistData(), B.DistData()))
        A.LockedAttach
        (B.Height(), B.Width(), B.Grid(), B.BlockHeight(), B.BlockWidth(),
         B.ColAlign(), B.RowAlign(), B.ColCut(), B.RowCut(),
         B.LockedBuffer(), B.LDim(), B.Root());
}

// Mixed
// -----
template<typename T>
void View
(BlockMatrix<T>& A,
 ElementalMatrix<T>& B)
{
    EL_DEBUG_CSE
        EL_DEBUG_ONLY(AssertSameDist(A.DistData(), B.DistData()))
        if (B.Locked())
            A.LockedAttach
                (B.Height(), B.Width(), B.Grid(),
                 1, 1, B.ColAlign(), B.RowAlign(), 0, 0,
                 B.LockedBuffer(), B.LDim(), B.Root());
        else
            A.Attach
                (B.Height(), B.Width(), B.Grid(),
                 1, 1, B.ColAlign(), B.RowAlign(), 0, 0,
                 B.Buffer(), B.LDim(), B.Root());
}

template<typename T>
void LockedView
(      BlockMatrix<T>& A,
       const ElementalMatrix<T>& B)
{
    EL_DEBUG_CSE
        EL_DEBUG_ONLY(AssertSameDist(A.DistData(), B.DistData()))
        A.LockedAttach
        (B.Height(), B.Width(), B.Grid(), 1, 1, B.ColAlign(), B.RowAlign(), 0, 0,
         B.LockedBuffer(), B.LDim(), B.Root());
}

template<typename T>
void View
(ElementalMatrix<T>& A,
 BlockMatrix<T>& B)
{
    EL_DEBUG_CSE
        EL_DEBUG_ONLY(AssertSameDist(A.DistData(), B.DistData()))
        if (B.BlockHeight() != 1 || B.BlockWidth() != 1)
            LogicError("Block size was ",B.BlockHeight()," x ",B.BlockWidth(),
                       " instead of 1x1");
    if (B.Locked())
        A.LockedAttach
            (B.Height(), B.Width(), B.Grid(), B.ColAlign(), B.RowAlign(),
             B.LockedBuffer(), B.LDim(), B.Root());
    else
        A.Attach
            (B.Height(), B.Width(), B.Grid(), B.ColAlign(), B.RowAlign(),
             B.Buffer(), B.LDim(), B.Root());
}

template<typename T>
void LockedView
(      ElementalMatrix<T>& A,
       const BlockMatrix<T>& B)
{
    EL_DEBUG_CSE
        EL_DEBUG_ONLY(AssertSameDist(A.DistData(), B.DistData()))
        if (B.BlockHeight() != 1 || B.BlockWidth() != 1)
            LogicError("Block size was ",B.BlockHeight()," x ",B.BlockWidth(),
                       " instead of 1x1");
    A.LockedAttach
        (B.Height(), B.Width(), B.Grid(), B.ColAlign(), B.RowAlign(),
         B.LockedBuffer(), B.LDim(), B.Root());
}

// AbstractDistMatrix
// ------------------
template<typename T>
void View
(AbstractDistMatrix<T>& A,
 AbstractDistMatrix<T>& B)
{
    auto AWrap = A.Wrap();
    auto BWrap = B.Wrap();
    if (AWrap == ELEMENT && BWrap == ELEMENT)
    {
        auto& ACast = static_cast<ElementalMatrix<T>&>(A);
        auto& BCast = static_cast<ElementalMatrix<T>&>(B);
        View(ACast, BCast);
    }
    else if (AWrap == ELEMENT && BWrap == BLOCK)
    {
        auto& ACast = static_cast<ElementalMatrix<T>&>(A);
        auto& BCast = static_cast<BlockMatrix<T>&>(B);
        View(ACast, BCast);
    }
    else if (AWrap == BLOCK && BWrap == ELEMENT)
    {
        auto& ACast = static_cast<BlockMatrix<T>&>(A);
        auto& BCast = static_cast<ElementalMatrix<T>&>(B);
        View(ACast, BCast);
    }
    else
    {
        auto& ACast = static_cast<BlockMatrix<T>&>(A);
        auto& BCast = static_cast<BlockMatrix<T>&>(B);
        View(ACast, BCast);
    }
}

template<typename T>
void LockedView
(      AbstractDistMatrix<T>& A,
       const AbstractDistMatrix<T>& B)
{
    auto AWrap = A.Wrap();
    auto BWrap = B.Wrap();
    if (AWrap == ELEMENT && BWrap == ELEMENT)
    {
        auto& ACast = static_cast<ElementalMatrix<T>&>(A);
        auto& BCast = static_cast<const ElementalMatrix<T>&>(B);
        LockedView(ACast, BCast);
    }
    else if (AWrap == ELEMENT && BWrap == BLOCK)
    {
        auto& ACast = static_cast<ElementalMatrix<T>&>(A);
        auto& BCast = static_cast<const BlockMatrix<T>&>(B);
        LockedView(ACast, BCast);
    }
    else if (AWrap == BLOCK && BWrap == ELEMENT)
    {
        auto& ACast = static_cast<BlockMatrix<T>&>(A);
        auto& BCast = static_cast<const ElementalMatrix<T>&>(B);
        LockedView(ACast, BCast);
    }
    else
    {
        auto& ACast = static_cast<BlockMatrix<T>&>(A);
        auto& BCast = static_cast<const BlockMatrix<T>&>(B);
        LockedView(ACast, BCast);
    }
}

// View a contiguous submatrix
// ===========================

// (Sequential) Matrix
// -------------------

template<typename T, Device D>
void View(Matrix<T,D>& A, Matrix<T,D>& B,
          Int i, Int j,
          Int height, Int width)
{
    EL_DEBUG_CSE;
#ifndef EL_RELEASE
    if (i < 0 || j < 0)
        LogicError("Indices must be non-negative");
    if (height < 0 || width < 0)
        LogicError("Height and width must be non-negative");
    if ((i+height) > B.Height() || (j+width) > B.Width())
        LogicError
            ("Trying to view outside of a Matrix: (",i,",",j,") up to (",
             i+height-1,",",j+width-1,") of ",B.Height()," x ",B.Width(),
             " Matrix");
#endif // EL_RELEASE

    if (B.Locked())
        A.LockedAttach(height, width, B.LockedBuffer(i,j), B.LDim());
    else
        A.Attach(height, width, B.Buffer(i,j), B.LDim());
}

template<typename T, Device D>
void LockedView(Matrix<T,D>& A, Matrix<T,D> const& B,
                Int i, Int j,
                Int height, Int width)
{
    EL_DEBUG_CSE;
#ifndef EL_RELEASE
    if (i < 0 || j < 0)
        LogicError("Indices must be non-negative");
    if (height < 0 || width < 0)
        LogicError("Height and width must be non-negative");
    if ((i+height) > B.Height() || (j+width) > B.Width())
        LogicError
            ("Trying to view outside of a Matrix: (",i,",",j,") up to (",
             i+height-1,",",j+width-1,") of ",B.Height()," x ",B.Width(),
             " Matrix");
#endif // !EL_RELEASE

    A.LockedAttach(height, width, B.LockedBuffer(i,j), B.LDim());
}

template<typename T, Device D>
void View(Matrix<T,D>& A, Matrix<T,D>& B,
          Range<Int> I, Range<Int> J)
{
    if (I.end == END)
        I.end = B.Height();
    if (J.end == END)
        J.end = B.Width();
    View(A, B, I.beg, J.beg, I.end-I.beg, J.end-J.beg);
}

template<typename T, Device D>
void LockedView(Matrix<T,D>& A, Matrix<T,D> const& B,
                Range<Int> I, Range<Int> J)
{
    if (I.end == END)
        I.end = B.Height();
    if (J.end == END)
        J.end = B.Width();
    LockedView(A, B, I.beg, J.beg, I.end-I.beg, J.end-J.beg);
}

// Return by value
// ^^^^^^^^^^^^^^^

template<typename T, Device D>
Matrix<T,D> View(Matrix<T,D>& B, Int i, Int j, Int height, Int width)
{
    Matrix<T,D> A;
    View(A, B, i, j, height, width);
    return A;
}

template<typename T, Device D>
Matrix<T,D> LockedView
(Matrix<T,D> const& B, Int i, Int j, Int height, Int width)
{
    Matrix<T,D> A;
    LockedView(A, B, i, j, height, width);
    return A;
}

template<typename T, Device D>
Matrix<T,D> View(Matrix<T,D>& B, Range<Int> I, Range<Int> J)
{
    if (I.end == END)
        I.end = B.Height();
    if (J.end == END)
        J.end = B.Width();
    return View(B, I.beg, J.beg, I.end-I.beg, J.end-J.beg);
}

template<typename T, Device D>
Matrix<T,D> LockedView(Matrix<T,D> const& B, Range<Int> I, Range<Int> J)
{
    if (I.end == END)
        I.end = B.Height();
    if (J.end == END)
        J.end = B.Width();
    return LockedView(B, I.beg, J.beg, I.end-I.beg, J.end-J.beg);
}

// Abstract Sequential Matrix
// --------------------------

template<typename T>
void View(AbstractMatrix<T>& A, AbstractMatrix<T>& B,
          Range<Int> I, Range<Int> J)
{
    if (A.GetDevice() != B.GetDevice())
        LogicError("View requires matching device types.");

    switch(A.GetDevice()) {
    case Device::CPU:
        View(static_cast<Matrix<T,Device::CPU>&>(A),
             static_cast<Matrix<T,Device::CPU>&>(B), I, J);
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        View(static_cast<Matrix<T,Device::GPU>&>(A),
             static_cast<Matrix<T,Device::GPU>&>(B), I, J);
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("Unsupported device type.");
    }
}

template<typename T>
void LockedView(AbstractMatrix<T>& A, AbstractMatrix<T> const& B,
                Range<Int> I, Range<Int> J)
{
    if (A.GetDevice() != B.GetDevice())
        LogicError("View requires matching device types.");

    switch(A.GetDevice()) {
    case Device::CPU:
      LockedView(static_cast<Matrix<T,Device::CPU>&>(A),
                 static_cast<const Matrix<T,Device::CPU>&>(B), I, J);
      break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
      LockedView(static_cast<Matrix<T,Device::GPU>&>(A),
                 static_cast<const Matrix<T,Device::GPU>&>(B), I, J);
      break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("Unsupported device type.");
    }
}

// ElementalMatrix
// ---------------

template<typename T>
void View
(ElementalMatrix<T>& A,
 ElementalMatrix<T>& B,
 Int i, Int j, Int height, Int width)
{
    EL_DEBUG_CSE
        EL_DEBUG_ONLY(
            AssertSameDist(A.DistData(), B.DistData());
            B.AssertValidSubmatrix(i, j, height, width);
            )
        const Int colAlign = B.RowOwner(i);
    const Int rowAlign = B.ColOwner(j);
    if (B.Participating())
    {
        const Int iLoc = B.LocalRowOffset(i);
        const Int jLoc = B.LocalColOffset(j);
        if (B.Locked())
            A.LockedAttach
                (height, width, B.Grid(), colAlign, rowAlign,
                 B.LockedBuffer(iLoc,jLoc), B.LDim(), B.Root());
        else
            A.Attach
                (height, width, B.Grid(), colAlign, rowAlign,
                 B.Buffer(iLoc,jLoc), B.LDim(), B.Root());
    }
    else
    {
        if (B.Locked())
            A.LockedAttach
                (height, width, B.Grid(),
                 colAlign, rowAlign, 0, B.LDim(), B.Root());
        else
            A.Attach
                (height, width, B.Grid(),
                 colAlign, rowAlign, 0, B.LDim(), B.Root());
    }
}

template<typename T>
void LockedView
(      ElementalMatrix<T>& A,
       const ElementalMatrix<T>& B,
       Int i, Int j, Int height, Int width)
{
    EL_DEBUG_CSE
        EL_DEBUG_ONLY(
            AssertSameDist(A.DistData(), B.DistData());
            B.AssertValidSubmatrix(i, j, height, width);
            )
        const Int colAlign = B.RowOwner(i);
    const Int rowAlign = B.ColOwner(j);
    if (B.Participating())
    {
        const Int iLoc = B.LocalRowOffset(i);
        const Int jLoc = B.LocalColOffset(j);
        A.LockedAttach
            (height, width, B.Grid(), colAlign, rowAlign,
             B.LockedBuffer(iLoc,jLoc), B.LDim(), B.Root());
    }
    else
    {
        A.LockedAttach
            (height, width, B.Grid(), colAlign, rowAlign, 0, B.LDim(), B.Root());
    }
}

template<typename T>
void View
(ElementalMatrix<T>& A,
 ElementalMatrix<T>& B,
 Range<Int> I, Range<Int> J)
{
    if (I.end == END)
        I.end = B.Height();
    if (J.end == END)
        J.end = B.Width();
    View(A, B, I.beg, J.beg, I.end-I.beg, J.end-J.beg);
}

template<typename T>
void LockedView
(      ElementalMatrix<T>& A,
       const ElementalMatrix<T>& B,
       Range<Int> I, Range<Int> J)
{
    if (I.end == END)
        I.end = B.Height();
    if (J.end == END)
        J.end = B.Width();
    LockedView(A, B, I.beg, J.beg, I.end-I.beg, J.end-J.beg);
}

// Return by value
// ^^^^^^^^^^^^^^^

template<typename T,Dist U,Dist V,DistWrap wrapType,Device D>
DistMatrix<T,U,V,wrapType,D> View
(DistMatrix<T,U,V,wrapType,D>& B, Int i, Int j, Int height, Int width)
{
    DistMatrix<T,U,V,wrapType,D> A(B.Grid());
    View(A, B, i, j, height, width);
    return A;
}

template<typename T,Dist U,Dist V,DistWrap wrapType,Device D>
DistMatrix<T,U,V,wrapType,D> LockedView
(const DistMatrix<T,U,V,wrapType,D>& B, Int i, Int j, Int height, Int width)
{
    DistMatrix<T,U,V,wrapType,D> A(B.Grid());
    LockedView(A, B, i, j, height, width);
    return A;
}

template<typename T,Dist U,Dist V,DistWrap wrapType,Device D>
DistMatrix<T,U,V,wrapType,D> View
(DistMatrix<T,U,V,wrapType,D>& B, Range<Int> I, Range<Int> J)
{
    if (I.end == END)
        I.end = B.Height();
    if (J.end == END)
        J.end = B.Width();
    return View(B, I.beg, J.beg, I.end-I.beg, J.end-J.beg);
}

template<typename T,Dist U,Dist V,DistWrap wrapType,Device D>
DistMatrix<T,U,V,wrapType,D> LockedView
(const DistMatrix<T,U,V,wrapType,D>& B, Range<Int> I, Range<Int> J)
{
    if (I.end == END)
        I.end = B.Height();
    if (J.end == END)
        J.end = B.Width();
    return LockedView(B, I.beg, J.beg, I.end-I.beg, J.end-J.beg);
}

// BlockMatrix
// -----------

template<typename T>
void View
(BlockMatrix<T>& A,
 BlockMatrix<T>& B,
 Int i,
 Int j,
 Int height,
 Int width)
{
    EL_DEBUG_CSE
        EL_DEBUG_ONLY(
            AssertSameDist(A.DistData(), B.DistData());
            B.AssertValidSubmatrix(i, j, height, width);
            )
        const Int iLoc = B.LocalRowOffset(i);
    const Int jLoc = B.LocalColOffset(j);
    if (B.Locked())
        A.LockedAttach
            (height, width, B.Grid(),
             B.BlockHeight(), B.BlockWidth(),
             B.RowOwner(i), B.ColOwner(j),
             Mod(B.ColCut()+i,B.BlockHeight()),
             Mod(B.RowCut()+j,B.BlockWidth()),
             B.LockedBuffer(iLoc,jLoc), B.LDim(), B.Root());
    else
        A.Attach
            (height, width, B.Grid(),
             B.BlockHeight(), B.BlockWidth(),
             B.RowOwner(i), B.ColOwner(j),
             Mod(B.ColCut()+i,B.BlockHeight()),
             Mod(B.RowCut()+j,B.BlockWidth()),
             B.Buffer(iLoc,jLoc), B.LDim(), B.Root());
}

template<typename T>
void LockedView
(      BlockMatrix<T>& A,
       const BlockMatrix<T>& B,
       Int i, Int j, Int height, Int width)
{
    EL_DEBUG_CSE
        EL_DEBUG_ONLY(
            AssertSameDist(A.DistData(), B.DistData());
            B.AssertValidSubmatrix(i, j, height, width);
            )
        const Int iLoc = B.LocalRowOffset(i);
    const Int jLoc = B.LocalColOffset(j);
    A.LockedAttach
        (height, width, B.Grid(),
         B.BlockHeight(), B.BlockWidth(),
         B.RowOwner(i), B.ColOwner(j),
         Mod(B.ColCut()+i,B.BlockHeight()),
         Mod(B.RowCut()+j,B.BlockWidth()),
         B.LockedBuffer(iLoc,jLoc), B.LDim(), B.Root());
}

template<typename T>
void View
(BlockMatrix<T>& A,
 BlockMatrix<T>& B,
 Range<Int> I, Range<Int> J)
{
    if (I.end == END)
        I.end = B.Height();
    if (J.end == END)
        J.end = B.Width();
    View(A, B, I.beg, J.beg, I.end-I.beg, J.end-J.beg);
}

template<typename T>
void LockedView
(      BlockMatrix<T>& A,
       const BlockMatrix<T>& B,
       Range<Int> I, Range<Int> J)
{
    if (I.end == END)
        I.end = B.Height();
    if (J.end == END)
        J.end = B.Width();
    LockedView(A, B, I.beg, J.beg, I.end-I.beg, J.end-J.beg);
}

// AbstractDistMatrix
// ------------------
template<typename T>
void View
(AbstractDistMatrix<T>& A,
 AbstractDistMatrix<T>& B,
 Int i, Int j, Int height, Int width)
{
    auto AWrap = A.Wrap();
    auto BWrap = B.Wrap();
    if (AWrap == ELEMENT && BWrap == ELEMENT)
    {
        auto& ACast = static_cast<ElementalMatrix<T>&>(A);
        auto& BCast = static_cast<ElementalMatrix<T>&>(B);
        View(ACast, BCast, i, j, height, width);
    }
    else if (AWrap == ELEMENT && BWrap == BLOCK)
    {
        auto& ACast = static_cast<ElementalMatrix<T>&>(A);
        auto& BCast = static_cast<BlockMatrix<T>&>(B);
        View(ACast, BCast, i, j, height, width);
    }
    else if (AWrap == BLOCK && BWrap == ELEMENT)
    {
        auto& ACast = static_cast<BlockMatrix<T>&>(A);
        auto& BCast = static_cast<ElementalMatrix<T>&>(B);
        View(ACast, BCast, i, j, height, width);
    }
    else
    {
        auto& ACast = static_cast<BlockMatrix<T>&>(A);
        auto& BCast = static_cast<BlockMatrix<T>&>(B);
        View(ACast, BCast, i, j, height, width);
    }
}

template<typename T>
void LockedView
(      AbstractDistMatrix<T>& A,
       const AbstractDistMatrix<T>& B,
       Int i, Int j, Int height, Int width)
{
    auto AWrap = A.Wrap();
    auto BWrap = B.Wrap();
    if (AWrap == ELEMENT && BWrap == ELEMENT)
    {
        auto& ACast = static_cast<ElementalMatrix<T>&>(A);
        auto& BCast = static_cast<const ElementalMatrix<T>&>(B);
        LockedView(ACast, BCast, i, j, height, width);
    }
    else if (AWrap == ELEMENT && BWrap == BLOCK)
    {
        auto& ACast = static_cast<ElementalMatrix<T>&>(A);
        auto& BCast = static_cast<const BlockMatrix<T>&>(B);
        LockedView(ACast, BCast, i, j, height, width);
    }
    else if (AWrap == BLOCK && BWrap == ELEMENT)
    {
        auto& ACast = static_cast<BlockMatrix<T>&>(A);
        auto& BCast = static_cast<const ElementalMatrix<T>&>(B);
        LockedView(ACast, BCast, i, j, height, width);
    }
    else
    {
        auto& ACast = static_cast<BlockMatrix<T>&>(A);
        auto& BCast = static_cast<const BlockMatrix<T>&>(B);
        LockedView(ACast, BCast, i, j, height, width);
    }
}

template<typename T>
void View
(AbstractDistMatrix<T>& A,
 AbstractDistMatrix<T>& B,
 Range<Int> I, Range<Int> J)
{
    if (I.end == END)
        I.end = B.Height();
    if (J.end == END)
        J.end = B.Width();
    View(A, B, I.beg, J.beg, I.end-I.beg, J.end-J.beg);
}

template<typename T>
void LockedView
(      AbstractDistMatrix<T>& A,
       const AbstractDistMatrix<T>& B,
       Range<Int> I, Range<Int> J)
{
    if (I.end == END)
        I.end = B.Height();
    if (J.end == END)
        J.end = B.Width();
    LockedView(A, B, I.beg, J.beg, I.end-I.beg, J.end-J.beg);
}

#ifdef EL_INSTANTIATE_CORE
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  /* View an entire matrix
     ===================== */ \
  /* Sequential matrix
     ----------------- */ \
  EL_EXTERN template void View(Matrix<T>& A, Matrix<T>& B); \
  EL_EXTERN template void LockedView(Matrix<T>& A, const Matrix<T>& B); \
  EL_EXTERN template Matrix<T> View(Matrix<T>& B); \
  EL_EXTERN template Matrix<T> LockedView(const Matrix<T>& B); \
  /* Abstract Sequential Matrix
     -------------------------- */ \
  EL_EXTERN template void View(AbstractMatrix<T>& A, AbstractMatrix<T>& B); \
  EL_EXTERN template void LockedView(AbstractMatrix<T>& A, const AbstractMatrix<T>& B); \
  /* ElementalMatrix
     --------------- */ \
  EL_EXTERN template void View \
  (ElementalMatrix<T>& A, ElementalMatrix<T>& B); \
  EL_EXTERN template void LockedView \
  (ElementalMatrix<T>& A, const ElementalMatrix<T>& B); \
  /* BlockMatrix
     ----------- */ \
  EL_EXTERN template void View \
  (BlockMatrix<T>& A, BlockMatrix<T>& B); \
  EL_EXTERN template void LockedView \
  (BlockMatrix<T>& A, const BlockMatrix<T>& B); \
  /* Mixed
     ----- */ \
  EL_EXTERN template void View \
  (BlockMatrix<T>& A, ElementalMatrix<T>& B); \
  EL_EXTERN template void LockedView \
  (BlockMatrix<T>& A, const ElementalMatrix<T>& B); \
  EL_EXTERN template void View \
  (ElementalMatrix<T>& A, BlockMatrix<T>& B); \
  EL_EXTERN template void LockedView \
  (ElementalMatrix<T>& A, const BlockMatrix<T>& B); \
  /* AbstractDistMatrix
     ------------------ */ \
  EL_EXTERN template void View \
  (AbstractDistMatrix<T>& A, AbstractDistMatrix<T>& B); \
  EL_EXTERN template void LockedView \
  (AbstractDistMatrix<T>& A, const AbstractDistMatrix<T>& B); \
  /* View a submatrix
     ================ */ \
  EL_EXTERN template void View \
  (Matrix<T>& A, \
    Matrix<T>& B, \
    Int i, Int j, \
    Int height, Int width); \
  EL_EXTERN template void LockedView \
  (      Matrix<T>& A, \
    const Matrix<T>& B, \
    Int i, Int j, \
    Int height, Int width); \
  EL_EXTERN template void View \
  (Matrix<T>& A, \
    Matrix<T>& B, \
    Range<Int> I, Range<Int> J); \
  EL_EXTERN template void LockedView \
  (      Matrix<T>& A, \
    const Matrix<T>& B, \
    Range<Int> I, Range<Int> J); \
  EL_EXTERN template Matrix<T> View \
  (Matrix<T>& B, Int i, Int j, Int height, Int width); \
  EL_EXTERN template Matrix<T> LockedView \
  (const Matrix<T>& B, Int i, Int j, Int height, Int width); \
  EL_EXTERN template Matrix<T> View \
  (Matrix<T>& B, Range<Int> I, Range<Int> J); \
  EL_EXTERN template Matrix<T> LockedView \
  (const Matrix<T>& B, Range<Int> I, Range<Int> J); \
  /* Abstract Sequential Matrix
     -------------------------- */ \
  EL_EXTERN template void View \
  (AbstractMatrix<T>& A, \
    AbstractMatrix<T>& B, \
    Range<Int> I, Range<Int> J); \
  EL_EXTERN template void LockedView \
  (AbstractMatrix<T>& A, \
    AbstractMatrix<T> const& B, \
    Range<Int> I, Range<Int> J); \
  /* ElementalMatrix
     --------------- */ \
  EL_EXTERN template void View \
  (ElementalMatrix<T>& A, \
    ElementalMatrix<T>& B, \
    Int i, Int j, Int height, Int width); \
  EL_EXTERN template void LockedView \
  (      ElementalMatrix<T>& A, \
    const ElementalMatrix<T>& B, \
    Int i, Int j, Int height, Int width); \
  EL_EXTERN template void View \
  (ElementalMatrix<T>& A, \
    ElementalMatrix<T>& B,  \
    Range<Int> I, Range<Int> J); \
  EL_EXTERN template void LockedView \
  (      ElementalMatrix<T>& A, \
    const ElementalMatrix<T>& B, \
    Range<Int> I, Range<Int> J); \
  /* BlockMatrix
     ----------- */ \
  EL_EXTERN template void View \
  (BlockMatrix<T>& A, \
    BlockMatrix<T>& B, \
    Int i, \
    Int j, \
    Int height, \
    Int width); \
  EL_EXTERN template void LockedView \
  (      BlockMatrix<T>& A, \
    const BlockMatrix<T>& B, \
    Int i, Int j, Int height, Int width); \
  EL_EXTERN template void View \
  (BlockMatrix<T>& A, \
    BlockMatrix<T>& B, \
    Range<Int> I, Range<Int> J); \
  EL_EXTERN template void LockedView \
  (      BlockMatrix<T>& A, \
    const BlockMatrix<T>& B, \
    Range<Int> I, Range<Int> J); \
  /* AbstractDistMatrix
     ------------------ */ \
  EL_EXTERN template void View \
  (AbstractDistMatrix<T>& A, \
    AbstractDistMatrix<T>& B, \
    Int i, Int j, Int height, Int width); \
  EL_EXTERN template void LockedView \
  (      AbstractDistMatrix<T>& A, \
    const AbstractDistMatrix<T>& B, \
    Int i, Int j, Int height, Int width); \
  EL_EXTERN template void View \
  (AbstractDistMatrix<T>& A, \
    AbstractDistMatrix<T>& B, \
    Range<Int> I, Range<Int> J); \
  EL_EXTERN template void LockedView \
  (      AbstractDistMatrix<T>& A, \
    const AbstractDistMatrix<T>& B, \
    Range<Int> I, Range<Int> J);

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_VIEW_IMPL_HPP
