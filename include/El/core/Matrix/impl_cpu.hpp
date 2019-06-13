/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_MATRIX_IMPL_CPU_HPP_
#define EL_MATRIX_IMPL_CPU_HPP_

#include <El/hydrogen_config.h>

#ifdef HYDROGEN_HAVE_GPU
#include <hydrogen/device/GPU.hpp>
#endif // HYDROGEN_HAVE_GPU

namespace El
{

// Public routines
// ###############

// Constructors and destructors
// ============================

template <typename T>
Matrix<T, Device::CPU>::Matrix() { }

template <typename T>
Matrix<T, Device::CPU>::Matrix(
    size_type height, size_type width, size_type leadingDimension)
    : AbstractMatrix<T>{height, width, leadingDimension}
{
    memory_.Require(this->LDim()*this->Width());
    data_ = memory_.Buffer();
}

template <typename T>
Matrix<T, Device::CPU>::Matrix(
    size_type height, size_type width, value_type const* buffer,
    size_type leadingDimension)
    : AbstractMatrix<T>{LOCKED_VIEW, height, width, leadingDimension},
      data_(const_cast<T*>(buffer))
{
}

template <typename T>
Matrix<T, Device::CPU>::Matrix(
    size_type height, size_type width, value_type* buffer,
    size_type leadingDimension)
    : AbstractMatrix<T>{VIEW,height,width,leadingDimension},
      data_(buffer)
{
}

template <typename T>
Matrix<T, Device::CPU>::Matrix(Matrix<T, Device::CPU> const& A)
    : Matrix{A.Height(), A.Width(), A.Height()}
{
    EL_DEBUG_CSE;
    Copy(A, *this);
}

#ifdef HYDROGEN_HAVE_CUDA
template <typename T>
Matrix<T, Device::CPU>::Matrix(Matrix<T, Device::GPU> const& A)
    : Matrix{A.Height(), A.Width(), A.LDim()}
{
    EL_DEBUG_CSE;
    auto stream = gpu::DefaultSyncInfo().Stream();
    H_CHECK_CUDA(cudaMemcpy2DAsync(data_, this->LDim()*sizeof(T),
                                    A.LockedBuffer(), A.LDim()*sizeof(T),
                                    A.Height()*sizeof(T), A.Width(),
                                    cudaMemcpyDeviceToHost,
                                    stream));
    H_CHECK_CUDA(cudaStreamSynchronize(stream));
}
#endif // HYDROGEN_HAVE_CUDA

template <typename T>
Matrix<T, Device::CPU>::Matrix(Matrix<T, Device::CPU>&& A) EL_NO_EXCEPT
    : AbstractMatrix<T>{std::move(A)},
    memory_{std::move(A.memory_)}, data_{A.data_}
{
    A.data_ = nullptr;
}

template <typename T>
Matrix<T, Device::CPU>::~Matrix() { }

// Assignment and reconfiguration
// ==============================


template <typename T>
void Matrix<T, Device::CPU>::Attach(
    size_type height, size_type width, value_type* buffer,
    size_type leadingDimension)
{
    EL_DEBUG_CSE;
#ifndef EL_RELEASE
    if (this->FixedSize())
        LogicError("Cannot attach a new buffer to a view with fixed size");
#endif
    Attach_(height, width, buffer, leadingDimension);
}

template <typename T>
void Matrix<T, Device::CPU>::LockedAttach(
    size_type height, size_type width, value_type const* buffer,
    size_type leadingDimension)
{
    EL_DEBUG_CSE;
#ifndef EL_RELEASE
    if (this->FixedSize())
        LogicError("Cannot attach a new buffer to a view with fixed size");
#endif
    LockedAttach_(height, width, buffer, leadingDimension);
}

// Operator overloading
// ====================

// Return a view
// -------------
template <typename T>
Matrix<T, Device::CPU>
Matrix<T, Device::CPU>::operator()(Range<Int> I, Range<Int> J)
{
    EL_DEBUG_CSE;
    if (this->Locked())
        return LockedView(*this, I, J);
    else
        return View(*this, I, J);
}

template <typename T>
const Matrix<T, Device::CPU>
Matrix<T, Device::CPU>::operator()(Range<Int> I, Range<Int> J) const
{
    EL_DEBUG_CSE;
    return LockedView(*this, I, J);
}

// Return a (potentially non-contiguous) subset of indices
// -------------------------------------------------------
template <typename T>
Matrix<T, Device::CPU> Matrix<T, Device::CPU>::operator()
(Range<Int> I, vector<Int> const& J) const
{
    EL_DEBUG_CSE;
    Matrix<T, Device::CPU> ASub;
    GetSubmatrix(*this, I, J, ASub);
    return ASub;
}

template <typename T>
Matrix<T, Device::CPU> Matrix<T, Device::CPU>::operator()
(vector<Int> const& I, Range<Int> J) const
{
    EL_DEBUG_CSE;
    Matrix<T, Device::CPU> ASub;
    GetSubmatrix(*this, I, J, ASub);
    return ASub;
}

template <typename T>
Matrix<T, Device::CPU> Matrix<T, Device::CPU>::operator()
(vector<Int> const& I, vector<Int> const& J) const
{
    EL_DEBUG_CSE;
    Matrix<T, Device::CPU> ASub;
    GetSubmatrix(*this, I, J, ASub);
    return ASub;
}

// Make a copy
// -----------
template <typename T>
Matrix<T, Device::CPU>&
Matrix<T, Device::CPU>::operator=(Matrix<T, Device::CPU> const& A)
{
    EL_DEBUG_CSE;
    Matrix<T, Device::CPU>{A}.Swap(*this);
    return *this;
}

// Move assignment
// ---------------
template <typename T>
Matrix<T, Device::CPU>&
Matrix<T, Device::CPU>::operator=(Matrix<T, Device::CPU>&& A)
{
    EL_DEBUG_CSE;
    // "Move-and-swap"
    Matrix<T, Device::CPU>{std::move(A)}.Swap(*this);
    return *this;
}

// Basic queries
// =============

template <typename T>
auto Matrix<T, Device::CPU>::MemorySize() const EL_NO_EXCEPT
    -> size_type
{
    return memory_.Size();
}

template <typename T>
Device Matrix<T, Device::CPU>::GetDevice() const EL_NO_EXCEPT
{
    return this->MyDevice();
}

template <typename T>
T* Matrix<T, Device::CPU>::Buffer() EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifndef EL_RELEASE
    if (this->Locked())
        LogicError("Cannot return non-const buffer of locked Matrix");
#endif
    return data_;
}

template <typename T>
T* Matrix<T, Device::CPU>::Buffer(Int i, Int j) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifndef EL_RELEASE
    if (this->Locked())
        LogicError("Cannot return non-const buffer of locked Matrix");
#endif
    if (data_ == nullptr)
        return nullptr;
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    return &data_[i+j*this->LDim()];
}

template <typename T>
const T* Matrix<T, Device::CPU>::LockedBuffer() const EL_NO_EXCEPT
{ return data_; }

template <typename T>
const T*
Matrix<T, Device::CPU>::LockedBuffer(Int i, Int j) const EL_NO_EXCEPT
{
    EL_DEBUG_CSE;
    if (data_ == nullptr)
        return nullptr;
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    return &data_[i+j*this->LDim()];
}

// Advanced functions
// ==================

template <typename T>
void Matrix<T, Device::CPU>::SetMemoryMode(memory_mode_type mode)
{
    const auto oldBuffer = memory_.Buffer();
    memory_.SetMode(mode);
    if (data_ == oldBuffer)
        data_ = memory_.Buffer();
}

template <typename T>
auto Matrix<T, Device::CPU>::MemoryMode() const EL_NO_EXCEPT
    -> memory_mode_type
{ return memory_.Mode(); }

// Single-entry manipulation
// =========================

template <typename T>
T Matrix<T, Device::CPU>::Get(Int i, Int j) const
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifdef HYDROGEN_DO_BOUNDS_CHECKING
    this->AssertValidEntry(i, j);
#endif
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    return CRef(i, j);
}

template <typename T>
T Matrix<T, Device::CPU>::do_get_(
    index_type const& i, index_type const& j) const
{
    return this->Get(i,j);
}

template <typename T>
Base<T> Matrix<T, Device::CPU>::GetRealPart(Int i, Int j) const
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifdef HYDROGEN_DO_BOUNDS_CHECKING
    this->AssertValidEntry(i, j);
#endif
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    return El::RealPart(CRef(i, j));
}

template <typename T>
Base<T> Matrix<T, Device::CPU>::GetImagPart(Int i, Int j) const
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifdef HYDROGEN_DO_BOUNDS_CHECKING
    this->AssertValidEntry(i, j);
#endif
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    return El::ImagPart(CRef(i, j));
}

template <typename T>
void Matrix<T, Device::CPU>::Set(Int i, Int j, T const& alpha)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifdef HYDROGEN_DO_BOUNDS_CHECKING
    this->AssertValidEntry(i, j);
#endif
#ifndef EL_RELEASE
    if (this->Locked())
        LogicError("Cannot modify data of locked matrices");
#endif
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    Ref(i, j) = alpha;
}

template <typename T>
void Matrix<T, Device::CPU>::do_set_(
    index_type const& i, index_type const& j, T const& alpha)
{
    this->Set(i,j,alpha);
}

template <typename T>
void Matrix<T, Device::CPU>::Set(Entry<T> const& entry)
EL_NO_RELEASE_EXCEPT
{
    Set(entry.i, entry.j, entry.value);
}

template <typename T>
void
Matrix<T, Device::CPU>::SetRealPart(
    Int i, Int j, Base<T> const& alpha)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifdef HYDROGEN_DO_BOUNDS_CHECKING
    this->AssertValidEntry(i, j);
#endif
#ifndef EL_RELEASE
    if (this->Locked())
        LogicError("Cannot modify data of locked matrices");
#endif
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    El::SetRealPart(Ref(i, j), alpha);
}

template <typename T>
void Matrix<T, Device::CPU>::SetRealPart(Entry<Base<T>> const& entry)
EL_NO_RELEASE_EXCEPT
{
    SetRealPart(entry.i, entry.j, entry.value);
}

template <typename T>
void
Matrix<T, Device::CPU>::SetImagPart(Int i, Int j, Base<T> const& alpha)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifdef HYDROGEN_DO_BOUNDS_CHECKING
    this->AssertValidEntry(i, j);
#endif
#ifndef EL_RELEASE
    if (this->Locked())
        LogicError("Cannot modify data of locked matrices");
#endif
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    El::SetImagPart(Ref(i, j), alpha);
}

template <typename T>
void Matrix<T, Device::CPU>::SetImagPart(Entry<Base<T>> const& entry)
EL_NO_RELEASE_EXCEPT
{
    SetImagPart(entry.i, entry.j, entry.value);
}

template <typename T>
void Matrix<T, Device::CPU>::Update(Int i, Int j, T const& alpha)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifdef HYDROGEN_DO_BOUNDS_CHECKING
    this->AssertValidEntry(i, j);
#endif
#ifndef EL_RELEASE
    if (this->Locked())
        LogicError("Cannot modify data of locked matrices");
#endif // !EL_RELEASE
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    Ref(i, j) += alpha;
}

template <typename T>
void Matrix<T, Device::CPU>::Update(Entry<T> const& entry)
EL_NO_RELEASE_EXCEPT
{
    Update(entry.i, entry.j, entry.value);
}

template <typename T>
void
Matrix<T, Device::CPU>::UpdateRealPart(Int i, Int j, Base<T> const& alpha)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifdef HYDROGEN_DO_BOUNDS_CHECKING
    this->AssertValidEntry(i, j);
#endif
#ifndef EL_RELEASE
    if (this->Locked())
        LogicError("Cannot modify data of locked matrices");
#endif // !EL_RELEASE
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    El::UpdateRealPart(Ref(i, j), alpha);
}

template <typename T>
void Matrix<T, Device::CPU>::UpdateRealPart(Entry<Base<T>> const& entry)
EL_NO_RELEASE_EXCEPT
{
    UpdateRealPart(entry.i, entry.j, entry.value);
}

template <typename T>
void
Matrix<T, Device::CPU>::UpdateImagPart(Int i, Int j, Base<T> const& alpha)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifdef HYDROGEN_DO_BOUNDS_CHECKING
    this->AssertValidEntry(i, j);
#endif
#ifndef EL_RELEASE
    if (this->Locked())
        LogicError("Cannot modify data of locked matrices");
#endif // EL_RELEASE

    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    El::UpdateImagPart(Ref(i, j), alpha);
}

template <typename T>
void Matrix<T, Device::CPU>::UpdateImagPart(Entry<Base<T>> const& entry)
EL_NO_RELEASE_EXCEPT
{
    UpdateImagPart(entry.i, entry.j, entry.value);
}

template <typename T>
void Matrix<T, Device::CPU>::MakeReal(Int i, Int j)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifdef HYDROGEN_DO_BOUNDS_CHECKING
    this->AssertValidEntry(i, j);
#endif
#ifndef EL_RELEASE
    if (this->Locked())
        LogicError("Cannot modify data of locked matrices");
#endif

    Set(i, j, GetRealPart(i,j));
}

template <typename T>
void Matrix<T, Device::CPU>::Conjugate(Int i, Int j)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifdef HYDROGEN_DO_BOUNDS_CHECKING
    this->AssertValidEntry(i, j);
#endif
#ifndef EL_RELEASE
    if (this->Locked())
        LogicError("Cannot modify data of locked matrices");
#endif
    Set(i, j, El::Conj(Get(i,j)));
}

// Private routines
// ################

// Exchange metadata with another matrix
// =====================================
template <typename T>
void Matrix<T, Device::CPU>::Swap(
    Matrix<T, Device::CPU>& A) EL_NO_EXCEPT
{
    EL_DEBUG_CSE;
    this->SwapMetadata_(A);
    SwapImpl_(A);
}

template <typename T>
void Matrix<T, Device::CPU>::SwapImpl_(
    Matrix<T, Device::CPU>& A) EL_NO_EXCEPT
{
    EL_DEBUG_CSE;
    memory_.ShallowSwap(A.memory_);
    std::swap(data_, A.data_);
}

template <typename T>
void Matrix<T, Device::CPU>::do_swap_(AbstractMatrix<T>& A)
{
    EL_DEBUG_CSE;
    if (A.GetDevice() == Device::CPU)
        SwapImpl_(static_cast<Matrix<T, Device::CPU>&>(A));
    else
        LogicError("Source of swap does not have the same device.");
}

// Reconfigure without error-checking
// ==================================

template <typename T>
void Matrix<T, Device::CPU>::do_empty_(bool freeMemory)
{
    EL_DEBUG_CSE;
    if (freeMemory)
        memory_.Empty();
    data_ = nullptr;
}

template <typename T>
void Matrix<T, Device::CPU>::Attach_(
    size_type height, size_type width, value_type* buffer,
    size_type leadingDimension)
{
    // This is no longer locked. But it is viewing.
    this->SetViewType(
        static_cast<El::ViewType>((this->ViewType() & ~LOCKED_OWNER) | VIEW));
    this->SetSize_(height, width, leadingDimension);
    data_ = buffer;
}

template <typename T>
void Matrix<T, Device::CPU>::LockedAttach_(
    size_type height, size_type width, value_type const* buffer,
    size_type leadingDimension)
{
    // This is now locked and viewing.
    this->SetViewType(
        static_cast<El::ViewType>(this->ViewType() | LOCKED_VIEW));
    this->SetSize_(height, width, leadingDimension);
    data_ = const_cast<T*>(buffer);
}

// Return a reference to a single entry without error-checking
// ===========================================================
template <typename T>
T const& Matrix<T, Device::CPU>::CRef(Int i, Int j) const
EL_NO_RELEASE_EXCEPT
{
    return data_[i+j*this->LDim()];
}

template <typename T>
T const& Matrix<T, Device::CPU>::operator()(Int i, Int j) const
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifdef HYDROGEN_DO_BOUNDS_CHECKING
    this->AssertValidEntry(i, j);
#endif
    return data_[i+j*this->LDim()];
}

template <typename T>
T& Matrix<T, Device::CPU>::Ref(Int i, Int j)
EL_NO_RELEASE_EXCEPT
{
    return data_[i+j*this->LDim()];
}

template <typename T>
T& Matrix<T, Device::CPU>::operator()(Int i, Int j)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifdef HYDROGEN_DO_BOUNDS_CHECKING
    this->AssertValidEntry(i, j);
#endif
#ifndef EL_RELEASE
    if (this->Locked())
        LogicError("Cannot modify data of locked matrices");
#endif
    return data_[i+j*this->LDim()];
}

template <typename T>
void Matrix<T, Device::CPU>::do_resize_(
    size_type const& /*height*/, size_type const& width,
    size_type const& ldim)
{
    data_ = memory_.Require(ldim * width);
}

// For supporting duck typing
// ==========================
template <typename T>
Matrix<T, Device::CPU>::Matrix(El::Grid const& grid)
{
    EL_DEBUG_CSE;
#ifndef EL_RELEASE
      if (grid != El::Grid::Trivial())
          LogicError("Tried to construct a Matrix with a nontrivial Grid");
#endif
}

template <typename T>
void Matrix<T, Device::CPU>::SetGrid(El::Grid const& grid)
{
    EL_DEBUG_CSE;
#ifndef EL_RELEASE
      if (grid != El::Grid::Trivial())
          LogicError("Tried to assign nontrivial Grid to Matrix");
#endif
}

template <typename T>
El::Grid const& Matrix<T, Device::CPU>::Grid() const
{
    EL_DEBUG_CSE;
    return El::Grid::Trivial();
}

template <typename T>
void
Matrix<T, Device::CPU>::Align(Int colAlign, Int rowAlign, bool constrain)
{
    EL_DEBUG_CSE;
#ifndef EL_RELEASE
      if (colAlign != 0 || rowAlign != 0)
          LogicError("Attempted to impose nontrivial alignment on Matrix");

#endif
}

template <typename T>
int Matrix<T, Device::CPU>::ColAlign() const EL_NO_EXCEPT { return 0; }
template <typename T>
int Matrix<T, Device::CPU>::RowAlign() const EL_NO_EXCEPT { return 0; }

#ifdef EL_INSTANTIATE_CORE
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) EL_EXTERN template class Matrix<T>;
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT

#ifdef HYDROGEN_HAVE_HALF
PROTO(cpu_half_type)
#endif
PROTO(uint8_t)

#include <El/macros/Instantiate.h>

#undef EL_EXTERN
} // namespace El

#endif // ifndef EL_MATRIX_IMPL_CPU_HPP_
