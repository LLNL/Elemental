/*
  Copyright (c) 2009-2016, Jack Poulson
  All rights reserved.

  This file is part of Elemental and is under the BSD 2-Clause License,
  which can be found in the LICENSE file in the root directory, or at
  http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_MATRIX_IMPL_GPU_HPP_
#define EL_MATRIX_IMPL_GPU_HPP_

#include <El/blas_like/level1/decl.hpp>

namespace El
{
//
// Public routines
//
// Constructors and destructors
//

template <typename T>
Matrix<T, Device::GPU>::Matrix() { }

template <typename T>
Matrix<T, Device::GPU>::Matrix(
    size_type height, size_type width, size_type leadingDimension)
    : AbstractMatrix<T>{height,width,leadingDimension}
{
    memory_.Require(this->LDim()*this->Width());
    data_ = memory_.Buffer();
}

template <typename T>
Matrix<T, Device::GPU>::Matrix(
    size_type height, size_type width, value_type const* buffer,
    size_type leadingDimension)
    : AbstractMatrix<T>{LOCKED_VIEW,height,width,leadingDimension},
    data_{const_cast<T*>(buffer)}
{
}

template <typename T>
Matrix<T, Device::GPU>::Matrix
(size_type height, size_type width, value_type* buffer, size_type leadingDimension)
    : AbstractMatrix<T>{VIEW,height,width,leadingDimension},
      data_(buffer)
{
}

template <typename T>
Matrix<T, Device::GPU>::Matrix(Matrix<T, Device::GPU> const& A)
    : Matrix{A.Height(), A.Width(), A.Height()}
{
    EL_DEBUG_CSE;
    ::El::Copy(A, *this);
}

template <typename T>
Matrix<T, Device::GPU>::Matrix(Matrix<T, Device::CPU> const& A)
    : Matrix{A.Height(), A.Width(), A.LDim()}
{
    EL_DEBUG_CSE;
    auto syncinfo = SyncInfoFromMatrix(*this);

    gpu::Copy2DToDevice(
        A.LockedBuffer(), A.LDim(),
        data_, this->LDim(),
        A.Height(), A.Width(),
        syncinfo);

    Synchronize(syncinfo);
}

template <typename T>
Matrix<T, Device::GPU>&
Matrix<T, Device::GPU>::operator=(Matrix<T, Device::CPU> const& A)
{
    auto A_new = Matrix<T, Device::GPU>(A);
    *this = std::move(A_new);
    return *this;
}

template <typename T>
Matrix<T, Device::GPU>::Matrix(Matrix<T, Device::GPU>&& A) EL_NO_EXCEPT
    : AbstractMatrix<T>(std::move(A)),
      memory_{std::move(A.memory_)}, data_{A.data_}
{
    A.data_ = nullptr;
}

template <typename T>
Matrix<T, Device::GPU>::~Matrix() { }

template <typename T>
std::unique_ptr<AbstractMatrix<T>>
Matrix<T, Device::GPU>::Copy() const
{
    return std::unique_ptr<AbstractMatrix<T>>{
        new Matrix<T,Device::GPU>(*this)};
}

template <typename T>
std::unique_ptr<AbstractMatrix<T>>
Matrix<T, Device::GPU>::Construct() const
{
    return std::unique_ptr<AbstractMatrix<T>>{
        new Matrix<T,Device::GPU>{}};
}

template <typename T>
void Matrix<T, Device::GPU>::Attach
(Int height, Int width, T* buffer, Int leadingDimension)
{
    EL_DEBUG_CSE;
    Attach_(height, width, buffer, leadingDimension);
}

template <typename T>
void Matrix<T, Device::GPU>::LockedAttach
(Int height, Int width, T const* buffer, Int leadingDimension)
{
    EL_DEBUG_CSE;
    LockedAttach_(height, width, buffer, leadingDimension);
}

// Operator overloading
// ====================

// Return a view
// -------------
template <typename T>
Matrix<T, Device::GPU>
Matrix<T, Device::GPU>::operator()(Range<Int> I, Range<Int> J)
{
    EL_DEBUG_CSE;
    if (this->Locked())
        return LockedView(*this, I, J);
    else
        return View(*this, I, J);
}

template <typename T>
const Matrix<T, Device::GPU>
Matrix<T, Device::GPU>::operator()(Range<Int> I, Range<Int> J) const
{
    EL_DEBUG_CSE;
    return LockedView(*this, I, J);
}

// Make a copy
// -----------
template <typename T>
Matrix<T, Device::GPU>&
Matrix<T, Device::GPU>::operator=(Matrix<T, Device::GPU> const& A)
{
    EL_DEBUG_CSE;
    Matrix<T, Device::GPU>{A}.Swap(*this);
    return *this;
}

// Move assignment
// ---------------
template <typename T>
Matrix<T, Device::GPU>&
Matrix<T, Device::GPU>::operator=(Matrix<T, Device::GPU>&& A)
{
    EL_DEBUG_CSE;
    // "Move-and-swap"
    Matrix<T, Device::GPU>{std::move(A)}.Swap(*this);
    return *this;
}

// Basic queries
// -------------

template <typename T>
auto Matrix<T, Device::GPU>::MemorySize() const EL_NO_EXCEPT
    -> size_type
{
    return memory_.Size();
}

template <typename T>
Device Matrix<T, Device::GPU>::GetDevice() const EL_NO_EXCEPT
{
    return this->MyDevice();
}

template <typename T>
void Matrix<T, Device::GPU>::do_empty_(bool freeMemory)
{
    if (freeMemory)
        memory_.Empty();
    data_ = nullptr;
}

template <typename T>
void Matrix<T, Device::GPU>::do_resize_(
    size_type const& /*height*/, size_type const& width,
    size_type const& ldim)
{
    data_ = memory_.Require(ldim * width);
}

template <typename T>
T* Matrix<T, Device::GPU>::Buffer() EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
#ifndef EL_RELEASE
        if (this->Locked())
            LogicError("Cannot return non-const buffer of locked Matrix");
#endif
        return data_;
}

template <typename T>
T* Matrix<T, Device::GPU>::Buffer(Int i, Int j) EL_NO_RELEASE_EXCEPT
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
const T* Matrix<T, Device::GPU>::LockedBuffer() const EL_NO_EXCEPT
{ return data_; }

template <typename T>
const T*
Matrix<T, Device::GPU>::LockedBuffer(Int i, Int j) const EL_NO_EXCEPT
{
    EL_DEBUG_CSE;
    if (data_ == nullptr)
        return nullptr;
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    return &data_[i+j*this->LDim()];
}

// Advanced functions
// ------------------

template <typename T>
void Matrix<T, Device::GPU>::SetMemoryMode(memory_mode_type mode)
{
    const auto oldBuffer = memory_.Buffer();
    memory_.SetMode(mode);
    if (data_ == oldBuffer)
        data_ = memory_.Buffer();
}

template <typename T>
auto Matrix<T, Device::GPU>::MemoryMode() const EL_NO_EXCEPT
    -> memory_mode_type
{
    return memory_.Mode();
}

// Single-entry manipulation
// =========================

template <typename T>
T Matrix<T, Device::GPU>::Get(Int i, Int j) const
{
    EL_DEBUG_CSE;
#ifdef HYDROGEN_DO_BOUNDS_CHECKING
    this->AssertValidEntry(i, j);
#endif
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    auto syncinfo = SyncInfoFromMatrix(*this);
    T val;
    gpu::Copy1DToHost(&data_[i+j*this->LDim()], &val, 1, syncinfo);
    Synchronize(syncinfo);
    return val;
}

template <typename T>
T Matrix<T, Device::GPU>::do_get_(
    index_type const& i, index_type const& j) const
{
    return this->Get(i,j);
}

template <typename T>
Base<T> Matrix<T, Device::GPU>::GetRealPart(Int, Int) const
{
    EL_DEBUG_CSE;
    LogicError("GetRealPart no longer supported on GPU.");
    return Base<T>{};
}

template <typename T>
Base<T> Matrix<T, Device::GPU>::GetImagPart(Int, Int) const
{
    EL_DEBUG_CSE;
    LogicError("GetImagPart no longer supported on GPU.");
    return Base<T>{};
}

template <typename T>
void Matrix<T, Device::GPU>::Set(Int i, Int j, T const& alpha)
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

    auto syncinfo = SyncInfoFromMatrix(*this);
    gpu::Copy1DToDevice(&alpha, &data_[i+j*this->LDim()], 1, syncinfo);
    Synchronize(syncinfo);
}

template <typename T>
void Matrix<T, Device::GPU>::do_set_(
    index_type const& i, index_type const& j, T const& alpha)
{
    this->Set(i,j,alpha);
}

template <typename T>
void Matrix<T, Device::GPU>::Set(Entry<T> const& entry)
{ Set(entry.i, entry.j, entry.value); }

template <typename T>
void
Matrix<T, Device::GPU>::SetRealPart(
    Int, Int, Base<T> const&)
{
    EL_DEBUG_CSE;
    LogicError("SetRealPart no longer supported on GPU.");
}

template <typename T>
void Matrix<T, Device::GPU>::SetRealPart(Entry<Base<T>> const& entry)
{ SetRealPart(entry.i, entry.j, entry.value); }

template <typename T>
void
Matrix<T, Device::GPU>::SetImagPart(Int, Int, Base<T> const&)
{
    EL_DEBUG_CSE;
    LogicError("SetImagPart no longer supported on GPU.");
}

template <typename T>
void Matrix<T, Device::GPU>::SetImagPart(Entry<Base<T>> const& entry)
{
    SetImagPart(entry.i, entry.j, entry.value);
}
template <typename T>
void Matrix<T, Device::GPU>::Update(Int, Int, T const&)
{
    EL_DEBUG_CSE;
    LogicError("SetImagPart no longer supported on GPU.");

}

template <typename T>
void Matrix<T, Device::GPU>::Update(Entry<T> const& entry)
{ Update(entry.i, entry.j, entry.value); }

template <typename T>
void
Matrix<T, Device::GPU>::UpdateRealPart(Int i, Int j, Base<T> const& alpha)
{
    EL_DEBUG_CSE;
    LogicError("UpdateRealPart no longer supported on GPU.");
}

template <typename T>
void Matrix<T, Device::GPU>::UpdateRealPart(Entry<Base<T>> const& entry)
{
    UpdateRealPart(entry.i, entry.j, entry.value);
}

template <typename T>
void
Matrix<T, Device::GPU>::UpdateImagPart(Int, Int, Base<T> const&)
{
    EL_DEBUG_CSE;
    LogicError("UpdateImagPart no longer supported on GPU.");
}

template <typename T>
void Matrix<T, Device::GPU>::UpdateImagPart(Entry<Base<T>> const& entry)
{ UpdateImagPart(entry.i, entry.j, entry.value); }

template <typename T>
void Matrix<T, Device::GPU>::MakeReal(Int, Int)
{
    EL_DEBUG_CSE;
    LogicError("MakeReal no longer supported on GPU.");
}

template <typename T>
void Matrix<T, Device::GPU>::Conjugate(Int, Int)
{
    EL_DEBUG_CSE;
    LogicError("Conjugate no longer supported on GPU.");
}

// Exchange metadata with another matrix
// =====================================
template <typename T>
void Matrix<T, Device::GPU>::Swap(
    Matrix<T, Device::GPU>& A) EL_NO_EXCEPT
{
    EL_DEBUG_CSE;
    this->SwapMetadata_(A);
    SwapImpl_(A);
}

template <typename T>
void Matrix<T, Device::GPU>::SwapImpl_(
    Matrix<T, Device::GPU>& A) EL_NO_EXCEPT
{
    EL_DEBUG_CSE;
    memory_.ShallowSwap(A.memory_);
    std::swap(data_, A.data_);
}

template <typename T>
void Matrix<T, Device::GPU>::do_swap_(AbstractMatrix<T>& A)
{
    EL_DEBUG_CSE;
    if (A.GetDevice() == Device::GPU)
        SwapImpl_(static_cast<Matrix<T, Device::GPU>&>(A));
    else
        LogicError("Source of swap does not have the same device.");
}

template <typename T>
void Matrix<T, Device::GPU>::Attach_(
    size_type height, size_type width, value_type* buffer,
    size_type leadingDimension)
{
    this->SetViewType(
        static_cast<El::ViewType>((this->ViewType() & ~LOCKED_OWNER) | VIEW));
    this->SetSize_(height, width, leadingDimension);

    data_ = buffer;
}

template <typename T>
void Matrix<T, Device::GPU>::LockedAttach_(
    size_type height, size_type width, value_type const* buffer,
    size_type leadingDimension)
{
    this->SetViewType(
        static_cast<El::ViewType>(this->ViewType() | LOCKED_VIEW));
    this->SetSize_(height, width, leadingDimension);

    data_ = const_cast<value_type*>(buffer);
}


// Return a reference to a single entry without error-checking
// ===========================================================
template <typename T>
T const& Matrix<T, Device::GPU>::CRef(Int i, Int j) const
{
    LogicError("Attempted to get reference to entry of a GPU matrix");
    return data_[0];
}

template <typename T>
T const& Matrix<T, Device::GPU>::operator()(Int i, Int j) const
{
    LogicError("Attempted to get reference to entry of a GPU matrix");
    return data_[0];
}

template <typename T>
T& Matrix<T, Device::GPU>::Ref(Int i, Int j)
{
    LogicError("Attempted to get reference to entry of a GPU matrix");
    return data_[0];
}

template <typename T>
T& Matrix<T, Device::GPU>::operator()(Int i, Int j)
{
    LogicError("Attempted to get reference to entry of a GPU matrix");
    return data_[0];
}

template <typename T>
SyncInfo<Device::GPU> Matrix<T, Device::GPU>::GetSyncInfo() const EL_NO_EXCEPT
{
    return sync_info_;
}

template <typename T>
void Matrix<T, Device::GPU>::SetSyncInfo(
    SyncInfo<Device::GPU> const& si) EL_NO_EXCEPT
{
    sync_info_.Merge(si);
}

#ifdef EL_INSTANTIATE_CORE
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) EL_EXTERN template class Matrix<T,Device::GPU>;
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT

// Right now, we don't acknowledge a world in which NVIDIA GPUs
// coexist on a platform with some other type of GPU (e.g., AMD)
#ifdef HYDROGEN_GPU_USE_FP16
PROTO(gpu_half_type)
#endif // HYDROGEN_GPU_USE_FP16

#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_MATRIX_IMPL_GPU_HPP_
