#ifndef EL_CORE_ABSTRACTMATRIX_IMPL_HPP_
#define EL_CORE_ABSTRACTMATRIX_IMPL_HPP_

#include "decl.hpp"
#include <El/blas_like/level1/decl.hpp>

namespace El
{

//
// Queries
//
template <typename T>
inline auto AbstractMatrix<T>::Height() const EL_NO_EXCEPT
    -> size_type
{
    return height_;
}

template <typename T>
inline auto AbstractMatrix<T>::Width() const EL_NO_EXCEPT
    -> size_type
{
    return width_;
}

template <typename T>
inline auto AbstractMatrix<T>::LDim() const EL_NO_EXCEPT
    -> size_type
{
    return leadingDimension_;
}

template <typename T>
inline auto
AbstractMatrix<T>::DiagonalLength(size_type offset) const EL_NO_EXCEPT
    -> size_type
{
    return El::DiagonalLength(height_, width_, offset);
}

template <typename T>
inline bool AbstractMatrix<T>::Viewing() const EL_NO_EXCEPT
{
    return IsViewing(viewType_);
}

template <typename T>
inline bool AbstractMatrix<T>::FixedSize() const EL_NO_EXCEPT
{
    return Viewing() || IsFixedSize(viewType_);
}

template <typename T>
inline bool AbstractMatrix<T>::Locked() const EL_NO_EXCEPT
{
    return IsLocked(viewType_);
}

template <typename T>
inline bool AbstractMatrix<T>::IsEmpty() const EL_NO_EXCEPT
{
    return height_ < 1 || width_ < 1;
}

template <typename T>
inline bool AbstractMatrix<T>::Contiguous() const EL_NO_EXCEPT
{
    return ((height_ == leadingDimension_)
            || (width_ == 1)
            || IsEmpty());
}

//
// Advanced queries
//

template <typename T>
inline El::ViewType AbstractMatrix<T>::ViewType() const EL_NO_EXCEPT
{
    return viewType_;
}

//
// Modifiers
//

template <typename T>
inline void AbstractMatrix<T>::FixSize() EL_NO_EXCEPT
{
    // A view is marked as fixed if its second bit is nonzero
    // (and OWNER_FIXED is zero except in its second bit).
    viewType_ = static_cast<El::ViewType>(viewType_ | OWNER_FIXED);
}

template <typename T>
inline void AbstractMatrix<T>::Empty(bool freeMemory)
{
    EL_DEBUG_CSE;
#ifndef EL_RELEASE
    if (this->FixedSize())
        LogicError("Cannot empty a fixed-size matrix");
#endif // !EL_RELEASE

    this->Empty_(freeMemory);
}

template <typename T>
void AbstractMatrix<T>::Resize(size_type height, size_type width)
{
    Resize(height, width, (this->Viewing() ? this->LDim() : height));
}

template <typename T>
inline void AbstractMatrix<T>::Resize(
    size_type height, size_type width, size_type leadingDimension)
{
    EL_DEBUG_CSE;
    leadingDimension = Max(leadingDimension, size_type{1});
    AssertValidDimensions(height, width, leadingDimension);
//    leadingDimension = Max(Max(leadingDimension, height), size_type{1});

    // This function is used in generic code and should be a valid
    // no-op for fixed-size matrices.
    auto const resize_would_happen =
        ((height != height_)
         || (width != width_)
         || (leadingDimension != leadingDimension_));

    if (resize_would_happen)
    {
        if (this->FixedSize())
            LogicError("Cannot resize a fixed-size matrix.");

        this->Resize_(height, width, leadingDimension);
    }
}

template <typename T>
void AbstractMatrix<T>::Swap(AbstractMatrix<T>& A)
{
    do_swap_(A); // Must ensure data moves
    SwapMetadata_(A);
}

template <typename T>
void AbstractMatrix<T>::ShallowSwap(AbstractMatrix<T>& A)
{
    this->Swap(A);
}

template <typename T>
inline void
AbstractMatrix<T>::SetViewType(El::ViewType viewType) EL_NO_EXCEPT
{
    viewType_ = viewType;
}

template <typename T>
inline void AbstractMatrix<T>::SetSize_(
    Int height, Int width, Int leadingDimension)
{
    height_ = height;
    width_ = width;
    leadingDimension_ = Max(leadingDimension, 1);
}

template <typename T>
inline void AbstractMatrix<T>::Empty_(bool freeMemory)
{
    // Set this to be neither locked nor viewing
    viewType_ = static_cast<El::ViewType>(viewType_ & ~LOCKED_VIEW);

    // Reset the default sizes
    this->SetSize_(0, 0, 1);

    // Clear any state in derived classes
    do_empty_(freeMemory);
}

template <typename T>
inline void AbstractMatrix<T>::Resize_(
    Int height, Int width, Int leadingDimension)
{
    leadingDimension = Max(leadingDimension, 1);

    // The following order of operations should be guaranteed to
    // ensure exception guarantees:
    //
    //   1. The underlying storage is resized if needed.
    //   2. The metadata is updated
    do_resize_(height, width, leadingDimension);
    this->SetSize_(height, width, leadingDimension);
}

template <typename T>
void AbstractMatrix<T>::SwapMetadata_(AbstractMatrix<T>& A) EL_NO_EXCEPT
{
    std::swap(viewType_, A.viewType_);
    std::swap(height_, A.height_);
    std::swap(width_, A.width_);
    std::swap(leadingDimension_, A.leadingDimension_);
}

//
// Protected constructors
//

template <typename T>
AbstractMatrix<T>::AbstractMatrix(Int height, Int width, Int ldim)
    : AbstractMatrix{OWNER, height, width, ldim}
{
}

template <typename T>
AbstractMatrix<T>::AbstractMatrix(
    El::ViewType view, Int height, Int width, Int ldim)
    : height_{height}, width_{width},
      leadingDimension_{Max(Max(height, ldim),1)},
      viewType_{view}
{
    EL_DEBUG_CSE;
#ifdef HYDROGEN_DO_BOUNDS_CHECKING
    AssertValidDimensions(this->Height(), this->Width(), this->LDim());
#endif
}

template <typename T>
void AbstractMatrix<T>::AssertValidEntry(index_type i, index_type j) const
{
    if ((i > this->Height())
        || (j > this->Width()))
        RuntimeError("Bad entry (",i,", ",j,"). Matrix is ",
                     this->Height(),"x",this->Width(),".");
}

// Single-entry manipulation
// =========================

template<typename T>
T AbstractMatrix<T>::Get(Int const& i, Int const& j) const
    EL_NO_RELEASE_EXCEPT
{
    return do_get_(i, j);
}

template<typename T>
void AbstractMatrix<T>::Set(Int const& i, Int const& j, T const& alpha)
    EL_NO_RELEASE_EXCEPT
{
    do_set_(i, j, alpha);
}

template<typename T>
void AbstractMatrix<T>::Set(Entry<T> const& entry) EL_NO_RELEASE_EXCEPT
{
    do_set_(entry.i, entry.j, entry.value);
}

// Operator overloading
// ====================

// Assignment
// ----------
template<typename T>
AbstractMatrix<T>&
AbstractMatrix<T>::operator=(AbstractMatrix<T> const& A)
{
    EL_DEBUG_CSE;
    ::El::Copy(A, *this);
    return *this;
}

}// namespace El
#endif // EL_CORE_ABSTRACTMATRIX_IMPL_HPP_
