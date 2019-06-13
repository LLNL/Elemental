/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_MATRIX_DECL_HPP
#define EL_MATRIX_DECL_HPP

#include <El/hydrogen_config.h>

#include <hydrogen/Device.hpp>

#ifdef HYDROGEN_HAVE_GPU
#include <hydrogen/device/GPU.hpp>
#endif // HYDROGEN_HAVE_GPU

#include <El/core/Grid.hpp>
#include <El/core/Memory.hpp>

namespace El
{

// Matrix base for arbitrary rings
template<typename T, Device Dev>
class Matrix;

// Specialization for CPU
template <typename T>
class Matrix<T, Device::CPU> : public AbstractMatrix<T>
{
public:
    using value_type = typename AbstractMatrix<T>::value_type;
    using index_type = typename AbstractMatrix<T>::index_type;
    using size_type = typename AbstractMatrix<T>::size_type;
    using memory_mode_type = typename AbstractMatrix<T>::memory_mode_type;
public:
    /** @name Constructors and destructors */
    ///@{

    /** @brief Create a 0x0 matrix. */
    Matrix();

    /** @brief Create a matrix with the specified dimensions and
     *         leading dimension.
     */
    Matrix(size_type height, size_type width,
           size_type leadingDimension=size_type{0});

    /** @brief Construct a matrix around an existing (possibly
     *         immutable) buffer.
     */
    Matrix(size_type height, size_type width,
           value_type* buffer, size_type leadingDimension);
    Matrix(size_type height, size_type width,
           value_type const* buffer, size_type leadingDimension);

    /** @brief Create a (deep) copy of a matrix.
     *
     *  This will always perform a deep copy of the underlying matrix
     *  data. This is most significant if the source matrix is a
     *  view. The newly constructed matrix will *not* be a view but
     *  will own a clean copy of the data.
     *
     *  The copy will be compressed into contiguous storage. That is,
     *  the matrix height will be equal to its leading dimension.
     */
    Matrix(Matrix<T, Device::CPU> const& A);

    /** @brief Move the metadata from a given matrix.
     *
     *  This is a simple move of the matrix data and metadata. Owners
     *  are moved into owners; views are moved into views.
     *
     *  @post The source matrix satisfies `A.IsEmpty()` returns @c true.
     */
    Matrix(Matrix<T, Device::CPU>&& A) EL_NO_EXCEPT;

    /** @brief Destructor */
    ~Matrix();

    ///@}
    /** @name Assignment operators **/
    ///@{

    /** @brief Copy assignment.
     *
     *  This will always perform a deep copy of the underlying matrix
     *  data. In particular, if source matrix is a view, the target
     *  matrix will become an owner of new data.
     */
    Matrix<T, Device::CPU>& operator=(Matrix<T, Device::CPU> const& A);

    /** @brief Move assignment.
     *
     *  This is a simple move of the matrix data and metadata. Owners
     *  are moved into owners; views are moved into views.
     *
     *  @post The source matrix satisfies `A.IsEmpty()` returns @c true.
     */
    Matrix<T, Device::CPU>& operator=(Matrix<T, Device::CPU>&& A);

#ifdef HYDROGEN_HAVE_GPU
    /** @brief Create a copy of a matrix from a GPU matrix */
    Matrix(Matrix<T, Device::GPU> const& A);

    /** @brief Assign by copying data from a GPU */
    Matrix<T, Device::CPU>& operator=(Matrix<T, Device::GPU> const& A);
#endif // HYDROGEN_HAVE_CUDA

    ///@}
    /** @name Modifiers */
    ///@{

    // Reconfigure around the given buffer, but do not assume ownership
    void Attach(size_type height, size_type width, value_type* buffer,
                size_type leadingDimension);
    void LockedAttach(size_type height, size_type width,
                      value_type const* buffer, size_type leadingDimension);

    void Swap(Matrix<T,Device::CPU>& A) EL_NO_EXCEPT;

    ///@}
    /** @name Non-owning views */
    ///@{

    /** @brief Get a view of contiguous subsets of rows and columns
     *
     *  The resulting matrix will still have column-stride of
     *  `this->LDim()`.
     */
    Matrix<T, Device::CPU> operator()(Range<Int> I, Range<Int> J);
    const Matrix<T, Device::CPU> operator()(
        Range<Int> I, Range<Int> J) const;

    ///@}
    /** @name Submatrix copies */
    ///@{

    /** @brief Get a copy of a (potentially non-contiguous) submatrix. */
    Matrix<T, Device::CPU> operator()(
        Range<Int> I, vector<Int> const& J) const;
    Matrix<T, Device::CPU> operator()(
        vector<Int> const& I, Range<Int> J) const;
    Matrix<T, Device::CPU> operator()(
        vector<Int> const& I, vector<Int> const& J) const;

    ///@}
    /** @name Basic queries */
    ///@{
    size_type MemorySize() const EL_NO_EXCEPT override;
    Device GetDevice() const EL_NO_EXCEPT override;
    constexpr Device MyDevice() const EL_NO_EXCEPT { return Device::CPU; }

    T* Buffer() EL_NO_RELEASE_EXCEPT override;
    T* Buffer(Int i, Int j) EL_NO_RELEASE_EXCEPT override;
    T const* LockedBuffer() const EL_NO_EXCEPT override;
    T const* LockedBuffer(Int i, Int j) const EL_NO_EXCEPT override;

    ///@}
    /** @name Advanced functions */
    ///@{

    void SetMemoryMode(memory_mode_type mode) override;
    memory_mode_type MemoryMode() const EL_NO_EXCEPT override;

    ///@}

    // Single-entry manipulation
    // =========================
    T Get(Int i, Int j=0) const EL_NO_RELEASE_EXCEPT;
    Base<T> GetRealPart(Int i, Int j=0) const EL_NO_RELEASE_EXCEPT;
    Base<T> GetImagPart(Int i, Int j=0) const EL_NO_RELEASE_EXCEPT;

    void Set(Int i, Int j, T const& alpha) EL_NO_RELEASE_EXCEPT;
    void Set(Entry<T> const& entry) EL_NO_RELEASE_EXCEPT;

    void SetRealPart(Int i, Int j, Base<T> const& alpha) EL_NO_RELEASE_EXCEPT;
    void SetImagPart(Int i, Int j, Base<T> const& alpha) EL_NO_RELEASE_EXCEPT;

    void SetRealPart(Entry<Base<T>> const& entry) EL_NO_RELEASE_EXCEPT;
    void SetImagPart(Entry<Base<T>> const& entry) EL_NO_RELEASE_EXCEPT;

    void Update(Int i, Int j, T const& alpha) EL_NO_RELEASE_EXCEPT;
    void Update(Entry<T> const& entry) EL_NO_RELEASE_EXCEPT;

    void UpdateRealPart(Int i, Int j, Base<T> const& alpha) EL_NO_RELEASE_EXCEPT;
    void UpdateImagPart(Int i, Int j, Base<T> const& alpha) EL_NO_RELEASE_EXCEPT;

    void UpdateRealPart(Entry<Base<T>> const& entry) EL_NO_RELEASE_EXCEPT;
    void UpdateImagPart(Entry<Base<T>> const& entry) EL_NO_RELEASE_EXCEPT;

    void MakeReal(Int i, Int j) EL_NO_RELEASE_EXCEPT;
    void Conjugate(Int i, Int j) EL_NO_RELEASE_EXCEPT;

    // Return a reference to a single entry without error-checking
    // -----------------------------------------------------------
    inline T const& CRef(Int i, Int j=0) const EL_NO_RELEASE_EXCEPT override;
    inline T const& operator()(Int i, Int j=0) const EL_NO_RELEASE_EXCEPT override;

    inline T& Ref(Int i, Int j=0) EL_NO_RELEASE_EXCEPT override;
    inline T& operator()(Int i, Int j=0) EL_NO_RELEASE_EXCEPT override;

private:

    T do_get_(index_type const& i, index_type const& j) const override;
    void do_set_(
        index_type const& i, index_type const& j, T const& val) override;

    // Exchange metadata with another matrix
    // =====================================
    void SwapImpl_(Matrix<T, Device::CPU>& A) EL_NO_EXCEPT;

    // Reconfigure without error-checking
    // ==================================
    void do_empty_(bool freeMemory) override;
    void do_resize_(
        size_type const&, size_type const&, size_type const&) override;
    void do_swap_(AbstractMatrix<T>& A) override;

    void Attach_(
        size_type height, size_type width, value_type* buffer,
        size_type leadingDimension);
    void LockedAttach_(
        size_type height, size_type width, value_type const* buffer,
        size_type leadingDimension);

    // Friend declarations
    // ===================
    template<typename S, Device D> friend class Matrix;
    template<typename S> friend class AbstractDistMatrix;
    template<typename S> friend class ElementalMatrix;
    template<typename S> friend class BlockMatrix;

    // For supporting duck typing
    // ==========================
    // The following are provided in order to aid duck-typing over
    // {Matrix, DistMatrix, etc.}.

    // This is equivalent to the trivial constructor in functionality
    // (though an error is thrown if 'grid' is not equal to 'Grid::Trivial()').
    explicit Matrix(El::Grid const& grid);

    // This is a no-op
    // (though an error is thrown if 'grid' is not equal to 'Grid::Trivial()').
    void SetGrid(El::Grid const& grid);

    // This always returns 'Grid::Trivial()'.
    El::Grid const& Grid() const;

    // This is a no-op
    // (though an error is thrown if 'colAlign' or 'rowAlign' is not zero).
    void Align(Int colAlign, Int rowAlign, bool constrain=true);

    // These always return 0.
    int ColAlign() const EL_NO_EXCEPT;
    int RowAlign() const EL_NO_EXCEPT;

private:
    // Member variables
    // ================
    Memory<T,Device::CPU> memory_;
    // Const-correctness is internally managed to avoid the need for storing
    // two separate pointers with different 'const' attributes
    T* data_ = nullptr;
};

template <typename T, Device D>
void SetSyncInfo(Matrix<T,D>&, SyncInfo<D> const&)
{}

template <typename T>
SyncInfo<Device::CPU> SyncInfoFromMatrix(Matrix<T,Device::CPU> const& mat)
{
    return SyncInfo<Device::CPU>{};
}

#ifdef HYDROGEN_HAVE_GPU
// GPU version
template <typename T>
class Matrix<T, Device::GPU> : public AbstractMatrix<T>
{
public:
    using value_type = typename AbstractMatrix<T>::value_type;
    using index_type = typename AbstractMatrix<T>::index_type;
    using size_type = typename AbstractMatrix<T>::size_type;
    using memory_mode_type = typename AbstractMatrix<T>::memory_mode_type;
public:
    /** @name Constructors and destructors */
    ///@{

    /** @brief Create a 0x0 matrix */
    Matrix();

    /** @brief Create a matrix with the specified dimensions and leading dimension */
    Matrix(size_type height, size_type width,
           size_type leadingDimension=size_type{0});

    /** @brief Construct a matrix around an existing (possibly immutable) buffer */
    Matrix(size_type height, size_type width, value_type const* buffer,
           size_type leadingDimension);
    Matrix(size_type height, size_type width, value_type* buffer,
           size_type leadingDimension);

    /** @brief Create a copy of a matrix */
    Matrix(Matrix<T, Device::GPU> const& A);

    /** @brief Create a copy of a matrix from a CPU matrix */
    Matrix(Matrix<T, Device::CPU> const& A);

    /** @brief Move the metadata from a given matrix */
    Matrix(Matrix<T, Device::GPU>&& A) EL_NO_EXCEPT;

    /** @brief Destructor */
    ~Matrix();

    /** @brief Copy assignment */
    Matrix<T, Device::GPU>& operator=(
        Matrix<T, Device::GPU> const& A);

    /** @brief Assign by copying data from a CPU matrix */
    Matrix<T, Device::GPU>& operator=(
        Matrix<T, Device::CPU> const& A);

    /** @brief Move assignment */
    Matrix<T, Device::GPU>& operator=(Matrix<T, Device::GPU>&& A);

    ///@}
    /** @name Basic queries */
    ///@{

    size_type MemorySize() const EL_NO_EXCEPT override;
    Device GetDevice() const EL_NO_EXCEPT override;
    constexpr Device MyDevice() const EL_NO_EXCEPT { return Device::GPU; }

    T* Buffer() EL_NO_RELEASE_EXCEPT override;
    T* Buffer(Int i, Int j) EL_NO_RELEASE_EXCEPT override;
    T const* LockedBuffer() const EL_NO_EXCEPT override;
    T const*
    LockedBuffer(Int i, Int j) const EL_NO_EXCEPT override;

    ///@}
    /** @name Advanced functions */
    ///@{
    void SetMemoryMode(memory_mode_type mode) override;
    memory_mode_type MemoryMode() const EL_NO_EXCEPT override;
    ///@}
    /** @name Modifiers */
    ///@{

    // Reconfigure around the given buffer, but do not assume ownership
    void Attach(Int height, Int width, T* buffer, Int leadingDimension);
    void LockedAttach(
        Int height, Int width, T const* buffer, Int leadingDimension);

    void Swap(Matrix<T,Device::GPU>& A) EL_NO_EXCEPT;

    ///@}
    /** @name Non-owning views */
    ///@{

    /** @brief Get a view of contiguous subsets of rows and columns
     *
     *  The resulting matrix will still have column-stride of
     *  `this->LDim()`.
     */
    Matrix<T, Device::GPU> operator()(Range<Int> I, Range<Int> J);
    const Matrix<T, Device::GPU>
    operator()(Range<Int> I, Range<Int> J) const;

    ///@}
    /** @name Single-entry manipulation
     *  @deprecated All single-entry manipulations to GPU matrices
     *  should be considered deprecated.
     */
    ///@{

    // FIXME (trb 03/07/18): This is a phenomenally bad idea. This
    // access should be granted for kernels only, if we were to
    // offload these objects directly to device (also probably a bad
    // idea). As is, if the impls didn't just throw, this would
    // require a device sync after every call.

    H_DEPRECATED("Single-entry access to GPU matrices will be removed soon.")
    T Get(Int i, Int j=0) const;

    H_DEPRECATED("Single-entry access to GPU matrices will be removed soon.")
    Base<T> GetRealPart(Int i, Int j=0) const;
    H_DEPRECATED("Single-entry access to GPU matrices will be removed soon.")
    Base<T> GetImagPart(Int i, Int j=0) const;

    H_DEPRECATED("Single-entry access to GPU matrices will be removed soon.")
    void Set(Int i, Int j, T const& alpha);
    H_DEPRECATED("Single-entry access to GPU matrices will be removed soon.")
    void Set(Entry<T> const& entry);

    H_DEPRECATED("Single-entry access to GPU matrices will be removed soon.")
    void SetRealPart(Int i, Int j, Base<T> const& alpha);
    H_DEPRECATED("Single-entry access to GPU matrices will be removed soon.")
    void SetImagPart(Int i, Int j, Base<T> const& alpha);

    H_DEPRECATED("Single-entry access to GPU matrices will be removed soon.")
    void SetRealPart(Entry<Base<T>> const& entry);
    H_DEPRECATED("Single-entry access to GPU matrices will be removed soon.")
    void SetImagPart(Entry<Base<T>> const& entry);

    H_DEPRECATED("Single-entry access to GPU matrices will be removed soon.")
    void Update(Int i, Int j, T const& alpha);
    H_DEPRECATED("Single-entry access to GPU matrices will be removed soon.")
    void Update(Entry<T> const& entry);

    H_DEPRECATED("Single-entry access to GPU matrices will be removed soon.")
    void UpdateRealPart(Int i, Int j, Base<T> const& alpha);
    H_DEPRECATED("Single-entry access to GPU matrices will be removed soon.")
    void UpdateImagPart(Int i, Int j, Base<T> const& alpha);

    H_DEPRECATED("Single-entry access to GPU matrices will be removed soon.")
    void UpdateRealPart(Entry<Base<T>> const& entry);
    H_DEPRECATED("Single-entry access to GPU matrices will be removed soon.")
    void UpdateImagPart(Entry<Base<T>> const& entry);

    H_DEPRECATED("Single-entry access to GPU matrices will be removed soon.")
    void MakeReal(Int i, Int j);
    H_DEPRECATED("Single-entry access to GPU matrices will be removed soon.")
    void Conjugate(Int i, Int j);

    // Return a reference to a single entry without error-checking
    // -----------------------------------------------------------
    H_DEPRECATED("Single-entry access to GPU matrices will be removed soon.")
    inline T const& CRef(Int i, Int j=0) const override;
    H_DEPRECATED("Single-entry access to GPU matrices will be removed soon.")
    inline T const& operator()(Int i, Int j=0) const override;

    H_DEPRECATED("Single-entry access to GPU matrices will be removed soon.")
    inline T& Ref(Int i, Int j=0) override;
    H_DEPRECATED("Single-entry access to GPU matrices will be removed soon.")
    inline T& operator()(Int i, Int j=0) override;

    ///@}
    /** @name Synchronization semantics */
    ///@{

    SyncInfo<Device::GPU> GetSyncInfo() const EL_NO_EXCEPT;
    void SetSyncInfo(SyncInfo<Device::GPU> const&) EL_NO_EXCEPT;

    ///@}

private:

    void do_empty_(bool freeMemory) override;
    void do_resize_(
        size_type const&, size_type const&, size_type const&) override;

    T do_get_(index_type const& i, index_type const& j) const override;
    void do_set_(
        index_type const& i, index_type const& j, T const& val) override;

    void do_swap_(AbstractMatrix<T>& A) override;

    void Attach_(
        size_type height, size_type width, value_type* buffer,
        size_type leadingDimension);
    void LockedAttach_(
        size_type height, size_type width,
        const value_type* buffer, size_type leadingDimension);

    void SwapImpl_(Matrix<T, Device::GPU>& A) EL_NO_EXCEPT;

    template<typename S, Device D> friend class Matrix;
    template<typename S> friend class AbstractDistMatrix;
    template<typename S> friend class ElementalMatrix;
    template<typename S> friend class BlockMatrix;

private:

    Memory<T,Device::GPU> memory_;

    T* data_=nullptr;

    SyncInfo<Device::GPU> sync_info_ = gpu::DefaultSyncInfo();

};// class Matrix<T,Device::GPU>

template <typename T>
SyncInfo<Device::GPU> SyncInfoFromMatrix(Matrix<T,Device::GPU> const& mat)
{
    return mat.GetSyncInfo();
}

template <typename T>
void SetSyncInfo(
    Matrix<T,Device::GPU>& mat, SyncInfo<Device::GPU> const& syncInfo)
{
    mat.SetSyncInfo(syncInfo);
}
#endif // HYDROGEN_HAVE_GPU

} // namespace El

#endif // ifndef EL_MATRIX_DECL_HPP
