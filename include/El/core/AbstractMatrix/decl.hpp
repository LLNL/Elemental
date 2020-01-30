#ifndef EL_CORE_ABSTRACTMATRIX_DECL_HPP_
#define EL_CORE_ABSTRACTMATRIX_DECL_HPP_

namespace El
{

template <typename T>
class AbstractMatrix
{
public:
    /** @name Public types */
    ///@{
    using value_type = T;
    using index_type = Int;
    using size_type = Int;
    using memory_mode_type = unsigned int;
    ///@}

public:

    /** @name Constructors and destructor */
    ///@{

    /** @brief Construct an empty AbstractMatrix. */
    AbstractMatrix() = default;

    /** @brief Copy-construct an AbstractMatrix.
     *
     *  Views will be copied as views.
     */
    AbstractMatrix(AbstractMatrix<T> const&) = default;

    /** @brief Move-construct an AbstractMatrix.
     *
     *  Views will be moved into views.
     */
    AbstractMatrix(AbstractMatrix<T>&&) = default;

    /** @brief Copy-assign an AbstractMatrix.
     *
     *  This will always perform a deep copy of the underlying matrix
     *  data. That is, views will have their data deep-copied into a
     *  new owning matrix.
     */
    AbstractMatrix<T>& operator=(AbstractMatrix<T> const&);

    /** @brief Destroy the AbstractMatrix. */
    virtual ~AbstractMatrix() = default;

    ///@}
    /** @name Abstract Copies. */
    ///@{

    /** @brief Copy the underlying matrix into a new matrix with the
     *         same type and device.
     *
     *  @return A newly-allocated matrix that is a deep copy of this
     *          matrix with the same device type.
     *  @warning This could dynamically allocate memory.
     */
    virtual std::unique_ptr<AbstractMatrix<T>> Copy() const = 0;

    /** @brief Construct an empty matrix with the same type and device.
     *
     *  @return A default-constructed (i.e., empty) matrix with the
     *          same device allocation as this matrix.
     */
    virtual std::unique_ptr<AbstractMatrix<T>> Construct() const = 0;

    ///@}
    /** @name Queries */
    ///@{

    /** @brief Get the number of rows in the matrix. */
    size_type Height() const EL_NO_EXCEPT;

    /** @brief Get the number of columns in the matrix. */
    size_type Width() const EL_NO_EXCEPT;

    /** @brief Get the leading dimension of the memory.
     *
     *  It is assumed that the matrix memory is stored column-major;
     *  this information is required for most BLAS interactions, as
     *  well as numerous internal kernels.
     */
    size_type LDim() const EL_NO_EXCEPT;

    /** @brief Get the size of the underlying memory. */
    virtual size_type MemorySize() const EL_NO_EXCEPT = 0;

    /** @brief Get the the length of the matrix diagonal.
     *
     *  @param[in] offset If positive, a column offset. If negative, a
     *                    row offset.
     *
     *  @return The length of the diagonal of the submatrix
     *          A[:, offset:] if offset > 0 or
     *          A[-offset:, :] if offset <= 0.
     */
    size_type DiagonalLength(
        size_type offset = size_type{0}) const EL_NO_EXCEPT;

    /** @brief Test if the matrix is a view.
     *
     *  Views are considered to have fixed size. Smaller views may be
     *  constructed by creating new views with smaller dimensions.
     */
    bool Viewing() const EL_NO_EXCEPT;

    /** @brief Test if the matrix has a fixed size.
     *
     *  Any AbstractMatrix may have a fixed size. This will prevent
     *  resizing. All AbstractMatrix objects that represent views have
     *  fixed size by definition.
     */
    bool FixedSize() const EL_NO_EXCEPT;

    /** @brief Test if the matrix is logically @c const.
     *
     *  If @c this->Locked(), certain methods that are not marked @c
     *  const may throw with exceptions indicating that a logical @c
     *  const condition would be violated.
     */
    bool Locked() const EL_NO_EXCEPT;

    /** @brief Test if the matrix is empty (i.e., has size 0x0). */
    bool IsEmpty() const EL_NO_EXCEPT;

    /** @brief Test if the matrix memory is contiguous.
     *
     *  A matrix is considered contiguous if any of the following hold:
     *    - The matrix height equals its leading dimension.
     *    - The matrix has width 1.
     *    - The matrix is empty.
     */
    bool Contiguous() const EL_NO_EXCEPT;

    ///@}
    /** @name Advanced queries */
    ///@{

    /** @brief Get the ViewType of the matrix. */
    El::ViewType ViewType() const EL_NO_EXCEPT;

    /** @brief Get the device type on which memory is resident. */
    virtual Device GetDevice() const EL_NO_EXCEPT = 0;

    /** @brief Get the underlying memory mode.
     *
     *  @todo (trb) Why is this exposed here? This is intimately tied
     *  to device allocations and the like. I'm skeptical this is
     *  required at this level of the interface.
     */
    virtual memory_mode_type MemoryMode() const EL_NO_EXCEPT = 0;

    ///@}
    /** @name Modifiers */
    ///@{

    /** @brief Permanently fix the size of the matrix.
     *
     *  Any further call to Resize that with different size
     *  information will result in an exception.
     */
    void FixSize() EL_NO_EXCEPT;

    /** @brief Reset the matrix to have size 0x0, optionally freeing
     *         underlying memory.
     */
    void Empty(bool freeMemory=true);

    /** @brief Resize the Matrix.
     *
     *  @param height The number of rows in the matrix.
     *  @param width  The number of columns in the matrix.
     *
     *  @throws A bunch of things
     */
    void Resize(size_type height, size_type width);

    /** @brief Resize the Matrix.
     *
     *  @param height The number of rows in the matrix.
     *  @param width  The number of columns in the matrix.
     *  @param ldim   The leading dimension of the underlying memory.
     *
     *  @throws A bunch of things
     */
    void Resize(size_type height, size_type width, size_type ldim);

    /** @brief Exchange data with another AbstractMatrix.
     *
     *  @throws El::DeviceError if @c this and @c other are on
     *          different devices.
     */
    void Swap(AbstractMatrix<T>& other);

    // Advanced functions
    void SetViewType(El::ViewType viewType) EL_NO_EXCEPT;
    virtual void SetMemoryMode(memory_mode_type mode) = 0;

    ///@}
    /** @name Dimensional assertions */
    ///@{

    void AssertValidEntry(index_type i, index_type j) const;

    ///@}
    /** @name Buffer access */
    ///@{

    /** @brief Get a pointer to the beginning of the raw memory
     *         region for this matrix.
     *
     *  This is equivalent to `Buffer(0,0)`.
     */
    virtual T* Buffer() EL_NO_RELEASE_EXCEPT = 0;

    /** @brief Get a pointer to the raw memory beginning at the
     *         specified row and column.
     *
     *  @warning This routine is never bounds-checked.
     *
     *  @param i The row index of the first element.
     *  @param j The column index of the first element.
     */
    virtual T* Buffer(index_type i, index_type j) EL_NO_RELEASE_EXCEPT = 0;

    /** @brief Get a read-only pointer to the beginning of the raw
     *         memory region for this matrix.
     *
     *  This is equivalent to `LockedBuffer(0,0)`.
     */
    virtual T const* LockedBuffer() const EL_NO_EXCEPT = 0;

    /** @brief Get a read-only pointer to the raw memory beginning at
     *         the specified row and column.
     *
     *  @warning This routine is never bounds-checked.
     *
     *  @param i The row index of the first element.
     *  @param j The column index of the first element.
     */
    virtual T const* LockedBuffer(index_type i, index_type j) const EL_NO_EXCEPT = 0;

    ///@}
    /** @name Deprecated functions */
    ///@}

    /** @brief Alias for Swap().
     *
     *  @deprecated Prefer Swap() instead.
     */
    H_DEPRECATED("Use Swap() instead.")
    void ShallowSwap(AbstractMatrix<T>& A);

    // Type conversion
    H_DEPRECATED("Extremely dangerous. Will be removed soon.")
    operator Matrix<T, Device::CPU>& ()
    {
        if(this->GetDevice() != Device::CPU)
        {
            LogicError("Illegal conversion from AbstractMatrix to "
                       "incompatible CPU Matrix reference.");
        }
        return static_cast<Matrix<T, Device::CPU>&>(*this);
    }
    H_DEPRECATED("Extremely dangerous. Will be removed soon.")
    operator Matrix<T, Device::CPU>const& () const
    {
        if(this->GetDevice() != Device::CPU)
        {
            LogicError("Illegal conversion from AbstractMatrix to "
                       "incompatible CPU Matrix const reference.");
        }
        return static_cast<const Matrix<T, Device::CPU>&>(*this);
    }

#ifdef HYDROGEN_HAVE_GPU
    H_DEPRECATED("Extremely dangerous. Will be removed soon.")
    operator Matrix<T, Device::GPU>& ()
    {
        if(this->GetDevice() != Device::GPU)
        {
            LogicError("Illegal conversion from AbstractMatrix to "
                       "incompatible GPU Matrix reference.");
        }
        return static_cast<Matrix<T, Device::GPU>&>(*this);
    }
    H_DEPRECATED("Extremely dangerous. Will be removed soon.")
    operator Matrix<T, Device::GPU>const& () const
    {
        if(this->GetDevice() != Device::GPU)
        {
            LogicError("Illegal conversion from AbstractMatrix to "
                       "incompatible GPU Matrix const reference.");
        }
        return static_cast<const Matrix<T, Device::GPU>&>(*this);
    }
#endif // HYDROGEN_HAVE_GPU

    // Single-entry manipulation
    // =========================
    H_DEPRECATED("Will be removed soon.")
    T Get(size_type const& i, size_type const& j=size_type{0})
        const EL_NO_RELEASE_EXCEPT;

    H_DEPRECATED("Will be removed soon.")
    void Set(size_type const& i, size_type const& j, T const& alpha)
        EL_NO_RELEASE_EXCEPT;

    H_DEPRECATED("Will be removed soon.")
    void Set(Entry<T> const& entry) EL_NO_RELEASE_EXCEPT;

    // Return a reference to a single entry without error-checking
    // -----------------------------------------------------------
    H_DEPRECATED("Will be removed soon.")
    virtual T const& CRef(Int i, Int j=0) const = 0;
    H_DEPRECATED("Will be removed soon.")
    virtual T const& operator()(Int i, Int j=0) const = 0;

    H_DEPRECATED("Will be removed soon.")
    virtual T& Ref(Int i, Int j=0) = 0;
    H_DEPRECATED("Will be removed soon.")
    virtual T& operator()(Int i, Int j=0) = 0;
    ///@}

protected:

    /** @brief Construct a new data-owning AbstractMatrix with the
     *         given metadata.
     */
    AbstractMatrix(size_type height, size_type width, size_type ldim);

    /** @brief Construct a new AbstractMatrix with the given metadata
     *         and view type.
     */
    AbstractMatrix(
        El::ViewType view, size_type height, size_type width, size_type ldim);

    /** @brief Move-assign an AbstractMatrix.
     *
     *  Views will be moved into views.
     */
    AbstractMatrix<T>& operator=(AbstractMatrix<T>&&) = default;

    /** @brief Swap metadata only.
     *
     *  This is useful for derived copies implementing noexcept swaps.
     */
    void SwapMetadata_(AbstractMatrix<T>& A) EL_NO_EXCEPT;

    /** @brief This is needed for attach and lockedattach */
    void SetSize_(Int height, Int width, Int ldim);
private:

    /** @name Virtual functions */
    ///@{

    // Operations on indexed data managed by the derived class. Note
    // that bounds-checking is handled at the base-class level.
    virtual T do_get_(index_type const& i, index_type const& j) const = 0;
    virtual void do_set_(
        index_type const& i, index_type const& j, T const& val) = 0;

    // Operations that involve both data and metadata of the
    // matrix. This class handles the metadata and defers data
    // handling to derived classes. These represent the derived
    // class's obligations for these operations.
    virtual void do_empty_(bool) = 0;

    // This class should take arguments viewed as "tentative". That
    // way, we don't need to worry about correcting metadata if a
    // reallocation fails.
    virtual void do_resize_(
        size_type const& height, size_type const& width,
        size_type const& ldim) = 0;
    virtual void do_swap_(AbstractMatrix<T>&) = 0;
    ///@}

private:

    // FIXME (trb): It would be cool if these were not friends.
    template<typename S> friend class AbstractDistMatrix;
    template<typename S> friend class ElementalMatrix;
    template<typename S> friend class BlockMatrix;

    // These don't have debugging checks
    void Empty_(bool freeMemory=true);
    void Resize_(Int height, Int width, Int leadingDimension);

private:

    size_type height_ = size_type{0};
    size_type width_ = size_type{0};
    size_type leadingDimension_ = size_type{1};
    El::ViewType viewType_=OWNER;

};// class AbstractMatrix

/** @brief Assert that the the height/width/ldim combination is valid.
 *
 *  All dimensions must be nonnegative; leading dimension must be
 *  positive. Leading dimension must be at least height.
 *
 *  @throws std::logic_error if any of the above conditions are violated.
 */
template <typename SizeT, typename=EnableIf<std::is_signed<SizeT>>>
void AssertValidDimensions(
    SizeT const& height, SizeT const& width, SizeT const& leadingDimension)
{
    EL_DEBUG_CSE;
    if (height < 0 || width < 0)
        LogicError("Height and width must be non-negative");
    if (leadingDimension < height)
        LogicError("Leading dimension must be no less than height");
    if (leadingDimension <= 0)
        LogicError("Leading dimension must be greater than zero "
                   "(for BLAS compatibility)");
}

template <typename SizeT, typename=DisableIf<std::is_signed<SizeT>>, typename=void>
void AssertValidDimensions(
    SizeT const& height, SizeT const& width, SizeT const& leadingDimension)
{
    EL_DEBUG_CSE;
    if (leadingDimension < height)
        LogicError("Leading dimension must be no less than height");
    if (leadingDimension == 0)
        LogicError("Leading dimension must be greater than zero "
                   "(for BLAS compatibility)");
}

}// namespace El
#endif // EL_CORE_ABSTRACTMATRIX_DECL_HPP_
