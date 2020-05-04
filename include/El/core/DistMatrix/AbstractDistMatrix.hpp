/*
  Copyright (c) 2009-2016, Jack Poulson
  All rights reserved.

  This file is part of Elemental and is under the BSD 2-Clause License,
  which can be found in the LICENSE file in the root directory, or at
  http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_DISTMATRIX_ABSTRACT_HPP
#define EL_DISTMATRIX_ABSTRACT_HPP

#include <functional>
#include <typeinfo>

namespace El
{

struct DistData;

/** @class BaseDistMatrix
 *  @brief A basic description of a distributed matrix
 */
class BaseDistMatrix
{
public:
    /** @brief Destructor */
    virtual ~BaseDistMatrix() = default;
    /** @brief The dimension of the range space */
    virtual Int Height() const noexcept = 0;
    /** @brief The dimension of the domain space */
    virtual Int Width() const noexcept = 0;
    /** @brief A description of the distribution information */
    virtual El::DistData DistData() const noexcept = 0;
};// class BaseDistMatrix

template<typename Ring>
class AbstractDistMatrix : public BaseDistMatrix
{
public:
    // Typedefs
    // ========
    typedef AbstractDistMatrix<Ring> type;

    // Constructors and destructors
    // ============================
    // Move constructor
    AbstractDistMatrix(type&& A) EL_NO_EXCEPT;

    virtual ~AbstractDistMatrix();

    // Assignment and reconfiguration
    // ==============================
    void EmptyData(bool freeMemory=true);
    void SetGrid(const El::Grid& grid);

    void MakeSizeConsistent(bool includingViewers=false);

    void SetRoot(int root, bool constrain=true);

    // Operator overloading
    // ====================

    // Move assignment
    // ---------------
    type& operator=(type&& A);

    // Rescaling
    // ---------
    const type& operator*=(Ring alpha);

    // Basic queries
    // =============

    // Global matrix information
    // -------------------------
    Int Height() const EL_NO_EXCEPT override;
    Int Width() const EL_NO_EXCEPT override;
    Int DiagonalLength(Int offset=0) const EL_NO_EXCEPT;
    bool Viewing() const EL_NO_EXCEPT;
    bool Locked() const EL_NO_EXCEPT;
    bool IsEmpty() const EL_NO_EXCEPT;

    // Local matrix information
    // ------------------------
    Int LocalHeight() const EL_NO_EXCEPT;
    Int LocalWidth() const EL_NO_EXCEPT;
    Int LDim() const EL_NO_EXCEPT;
    bool Contiguous() const EL_NO_EXCEPT;

    size_t AllocatedMemory() const EL_NO_EXCEPT;

    Ring* Buffer() EL_NO_RELEASE_EXCEPT;
    Ring* Buffer(Int iLoc, Int jLoc) EL_NO_RELEASE_EXCEPT;

    const Ring* LockedBuffer() const EL_NO_EXCEPT;
    const Ring* LockedBuffer(Int iLoc, Int jLoc) const EL_NO_EXCEPT;

    virtual El::AbstractMatrix<Ring>& Matrix() EL_NO_EXCEPT = 0;
    virtual const El::AbstractMatrix<Ring>& LockedMatrix() const EL_NO_EXCEPT = 0;

    // Distribution information
    // ------------------------
    const El::Grid& Grid() const EL_NO_EXCEPT;

    int ColAlign() const EL_NO_EXCEPT;
    int RowAlign() const EL_NO_EXCEPT;
    int ColShift() const EL_NO_EXCEPT;
    int RowShift() const EL_NO_EXCEPT;
    bool ColConstrained() const EL_NO_EXCEPT;
    bool RowConstrained() const EL_NO_EXCEPT;
    bool RootConstrained() const EL_NO_EXCEPT;
    bool Participating() const EL_NO_RELEASE_EXCEPT;
    int Root() const EL_NO_EXCEPT;

    Int LocalRow(Int i) const EL_NO_RELEASE_EXCEPT;
    Int LocalCol(Int j) const EL_NO_RELEASE_EXCEPT;
    Int LocalRow(Int i, int rowOwner) const EL_NO_RELEASE_EXCEPT;
    Int LocalCol(Int j, int colOwner) const EL_NO_RELEASE_EXCEPT;
    int Owner(Int i, Int j) const EL_NO_EXCEPT;
    bool IsLocalRow(Int i) const EL_NO_RELEASE_EXCEPT;
    bool IsLocalCol(Int j) const EL_NO_RELEASE_EXCEPT;
    bool IsLocal(Int i, Int j) const EL_NO_RELEASE_EXCEPT;

    El::DistData DistData() const EL_NO_EXCEPT override;

    // Single-entry manipulation
    // =========================

    // Global entry manipulation
    // -------------------------
    // NOTE: Local entry manipulation is often much faster and should be
    //       preferred in most circumstances where performance matters.

    virtual Ring Get(Int i, Int j) const EL_NO_RELEASE_EXCEPT = 0;

    virtual Base<Ring> GetRealPart(Int i, Int j) const EL_NO_RELEASE_EXCEPT = 0;
    virtual Base<Ring> GetImagPart(Int i, Int j) const EL_NO_RELEASE_EXCEPT = 0;

    virtual void Set(Int i, Int j, Ring alpha) EL_NO_RELEASE_EXCEPT = 0;
    virtual void Set(const Entry<Ring>& entry) EL_NO_RELEASE_EXCEPT = 0;

    virtual void SetRealPart(Int i, Int j, Base<Ring> alpha) EL_NO_RELEASE_EXCEPT = 0;
    virtual void SetImagPart (Int i, Int j, Base<Ring> alpha) EL_NO_RELEASE_EXCEPT = 0;

    virtual void SetRealPart(const Entry<Base<Ring>>& entry) EL_NO_RELEASE_EXCEPT = 0;
    virtual void SetImagPart(const Entry<Base<Ring>>& entry) EL_NO_RELEASE_EXCEPT = 0;

    virtual void Update(Int i, Int j, Ring alpha) EL_NO_RELEASE_EXCEPT = 0;
    virtual void Update(const Entry<Ring>& entry) EL_NO_RELEASE_EXCEPT = 0;

    virtual void UpdateRealPart(Int i, Int j, Base<Ring> alpha) EL_NO_RELEASE_EXCEPT = 0;
    virtual void UpdateImagPart(Int i, Int j, Base<Ring> alpha) EL_NO_RELEASE_EXCEPT = 0;

    virtual void UpdateRealPart(const Entry<Base<Ring>>& entry) EL_NO_RELEASE_EXCEPT = 0;
    virtual void UpdateImagPart(const Entry<Base<Ring>>& entry) EL_NO_RELEASE_EXCEPT = 0;

    virtual void MakeReal(Int i, Int j) EL_NO_RELEASE_EXCEPT = 0;
    virtual void Conjugate(Int i, Int j) EL_NO_RELEASE_EXCEPT = 0;

    // Batch updating of remote entries
    // ---------------------------------
    virtual void Reserve(Int numRemoteEntries) = 0;
    virtual void QueueUpdate(const Entry<Ring>& entry) EL_NO_RELEASE_EXCEPT = 0;
    virtual void QueueUpdate(Int i, Int j, Ring value) EL_NO_RELEASE_EXCEPT = 0;
    virtual void ProcessQueues(bool includeViewers=true) = 0;

    // Batch extraction of remote entries
    // ----------------------------------
    virtual void ReservePulls(Int numPulls) const = 0;
    virtual void QueuePull(Int i, Int j) const EL_NO_RELEASE_EXCEPT = 0;
    virtual void ProcessPullQueue(Ring* pullBuf, bool includeViewers=true) const = 0;
    virtual void ProcessPullQueue(
        vector<Ring>& pullBuf, bool includeViewers=true) const = 0;

    // Local entry manipulation
    // ------------------------
    // NOTE: Clearly each of the following routines could instead be performed
    //       via composing [Locked]Matrix() with the corresponding local
    //       routine, but a large amount of code might need to change if
    //       these were removed.

    virtual Ring GetLocal(Int iLoc, Int jLoc) const EL_NO_RELEASE_EXCEPT = 0;
    virtual Base<Ring> GetLocalRealPart(Int iLoc, Int jLoc) const EL_NO_RELEASE_EXCEPT = 0;
    virtual Base<Ring> GetLocalImagPart(Int iLoc, Int jLoc) const EL_NO_RELEASE_EXCEPT = 0;

    virtual void SetLocal(Int iLoc, Int jLoc, Ring alpha) EL_NO_RELEASE_EXCEPT = 0;
    virtual void SetLocal(Entry<Ring> const& localEntry) EL_NO_RELEASE_EXCEPT = 0;

    virtual void SetLocalRealPart(Int iLoc, Int jLoc, Base<Ring> alpha)
        EL_NO_RELEASE_EXCEPT = 0;
    virtual void SetLocalImagPart(Int iLoc, Int jLoc, Base<Ring> alpha)
        EL_NO_RELEASE_EXCEPT = 0;

    virtual void SetLocalRealPart(const Entry<Base<Ring>>& localEntry)
        EL_NO_RELEASE_EXCEPT = 0;
    virtual void SetLocalImagPart(const Entry<Base<Ring>>& localEntry)
        EL_NO_RELEASE_EXCEPT = 0;

    virtual void UpdateLocal
    (Int iLoc, Int jLoc, Ring alpha) EL_NO_RELEASE_EXCEPT = 0;
    virtual void UpdateLocal
    (const Entry<Ring>& localEntry) EL_NO_RELEASE_EXCEPT = 0;

    virtual void UpdateLocalRealPart(Int iLoc, Int jLoc, Base<Ring> alpha)
        EL_NO_RELEASE_EXCEPT = 0;
    virtual void UpdateLocalImagPart(Int iLoc, Int jLoc, Base<Ring> alpha)
        EL_NO_RELEASE_EXCEPT = 0;

    virtual void UpdateLocalRealPart(const Entry<Base<Ring>>& localEntry)
        EL_NO_RELEASE_EXCEPT = 0;
    virtual void UpdateLocalImagPart(const Entry<Base<Ring>>& localEntry)
        EL_NO_RELEASE_EXCEPT = 0;

    virtual void MakeLocalReal(Int iLoc, Int jLoc) EL_NO_RELEASE_EXCEPT = 0;
    virtual void ConjugateLocal(Int iLoc, Int jLoc) EL_NO_RELEASE_EXCEPT = 0;

    // Assertions
    // ==========
    void AssertNotLocked() const;
    void AssertNotStoringData() const;
    void AssertValidEntry(Int i, Int j) const;
    void AssertValidSubmatrix(Int i, Int j, Int height, Int width) const;
    void AssertSameSize(Int height, Int width) const;

    //
    // Virtual functions
    //
    virtual void Empty(bool freeMemory=true);
    virtual void FreeAlignments();

    // Pure virtual functions

    virtual type* Copy() const = 0;
    virtual type* Construct(const El::Grid& grid, int root) const = 0;
    virtual type* ConstructTranspose(const El::Grid& grid, int root) const = 0;
    virtual type* ConstructDiagonal(const El::Grid& grid, int root) const = 0;
    virtual std::unique_ptr<type> ConstructWithNewDevice(Device D2) const = 0;
    // TODO(poulson): ConstructPartialCol and friends?

    virtual void AlignWith(
        const El::DistData& data,
        bool constrain=true, bool allowMismatch=false) = 0;
    virtual void AlignColsWith(
        const El::DistData& data,
        bool constrain=true, bool allowMismatch=false) = 0;
    virtual void AlignRowsWith(
        const El::DistData& data,
        bool constrain=true, bool allowMismatch=false) = 0;

    virtual void Resize(Int height, Int width) = 0;
    virtual void Resize(Int height, Int width, Int ldim) = 0;

    virtual Int BlockHeight() const EL_NO_EXCEPT = 0;
    virtual Int BlockWidth() const EL_NO_EXCEPT = 0;
    virtual Int ColCut() const EL_NO_EXCEPT = 0;
    virtual Int RowCut() const EL_NO_EXCEPT = 0;

    virtual int RowOwner(Int i) const EL_NO_EXCEPT = 0;
    virtual int ColOwner(Int j) const EL_NO_EXCEPT = 0;
    virtual Int LocalRowOffset(Int i) const EL_NO_EXCEPT = 0;
    virtual Int LocalColOffset(Int j) const EL_NO_EXCEPT = 0;
    virtual Int LocalRowOffset(Int i, int rowOwner) const EL_NO_EXCEPT = 0;
    virtual Int LocalColOffset(Int j, int colOwner) const EL_NO_EXCEPT = 0;
    virtual Int GlobalRow(Int iLoc) const EL_NO_EXCEPT = 0;
    virtual Int GlobalCol(Int jLoc) const EL_NO_EXCEPT = 0;

    // NOTE: These are all clearly equivalent to composing mpi::Rank
    //       with ColComm(), RowComm(), etc., but it is not clear that
    //       they should be removed just yet.
    virtual int ColRank() const EL_NO_RELEASE_EXCEPT = 0;
    virtual int RowRank() const EL_NO_RELEASE_EXCEPT = 0;
    virtual int PartialColRank() const EL_NO_RELEASE_EXCEPT = 0;
    virtual int PartialRowRank() const EL_NO_RELEASE_EXCEPT = 0;
    virtual int PartialUnionColRank() const EL_NO_RELEASE_EXCEPT = 0;
    virtual int PartialUnionRowRank() const EL_NO_RELEASE_EXCEPT = 0;
    virtual int DistRank() const EL_NO_RELEASE_EXCEPT = 0;
    virtual int CrossRank() const EL_NO_RELEASE_EXCEPT = 0;
    virtual int RedundantRank() const EL_NO_RELEASE_EXCEPT = 0;

    virtual Dist ColDist() const EL_NO_EXCEPT = 0;
    virtual Dist RowDist() const EL_NO_EXCEPT = 0;
    virtual Dist CollectedColDist() const EL_NO_EXCEPT = 0;
    virtual Dist CollectedRowDist() const EL_NO_EXCEPT = 0;
    virtual Dist PartialColDist() const EL_NO_EXCEPT = 0;
    virtual Dist PartialRowDist() const EL_NO_EXCEPT = 0;
    virtual Dist PartialUnionColDist() const EL_NO_EXCEPT = 0;
    virtual Dist PartialUnionRowDist() const EL_NO_EXCEPT = 0;
    virtual DistWrap Wrap() const EL_NO_EXCEPT = 0;

    virtual mpi::Comm const& ColComm() const EL_NO_EXCEPT = 0;
    virtual mpi::Comm const& RowComm() const EL_NO_EXCEPT = 0;
    virtual mpi::Comm const& PartialColComm() const EL_NO_EXCEPT = 0;
    virtual mpi::Comm const& PartialRowComm() const EL_NO_EXCEPT = 0;
    virtual mpi::Comm const& PartialUnionColComm() const EL_NO_EXCEPT = 0;
    virtual mpi::Comm const& PartialUnionRowComm() const EL_NO_EXCEPT = 0;
    virtual mpi::Comm const& DistComm() const EL_NO_EXCEPT = 0;
    virtual mpi::Comm const& CrossComm() const EL_NO_EXCEPT = 0;
    virtual mpi::Comm const& RedundantComm() const EL_NO_EXCEPT = 0;

    virtual int ColStride() const EL_NO_EXCEPT = 0;
    virtual int RowStride() const EL_NO_EXCEPT = 0;
    virtual int PartialColStride() const EL_NO_EXCEPT = 0;
    virtual int PartialRowStride() const EL_NO_EXCEPT = 0;
    virtual int PartialUnionColStride() const EL_NO_EXCEPT = 0;
    virtual int PartialUnionRowStride() const EL_NO_EXCEPT = 0;
    virtual int DistSize() const EL_NO_EXCEPT = 0;
    virtual int CrossSize() const EL_NO_EXCEPT = 0;
    virtual int RedundantSize() const EL_NO_EXCEPT = 0;

    // Diagonal manipulation
    // =====================
    virtual bool DiagonalAlignedWith
    (const El::DistData& d, Int offset=0) const = 0;
    virtual int DiagonalRoot(Int offset=0) const = 0;
    virtual int DiagonalAlign(Int offset=0) const = 0;

    virtual Device GetLocalDevice() const EL_NO_EXCEPT = 0;

    // Static functions
    // ================

    // Instantiate with runtime template arguments
    static type* Instantiate
    (const El::Grid& grid=Grid::Default(), int root=0,
     Dist colDist=MC, Dist rowDist=MR, DistWrap wrap=ELEMENT,
     Device dev=Device::CPU);
    static type* Instantiate(const El::DistData& data);

private:
    // Member variables
    // ================

    // Global and local matrix information
    // -----------------------------------
    ViewType viewType_=OWNER;
    Int height_=0, width_=0;

    bool colConstrained_=false,
        rowConstrained_=false,
        rootConstrained_=false;
    int colAlign_=0,
        rowAlign_=0,
        colShift_=0,
        rowShift_=0;
    int root_=0;

    // An observing pointer to a pre-existing grid.
    const El::Grid* grid_=nullptr;

protected:

    // Protected constructors
    // ======================
    // Create a 0 x 0 distributed matrix
    AbstractDistMatrix(const El::Grid& grid=Grid::Default(), int root=0);

    // Modify the distribution metadata
    // ================================
    virtual void SetShifts();
    virtual void SetColShift();
    virtual void SetRowShift();

private:
    // Exchange metadata with another matrix
    // =====================================
    void ShallowSwap(type& A);

    virtual void do_empty_data_() = 0;

    template<typename S> friend class AbstractDistMatrix;
    template<typename S> friend class ElementalMatrix;
    template<typename S> friend class BlockMatrix;
};

struct DistData
{
    Dist colDist, rowDist;
    Int blockHeight, blockWidth;
    int colAlign, rowAlign;
    Int colCut, rowCut;
    int root;  // relevant for [o ,o ]/[MD,* ]/[* ,MD]
    const Grid* grid;
    Device device;
    std::reference_wrapper<std::type_info const> held_type_info;

    DistData() : held_type_info(typeid(BaseDistMatrix)) { }

    template<typename Ring>
    DistData(const AbstractDistMatrix<Ring>& A)
        : colDist(A.ColDist()), rowDist(A.RowDist()),
          blockHeight(A.BlockHeight()), blockWidth(A.BlockWidth()),
          colAlign(A.ColAlign()), rowAlign(A.RowAlign()),
          colCut(A.ColCut()), rowCut(A.RowCut()),
          root(A.Root()), grid(&A.Grid()), device(A.GetLocalDevice()),
          held_type_info(typeid(Ring))
    { }
};
inline bool operator==(const DistData& A, const DistData& B)
{ return ((A.colDist == B.colDist) &&
          (A.rowDist == B.rowDist) &&
          (A.blockHeight == B.blockHeight) &&
          (A.blockWidth == B.blockWidth) &&
          (A.colAlign == B.colAlign) &&
          (A.rowAlign == B.rowAlign) &&
          (A.colCut == B.colCut) &&
          (A.rowCut == B.rowCut) &&
          (A.root == B.root) &&
          (A.grid == B.grid) &&
          (A.device == B.device) &&
          (static_cast<std::type_info const&>(A.held_type_info)
           == static_cast<std::type_info const&>(B.held_type_info))); }
inline bool operator!=(const DistData& A, const DistData& B)
{ return !(A == B); }

} // namespace El

#endif // ifndef EL_DISTMATRIX_ABSTRACT_HPP
