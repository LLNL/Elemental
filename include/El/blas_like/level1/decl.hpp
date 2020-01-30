/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS1_DECL_HPP
#define EL_BLAS1_DECL_HPP

namespace El {

// TODO(poulson):
// More 'Contract' routines, e.g., {Contract,ContractedAxpy},
// which sum results over the teams of processes that shared data in the
// original distribution but do not in the final distribution.
// For example, a contraction of the form (U,Collect(V)) -> (U,V)
// would perform the equivalent of an MPI_Reduce_scatter summation over
// the team of processes defining the 'V' row distribution.

// Adjoint
// =======
template<typename Ring>
void Adjoint( const Matrix<Ring>& A, Matrix<Ring>& B );
template<typename Ring>
void Adjoint( const ElementalMatrix<Ring>& A, ElementalMatrix<Ring>& B );
template<typename Ring>
void Adjoint( const BlockMatrix<Ring>& A, BlockMatrix<Ring>& B );
template<typename Ring>
void Adjoint( const AbstractDistMatrix<Ring>& A, AbstractDistMatrix<Ring>& B );

// AdjointContract
// ===============
template<typename Ring>
void AdjointContract
( const ElementalMatrix<Ring>& A,
        ElementalMatrix<Ring>& B );
template<typename Ring>
void AdjointContract
( const BlockMatrix<Ring>& A,
        BlockMatrix<Ring>& B );

// AdjointAxpy
// ===========
template<typename Ring1,typename Ring2>
void AdjointAxpy
( Ring2 alpha, const Matrix<Ring1>& X, Matrix<Ring1>& Y );
template<typename Ring1,typename Ring2>
void AdjointAxpy
( Ring2 alpha, const ElementalMatrix<Ring1>& X, ElementalMatrix<Ring1>& Y );
template<typename Ring1,typename Ring2>
void AdjointAxpy
( Ring2 alpha, const BlockMatrix<Ring1>& X, BlockMatrix<Ring1>& Y );

// AdjointAxpyContract
// ===================
template<typename Ring>
void AdjointAxpyContract
( Ring alpha, const ElementalMatrix<Ring>& A,
                    ElementalMatrix<Ring>& B );
template<typename Ring>
void AdjointAxpyContract
( Ring alpha, const BlockMatrix<Ring>& A,
                    BlockMatrix<Ring>& B );

// AllReduce
// =========
template<typename T>
void AllReduce( AbstractMatrix<T>& A, mpi::Comm const& comm, mpi::Op op=mpi::SUM );
template<typename T, Device D, typename=EnableIf<IsDeviceValidType<T,D>>>
void AllReduce( Matrix<T,D>& A, mpi::Comm const& comm, mpi::Op op=mpi::SUM );
template<typename T, Device D,
         typename=DisableIf<IsDeviceValidType<T,D>>,typename=void>
void AllReduce( Matrix<T,D>& A, mpi::Comm const& comm, mpi::Op op=mpi::SUM );
template<typename T>
void AllReduce( AbstractDistMatrix<T>& A, mpi::Comm const& comm, mpi::Op op=mpi::SUM );

// Axpy
// ====
template<typename Ring1,typename Ring2>
void Axpy(
    Ring2 alpha, const AbstractMatrix<Ring1>& X, AbstractMatrix<Ring1>& Y );
template<typename Ring1,typename Ring2>
void Axpy( Ring2 alpha, const Matrix<Ring1>& X, Matrix<Ring1>& Y );
template<typename Ring1,typename Ring2>
void Axpy
( Ring2 alpha, const ElementalMatrix<Ring1>& X, ElementalMatrix<Ring1>& Y );
template<typename Ring1,typename Ring2>
void Axpy( Ring2 alpha, const BlockMatrix<Ring1>& X, BlockMatrix<Ring1>& Y );
template<typename Ring1,typename Ring2>
void Axpy
( Ring2 alpha, const AbstractDistMatrix<Ring1>& X,
                     AbstractDistMatrix<Ring1>& Y );

namespace axpy {
namespace util {

template<typename Ring>
void InterleaveMatrixUpdate(
    Ring alpha, Int localHeight, Int localWidth,
    Ring const* A, Int colStrideA, Int rowStrideA,
    Ring* B, Int colStrideB, Int rowStrideB,
    SyncInfo<Device::CPU>);

#ifdef HYDROGEN_HAVE_GPU
template<typename Ring>
void InterleaveMatrixUpdate(
    Ring alpha, Int localHeight, Int localWidth,
    Ring const* A, Int colStrideA, Int rowStrideA,
    Ring* B, Int colStrideB, Int rowStrideB,
    SyncInfo<Device::GPU>);
#endif // HYDROGEN_HAVE_GPU

template<typename Ring, Device D=Device::CPU>
void UpdateWithLocalData(
    Ring alpha, ElementalMatrix<Ring> const& A,
    DistMatrix<Ring,STAR,STAR,ELEMENT,D>& B );

} // namespace util
} // namespace axpy

// AxpyContract
// ============

template<typename Ring>
void AxpyContract
( Ring alpha, const ElementalMatrix<Ring>& A, ElementalMatrix<Ring>& B );
template<typename Ring>
void AxpyContract
( Ring alpha, const BlockMatrix<Ring>& A, BlockMatrix<Ring>& B );

// AxpyTrapezoid
// =============
template<typename Ring1,typename Ring2>
void AxpyTrapezoid
( UpperOrLower uplo, Ring2 alpha,
  const Matrix<Ring1>& X, Matrix<Ring1>& Y, Int offset=0 );
template<typename Ring1,typename Ring2>
void AxpyTrapezoid
( UpperOrLower uplo, Ring2 alpha,
  const ElementalMatrix<Ring1>& X, ElementalMatrix<Ring1>& Y, Int offset=0 );
template<typename Ring1,typename Ring2>
void AxpyTrapezoid
( UpperOrLower uplo, Ring2 alpha,
  const BlockMatrix<Ring1>& X, BlockMatrix<Ring1>& Y, Int offset=0 );

template<typename Ring>
void LocalAxpyTrapezoid
( UpperOrLower uplo, Ring alpha,
  const AbstractDistMatrix<Ring>& X,
        AbstractDistMatrix<Ring>& Y, Int offset=0 );

// Broadcast
// =========
template<typename T, Device D>
void Broadcast( Matrix<T, D>& A, mpi::Comm const& comm, int rank=0 );
template<typename T>
void Broadcast( AbstractMatrix<T>& A, mpi::Comm const& comm, int rank=0 );
template<typename T>
void Broadcast( AbstractDistMatrix<T>& A, mpi::Comm const& comm, int rank=0 );

// Send
// ====
template <typename T>
void Send(AbstractMatrix<T> const& A, mpi::Comm const& comm, int destination);
template <typename T, Device D>
void Send(Matrix<T,D> const& A, mpi::Comm const& comm, int destination);

// SendRecv
// ========
template<typename T>
void SendRecv
( const AbstractMatrix<T>& A, AbstractMatrix<T>& B, mpi::Comm const& comm,
  int sendRank, int recvRank );
template<typename T, Device D>
void SendRecv
( const Matrix<T,D>& A, Matrix<T,D>& B, mpi::Comm const& comm,
  int sendRank, int recvRank );

// Recv
// ====
// On entry, the matrix 'A' must be of the correct size and will be overwritten
// on exit (without changing the leading dimension).
template <typename T>
void Recv(AbstractMatrix<T>& A, mpi::Comm const& comm, int source);
template<typename T, Device D>
void Recv(Matrix<T,D>& A, mpi::Comm const& comm, int source );

// Column norms
// ============

// One norms
// ---------
// TODO(poulson)

// Two norms
// ---------
template<typename Field>
void ColumnTwoNorms
( const Matrix<Field>& X, Matrix<Base<Field>>& norms );
template<typename Field,Dist U,Dist V,DistWrap W>
void ColumnTwoNorms
( const DistMatrix<Field,U,V,W>& X, DistMatrix<Base<Field>,V,STAR,W>& norms );

// Separated complex data
// ^^^^^^^^^^^^^^^^^^^^^^
template<typename Real,
         typename=EnableIf<IsReal<Real>>>
void ColumnTwoNorms
( const Matrix<Real>& XReal,
  const Matrix<Real>& XImag,
        Matrix<Real>& norms );
template<typename Real,Dist U,Dist V,
         typename=EnableIf<IsReal<Real>>>
void ColumnTwoNorms
( const DistMatrix<Real,U,V>& XReal,
  const DistMatrix<Real,U,V>& XImag,
        DistMatrix<Real,V,STAR>& norms );

// Max norms
// ---------
template<typename Ring>
void ColumnMaxNorms
( const Matrix<Ring>& X, Matrix<Base<Ring>>& norms );
template<typename Ring,Dist U,Dist V,DistWrap W>
void ColumnMaxNorms
( const DistMatrix<Ring,U,V,W>& X, DistMatrix<Base<Ring>,V,STAR,W>& norms );

// Column minimum absolute values
// ==============================
// NOTE: While this is not a norm, it is often colloquially referred to as the
// "-infinity" norm
template<typename Ring>
void ColumnMinAbs
( const Matrix<Ring>& X, Matrix<Base<Ring>>& mins );
template<typename Ring,Dist U,Dist V>
void ColumnMinAbs
( const DistMatrix<Ring,U,V>& X, DistMatrix<Base<Ring>,V,STAR>& mins );

template<typename Ring>
void ColumnMinAbsNonzero
( const Matrix<Ring>& X,
  const Matrix<Base<Ring>>& upperBounds,
        Matrix<Base<Ring>>& mins );
template<typename Ring,Dist U,Dist V>
void ColumnMinAbsNonzero
( const DistMatrix<Ring,U,V>& X,
  const DistMatrix<Base<Ring>,V,STAR>& upperBounds,
        DistMatrix<Base<Ring>,V,STAR>& mins );

// Row norms
// =========

// One norm
// --------
// TODO(poulson)

// Two-norm
// --------
template<typename Field>
void RowTwoNorms
( const Matrix<Field>& X, Matrix<Base<Field>>& norms );
template<typename Field,Dist U,Dist V>
void RowTwoNorms
( const DistMatrix<Field,U,V>& X, DistMatrix<Base<Field>,U,STAR>& norms );

// Max norm
// --------
template<typename Ring>
void RowMaxNorms
( const Matrix<Ring>& X, Matrix<Base<Ring>>& norms );
template<typename Ring,Dist U,Dist V>
void RowMaxNorms
( const DistMatrix<Ring,U,V>& X, DistMatrix<Base<Ring>,U,STAR>& norms );

// Row minimum absolute values
// ===========================
// NOTE: While this is not a norm, it is often colloquially referred to as the
// "-infinity" norm
template<typename Ring>
void RowMinAbs
( const Matrix<Ring>& X, Matrix<Base<Ring>>& mins );
template<typename Ring,Dist U,Dist V>
void RowMinAbs
( const DistMatrix<Ring,U,V>& X, DistMatrix<Base<Ring>,U,STAR>& mins );

template<typename Ring>
void RowMinAbsNonzero
( const Matrix<Ring>& X,
  const Matrix<Base<Ring>>& upperBounds,
        Matrix<Base<Ring>>& mins );
template<typename Ring,Dist U,Dist V>
void RowMinAbsNonzero
( const DistMatrix<Ring,U,V>& X,
  const DistMatrix<Base<Ring>,U,STAR>& upperBounds,
        DistMatrix<Base<Ring>,U,STAR>& mins );

// Concatenation
// =============

// Horizontal concatenation: C := [A, B]
// -------------------------------------
template<typename T>
void HCat
( const Matrix<T>& A,
  const Matrix<T>& B,
        Matrix<T>& C );
template<typename T>
void HCat
( const AbstractDistMatrix<T>& A,
  const AbstractDistMatrix<T>& B,
        AbstractDistMatrix<T>& C );

// Vertical concatenation: C := [A; B]
// -----------------------------------
template<typename T>
void VCat
( const Matrix<T>& A,
  const Matrix<T>& B,
        Matrix<T>& C );
template<typename T>
void VCat
( const AbstractDistMatrix<T>& A,
  const AbstractDistMatrix<T>& B,
        AbstractDistMatrix<T>& C );

// Conjugate
// =========
template<typename RealRing>
void Conjugate( Matrix<RealRing>& A );
template<typename Real>
void Conjugate( Matrix<Complex<Real>>& A );

template<typename T>
void Conjugate( const Matrix<T>& A, Matrix<T>& B );

template<typename T>
void Conjugate( AbstractDistMatrix<T>& A );
template<typename T>
void Conjugate( const AbstractDistMatrix<T>& A, AbstractDistMatrix<T>& B );

// ConjugateDiagonal
// =================
template<typename T>
void ConjugateDiagonal( Matrix<T>& A, Int offset=0 );
template<typename T>
void ConjugateDiagonal( AbstractDistMatrix<T>& A, Int offset=0 );

// ConjugateSubmatrix
// ==================
template<typename T>
void ConjugateSubmatrix
( Matrix<T>& A, const vector<Int>& I, const vector<Int>& J );
template<typename T>
void ConjugateSubmatrix
( AbstractDistMatrix<T>& A,
  const vector<Int>& I, const vector<Int>& J );

// Contract
// ========
template<typename T>
void Contract( const ElementalMatrix<T>& A, ElementalMatrix<T>& B );
template<typename T>
void Contract( const BlockMatrix<T>& A, BlockMatrix<T>& B );

// Copy
// ====
class BaseDistMatrix;

namespace copy {
namespace util {

template <typename T, typename=EnableIf<IsStorageType<T,Device::CPU>>>
void InterleaveMatrix(
    Int height, Int width,
    T const* A, Int colStrideA, Int rowStrideA,
    T* B, Int colStrideB, Int rowStrideB,
    SyncInfo<Device::CPU>);

template <typename T, typename=EnableIf<IsStorageType<T,Device::CPU>>>
void RowStridedPack(
    Int height, Int width,
    Int rowAlign, Int rowStride,
    const T* A, Int ALDim,
    T* BPortions, Int portionSize,
    SyncInfo<Device::CPU> );

template <typename T, typename=EnableIf<IsStorageType<T,Device::CPU>>>
void RowStridedUnpack(
 Int height, Int width,
  Int rowAlign, Int rowStride,
  T const* APortions, Int portionSize,
        T* B, Int BLDim,
  SyncInfo<Device::CPU> );

template <typename T, typename=EnableIf<IsStorageType<T,Device::CPU>>>
void PartialRowStridedPack(
    Int height, Int width,
    Int rowAlign, Int rowStride,
    Int rowStrideUnion, Int rowStridePart, Int rowRankPart,
    Int rowShiftA,
    T const* A, Int ALDim,
    T* BPortions, Int portionSize,
    SyncInfo<Device::CPU> );

template <typename T, typename=EnableIf<IsStorageType<T,Device::CPU>>>
void PartialRowStridedUnpack(
    Int height, Int width,
    Int rowAlign, Int rowStride,
    Int rowStrideUnion, Int rowStridePart, Int rowRankPart,
    Int rowShiftB,
    const T* APortions, Int portionSize,
    T* B, Int BLDim,
    SyncInfo<Device::CPU> );

#ifdef HYDROGEN_HAVE_GPU
template <typename T, typename=EnableIf<IsStorageType<T,Device::GPU>>>
void InterleaveMatrix(
    Int height, Int width,
    T const* A, Int colStrideA, Int rowStrideA,
    T* B, Int colStrideB, Int rowStrideB,
    SyncInfo<Device::GPU>);

template <typename T, typename=EnableIf<IsStorageType<T,Device::GPU>>>
void RowStridedPack(
    Int height, Int width,
    Int rowAlign, Int rowStride,
    const T* A, Int ALDim,
    T* BPortions, Int portionSize,
    SyncInfo<Device::GPU> );

template <typename T, typename=EnableIf<IsStorageType<T,Device::GPU>>>
void RowStridedUnpack(
 Int height, Int width,
  Int rowAlign, Int rowStride,
  T const* APortions, Int portionSize,
        T* B, Int BLDim,
  SyncInfo<Device::GPU> );

template <typename T, typename=EnableIf<IsStorageType<T,Device::GPU>>>
void PartialRowStridedPack(
    Int height, Int width,
    Int rowAlign, Int rowStride,
    Int rowStrideUnion, Int rowStridePart, Int rowRankPart,
    Int rowShiftA,
    T const* A, Int ALDim,
    T* BPortions, Int portionSize,
    SyncInfo<Device::GPU> );

template <typename T, typename=EnableIf<IsStorageType<T,Device::GPU>>>
void PartialRowStridedUnpack(
    Int height, Int width,
    Int rowAlign, Int rowStride,
    Int rowStrideUnion, Int rowStridePart, Int rowRankPart,
    Int rowShiftB,
    const T* APortions, Int portionSize,
    T* B, Int BLDim,
    SyncInfo<Device::GPU> );

#endif // HYDROGEN_HAVE_GPU

template <typename T, Device D,
          typename=DisableIf<IsStorageType<T,D>>, typename=void>
void InterleaveMatrix(
    Int const&, Int const&,
    T const*, Int const&, Int const&,
    T const*, Int const&, Int const&,
    SyncInfo<D> const&);

template <typename T, Device D,
          typename=DisableIf<IsStorageType<T,D>>, typename=void>
void RowStridedPack(
    Int const&, Int const&, Int const&, Int const&,
    T const*, Int const&, T const*, Int const&,
    SyncInfo<D> const&);

template <typename T, Device D,
          typename=DisableIf<IsStorageType<T,D>>, typename=void>
void RowStridedUnpack(
    Int const&, Int const&, Int const&, Int const&,
    T const*, Int const&, T const*, Int const&,
    SyncInfo<D> const&);

template <typename T, Device D,
          typename=DisableIf<IsStorageType<T,D>>, typename=void>
void PartialRowStridedPack(
    Int const&, Int const&, Int const&, Int const&,
    Int const&, Int const&, Int const&, Int const&,
    T const*, Int const&, T const*, Int const&,
    SyncInfo<D> const&);

template <typename T, Device D,
          typename=DisableIf<IsStorageType<T,D>>, typename=void>
void PartialRowStridedUnpack(
    Int const&, Int const&, Int const&, Int const&,
    Int const&, Int const&, Int const&, Int const&,
    T const*, Int const&, T const*, Int const&,
    SyncInfo<D> const&);

template <typename T, Device D,
          typename=DisableIf<IsStorageType<T,D>>,
          typename=void>
void PartialColStridedColumnPack(
    Int const&, Int const&, Int const&,
    Int const&, Int const&, Int const&, Int const&,
    T const*, T const*, Int const&,
    SyncInfo<D> const&);

template <typename T, Device D,
          typename=DisableIf<IsStorageType<T,D>>,
          typename=void>
void ColStridedPack(
    Int const&, Int const&, Int const&, Int const&,
    T const*, Int const&, T const*, Int const&, SyncInfo<D> const&);

template <typename T, Device D,
          typename=DisableIf<IsStorageType<T,D>>,
          typename=void>
void ColStridedUnpack(
    Int const&, Int const&, Int const&, Int const&,
    T const*, Int const&, T const*, Int const&, SyncInfo<D> const&);

template <typename T, Device D, typename=EnableIf<IsStorageType<T,D>>>
void ColStridedPack(
    Int height, Int width,
    Int colAlign, Int colStride,
    T const* A, Int ALDim,
    T* BPortions, Int portionSize,
    SyncInfo<D>);

template <typename T, Device D, typename=EnableIf<IsStorageType<T,D>>>
void ColStridedColumnPack(
    Int height,
    Int colAlign, Int colStride,
    T const* A,
    T* BPortions, Int portionSize,
    SyncInfo<D> );

template<typename T, Device D=Device::CPU,
         typename=EnableIf<IsStorageType<T,D>>>
void ColStridedUnpack(
    Int height, Int width,
    Int colAlign, Int colStride,
    T const* APortions, Int portionSize,
    T* B, Int BLDim,
    SyncInfo<D> );

template <typename T, Device D=Device::CPU,
         typename=EnableIf<IsStorageType<T,D>>>
void PartialColStridedPack(
    Int height, Int width,
    Int colAlign, Int colStride,
    Int colStrideUnion, Int colStridePart, Int colRankPart,
    Int colShiftA,
    T const* A, Int ALDim,
    T* BPortions, Int portionSize,
    SyncInfo<D> );

template <typename T, Device D=Device::CPU,
          typename=EnableIf<IsStorageType<T,D>>>
void PartialColStridedUnpack(
    Int height, Int width,
    Int colAlign, Int colStride,
    Int colStrideUnion, Int colStridePart, Int colRankPart,
    Int colShiftB,
    T const* APortions, Int portionSize,
    T* B, Int BLDim,
    SyncInfo<D> );

template <typename T, Device D,
          typename=EnableIf<IsStorageType<T,D>>>
void PartialColStridedColumnPack(
    Int height,
    Int colAlign, Int colStride,
    Int colStrideUnion, Int colStridePart, Int colRankPart,
    Int colShiftA,
    T const* A,
    T* BPortions, Int portionSize,
    SyncInfo<D>);

template <typename T, Device D,
          typename=EnableIf<IsStorageType<T,D>>>
void PartialColStridedColumnUnpack(
    Int height,
    Int colAlign, Int colStride,
    Int colStrideUnion, Int colStridePart, Int colRankPart,
    Int colShiftB,
    T const* APortions, Int portionSize,
    T* B, SyncInfo<D> );

template<typename T, Device D,
         typename=EnableIf<IsStorageType<T,D>>>
void StridedPack(
    Int height, Int width,
    Int colAlign, Int colStride,
    Int rowAlign, Int rowStride,
    T const* A, Int ALDim,
    T* BPortions, Int portionSize,
    SyncInfo<D> );

template<typename T, Device D,
         typename=EnableIf<IsStorageType<T,D>>>
void StridedUnpack(
    Int height, Int width,
    Int colAlign, Int colStride,
    Int rowAlign, Int rowStride,
    T const* APortions, Int portionSize,
    T* B, Int BLDim,
    SyncInfo<D> );


} // namespace util
} // namespace copy

} // namespace El

#include "Copy.hpp"

namespace El
{

// DiagonalScale
// =============
#ifdef HYDROGEN_HAVE_GPU
template<typename T, typename=EnableIf<IsDeviceValidType<T,Device::GPU>>>
void DiagonalScale
( LeftOrRight side, Orientation orientation,
  Matrix<T,Device::GPU> const& d, Matrix<T,Device::GPU>& A );
template<typename T, typename=DisableIf<IsDeviceValidType<T,Device::GPU>>,
         typename=void>
void DiagonalScale
( LeftOrRight side, Orientation orientation,
  Matrix<T,Device::GPU> const& d, Matrix<T,Device::GPU>& A );
#endif // HYDROGEN_HAVE_GPU

template<typename TDiag,typename T>
void DiagonalScale(
    LeftOrRight side, Orientation orientation,
    Matrix<TDiag, Device::CPU> const& d, Matrix<T, Device::CPU>& A );

template<typename T>
void DiagonalScale(
    LeftOrRight side, Orientation orientation,
    AbstractMatrix<T> const& d, AbstractMatrix<T>& A );

template<typename TDiag, typename T, Dist U, Dist V,
         DistWrap wrapType=ELEMENT,
         Device D=Device::CPU,
         typename=EnableIf<And<IsDeviceValidType<T,D>,IsDeviceValidType<TDiag,D>>>>
void DiagonalScale(
    LeftOrRight side, Orientation orientation,
    const AbstractDistMatrix<TDiag>& d, DistMatrix<T,U,V,wrapType,D>& A);

template<typename TDiag, typename T, Dist U, Dist V,
         DistWrap wrapType=ELEMENT,
         Device D=Device::CPU,
         typename=DisableIf<And<IsDeviceValidType<T,D>,IsDeviceValidType<TDiag,D>>>,
         typename=void>
void DiagonalScale(
    LeftOrRight side, Orientation orientation,
    const AbstractDistMatrix<TDiag>& d, DistMatrix<T,U,V,wrapType,D>& A);

template<typename TDiag,typename T>
void DiagonalScale
( LeftOrRight side, Orientation orientation,
  const AbstractDistMatrix<TDiag>& d, AbstractDistMatrix<T>& A );

// DiagonalScaleTrapezoid
// ======================
template<typename TDiag,typename T>
void DiagonalScaleTrapezoid
( LeftOrRight side, UpperOrLower uplo, Orientation orientation,
  const Matrix<TDiag>& d, Matrix<T>& A, Int offset=0 );

template<typename TDiag,typename T,Dist U,Dist V>
void DiagonalScaleTrapezoid
( LeftOrRight side, UpperOrLower uplo, Orientation orientation,
  const AbstractDistMatrix<TDiag>& d,
        DistMatrix<T,U,V>& A, Int offset=0 );
template<typename TDiag,typename T,Dist U,Dist V>
void DiagonalScaleTrapezoid
( LeftOrRight side, UpperOrLower uplo, Orientation orientation,
  const AbstractDistMatrix<TDiag>& d,
        DistMatrix<T,U,V,BLOCK>& A, Int offset=0 );

template<typename TDiag,typename T>
void DiagonalScaleTrapezoid
( LeftOrRight side, UpperOrLower uplo, Orientation orientation,
  const AbstractDistMatrix<TDiag>& d,
        AbstractDistMatrix<T>& A, Int offset=0 );

// DiagonalSolve
// =============
template<typename FieldDiag,typename Field>
void DiagonalSolve
( LeftOrRight side, Orientation orientation,
  const Matrix<FieldDiag>& d, Matrix<Field>& A, bool checkIfSingular=true );
template<typename Field>
void SymmetricDiagonalSolve( const Matrix<Base<Field>>& d, Matrix<Field>& A );

template<typename FieldDiag,typename Field,Dist U,Dist V>
void DiagonalSolve
( LeftOrRight side, Orientation orientation,
  const AbstractDistMatrix<FieldDiag>& d,
        DistMatrix<Field,U,V>& A,
  bool checkIfSingular=true );
template<typename FieldDiag,typename Field,Dist U,Dist V>
void DiagonalSolve
( LeftOrRight side, Orientation orientation,
  const AbstractDistMatrix<FieldDiag>& d,
        DistMatrix<Field,U,V,BLOCK>& A,
  bool checkIfSingular=true );

template<typename FieldDiag,typename Field>
void DiagonalSolve
( LeftOrRight side, Orientation orientation,
  const AbstractDistMatrix<FieldDiag>& d,
        AbstractDistMatrix<Field>& A,
  bool checkIfSingular=true );

// Dot
// ===
template<typename T, Device D>
T Dot( const Matrix<T, D>& A, const Matrix<T, D>& B );
template<typename T>
T Dot( const AbstractMatrix<T>& A, const AbstractMatrix<T>& B );
template<typename T>
T Dot( const AbstractDistMatrix<T>& A, const AbstractDistMatrix<T>& B );

// Dotu
// ====
template<typename T>
T Dotu( const Matrix<T>& A, const Matrix<T>& B );
template<typename T>
T Dotu( const ElementalMatrix<T>& A, const ElementalMatrix<T>& B );

// EntrywiseFill
// =============
template<typename T>
void EntrywiseFill( Matrix<T>& A, function<T(void)> func );
template<typename T>
void EntrywiseFill( AbstractDistMatrix<T>& A, function<T(void)> func );
#ifdef HYDROGEN_HAVE_GPU
template<typename T>
void EntrywiseFill( Matrix<T,Device::GPU>& A, function<T(void)> func );
#endif // HYDROGEN_HAVE_GPU

// EntrywiseMap
// ============
template<typename T>
void EntrywiseMap( AbstractMatrix<T>& A, function<T(const T&)> func );
template<typename T>
void EntrywiseMap( AbstractDistMatrix<T>& A, function<T(const T&)> func );

template<typename S,typename T>
void EntrywiseMap
( const AbstractMatrix<S>& A, AbstractMatrix<T>& B, function<T(const S&)> func );
template<typename S,typename T>
void EntrywiseMap
( const AbstractDistMatrix<S>& A, AbstractDistMatrix<T>& B,
  function<T(const S&)> func );

// Fill
// ====
template<typename T>
void Fill( AbstractMatrix<T>& A, T alpha );
template<typename T>
void Fill( AbstractDistMatrix<T>& A, T alpha );

// FillDiagonal
// ============
template<typename T>
void FillDiagonal( Matrix<T>& A, T alpha, Int offset=0 );
template<typename T>
void FillDiagonal( AbstractDistMatrix<T>& A, T alpha, Int offset=0 );

// GetDiagonal
// ===========
template<typename T>
void GetDiagonal
( const Matrix<T>& A, Matrix<T>& d, Int offset=0 );
template<typename T>
void GetRealPartOfDiagonal
( const Matrix<T>& A, Matrix<Base<T>>& d, Int offset=0 );
template<typename T>
void GetImagPartOfDiagonal
( const Matrix<T>& A, Matrix<Base<T>>& d, Int offset=0 );

template<typename T>
Matrix<T> GetDiagonal( const Matrix<T>& A, Int offset=0 );
template<typename T>
Matrix<Base<T>> GetRealPartOfDiagonal( const Matrix<T>& A, Int offset=0 );
template<typename T>
Matrix<Base<T>> GetImagPartOfDiagonal( const Matrix<T>& A, Int offset=0 );

template<typename T,Dist U,Dist V,DistWrap wrap>
void GetDiagonal
( const DistMatrix<T,U,V,wrap>& A,
  AbstractDistMatrix<T>& d, Int offset=0 );
template<typename T,Dist U,Dist V,DistWrap wrap>
void GetRealPartOfDiagonal
( const DistMatrix<T,U,V,wrap>& A,
  AbstractDistMatrix<Base<T>>& d, Int offset=0 );
template<typename T,Dist U,Dist V,DistWrap wrap>
void GetImagPartOfDiagonal
( const DistMatrix<T,U,V,wrap>& A,
  AbstractDistMatrix<Base<T>>& d, Int offset=0 );
template<typename T>
void GetDiagonal
( const AbstractDistMatrix<T>& A,
  AbstractDistMatrix<T>& d, Int offset=0 );
template<typename T>
void GetRealPartOfDiagonal
( const AbstractDistMatrix<T>& A,
  AbstractDistMatrix<Base<T>>& d, Int offset=0 );
template<typename T>
void GetImagPartOfDiagonal
( const AbstractDistMatrix<T>& A,
  AbstractDistMatrix<Base<T>>& d, Int offset=0 );

template<typename T,Dist U,Dist V>
DistMatrix<T,DiagCol<U,V>(),DiagRow<U,V>()>
GetDiagonal( const DistMatrix<T,U,V>& A, Int offset=0 );
template<typename T,Dist U,Dist V>
DistMatrix<T,MC,STAR>
GetDiagonal( const DistMatrix<T,U,V,BLOCK>& A, Int offset=0 );

template<typename T,Dist U,Dist V>
DistMatrix<Base<T>,DiagCol<U,V>(),DiagRow<U,V>()>
GetRealPartOfDiagonal( const DistMatrix<T,U,V>& A, Int offset=0 );
template<typename T,Dist U,Dist V>
DistMatrix<Base<T>,MC,STAR>
GetRealPartOfDiagonal( const DistMatrix<T,U,V,BLOCK>& A, Int offset=0 );

template<typename T,Dist U,Dist V>
DistMatrix<Base<T>,DiagCol<U,V>(),DiagRow<U,V>()>
GetImagPartOfDiagonal( const DistMatrix<T,U,V>& A, Int offset=0 );
template<typename T,Dist U,Dist V>
DistMatrix<Base<T>,MC,STAR>
GetImagPartOfDiagonal( const DistMatrix<T,U,V,BLOCK>& A, Int offset=0 );

// GetMappedDiagonal
// =================
template<typename T,typename S>
void GetMappedDiagonal
( const Matrix<T>& A, Matrix<S>& d, function<S(const T&)> func, Int offset=0 );
template<typename T,typename S,Dist U,Dist V>
void GetMappedDiagonal
( const DistMatrix<T,U,V>& A, AbstractDistMatrix<S>& d,
  function<S(const T&)> func, Int offset=0 );
template<typename T,typename S,Dist U,Dist V>
void GetMappedDiagonal
( const DistMatrix<T,U,V,BLOCK>& A, AbstractDistMatrix<S>& d,
  function<S(const T&)> func, Int offset=0 );


// GetSubmatrix
// ============

// Return a view
// ----------
template<typename T>
void GetSubmatrix
( const Matrix<T>& A,
        Range<Int> I,
        Range<Int> J,
        Matrix<T>& ASub );
template<typename T>
void GetSubmatrix
( const ElementalMatrix<T>& A,
        Range<Int> I,
        Range<Int> J,
        ElementalMatrix<T>& ASub );

// Noncontiguous
// -------------
template<typename T>
void GetSubmatrix
( const Matrix<T>& A,
        Range<Int> I,
  const vector<Int>& J,
        Matrix<T>& ASub );
template<typename T>
void GetSubmatrix
( const Matrix<T>& A,
  const vector<Int>& I,
  const Range<Int> J,
        Matrix<T>& ASub );
template<typename T>
void GetSubmatrix
( const Matrix<T>& A,
  const vector<Int>& I,
  const vector<Int>& J,
        Matrix<T>& ASub );

template<typename T>
void GetSubmatrix
( const AbstractDistMatrix<T>& A,
        Range<Int> I,
  const vector<Int>& J,
        AbstractDistMatrix<T>& ASub );
template<typename T>
void GetSubmatrix
( const AbstractDistMatrix<T>& A,
  const vector<Int>& I,
        Range<Int> J,
        AbstractDistMatrix<T>& ASub );
template<typename T>
void GetSubmatrix
( const AbstractDistMatrix<T>& A,
  const vector<Int>& I,
  const vector<Int>& J,
        AbstractDistMatrix<T>& ASub );

// Hadamard
// ========
template<typename T>
void Hadamard( const AbstractMatrix<T>& A, const AbstractMatrix<T>& B,
               AbstractMatrix<T>& C );
template<typename T>
void Hadamard
( const AbstractDistMatrix<T>& A,
  const AbstractDistMatrix<T>& B,
        AbstractDistMatrix<T>& C );

// HilbertSchmidt
// ==============
template<typename T>
T HilbertSchmidt( const AbstractMatrix<T>& A, const AbstractMatrix<T>& B );
template<typename T>
T HilbertSchmidt
( const AbstractDistMatrix<T>& A, const AbstractDistMatrix<T>& C );

// Imaginary part
// ==============
template<typename T>
void ImagPart
( const Matrix<T>& A, Matrix<Base<T>>& AImag );
template<typename T>
void ImagPart
( const AbstractDistMatrix<T>& A, AbstractDistMatrix<Base<T>>& AImag );
/* TODO(poulson): Sparse versions */

// IndexDependentFill
// ==================
template<typename T>
void IndexDependentFill( Matrix<T>& A, function<T(Int,Int)> func );
template<typename T>
void IndexDependentFill
( AbstractDistMatrix<T>& A, function<T(Int,Int)> func );

// IndexDependentMap
// =================
template<typename T>
void IndexDependentMap( Matrix<T>& A, function<T(Int,Int,const T&)> func );
template<typename T>
void IndexDependentMap
( AbstractDistMatrix<T>& A, function<T(Int,Int,const T&)> func );

template<typename S,typename T>
void IndexDependentMap
( const Matrix<S>& A,
        Matrix<T>& B,
        function<T(Int,Int,const S&)> func );
template<typename S,typename T,Dist U,Dist V,DistWrap wrap>
void IndexDependentMap
( const DistMatrix<S,U,V,wrap>& A,
        DistMatrix<T,U,V,wrap>& B,
        function<T(Int,Int,const S&)> func );
template<typename S,typename T,Dist U,Dist V>
void IndexDependentMap
( const AbstractDistMatrix<S>& A,
        DistMatrix<T,U,V>& B,
        function<T(Int,Int,const S&)> func );
template<typename S,typename T,Dist U,Dist V>
void IndexDependentMap
( const AbstractDistMatrix<S>& A,
        DistMatrix<T,U,V,BLOCK>& B,
        function<T(Int,Int,const S&)> func );

// Kronecker product
// =================
template<typename T>
void Kronecker( const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C );
template<typename T>
void Kronecker
( const Matrix<T>& A, const Matrix<T>& B, ElementalMatrix<T>& C );

// MakeHermitian
// =============
template<typename T>
void MakeHermitian( UpperOrLower uplo, Matrix<T>& A );
template<typename T>
void MakeHermitian( UpperOrLower uplo, ElementalMatrix<T>& A );

// MakeDiagonalReal
// ================
template<typename T>
void MakeDiagonalReal( Matrix<T>& A, Int offset=0 );
template<typename T>
void MakeDiagonalReal( AbstractDistMatrix<T>& A, Int offset=0 );

// MakeReal
// ========
template<typename Real>
void MakeReal( Matrix<Real>& A );
template<typename Real>
void MakeReal( Matrix<Complex<Real>>& A );
template<typename T>
void MakeReal( AbstractDistMatrix<T>& A );

// MakeSubmatrixReal
// ================
template<typename T>
void MakeSubmatrixReal
( Matrix<T>& A, const vector<Int>& I, const vector<Int>& J );
template<typename T>
void MakeSubmatrixReal
( AbstractDistMatrix<T>& A, const vector<Int>& I, const vector<Int>& J );

// MakeSymmetric
// =============
template<typename T>
void MakeSymmetric( UpperOrLower uplo, Matrix<T>& A, bool conjugate=false );
template<typename T>
void MakeSymmetric
( UpperOrLower uplo, ElementalMatrix<T>& A, bool conjugate=false );

// MakeTrapezoidal
// ===============
template<typename T>
void MakeTrapezoidal( UpperOrLower uplo, Matrix<T>& A, Int offset=0 );
template<typename T>
void MakeTrapezoidal
( UpperOrLower uplo, AbstractDistMatrix<T>& A, Int offset=0 );

// Max
// ===
template<typename Real,
         typename=EnableIf<IsReal<Real>>>
Real Max( const Matrix<Real>& A );
template<typename Real,
         typename=EnableIf<IsReal<Real>>>
Real Max( const AbstractDistMatrix<Real>& A );

template<typename Real,
         typename=EnableIf<IsReal<Real>>>
Real SymmetricMax( UpperOrLower uplo, const Matrix<Real>& A );
template<typename Real,
         typename=EnableIf<IsReal<Real>>>
Real SymmetricMax( UpperOrLower uplo, const AbstractDistMatrix<Real>& A );

// MaxLoc
// ======
template<typename Real,
         typename=EnableIf<IsReal<Real>>>
Entry<Real> MaxLoc( const Matrix<Real>& A );
template<typename Real,
         typename=EnableIf<IsReal<Real>>>
Entry<Real> MaxLoc( const AbstractDistMatrix<Real>& A );

template<typename Real,
         typename=EnableIf<IsReal<Real>>>
Entry<Real> SymmetricMaxLoc( UpperOrLower uplo, const Matrix<Real>& A );
template<typename Real,
         typename=EnableIf<IsReal<Real>>>
Entry<Real>
SymmetricMaxLoc( UpperOrLower uplo, const AbstractDistMatrix<Real>& A );

template<typename Real,
         typename=EnableIf<IsReal<Real>>>
ValueInt<Real> VectorMaxLoc( const Matrix<Real>& x );
template<typename Real,
         typename=EnableIf<IsReal<Real>>>
ValueInt<Real> VectorMaxLoc( const AbstractDistMatrix<Real>& x );

// MaxAbs
// ======
template<typename T>
Base<T> MaxAbs( const Matrix<T>& A );
template<typename T>
Base<T> MaxAbs( const AbstractDistMatrix<T>& A );

template<typename T>
Base<T> SymmetricMaxAbs( UpperOrLower uplo, const Matrix<T>& A );
template<typename T>
Base<T> SymmetricMaxAbs( UpperOrLower uplo, const AbstractDistMatrix<T>& A );

// MaxAbsLoc
// =========
template<typename T>
ValueInt<Base<T>> VectorMaxAbsLoc( const Matrix<T>& x );
template<typename T>
ValueInt<Base<T>> VectorMaxAbsLoc( const AbstractDistMatrix<T>& x );

template<typename T>
Entry<Base<T>> MaxAbsLoc( const Matrix<T>& A );
template<typename T>
Entry<Base<T>> MaxAbsLoc( const AbstractDistMatrix<T>& A );

template<typename T>
Entry<Base<T>>
SymmetricMaxAbsLoc( UpperOrLower uplo, const Matrix<T>& A );
template<typename T>
Entry<Base<T>>
SymmetricMaxAbsLoc( UpperOrLower uplo, const AbstractDistMatrix<T>& A );

// Min
// ===
template<typename Real,
         typename=EnableIf<IsReal<Real>>>
Real Min( const Matrix<Real>& A );
template<typename Real,
         typename=EnableIf<IsReal<Real>>>
Real Min( const AbstractDistMatrix<Real>& A );

template<typename Real,
         typename=EnableIf<IsReal<Real>>>
Real SymmetricMin( UpperOrLower uplo, const Matrix<Real>& A );
template<typename Real,
         typename=EnableIf<IsReal<Real>>>
Real SymmetricMin( UpperOrLower uplo, const AbstractDistMatrix<Real>& A );

// MinLoc
// ======
template<typename Real,
         typename=EnableIf<IsReal<Real>>>
ValueInt<Real> VectorMinLoc( const Matrix<Real>& x );
template<typename Real,
         typename=EnableIf<IsReal<Real>>>
ValueInt<Real> VectorMinLoc( const AbstractDistMatrix<Real>& x );

template<typename Real,
         typename=EnableIf<IsReal<Real>>>
Entry<Real> MinLoc( const Matrix<Real>& A );
template<typename Real,
         typename=EnableIf<IsReal<Real>>>
Entry<Real> MinLoc( const AbstractDistMatrix<Real>& A );

template<typename Real,
         typename=EnableIf<IsReal<Real>>>
Entry<Real> SymmetricMinLoc( UpperOrLower uplo, const Matrix<Real>& A );
template<typename Real,
         typename=EnableIf<IsReal<Real>>>
Entry<Real>
SymmetricMinLoc( UpperOrLower uplo, const AbstractDistMatrix<Real>& A );

// MinAbs
// ======
template<typename Ring>
Base<Ring> MinAbs( const Matrix<Ring>& A );
template<typename Ring>
Base<Ring> MinAbs( const AbstractDistMatrix<Ring>& A );

template<typename Ring>
Base<Ring> SymmetricMinAbs
( UpperOrLower uplo, const Matrix<Ring>& A );
template<typename Ring>
Base<Ring> SymmetricMinAbs
( UpperOrLower uplo, const AbstractDistMatrix<Ring>& A );

// MinAbsLoc
// =========
template<typename Ring>
ValueInt<Base<Ring>> VectorMinAbsLoc( const Matrix<Ring>& x );
template<typename Ring>
ValueInt<Base<Ring>> VectorMinAbsLoc( const AbstractDistMatrix<Ring>& x );

template<typename Ring>
Entry<Base<Ring>> MinAbsLoc( const Matrix<Ring>& A );
template<typename Ring>
Entry<Base<Ring>> MinAbsLoc( const AbstractDistMatrix<Ring>& A );

template<typename Ring>
Entry<Base<Ring>>
SymmetricMinAbsLoc( UpperOrLower uplo, const Matrix<Ring>& A );
template<typename Ring>
Entry<Base<Ring>>
SymmetricMinAbsLoc( UpperOrLower uplo, const AbstractDistMatrix<Ring>& A );

// Nrm2
// ====
template<typename Field>
Base<Field> Nrm2( const Matrix<Field>& x );
template<typename Field>
Base<Field> Nrm2( const AbstractDistMatrix<Field>& x );

// QuasiDiagonalScale
// ==================
template<typename Field,typename FieldMain>
void QuasiDiagonalScale
( LeftOrRight side, UpperOrLower uplo,
  const Matrix<FieldMain>& d,
  const Matrix<Field>& dSub,
        Matrix<Field>& X,
        bool conjugated=false );
template<typename Field,typename FieldMain,Dist U,Dist V>
void QuasiDiagonalScale
( LeftOrRight side, UpperOrLower uplo,
  const AbstractDistMatrix<FieldMain>& d,
  const AbstractDistMatrix<Field>& dSub,
        DistMatrix<Field,U,V>& X,
        bool conjugated=false );
template<typename Field,typename FieldMain,Dist U,Dist V>
void QuasiDiagonalScale
( LeftOrRight side, UpperOrLower uplo,
  const AbstractDistMatrix<FieldMain>& d,
  const AbstractDistMatrix<Field>& dSub,
        DistMatrix<Field,U,V,BLOCK>& X,
        bool conjugated=false );

template<typename Field,typename FieldMain,Dist U,Dist V>
void LeftQuasiDiagonalScale
( UpperOrLower uplo,
  const DistMatrix<FieldMain,U,STAR>& d,
  const DistMatrix<FieldMain,U,STAR>& dPrev,
  const DistMatrix<FieldMain,U,STAR>& dNext,
  const DistMatrix<Field,    U,STAR>& dSub,
  const DistMatrix<Field,    U,STAR>& dSubPrev,
  const DistMatrix<Field,    U,STAR>& dSubNext,
        DistMatrix<Field,U,V>& X,
  const DistMatrix<Field,U,V>& XPrev,
  const DistMatrix<Field,U,V>& XNext,
  bool conjugated=false );

template<typename Field,typename FieldMain,Dist U,Dist V>
void RightQuasiDiagonalScale
( UpperOrLower uplo,
  const DistMatrix<FieldMain,V,STAR>& d,
  const DistMatrix<FieldMain,V,STAR>& dPrev,
  const DistMatrix<FieldMain,V,STAR>& dNext,
  const DistMatrix<Field,    V,STAR>& dSub,
  const DistMatrix<Field,    V,STAR>& dSubPrev,
  const DistMatrix<Field,    V,STAR>& dSubNext,
        DistMatrix<Field,U,V>& X,
  const DistMatrix<Field,U,V>& XPrev,
  const DistMatrix<Field,U,V>& XNext,
  bool conjugated=false );

// QuasiDiagonalSolve
// ==================
template<typename Field,typename FieldMain>
void
QuasiDiagonalSolve
( LeftOrRight side, UpperOrLower uplo,
  const Matrix<FieldMain>& d, const Matrix<Field>& dSub,
  Matrix<Field>& X, bool conjugated=false );
template<typename Field,typename FieldMain,Dist U,Dist V>
void
QuasiDiagonalSolve
( LeftOrRight side, UpperOrLower uplo,
  const AbstractDistMatrix<FieldMain>& d,
  const AbstractDistMatrix<Field>& dSub,
        DistMatrix<Field,U,V>& X,
        bool conjugated=false );
template<typename Field,typename FieldMain,Dist U,Dist V>
void
QuasiDiagonalSolve
( LeftOrRight side, UpperOrLower uplo,
  const AbstractDistMatrix<FieldMain>& d,
  const AbstractDistMatrix<Field>& dSub,
        DistMatrix<Field,U,V,BLOCK>& X,
        bool conjugated=false );

template<typename Field,typename FieldMain,Dist U,Dist V>
void
LeftQuasiDiagonalSolve
( UpperOrLower uplo,
  const DistMatrix<FieldMain,U,STAR>& d,
  const DistMatrix<FieldMain,U,STAR>& dPrev,
  const DistMatrix<FieldMain,U,STAR>& dNext,
  const DistMatrix<Field,    U,STAR>& dSub,
  const DistMatrix<Field,    U,STAR>& dSubPrev,
  const DistMatrix<Field,    U,STAR>& dSubNext,
        DistMatrix<Field,U,V>& X,
  const DistMatrix<Field,U,V>& XPrev,
  const DistMatrix<Field,U,V>& XNext,
  bool conjugated=false );

template<typename Field,typename FieldMain,Dist U,Dist V>
void
RightQuasiDiagonalSolve
( UpperOrLower uplo,
  const DistMatrix<FieldMain,V,STAR>& d,
  const DistMatrix<FieldMain,V,STAR>& dPrev,
  const DistMatrix<FieldMain,V,STAR>& dNext,
  const DistMatrix<Field,    V,STAR>& dSub,
  const DistMatrix<Field,    V,STAR>& dSubPrev,
  const DistMatrix<Field,    V,STAR>& dSubNext,
        DistMatrix<Field,U,V>& X,
  const DistMatrix<Field,U,V>& XPrev,
  const DistMatrix<Field,U,V>& XNext,
  bool conjugated=false );

// Real part
// =========
template<typename T>
void RealPart( const Matrix<T>& A, Matrix<Base<T>>& AReal );
template<typename T>
void RealPart
( const AbstractDistMatrix<T>& A, AbstractDistMatrix<Base<T>>& AReal );
/* TODO(poulson): Sparse versions */

// Reshape
// =======
template<typename T>
void Reshape( Int m, Int n, const Matrix<T>& A, Matrix<T>& B );
template<typename T>
Matrix<T> Reshape( Int m, Int n, const Matrix<T>& A );

template<typename T>
void Reshape
( Int m, Int n, const AbstractDistMatrix<T>& A, AbstractDistMatrix<T>& B );
template<typename T>
DistMatrix<T> Reshape
( Int m, Int n, const AbstractDistMatrix<T>& A );

// Transform2x2
// ============

// Note that, if a1 and a2 are column vectors, the following overwrites
//
//   [a1, a2] := [a1, a2] G^T,
//
// and *not*
//
//   [a1, a2] := [a1, a2] G.
//
// In the case where a1 and a2 are row vectors, we are performing
//
//   [a1; a2] := G [a1; a2].
//
// It is recommended to use Transform2x2Rows or Transform2x2Cols instead.
//
template<typename T>
void Transform2x2
( const Matrix<T>& G,
        Matrix<T>& a1,
        Matrix<T>& a2 );
template<typename T>
void Transform2x2
( const Matrix<T>& G,
        AbstractDistMatrix<T>& a1,
        AbstractDistMatrix<T>& a2 );
template<typename T>
void Transform2x2
( const AbstractDistMatrix<T>& G,
        AbstractDistMatrix<T>& a1,
        AbstractDistMatrix<T>& a2 );

// A([i1,i2],:) := G A([i1,i2],:), where G is 2x2
// ----------------------------------------------
template<typename T>
void Transform2x2Rows
( const Matrix<T>& G,
        Matrix<T>& A, Int i1, Int i2 );
template<typename T>
void Transform2x2Rows
( const Matrix<T>& G,
        AbstractDistMatrix<T>& A, Int i1, Int i2 );
template<typename T>
void Transform2x2Rows
( const AbstractDistMatrix<T>& G,
        AbstractDistMatrix<T>& A, Int i1, Int i2 );

// A(:,[j1,j2]) := A(:,[j1,j2]) G, where G is 2x2
// ----------------------------------------------
template<typename T>
void Transform2x2Cols
( const Matrix<T>& G,
        Matrix<T>& A, Int j1, Int j2 );
template<typename T>
void Transform2x2Cols
( const Matrix<T>& G,
        AbstractDistMatrix<T>& A, Int j1, Int j2 );
template<typename T>
void Transform2x2Cols
( const AbstractDistMatrix<T>& G,
        AbstractDistMatrix<T>& A, Int j1, Int j2 );

// TODO(poulson): SymmetricTransform2x2?

// Rotate (via Givens)
// ===================
// NOTE: BLAS calls this 'rot'

// [a1; a2] := [c s; -conj(s) c] [a1; a2]
// --------------------------------------
template<typename Field>
void Rotate
( Base<Field> c, Field s,
  Matrix<Field>& a1, Matrix<Field>& a2 );
template<typename Field>
void Rotate
( Base<Field> c, Field s,
  AbstractDistMatrix<Field>& a1, AbstractDistMatrix<Field>& a2 );

// A([i1,i2],:) := [c s; -conj(s) c] A([i1,i2],:)
// ----------------------------------------------
template<typename Field>
void RotateRows
( Base<Field> c, Field s,
  Matrix<Field>& A, Int i1, Int i2 );
template<typename Field>
void RotateRows
( Base<Field> c, Field s,
  AbstractDistMatrix<Field>& A, Int i1, Int i2 );

// A(:,[j1,j2]) := A(:,[j1,j2]) [c s; -conj(s), c]
// -----------------------------------------------
template<typename Field>
void RotateCols
( Base<Field> c, Field s,
  Matrix<Field>& A, Int j1, Int j2 );
template<typename Field>
void RotateCols
( Base<Field> c, Field s,
  AbstractDistMatrix<Field>& A, Int j1, Int j2 );

// TODO(poulson): SymmetricRotation?

// Round
// =====
// Round each entry to the nearest integer
template <typename T>
void Round(AbstractMatrix<T>& A);

template<typename T>
void Round( Matrix<T>& A );
template<>
void Round( Matrix<Int>& A );
#ifdef HYDROGEN_HAVE_MPC
template<>
void Round( Matrix<BigInt>& A );
#endif
template<typename T>
void Round( AbstractDistMatrix<T>& A );

// Scale
// =====
// TODO(poulson): Force S=T?
template<typename T,typename S>
void Scale( S alpha, AbstractMatrix<T>& A );
template<typename T,typename S>
void Scale( S alpha, AbstractDistMatrix<T>& A );

template<typename Real,typename S,
         typename=EnableIf<IsReal<Real>>>
void Scale( S alpha, AbstractMatrix<Real>& AReal, AbstractMatrix<Real>& AImag );
template<typename Real,typename S,
         typename=EnableIf<IsReal<Real>>>
void Scale
( S alpha, AbstractDistMatrix<Real>& AReal, AbstractDistMatrix<Real>& AImag );

template<typename Field>
void SafeScale
( Base<Field> numerator, Base<Field> denominator,
  Matrix<Field>& A );
template<typename Field>
void SafeScale
( Base<Field> numerator, Base<Field> denominator,
  AbstractDistMatrix<Field>& A );

template<typename Field>
void SafeScaleTrapezoid
( Base<Field> numerator, Base<Field> denominator,
  UpperOrLower uplo, Matrix<Field>& A, Int offset=0 );
template<typename Field>
void SafeScaleTrapezoid
( Base<Field> numerator, Base<Field> denominator,
  UpperOrLower uplo, AbstractDistMatrix<Field>& A, Int offset=0 );

template<typename Field>
void SafeScaleHermitianTridiag
( Base<Field> numerator, Base<Field> denominator,
  Matrix<Base<Field>>& d, Matrix<Field>& e );

// ScaleTrapezoid
// ==============
template<typename T,typename S>
void ScaleTrapezoid( S alpha, UpperOrLower uplo, Matrix<T>& A, Int offset=0 );
template<typename T,typename S>
void ScaleTrapezoid
( S alpha, UpperOrLower uplo, AbstractDistMatrix<T>& A, Int offset=0 );

// SetDiagonal
// ===========
template<typename T>
void SetDiagonal
( Matrix<T>& A, const Matrix<T>& d, Int offset=0 );
template<typename T>
void SetRealPartOfDiagonal
( Matrix<T>& A, const Matrix<Base<T>>& d, Int offset=0 );
template<typename T>
void SetImagPartOfDiagonal
( Matrix<T>& A, const Matrix<Base<T>>& d, Int offset=0 );

template<typename T,Dist U,Dist V,DistWrap wrap>
void SetDiagonal
(       DistMatrix<T,U,V,wrap>& A,
  const AbstractDistMatrix<T>& d,
  Int offset=0 );
template<typename T,Dist U,Dist V,DistWrap wrap>
void SetRealPartOfDiagonal
(       DistMatrix<T,U,V,wrap>& A,
  const AbstractDistMatrix<Base<T>>& d,
  Int offset=0 );
template<typename T,Dist U,Dist V,DistWrap wrap>
void SetImagPartOfDiagonal
(       DistMatrix<T,U,V,wrap>& A,
  const AbstractDistMatrix<Base<T>>& d, Int offset=0 );
// Versions which will work for AbstractDistMatrix but which make use of a
// manual dynamic dispatch
template<typename T>
void SetDiagonal
(       AbstractDistMatrix<T>& A,
  const AbstractDistMatrix<T>& d, Int offset=0 );
template<typename T>
void SetRealPartOfDiagonal
(       AbstractDistMatrix<T>& A,
  const AbstractDistMatrix<Base<T>>& d, Int offset=0 );
template<typename T>
void SetImagPartOfDiagonal
(       AbstractDistMatrix<T>& A,
  const AbstractDistMatrix<Base<T>>& d, Int offset=0 );

// SetSubmatrix
// ============
template<typename T>
void SetSubmatrix
(       Matrix<T>& A,
  const vector<Int>& I, const vector<Int>& J,
  const Matrix<T>& ASub );
template<typename T>
void SetSubmatrix
(       AbstractDistMatrix<T>& A,
  const vector<Int>& I, const vector<Int>& J,
  const AbstractDistMatrix<T>& ASub );

// Swap
// ====
template<typename T>
void Swap( Orientation orientation, Matrix<T>& X, Matrix<T>& Y );
template<typename T>
void Swap
( Orientation orientation, AbstractDistMatrix<T>& X, AbstractDistMatrix<T>& Y );

template<typename T>
void RowSwap( Matrix<T>& A, Int to, Int from );
template<typename T>
void RowSwap( AbstractDistMatrix<T>& A, Int to, Int from );

template<typename T>
void ColSwap( Matrix<T>& A, Int to, Int from );
template<typename T>
void ColSwap( AbstractDistMatrix<T>& A, Int to, Int from );

template<typename T>
void SymmetricSwap
( UpperOrLower uplo, Matrix<T>& A, Int to, Int from, bool conjugate=false );
template<typename T>
void SymmetricSwap
( UpperOrLower uplo, AbstractDistMatrix<T>& A, Int to, Int from,
  bool conjugate=false );

template<typename T>
void HermitianSwap( UpperOrLower uplo, Matrix<T>& A, Int to, Int from );
template<typename T>
void HermitianSwap
( UpperOrLower uplo, AbstractDistMatrix<T>& A, Int to, Int from );

// Symmetric2x2Inv
// ===============
template<typename Field>
void Symmetric2x2Inv
( UpperOrLower uplo, Matrix<Field>& D, bool conjugate=false );

// Shift
// =====
template<typename T,typename S>
void Shift( Matrix<T>& A, S alpha );
template<typename T,typename S>
void Shift( AbstractDistMatrix<T>& A, S alpha );

// ShiftDiagonal
// =============
template<typename T,typename S>
void ShiftDiagonal( Matrix<T>& A, S alpha, Int offset=0 );
template<typename T,typename S>
void ShiftDiagonal( AbstractDistMatrix<T>& A, S alpha, Int offset=0 );

// Transpose
// =========
template<typename T>
void Transpose
( const Matrix<T>& A,
        Matrix<T>& B,
  bool conjugate=false );
#ifdef HYDROGEN_HAVE_GPU
template<typename T,typename=EnableIf<IsDeviceValidType<T,Device::GPU>>>
void Transpose
( Matrix<T,Device::GPU> const& A,
  Matrix<T,Device::GPU>& B,
  bool conjugate=false );
template<typename T,
         typename=DisableIf<IsDeviceValidType<T,Device::GPU>>,
         typename=void>
void Transpose
( Matrix<T,Device::GPU> const& A,
  Matrix<T,Device::GPU>& B,
  bool conjugate=false );
#endif // HYDROGEN_HAVE_GPU

template<typename T>
void Transpose
( const ElementalMatrix<T>& A,
        ElementalMatrix<T>& B,
  bool conjugate=false );
template<typename T>
void Transpose
( const BlockMatrix<T>& A,
        BlockMatrix<T>& B,
  bool conjugate=false );
template<typename T>
void Transpose
( const AbstractDistMatrix<T>& A,
        AbstractDistMatrix<T>& B,
  bool conjugate=false );

// TransposeContract
// =================
template<typename T>
void TransposeContract
( const ElementalMatrix<T>& A,
        ElementalMatrix<T>& B, bool conjugate=false );
template<typename T>
void TransposeContract
( const BlockMatrix<T>& A,
        BlockMatrix<T>& B, bool conjugate=false );

// TransposeAxpy
// =============
template<typename T,typename S>
void TransposeAxpy
( S alpha, const Matrix<T>& X, Matrix<T>& Y, bool conjugate=false );
template<typename T,typename S>
void TransposeAxpy
( S alpha, const ElementalMatrix<T>& X, ElementalMatrix<T>& Y,
  bool conjugate=false );

// TransposeAxpyContract
// =====================
template<typename T>
void TransposeAxpyContract
( T alpha, const ElementalMatrix<T>& A,
                 ElementalMatrix<T>& B, bool conjugate=false );
template<typename T>
void TransposeAxpyContract
( T alpha, const BlockMatrix<T>& A,
                 BlockMatrix<T>& B, bool conjugate=false );

// UpdateDiagonal
// ==============
template<typename T>
void UpdateDiagonal
( Matrix<T>& A, T alpha, const Matrix<T>& d, Int offset=0 );
template<typename T>
void UpdateRealPartOfDiagonal
( Matrix<T>& A, Base<T> alpha, const Matrix<Base<T>>& d, Int offset=0 );
template<typename T>
void UpdateImagPartOfDiagonal
( Matrix<T>& A, Base<T> alpha, const Matrix<Base<T>>& d, Int offset=0 );

template<typename T,Dist U,Dist V>
void UpdateDiagonal
( DistMatrix<T,U,V>& A, T alpha, const ElementalMatrix<T>& d,
  Int offset=0 );
template<typename T,Dist U,Dist V>
void UpdateRealPartOfDiagonal
( DistMatrix<T,U,V>& A, Base<T> alpha, const ElementalMatrix<Base<T>>& d,
  Int offset=0 );
template<typename T,Dist U,Dist V>
void UpdateImagPartOfDiagonal
( DistMatrix<T,U,V>& A, Base<T> alpha, const ElementalMatrix<Base<T>>& d,
  Int offset=0 );

// UpdateMappedDiagonal
// ====================
template<typename T,typename S>
void UpdateMappedDiagonal
( Matrix<T>& A, const Matrix<S>& d,
  function<void(T&,const S&)> func, Int offset=0 );
template<typename T,typename S,Dist U,Dist V>
void UpdateMappedDiagonal
( DistMatrix<T,U,V>& A, const AbstractDistMatrix<S>& d,
  function<void(T&,const S&)> func, Int offset=0 );
template<typename T,typename S,Dist U,Dist V>
void UpdateMappedDiagonal
( DistMatrix<T,U,V,BLOCK>& A, const AbstractDistMatrix<S>& d,
  function<void(T&,const S&)> func, Int offset=0 );

// UpdateSubmatrix
// ===============
template<typename T>
void UpdateSubmatrix
(       Matrix<T>& A,
  const vector<Int>& I, const vector<Int>& J,
  T alpha, const Matrix<T>& ASub );
template<typename T>
void UpdateSubmatrix
(       AbstractDistMatrix<T>& A,
  const vector<Int>& I, const vector<Int>& J,
  T alpha, const AbstractDistMatrix<T>& ASub );

// Zero
// ====
template<typename T>
void Zero_seq( AbstractMatrix<T>& A );
template<typename T>
void Zero_seq( AbstractDistMatrix<T>& A );
template<typename T>
void Zero( AbstractMatrix<T>& A );
template<typename T>
void Zero( AbstractDistMatrix<T>& A );

// Givens rotations
// ================
//
// Given phi and gamma, compute a Givens rotation such that
//
//  |       c   s | |   phi |  = | rho |, where c^2 + |s|^2 = 1
//  | -conj(s)  c | | gamma |    |  0  |
//
// This routine uses the stable approach suggested by Kahan and Demmel and
// returns the value rho.
//
template<typename Real>
Real Givens
( const Real& phi,
  const Real& gamma,
        Real& c,
        Real& s );
template<typename Real>
Complex<Real> Givens
( const Complex<Real>& phi,
  const Complex<Real>& gamma,
  Real& c,
  Complex<Real>& s );

} // namespace El

#endif // ifndef EL_BLAS1_DECL_HPP
