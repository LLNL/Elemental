/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS1_COPYINTERNAL_DECL_HPP
#define EL_BLAS1_COPYINTERNAL_DECL_HPP

namespace El {

namespace copy {

template<typename S,typename T,typename=EnableIf<CanCast<S,T>>>
void GeneralPurpose
( const AbstractDistMatrix<S>& A,
        AbstractDistMatrix<T>& B );
template<typename T,typename=EnableIf<IsBlasScalar<T>>>
void GeneralPurpose
( const AbstractDistMatrix<T>& A,
        AbstractDistMatrix<T>& B );

template<typename T>
void Exchange
( const ElementalMatrix<T>& A,
        ElementalMatrix<T>& B,
  int sendRank, int recvRank, mpi::Comm const& comm );

template<typename T,Dist U,Dist V,Device D1, Device D2>
void Translate( DistMatrix<T,U,V,ELEMENT,D1> const& A,
                DistMatrix<T,U,V,ELEMENT,D2>& B );
template<typename T,Dist U,Dist V>
void Translate
( const DistMatrix<T,U,V,BLOCK>& A, DistMatrix<T,U,V,BLOCK>& B );

template<typename T,Device D1,Device D2>
void TranslateBetweenGrids
( const DistMatrix<T,MC,MR,ELEMENT,D1>& A, DistMatrix<T,MC,MR,ELEMENT,D2>& B );
template<typename T,Device D1,Device D2>
void TranslateBetweenGrids
( DistMatrix<T,STAR,STAR,ELEMENT,D1> const& A,
  DistMatrix<T,STAR,STAR,ELEMENT,D2>& B );
// The fallback case that simply throws an exception
template<typename T,Dist U,Dist V,Device D1,Device D2>
void TranslateBetweenGrids
( const DistMatrix<T,U,V,ELEMENT,D1>& A,
  DistMatrix<T,U,V,ELEMENT,D2>& B );

// NOTE: Only instantiated for (U,V)=(MC,MR) and (U,V)=(MR,MC)
template<typename T,Dist U,Dist V,Device D>
void ColwiseVectorExchange
( DistMatrix<T,ProductDist<U,V>(),STAR,ELEMENT,D> const& A,
  DistMatrix<T,ProductDist<V,U>(),STAR,ELEMENT,D>& B );
template<typename T,Dist U,Dist V,Device D>
void RowwiseVectorExchange
( DistMatrix<T,STAR,ProductDist<U,V>(),ELEMENT,D> const& A,
  DistMatrix<T,STAR,ProductDist<V,U>(),ELEMENT,D>& B );

// NOTE: Only instantiated for (U,V)=(MC,MR) and (U,V)=(MR,MC)
template<typename T,Dist U,Dist V,
         typename=EnableIf<IsStorageType<T,Device::CPU>>>
void TransposeDist( DistMatrix<T,U,V,ELEMENT,Device::CPU> const& A,
                    DistMatrix<T,V,U,ELEMENT,Device::CPU>& B );

#ifdef HYDROGEN_HAVE_GPU
template<typename T,Dist U,Dist V,
         typename=EnableIf<IsStorageType<T,Device::GPU>>>
void TransposeDist( DistMatrix<T,U,V,ELEMENT,Device::GPU> const& A,
                    DistMatrix<T,V,U,ELEMENT,Device::GPU>& B );
#endif // HYDROGEN_HAVE_GPU

template<typename T,Dist U,Dist V,Device D,
         typename=DisableIf<IsStorageType<T,D>>,typename=void>
void TransposeDist( DistMatrix<T,U,V,ELEMENT,D> const& A,
                    DistMatrix<T,V,U,ELEMENT,D>& B );

template<typename T,Dist U,Dist V,Device D,
         typename=EnableIf<IsStorageType<T,D>>>
void Filter
( DistMatrix<T,Collect<U>(),Collect<V>(),ELEMENT,D> const& A,
  DistMatrix<T,U,V,ELEMENT,D>& B );
template<typename T,Dist U,Dist V,Device D,
         typename=DisableIf<IsStorageType<T,D>>,typename=void>
void Filter
( DistMatrix<T,Collect<U>(),Collect<V>(),ELEMENT,D> const& A,
  DistMatrix<T,U,V,ELEMENT,D>& B );
template<typename T,Dist U,Dist V>
void Filter
( const DistMatrix<T,Collect<U>(),Collect<V>(),BLOCK>& A,
        DistMatrix<T,        U,           V   ,BLOCK>& B );

// (V,Collect(U)) |-> (U,V)
template<typename T>
void ColFilter
( const ElementalMatrix<T>& A,
        ElementalMatrix<T>& B );
template<typename T>
void ColFilter
( const BlockMatrix<T>& A,
        BlockMatrix<T>& B );

// (U,Collect(V)) |-> (U,V)
template<typename T>
void RowFilter
( const ElementalMatrix<T>& A,
        ElementalMatrix<T>& B );
template<typename T>
void RowFilter
( const BlockMatrix<T>& A,
        BlockMatrix<T>& B );

// (Partial(U),V) |-> (U,V)
template<typename T>
void PartialColFilter
( const ElementalMatrix<T>& A, ElementalMatrix<T>& B );
template<typename T>
void PartialColFilter
( const BlockMatrix<T>& A, BlockMatrix<T>& B );

// (U,Partial(V)) |-> (U,V)
template<typename T>
void PartialRowFilter
( const ElementalMatrix<T>& A, ElementalMatrix<T>& B );
template<typename T>
void PartialRowFilter
( const BlockMatrix<T>& A, BlockMatrix<T>& B );

template<typename T,Dist U,Dist V,Device D>
void AllGather
( DistMatrix<T,        U,           V   ,ELEMENT,D> const& A,
  DistMatrix<T,Collect<U>(),Collect<V>(),ELEMENT,D>& B );
template<typename T,Dist U,Dist V>
void AllGather
( const DistMatrix<T,        U,           V   ,BLOCK>& A,
        DistMatrix<T,Collect<U>(),Collect<V>(),BLOCK>& B );

// (U,V) |-> (Collect(U),V)
template<typename T>
void ColAllGather
( const ElementalMatrix<T>& A, ElementalMatrix<T>& B );
template<typename T>
void ColAllGather
( const BlockMatrix<T>& A, BlockMatrix<T>& B );

// (U,V) |-> (U,Collect(V))
template<typename T>
void RowAllGather
( const ElementalMatrix<T>& A, ElementalMatrix<T>& B );
template<typename T>
void RowAllGather
( const BlockMatrix<T>& A, BlockMatrix<T>& B );

template<typename T,Dist U,Dist V,Device D>
void PartialColAllGather
( DistMatrix<T,        U,   V,ELEMENT,D> const& A,
  DistMatrix<T,Partial<U>(),V,ELEMENT,D>& B );
template<typename T,Dist U,Dist V>
void PartialColAllGather
( const DistMatrix<T,        U,   V,BLOCK>& A,
        DistMatrix<T,Partial<U>(),V,BLOCK>& B );

// (U,V) |-> (U,Partial(V))
template<typename T>
void PartialRowAllGather
( const ElementalMatrix<T>& A, ElementalMatrix<T>& B );
template<typename T>
void PartialRowAllGather
( const BlockMatrix<T>& A, BlockMatrix<T>& B );

template<typename T,Dist U,Dist V,Device D>
void ColAllToAllDemote
( DistMatrix<T,Partial<U>(),PartialUnionRow<U,V>(),ELEMENT,D> const& A,
  DistMatrix<T,        U,                     V   ,ELEMENT,D>& B );
template<typename T,Dist U,Dist V>
void ColAllToAllDemote
( const DistMatrix<T,Partial<U>(),PartialUnionRow<U,V>(),BLOCK>& A,
        DistMatrix<T,        U,                     V   ,BLOCK>& B );

template<typename T,Dist U,Dist V,Device D>
void RowAllToAllDemote
( DistMatrix<T,PartialUnionCol<U,V>(),Partial<V>(),ELEMENT,D> const& A,
  DistMatrix<T,U,V,ELEMENT,D>& B );
template<typename T,Dist U,Dist V>
void RowAllToAllDemote
( const DistMatrix<T,PartialUnionCol<U,V>(),Partial<V>(),BLOCK>& A,
        DistMatrix<T,                U,             V   ,BLOCK>& B );

template<typename T,Dist U,Dist V,Device D>
void ColAllToAllPromote
( DistMatrix<T,U,V,ELEMENT,D> const& A,
  DistMatrix<T,Partial<U>(),PartialUnionRow<U,V>(),ELEMENT,D>& B );
template<typename T,Dist U,Dist V>
void ColAllToAllPromote
( const DistMatrix<T,        U,                     V   ,BLOCK>& A,
        DistMatrix<T,Partial<U>(),PartialUnionRow<U,V>(),BLOCK>& B );

template<typename T,Dist U,Dist V,Device D>
void RowAllToAllPromote
( DistMatrix<T,U,V,ELEMENT,D> const& A,
  DistMatrix<T,PartialUnionCol<U,V>(),Partial<V>(),ELEMENT,D>& B );
template<typename T,Dist U,Dist V>
void RowAllToAllPromote
( const DistMatrix<T,                U,             V   ,BLOCK>& A,
        DistMatrix<T,PartialUnionCol<U,V>(),Partial<V>(),BLOCK>& B );

template<typename T, Device D>
void Gather
( ElementalMatrix<T> const& A,
  DistMatrix<T,CIRC,CIRC,ELEMENT,D>& B );
template<typename T>
void Gather
( const BlockMatrix<T>& A,
        DistMatrix<T,CIRC,CIRC,BLOCK>& B );

template<typename T,Device D>
void Scatter
( const DistMatrix<T,CIRC,CIRC,ELEMENT,D>& A,
        ElementalMatrix<T>& B );
template<typename T>
void Scatter
( const DistMatrix<T,CIRC,CIRC,BLOCK>& A,
        BlockMatrix<T>& B );

template<typename T,Device D>
void Scatter
( const DistMatrix<T,CIRC,CIRC,ELEMENT,D>& A,
        DistMatrix<T,STAR,STAR,ELEMENT,D>& B );
template<typename T>
void Scatter
( const DistMatrix<T,CIRC,CIRC,BLOCK>& A,
        DistMatrix<T,STAR,STAR,BLOCK>& B );

} // namespace copy

} // namespace El

#endif // ifndef EL_BLAS1_COPYINTERNAL_DECL_HPP
