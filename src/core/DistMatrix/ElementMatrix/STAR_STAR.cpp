/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like.hpp>

#define COLDIST STAR
#define ROWDIST STAR

#include "./setup.hpp"

namespace El
{

// Public section
// ##############

// Assignment and reconfiguration
// ==============================

// Make a copy
// -----------
template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,MC,MR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::AllGather(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,MC,STAR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::ColAllGather(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,STAR,MR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::RowAllGather(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,MD,STAR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::ColAllGather(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,STAR,MD,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::RowAllGather(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,MR,MC,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::AllGather(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,MR,STAR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::ColAllGather(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,STAR,MC,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::RowAllGather(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,VC,STAR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::ColAllGather(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,STAR,VC,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::RowAllGather(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,VR,STAR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::ColAllGather(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,STAR,VR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::RowAllGather(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,CIRC,CIRC,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::Scatter(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const ElementalMatrix<T>& A)
{
    EL_DEBUG_CSE
    #define GUARD(CDIST,RDIST,WRAP,DEVICE) \
      A.DistData().colDist == CDIST && A.DistData().rowDist == RDIST && \
      ELEMENT == WRAP && A.GetLocalDevice() == DEVICE
    #define PAYLOAD(CDIST,RDIST,WRAP,DEVICE) \
      auto& ACast = static_cast<const DistMatrix<T,CDIST,RDIST,ELEMENT,DEVICE>&>(A); \
      *this = ACast;
    #include "El/macros/DeviceGuardAndPayload.h"
    return *this;
}

// Basic queries
// =============
template <typename T, Device D>
mpi::Comm const& DM::DistComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }
template <typename T, Device D>
mpi::Comm const& DM::CrossComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }
template <typename T, Device D>
mpi::Comm const& DM::RedundantComm() const EL_NO_EXCEPT
{ return this->Grid().VCComm(); }

template <typename T, Device D>
mpi::Comm const& DM::ColComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }
template <typename T, Device D>
mpi::Comm const& DM::RowComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }

template <typename T, Device D>
mpi::Comm const& DM::PartialColComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }
template <typename T, Device D>
mpi::Comm const& DM::PartialRowComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }

template <typename T, Device D>
mpi::Comm const& DM::PartialUnionColComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }
template <typename T, Device D>
mpi::Comm const& DM::PartialUnionRowComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }

template <typename T, Device D>
int DM::DistSize() const EL_NO_EXCEPT { return 1; }
template <typename T, Device D>
int DM::CrossSize() const EL_NO_EXCEPT { return 1; }
template <typename T, Device D>
int DM::RedundantSize() const EL_NO_EXCEPT { return this->Grid().VCSize(); }

template <typename T, Device D>
int DM::ColStride() const EL_NO_EXCEPT { return 1; }
template <typename T, Device D>
int DM::RowStride() const EL_NO_EXCEPT { return 1; }
template <typename T, Device D>
int DM::PartialColStride() const EL_NO_EXCEPT { return 1; }
template <typename T, Device D>
int DM::PartialRowStride() const EL_NO_EXCEPT { return 1; }
template <typename T, Device D>
int DM::PartialUnionColStride() const EL_NO_EXCEPT { return 1; }
template <typename T, Device D>
int DM::PartialUnionRowStride() const EL_NO_EXCEPT { return 1; }

template <typename T, Device D>
int DM::DistRank() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? 0 : mpi::UNDEFINED); }
template <typename T, Device D>
int DM::CrossRank() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? 0 : mpi::UNDEFINED); }
template <typename T, Device D>
int DM::RedundantRank() const EL_NO_EXCEPT { return this->Grid().VCRank(); }

template <typename T, Device D>
int DM::ColRank() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? 0 : mpi::UNDEFINED); }
template <typename T, Device D>
int DM::RowRank() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? 0 : mpi::UNDEFINED); }
template <typename T, Device D>
int DM::PartialColRank() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? 0 : mpi::UNDEFINED); }
template <typename T, Device D>
int DM::PartialRowRank() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? 0 : mpi::UNDEFINED); }
template <typename T, Device D>
int DM::PartialUnionColRank() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? 0 : mpi::UNDEFINED); }
template <typename T, Device D>
int DM::PartialUnionRowRank() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? 0 : mpi::UNDEFINED); }

// Instantiate {Int,Real,Complex<Real>} for each Real in {float,double}
// ####################################################################

#define SELF(T,U,V) \
  template DistMatrix<T,COLDIST,ROWDIST>::DistMatrix \
  (const DistMatrix<T,U,V>& A);
#define OTHER(T,U,V) \
  template DistMatrix<T,COLDIST,ROWDIST>::DistMatrix \
  (const DistMatrix<T,U,V,BLOCK>& A); \
  template DistMatrix<T,COLDIST,ROWDIST>& \
           DistMatrix<T,COLDIST,ROWDIST>::operator= \
           (const DistMatrix<T,U,V,BLOCK>& A)
#define BOTH(T,U,V) \
  SELF(T,U,V) \
  OTHER(T,U,V)
#define PROTO(T) \
  template class DistMatrix<T,COLDIST,ROWDIST>; \
  BOTH(T,CIRC,CIRC); \
  BOTH(T,MC,  MR ); \
  BOTH(T,MC,  STAR); \
  BOTH(T,MD,  STAR); \
  BOTH(T,MR,  MC ); \
  BOTH(T,MR,  STAR); \
  BOTH(T,STAR,MC ); \
  BOTH(T,STAR,MD ); \
  BOTH(T,STAR,MR ); \
  OTHER(T,STAR,STAR); \
  BOTH(T,STAR,VC ); \
  BOTH(T,STAR,VR ); \
  BOTH(T,VC,  STAR); \
  BOTH(T,VR,  STAR);

#ifdef HYDROGEN_HAVE_GPU
#include "gpu_instantiate.h"

#define FULL_GPU_PROTO(T)                       \
  INST_DISTMATRIX_CLASS(T);                     \
  INST_COPY_AND_ASSIGN(T, CIRC, CIRC);          \
  INST_COPY_AND_ASSIGN(T, MC,     MR);          \
  INST_COPY_AND_ASSIGN(T, MC,   STAR);          \
  INST_COPY_AND_ASSIGN(T, MD,   STAR);          \
  INST_COPY_AND_ASSIGN(T, MR,   MC  );          \
  INST_COPY_AND_ASSIGN(T, MR,   STAR);          \
  INST_COPY_AND_ASSIGN(T, STAR, MC  );          \
  INST_COPY_AND_ASSIGN(T, STAR, MD  );          \
  INST_COPY_AND_ASSIGN(T, STAR, MR  );          \
  INST_COPY_AND_ASSIGN(T, STAR, VC  );          \
  INST_COPY_AND_ASSIGN(T, STAR, VR  );          \
  INST_COPY_AND_ASSIGN(T, VC,   STAR);          \
  INST_COPY_AND_ASSIGN(T, VR,   STAR)

#ifdef HYDROGEN_GPU_USE_FP16
PROTO(gpu_half_type)
FULL_GPU_PROTO(gpu_half_type);
#endif // HYDROGEN_GPU_USE_FP16

FULL_GPU_PROTO(float);
FULL_GPU_PROTO(double);
FULL_GPU_PROTO(El::Complex<float>);
FULL_GPU_PROTO(El::Complex<double>);

#undef FULL_GPU_PROTO
#undef INST_DISTMATRIX_CLASS
#undef INST_COPY_AND_ASSIGN
#undef INST_ASSIGN_OP
#undef INST_COPY_CTOR

#endif // HYDROGEN_HAVE_GPU

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#define EL_ENABLE_HALF
#include <El/macros/Instantiate.h>

} // namespace El
