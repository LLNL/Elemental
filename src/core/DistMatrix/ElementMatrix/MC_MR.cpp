/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like.hpp>

#define COLDIST MC
#define ROWDIST MR

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
DM& DM::operator=(const DistMatrix<T,MC,STAR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::RowFilter(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,STAR,MR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::ColFilter(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,MD,STAR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    // TODO: More efficient implementation
    copy::GeneralPurpose(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,STAR,MD,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    // TODO: More efficient implementation
    copy::GeneralPurpose(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,MR,MC,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    const Grid& grid = A.Grid();
    if (grid.Height() == grid.Width())
    {
        const int gridDim = grid.Height();
        const int sendRank = this->RowOwner(A.ColShift()) +
                             this->ColOwner(A.RowShift())*gridDim;
        const int recvRank = A.ColOwner(this->RowShift()) +
                             A.RowOwner(this->ColShift())*gridDim;
        copy::Exchange(A, *this, sendRank, recvRank, grid.VCComm());
    }
    else
    {
        copy::TransposeDist(A, *this);
    }
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,MR,STAR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    DistMatrix<T,VR,STAR,ELEMENT,D> A_VR_STAR(A);
    DistMatrix<T,VC,STAR,ELEMENT,D> A_VC_STAR(this->Grid());
    A_VC_STAR.AlignColsWith(*this);
    A_VC_STAR = A_VR_STAR;
    A_VR_STAR.Empty();
    *this = A_VC_STAR;
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,STAR,MC,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    DistMatrix<T,STAR,VC,ELEMENT,D> A_STAR_VC(A);
    DistMatrix<T,STAR,VR,ELEMENT,D> A_STAR_VR(this->Grid());
    A_STAR_VR.AlignRowsWith(*this);
    A_STAR_VR = A_STAR_VC;
    A_STAR_VC.Empty();
    *this = A_STAR_VR;
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,VC,STAR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::ColAllToAllPromote(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,STAR,VC,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    DistMatrix<T,STAR,VR,ELEMENT,D> A_STAR_VR(this->Grid());
    A_STAR_VR.AlignRowsWith(*this);
    A_STAR_VR = A;
    *this = A_STAR_VR;
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,VR,STAR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    DistMatrix<T,VC,STAR,ELEMENT,D> A_VC_STAR(this->Grid());
    A_VC_STAR.AlignColsWith(*this);
    A_VC_STAR = A;
    *this = A_VC_STAR;
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,STAR,VR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::RowAllToAllPromote(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,STAR,STAR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::Filter(A, *this);
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
    EL_DEBUG_CSE;
#define GUARD(CDIST,RDIST,WRAP,DEVICE)                                  \
    (A.DistData().colDist == CDIST) && (A.DistData().rowDist == RDIST) && \
        (ELEMENT == WRAP) && (A.GetLocalDevice() == DEVICE)
#define PAYLOAD(CDIST,RDIST,WRAP,DEVICE)                                \
    auto& ACast = static_cast<const DistMatrix<T,CDIST,RDIST,ELEMENT,DEVICE>&>(A); \
    *this = ACast;
#include "El/macros/DeviceGuardAndPayload.h"
    return *this;
}

// Basic queries
// =============
template <typename T, Device D>
mpi::Comm const& DM::DistComm() const EL_NO_EXCEPT { return this->Grid().VCComm(); }

template <typename T, Device D>
mpi::Comm const& DM::CrossComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }

template <typename T, Device D>
mpi::Comm const& DM::RedundantComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }

template <typename T, Device D>
mpi::Comm const& DM::ColComm() const EL_NO_EXCEPT { return this->Grid().MCComm(); }
template <typename T, Device D>
mpi::Comm const& DM::RowComm() const EL_NO_EXCEPT { return this->Grid().MRComm(); }

template <typename T, Device D>
mpi::Comm const& DM::PartialColComm() const EL_NO_EXCEPT { return this->ColComm(); }
template <typename T, Device D>
mpi::Comm const& DM::PartialRowComm() const EL_NO_EXCEPT { return this->RowComm(); }
template <typename T, Device D>
mpi::Comm const& DM::PartialUnionColComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }
template <typename T, Device D>
mpi::Comm const& DM::PartialUnionRowComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }

template <typename T, Device D>
int DM::ColStride() const EL_NO_EXCEPT { return this->Grid().MCSize(); }
template <typename T, Device D>
int DM::RowStride() const EL_NO_EXCEPT { return this->Grid().MRSize(); }
template <typename T, Device D>
int DM::DistSize() const EL_NO_EXCEPT { return this->Grid().VCSize(); }
template <typename T, Device D>
int DM::CrossSize() const EL_NO_EXCEPT { return 1; }
template <typename T, Device D>
int DM::RedundantSize() const EL_NO_EXCEPT { return 1; }
template <typename T, Device D>
int DM::PartialColStride() const EL_NO_EXCEPT { return this->ColStride(); }
template <typename T, Device D>
int DM::PartialRowStride() const EL_NO_EXCEPT { return this->RowStride(); }
template <typename T, Device D>
int DM::PartialUnionColStride() const EL_NO_EXCEPT { return 1; }
template <typename T, Device D>
int DM::PartialUnionRowStride() const EL_NO_EXCEPT { return 1; }

template <typename T, Device D>
int DM::ColRank() const EL_NO_EXCEPT { return this->Grid().MCRank(); }
template <typename T, Device D>
int DM::RowRank() const EL_NO_EXCEPT { return this->Grid().MRRank(); }
template <typename T, Device D>
int DM::DistRank() const EL_NO_EXCEPT { return this->Grid().VCRank(); }
template <typename T, Device D>
int DM::CrossRank() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? 0 : mpi::UNDEFINED); }
template <typename T, Device D>
int DM::RedundantRank() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? 0 : mpi::UNDEFINED); }
template <typename T, Device D>
int DM::PartialColRank() const EL_NO_EXCEPT { return this->ColRank(); }
template <typename T, Device D>
int DM::PartialRowRank() const EL_NO_EXCEPT { return this->RowRank(); }
template <typename T, Device D>
int DM::PartialUnionColRank() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? 0 : mpi::UNDEFINED); }
template <typename T, Device D>
int DM::PartialUnionRowRank() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? 0 : mpi::UNDEFINED); }

// Instantiate {Int,Real,Complex<Real>} for each Real in {float,double,half}
// ####################################################################

#define SELF(T,U,V,D)                                                    \
  template DistMatrix<T,COLDIST,ROWDIST,ELEMENT,D>::DistMatrix \
  (const DistMatrix<T,U,V,ELEMENT,D>& A)
#define OTHER(T,U,V,D)                                          \
  template DistMatrix<T,COLDIST,ROWDIST,ELEMENT,D>::DistMatrix \
  (const DistMatrix<T,U,V,BLOCK,D>& A);                         \
  template DistMatrix<T,COLDIST,ROWDIST,ELEMENT,D>&             \
      DistMatrix<T,COLDIST,ROWDIST,ELEMENT,D>::operator=    \
  (const DistMatrix<T,U,V,BLOCK,D>& A)
#define BOTH(T,U,V,D)                            \
    SELF(T,U,V,D);                               \
    OTHER(T,U,V,D)
#define PROTO(T)                                                        \
    template class DistMatrix<T,COLDIST,ROWDIST,ELEMENT,Device::CPU>;   \
    BOTH(T,CIRC,CIRC,Device::CPU);                                      \
    OTHER(T,MC,  MR ,Device::CPU);                                      \
    BOTH(T,MC,  STAR,Device::CPU);                                      \
    BOTH(T,MD,  STAR,Device::CPU);                                      \
    BOTH(T,MR,  MC  ,Device::CPU);                                      \
    BOTH(T,MR,  STAR,Device::CPU);                                      \
    BOTH(T,STAR,MC  ,Device::CPU);                                      \
    BOTH(T,STAR,MD  ,Device::CPU);                                      \
    BOTH(T,STAR,MR  ,Device::CPU);                                      \
    BOTH(T,STAR,STAR,Device::CPU);                                      \
    BOTH(T,STAR,VC  ,Device::CPU);                                      \
    BOTH(T,STAR,VR  ,Device::CPU);                                      \
    BOTH(T,VC,  STAR,Device::CPU);                                      \
    BOTH(T,VR,  STAR,Device::CPU);

#ifdef HYDROGEN_HAVE_GPU
#include "gpu_instantiate.h"

#define FULL_GPU_PROTO(T)                       \
  INST_DISTMATRIX_CLASS(T);                     \
  INST_COPY_AND_ASSIGN(T, CIRC, CIRC);          \
  INST_COPY_AND_ASSIGN(T, MC,   STAR);          \
  INST_COPY_AND_ASSIGN(T, MD,   STAR);          \
  INST_COPY_AND_ASSIGN(T, MR,   MC  );          \
  INST_COPY_AND_ASSIGN(T, MR,   STAR);          \
  INST_COPY_AND_ASSIGN(T, STAR, MC  );          \
  INST_COPY_AND_ASSIGN(T, STAR, MD  );          \
  INST_COPY_AND_ASSIGN(T, STAR, MR  );          \
  INST_COPY_AND_ASSIGN(T, STAR, STAR);          \
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
