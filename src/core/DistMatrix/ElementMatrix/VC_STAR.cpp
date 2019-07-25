/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like.hpp>

#define COLDIST VC
#define ROWDIST STAR

#include "./setup.hpp"

namespace El {

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
    copy::ColAllToAllDemote(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,MC,STAR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::PartialColFilter(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,STAR,MR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    DistMatrix<T,MC,MR,ELEMENT,D> A_MC_MR(A);
    *this = A_MC_MR;
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
    DistMatrix<T,VR,STAR,ELEMENT,D> A_VR_STAR(A);
    *this = A_VR_STAR;
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,MR,STAR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    DistMatrix<T,VR,STAR,ELEMENT,D> A_VR_STAR(A);
    *this = A_VR_STAR;
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,STAR,MC,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    DistMatrix<T,MR,MC,ELEMENT,D> A_MR_MC(A);
    DistMatrix<T,VR,STAR,ELEMENT,D> A_VR_STAR(A_MR_MC);
    A_MR_MC.Empty();
    *this = A_VR_STAR;
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,STAR,VC,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    DistMatrix<T,MR,MC,ELEMENT,D> A_MR_MC(A);
    DistMatrix<T,VR,STAR,ELEMENT,D> A_VR_STAR(A_MR_MC);
    A_MR_MC.Empty();
    *this = A_VR_STAR;
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,VR,STAR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::ColwiseVectorExchange<T,MR,MC>(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,STAR,VR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    DistMatrix<T,MC,MR,ELEMENT,D> A_MC_MR(A);
    *this = A_MC_MR;
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,STAR,STAR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::ColFilter(A, *this);
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
{ return this->Grid().VCComm(); }
template <typename T, Device D>
mpi::Comm const& DM::CrossComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }
template <typename T, Device D>
mpi::Comm const& DM::RedundantComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }

template <typename T, Device D>
mpi::Comm const& DM::ColComm() const EL_NO_EXCEPT
{ return this->Grid().VCComm(); }
template <typename T, Device D>
mpi::Comm const& DM::RowComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }

template <typename T, Device D>
mpi::Comm const& DM::PartialColComm() const EL_NO_EXCEPT
{ return this->Grid().MCComm(); }
template <typename T, Device D>
mpi::Comm const& DM::PartialUnionColComm() const EL_NO_EXCEPT
{ return this->Grid().MRComm(); }

template <typename T, Device D>
mpi::Comm const& DM::PartialRowComm() const EL_NO_EXCEPT
{ return this->RowComm(); }
template <typename T, Device D>
mpi::Comm const& DM::PartialUnionRowComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }

template <typename T, Device D>
int DM::DistSize() const EL_NO_EXCEPT
{ return this->Grid().VCSize(); }
template <typename T, Device D>
int DM::CrossSize() const EL_NO_EXCEPT
{ return 1; }
template <typename T, Device D>
int DM::RedundantSize() const EL_NO_EXCEPT
{ return 1; }

template <typename T, Device D>
int DM::ColStride() const EL_NO_EXCEPT
{ return this->Grid().VCSize(); }
template <typename T, Device D>
int DM::RowStride() const EL_NO_EXCEPT
{ return 1; }

template <typename T, Device D>
int DM::PartialColStride() const EL_NO_EXCEPT
{ return this->Grid().MCSize(); }
template <typename T, Device D>
int DM::PartialUnionColStride() const EL_NO_EXCEPT
{ return this->Grid().MRSize(); }

template <typename T, Device D>
int DM::PartialRowStride() const EL_NO_EXCEPT
{ return this->RowStride(); }
template <typename T, Device D>
int DM::PartialUnionRowStride() const EL_NO_EXCEPT
{ return 1; }

template <typename T, Device D>
int DM::DistRank() const EL_NO_EXCEPT
{ return this->Grid().VCRank(); }
template <typename T, Device D>
int DM::CrossRank() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? 0 : mpi::UNDEFINED); }
template <typename T, Device D>
int DM::RedundantRank() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? 0 : mpi::UNDEFINED); }

template <typename T, Device D>
int DM::ColRank() const EL_NO_EXCEPT
{ return this->Grid().VCRank(); }
template <typename T, Device D>
int DM::RowRank() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? 0 : mpi::UNDEFINED); }

template <typename T, Device D>
int DM::PartialColRank() const EL_NO_EXCEPT
{ return this->Grid().MCRank(); }
template <typename T, Device D>
int DM::PartialUnionColRank() const EL_NO_EXCEPT
{ return this->Grid().MRRank(); }

template <typename T, Device D>
int DM::PartialRowRank() const EL_NO_EXCEPT
{ return this->RowRank(); }
template <typename T, Device D>
int DM::PartialUnionRowRank() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? 0 : mpi::UNDEFINED); }

// Instantiate {Int,Real,Complex<Real>} for each Real in {float,double}
// ####################################################################

#define SELF(T,U,V,D)                                                     \
    template DistMatrix<T,COLDIST,ROWDIST,ELEMENT,D>::DistMatrix \
    (const DistMatrix<T,U,V,ELEMENT,D>& A)
#define OTHER(T,U,V,D)                                                    \
    template DistMatrix<T,COLDIST,ROWDIST,ELEMENT,D>::DistMatrix \
    (const DistMatrix<T,U,V,BLOCK,D>& A);                   \
    template DistMatrix<T,COLDIST,ROWDIST,ELEMENT,D>&         \
    DistMatrix<T,COLDIST,ROWDIST,ELEMENT,D>::operator=        \
    (const DistMatrix<T,U,V,BLOCK,D>& A)
#define BOTH(T,U,V,D)                             \
    SELF(T,U,V,D);                                \
    OTHER(T,U,V,D)
#define PROTO(T) \
    template class DistMatrix<T,COLDIST,ROWDIST,ELEMENT,Device::CPU>;   \
    BOTH(T,CIRC,CIRC,Device::CPU);                                      \
    BOTH(T,MC,  MR ,Device::CPU);                                       \
    BOTH(T,MC,  STAR,Device::CPU);                                      \
    BOTH(T,MD,  STAR,Device::CPU);                                      \
    BOTH(T,MR,  MC ,Device::CPU);                                       \
    BOTH(T,MR,  STAR,Device::CPU);                                      \
    BOTH(T,STAR,MC ,Device::CPU);                                       \
    BOTH(T,STAR,MD ,Device::CPU);                                       \
    BOTH(T,STAR,MR ,Device::CPU);                                       \
    BOTH(T,STAR,STAR,Device::CPU);                                      \
    BOTH(T,STAR,VC ,Device::CPU);                                       \
    BOTH(T,STAR,VR ,Device::CPU);                                       \
    OTHER(T,VC,  STAR,Device::CPU);                                     \
    BOTH(T,VR,  STAR,Device::CPU);

#ifdef HYDROGEN_HAVE_GPU
#define INSTGPU(T,U,V)                                                  \
    template DistMatrix<T,COLDIST,ROWDIST,ELEMENT,Device::GPU>::DistMatrix \
    (DistMatrix<T,U,V,ELEMENT,Device::CPU> const&);                     \
    template DistMatrix<T,COLDIST,ROWDIST,ELEMENT,Device::GPU>&         \
    DistMatrix<T,COLDIST,ROWDIST,ELEMENT,Device::GPU>::operator=        \
    (DistMatrix<T,U,V,ELEMENT,Device::CPU> const&);                     \
    template DistMatrix<T,COLDIST,ROWDIST,ELEMENT,Device::GPU>::DistMatrix \
    (DistMatrix<T,U,V,ELEMENT,Device::GPU> const&);                     \
                                                                        \
    template DistMatrix<T,COLDIST,ROWDIST,ELEMENT,Device::CPU>::DistMatrix \
    (DistMatrix<T,U,V,ELEMENT,Device::GPU> const&);                     \
    template DistMatrix<T,COLDIST,ROWDIST,ELEMENT,Device::CPU>&         \
    DistMatrix<T,COLDIST,ROWDIST,ELEMENT,Device::CPU>::operator=        \
    (DistMatrix<T,U,V,ELEMENT,Device::GPU> const&)

#ifdef HYDROGEN_GPU_USE_FP16
PROTO(gpu_half_type)
template class DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::GPU>;
INSTGPU(gpu_half_type,CIRC,CIRC);
INSTGPU(gpu_half_type,  MC,  MR);
INSTGPU(gpu_half_type,  MC,STAR);
INSTGPU(gpu_half_type,  MD,STAR);
INSTGPU(gpu_half_type,  MR,  MC);
INSTGPU(gpu_half_type,  MR,STAR);
INSTGPU(gpu_half_type,STAR,  MC);
INSTGPU(gpu_half_type,STAR,  MD);
INSTGPU(gpu_half_type,STAR,  MR);
INSTGPU(gpu_half_type,STAR,STAR);
INSTGPU(gpu_half_type,STAR,  VC);
INSTGPU(gpu_half_type,STAR,  VR);
INSTGPU(gpu_half_type,  VR,STAR);
template DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::GPU>&
DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::GPU>::operator=(
    DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::CPU> const&);
template DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::CPU>&
DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::CPU>::operator=(
    DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::GPU> const&);
#endif // HYDROGEN_GPU_USE_FP16

template class DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::GPU>;
INSTGPU(float,CIRC,CIRC);
INSTGPU(float,MC,  MR );
INSTGPU(float,MC,  STAR);
INSTGPU(float,MD,  STAR);
INSTGPU(float,MR,  MC );
INSTGPU(float,MR,  STAR);
INSTGPU(float,STAR,MC );
INSTGPU(float,STAR,MD );
INSTGPU(float,STAR,MR );
INSTGPU(float,STAR,STAR);
INSTGPU(float,STAR,VC );
INSTGPU(float,STAR,VR );
INSTGPU(float,VR,  STAR);
template DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::GPU>&
DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::GPU>::operator=(
    DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::CPU> const&);
template DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::CPU>&
DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::CPU>::operator=(
    DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::GPU> const&);

template class DistMatrix<double,COLDIST,ROWDIST,ELEMENT,Device::GPU>;
INSTGPU(double,CIRC,CIRC);
INSTGPU(double,MC,  MR );
INSTGPU(double,MC,  STAR);
INSTGPU(double,MD,  STAR);
INSTGPU(double,MR,  MC );
INSTGPU(double,MR,  STAR);
INSTGPU(double,STAR,MC );
INSTGPU(double,STAR,MD );
INSTGPU(double,STAR,MR );
INSTGPU(double,STAR,STAR);
INSTGPU(double,STAR,VC );
INSTGPU(double,STAR,VR );
INSTGPU(double,VR,  STAR);
template DistMatrix<double,COLDIST,ROWDIST,ELEMENT,Device::GPU>&
DistMatrix<double,COLDIST,ROWDIST,ELEMENT,Device::GPU>::operator=(
    DistMatrix<double,COLDIST,ROWDIST,ELEMENT,Device::CPU> const&);
template DistMatrix<double,COLDIST,ROWDIST,ELEMENT,Device::CPU>&
DistMatrix<double,COLDIST,ROWDIST,ELEMENT,Device::CPU>::operator=(
    DistMatrix<double,COLDIST,ROWDIST,ELEMENT,Device::GPU> const&);
#endif // HYDROGEN_HAVE_GPU

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#define EL_ENABLE_HALF
#include <El/macros/Instantiate.h>

} // namespace El
