/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like.hpp>

#define COLDIST MR
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
    DistMatrix<T,VC,STAR,ELEMENT,D> A_VC_STAR(A);
    DistMatrix<T,VR,STAR,ELEMENT,D> A_VR_STAR(this->Grid());
    A_VR_STAR.AlignColsWith(*this);
    A_VR_STAR = A_VC_STAR;
    A_VC_STAR.Empty();

    *this = A_VR_STAR;
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,MC,STAR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(AssertSameGrids(*this, A))
    const Grid& grid = A.Grid();
    if (grid.Height() == grid.Width())
    {
        const int gridDim = grid.Height();
        const int transposeRank =
            A.RowOwner(this->ColShift()) + gridDim*this->RowOwner(A.ColShift());
        copy::Exchange(A, *this, transposeRank, transposeRank, grid.VCComm());
    }
    else
    {
        DistMatrix<T,VC,STAR,ELEMENT,D> A_VC_STAR(A);
        DistMatrix<T,VR,STAR,ELEMENT,D> A_VR_STAR(grid);
        A_VR_STAR.AlignColsWith(*this);
        A_VR_STAR = A_VC_STAR;
        A_VC_STAR.Empty();
        *this = A_VR_STAR;
    }
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,STAR,MR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    DistMatrix<T,MC,MR,ELEMENT,D> A_MC_MR(A);
    DistMatrix<T,VC,STAR,ELEMENT,D> A_VC_STAR(A_MC_MR);
    A_MC_MR.Empty();

    DistMatrix<T,VR,STAR,ELEMENT,D> A_VR_STAR(this->Grid());
    A_VR_STAR.AlignColsWith(*this);
    A_VR_STAR = A_VC_STAR;
    A_VC_STAR.Empty();

    *this = A_VR_STAR;
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
    copy::RowAllGather(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,STAR,MC,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    DistMatrix<T,MR,MC,ELEMENT,D> A_MR_MC(A);
    *this = A_MR_MC;
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,VC,STAR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    DistMatrix<T,VR,STAR,ELEMENT,D> A_VR_STAR(this->Grid());
    A_VR_STAR.AlignColsWith(*this);
    A_VR_STAR = A;
    *this = A_VR_STAR;
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,STAR,VC,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    DistMatrix<T,MR,MC,ELEMENT,D> A_MR_MC(A);
    *this = A_MR_MC;
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,VR,STAR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    copy::PartialColAllGather(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const DistMatrix<T,STAR,VR,ELEMENT,D>& A)
{
    EL_DEBUG_CSE
    DistMatrix<T,STAR,VC,ELEMENT,D> A_STAR_VC(A);
    DistMatrix<T,MR,MC,ELEMENT,D> A_MR_MC(this->Grid());
    A_MR_MC.AlignColsWith(*this);
    A_MR_MC = A_STAR_VC;
    A_STAR_VC.Empty();

    *this = A_MR_MC;
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
    DistMatrix<T,MR,MC,ELEMENT,D> A_MR_MC(this->Grid());
    A_MR_MC.AlignWith(*this);
    A_MR_MC = A;
    *this = A_MR_MC;
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
{ return this->Grid().MRComm(); }

template <typename T, Device D>
mpi::Comm const& DM::CrossComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }

template <typename T, Device D>
mpi::Comm const& DM::RedundantComm() const EL_NO_EXCEPT
{ return this->Grid().MCComm(); }

template <typename T, Device D>
mpi::Comm const& DM::ColComm() const EL_NO_EXCEPT
{ return this->Grid().MRComm(); }
template <typename T, Device D>
mpi::Comm const& DM::RowComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }

template <typename T, Device D>
mpi::Comm const& DM::PartialColComm() const EL_NO_EXCEPT
{ return this->ColComm(); }
template <typename T, Device D>
mpi::Comm const& DM::PartialRowComm() const EL_NO_EXCEPT
{ return this->RowComm(); }

template <typename T, Device D>
mpi::Comm const& DM::PartialUnionColComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }
template <typename T, Device D>
mpi::Comm const& DM::PartialUnionRowComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }

template <typename T, Device D>
int DM::ColStride() const EL_NO_EXCEPT { return this->Grid().MRSize(); }
template <typename T, Device D>
int DM::RowStride() const EL_NO_EXCEPT { return 1; }
template <typename T, Device D>
int DM::DistSize() const EL_NO_EXCEPT { return this->Grid().MRSize(); }
template <typename T, Device D>
int DM::CrossSize() const EL_NO_EXCEPT { return 1; }
template <typename T, Device D>
int DM::RedundantSize() const EL_NO_EXCEPT { return this->Grid().MCSize(); }
template <typename T, Device D>
int DM::PartialColStride() const EL_NO_EXCEPT { return this->ColStride(); }
template <typename T, Device D>
int DM::PartialRowStride() const EL_NO_EXCEPT { return this->RowStride(); }
template <typename T, Device D>
int DM::PartialUnionColStride() const EL_NO_EXCEPT { return 1; }
template <typename T, Device D>
int DM::PartialUnionRowStride() const EL_NO_EXCEPT { return 1; }

template <typename T, Device D>
int DM::ColRank() const EL_NO_EXCEPT { return this->Grid().MRRank(); }
template <typename T, Device D>
int DM::RowRank() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? 0 : mpi::UNDEFINED); }
template <typename T, Device D>
int DM::DistRank() const EL_NO_EXCEPT { return this->Grid().MRRank(); }
template <typename T, Device D>
int DM::CrossRank() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? 0 : mpi::UNDEFINED); }
template <typename T, Device D>
int DM::RedundantRank() const EL_NO_EXCEPT { return this->Grid().MCRank(); }
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
  OTHER(T,MR,  STAR); \
  BOTH(T,STAR,MC ); \
  BOTH(T,STAR,MD ); \
  BOTH(T,STAR,MR ); \
  BOTH(T,STAR,STAR); \
  BOTH(T,STAR,VC ); \
  BOTH(T,STAR,VR ); \
  BOTH(T,VC,  STAR); \
  BOTH(T,VR,  STAR);

#ifdef HYDROGEN_HAVE_GPU
// Inter-device copy ctors
#ifdef HYDROGEN_GPU_USE_FP16
template DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::CPU>::DistMatrix(
    const DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::GPU>&);
#endif // HYDROGEN_GPU_USE_FP16
template DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::CPU>::DistMatrix(
    const DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::GPU>&);
template DistMatrix<double,COLDIST,ROWDIST,ELEMENT,Device::CPU>::DistMatrix(
    const DistMatrix<double,COLDIST,ROWDIST,ELEMENT,Device::GPU>&);

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
INSTGPU(gpu_half_type,STAR,  MC);
INSTGPU(gpu_half_type,STAR,  MD);
INSTGPU(gpu_half_type,STAR,  MR);
INSTGPU(gpu_half_type,STAR,STAR);
INSTGPU(gpu_half_type,STAR,  VC);
INSTGPU(gpu_half_type,STAR,  VR);
INSTGPU(gpu_half_type,  VC,STAR);
INSTGPU(gpu_half_type,  VR,STAR);
template DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::GPU>::DistMatrix(
    const DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::CPU>&);
template DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::GPU>&
DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::GPU>::operator=(
    DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::CPU> const&);
template DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::CPU>&
DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::CPU>::operator=(
    DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::GPU> const&);
#endif // HYDROGEN_GPU_USE_FP16

template class DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::GPU>;
INSTGPU(float,CIRC,CIRC);
INSTGPU(float,MC,  MR  );
INSTGPU(float,MC,  STAR);
INSTGPU(float,MD,  STAR);
INSTGPU(float,MR,  MC  );
INSTGPU(float,STAR,MC  );
INSTGPU(float,STAR,MD  );
INSTGPU(float,STAR,MR  );
INSTGPU(float,STAR,STAR);
INSTGPU(float,STAR,VC  );
INSTGPU(float,STAR,VR  );
INSTGPU(float,VC,  STAR);
INSTGPU(float,VR,  STAR);
template DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::GPU>::DistMatrix(
    const DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::CPU>&);
template DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::GPU>&
DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::GPU>::operator=(
    DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::CPU> const&);
template DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::CPU>&
DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::CPU>::operator=(
    DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::GPU> const&);

template class DistMatrix<double,COLDIST,ROWDIST,ELEMENT,Device::GPU>;
INSTGPU(double,CIRC,CIRC);
INSTGPU(double,MC,  MR  );
INSTGPU(double,MC,  STAR);
INSTGPU(double,MD,  STAR);
INSTGPU(double,MR,  MC  );
INSTGPU(double,STAR,MC  );
INSTGPU(double,STAR,MD  );
INSTGPU(double,STAR,MR  );
INSTGPU(double,STAR,STAR);
INSTGPU(double,STAR,VC  );
INSTGPU(double,STAR,VR  );
INSTGPU(double,VC,  STAR);
INSTGPU(double,VR,  STAR);
template DistMatrix<double,COLDIST,ROWDIST,ELEMENT,Device::GPU>::DistMatrix(
    const DistMatrix<double,COLDIST,ROWDIST,ELEMENT,Device::CPU>&);
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
