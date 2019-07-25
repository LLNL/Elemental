/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like.hpp>

#define COLDIST CIRC
#define ROWDIST CIRC

#include "./setup.hpp"

namespace El {

// Public section
// ##############

// Operator overloading
// ====================

template <typename T>
struct TC;

// Make a copy
// -----------
template <typename T, Device D>
DM& DM::operator=(const ElementalMatrix<T>& A)
{
    EL_DEBUG_CSE
    copy::Gather(A, *this);
    return *this;
}

// Basic queries
// =============
template <typename T, Device D>
mpi::Comm const& DM::CrossComm() const EL_NO_EXCEPT
{ return this->Grid().VCComm(); }

template <typename T, Device D>
mpi::Comm const& DM::ColComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }
template <typename T, Device D>
mpi::Comm const& DM::RowComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }
template <typename T, Device D>
mpi::Comm const& DM::DistComm() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL); }
template <typename T, Device D>
mpi::Comm const& DM::RedundantComm() const EL_NO_EXCEPT
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
int DM::ColStride() const EL_NO_EXCEPT { return 1; }
template <typename T, Device D>
int DM::RowStride() const EL_NO_EXCEPT { return 1; }
template <typename T, Device D>
int DM::DistSize() const EL_NO_EXCEPT { return 1; }
template <typename T, Device D>
int DM::CrossSize() const EL_NO_EXCEPT { return this->Grid().VCSize(); }
template <typename T, Device D>
int DM::RedundantSize() const EL_NO_EXCEPT { return 1; }
template <typename T, Device D>
int DM::PartialColStride() const EL_NO_EXCEPT { return 1; }
template <typename T, Device D>
int DM::PartialRowStride() const EL_NO_EXCEPT { return 1; }
template <typename T, Device D>
int DM::PartialUnionColStride() const EL_NO_EXCEPT { return 1; }
template <typename T, Device D>
int DM::PartialUnionRowStride() const EL_NO_EXCEPT { return 1; }

template <typename T, Device D>
int DM::ColRank() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? 0 : mpi::UNDEFINED); }
template <typename T, Device D>
int DM::RowRank() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? 0 : mpi::UNDEFINED); }
template <typename T, Device D>
int DM::DistRank() const EL_NO_EXCEPT
{ return (this->Grid().InGrid() ? 0 : mpi::UNDEFINED); }
template <typename T, Device D>
int DM::CrossRank() const EL_NO_EXCEPT { return this->Grid().VCRank(); }
template <typename T, Device D>
int DM::RedundantRank() const EL_NO_EXCEPT
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
  OTHER(T,CIRC,CIRC); \
  BOTH(T,MC,  MR ); \
  BOTH(T,MC,  STAR); \
  BOTH(T,MD,  STAR); \
  BOTH(T,MR,  MC ); \
  BOTH(T,MR,  STAR); \
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
template
    DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::CPU>::DistMatrix(
        const DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::GPU>&);

template DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::CPU>&
DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::CPU>::operator=(
    const DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::GPU>&);
#endif // HYDROGEN_GPU_USE_FP16

template DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::CPU>::DistMatrix(
    const DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::GPU>&);

template DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::CPU>&
DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::CPU>::operator=(
    const DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::GPU>&);

template DistMatrix<double,COLDIST,ROWDIST,ELEMENT,Device::CPU>::DistMatrix(
    const DistMatrix<double,COLDIST,ROWDIST,ELEMENT,Device::GPU>&);

template DistMatrix<double,COLDIST,ROWDIST,ELEMENT,Device::CPU>&
DistMatrix<double,COLDIST,ROWDIST,ELEMENT,Device::CPU>::operator=(
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
INSTGPU(gpu_half_type,  VC,STAR);
INSTGPU(gpu_half_type,  VR,STAR);
template DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::GPU>::DistMatrix(
    const DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::CPU>&);
template DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::GPU>&
DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::GPU>::operator=(
    const DistMatrix<gpu_half_type,COLDIST,ROWDIST,ELEMENT,Device::CPU>&);
#endif // HYDROGEN_GPU_USE_FP16

template class DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::GPU>;
INSTGPU(float,MC,  MR  );
INSTGPU(float,MC,  STAR);
INSTGPU(float,MD,  STAR);
INSTGPU(float,MR,  MC  );
INSTGPU(float,MR,  STAR);
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
    const DistMatrix<float,COLDIST,ROWDIST,ELEMENT,Device::CPU>&);

template class DistMatrix<double,COLDIST,ROWDIST,ELEMENT,Device::GPU>;
INSTGPU(double,MC,  MR  );
INSTGPU(double,MC,  STAR);
INSTGPU(double,MD,  STAR);
INSTGPU(double,MR,  MC  );
INSTGPU(double,MR,  STAR);
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
    const DistMatrix<double,COLDIST,ROWDIST,ELEMENT,Device::CPU>&);
#endif // HYDROGEN_HAVE_GPU

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#define EL_ENABLE_HALF
#include <El/macros/Instantiate.h>

} // namespace El
