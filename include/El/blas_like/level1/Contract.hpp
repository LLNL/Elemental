/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_CONTRACT_HPP
#define EL_BLAS_CONTRACT_HPP

namespace El {

template<typename T, Device D, typename=EnableIf<IsDeviceValidType<T,D>>>
void ContractDispatch
(const ElementalMatrix<T>& A,
        ElementalMatrix<T>& B)
{
    EL_DEBUG_CSE
    AssertSameGrids(A, B);
    const Dist U = B.ColDist();
    const Dist V = B.RowDist();
    // TODO: Shorten this implementation?
    if(A.ColDist() == U && A.RowDist() == V)
    {
        Copy(A, B);
    }
    else if(A.ColDist() == U && A.RowDist() == Partial(V))
    {
        B.AlignAndResize
        (A.ColAlign(), A.RowAlign(), A.Height(), A.Width(), false, false);
        Zero(B.Matrix());
        AxpyContract(TypeTraits<T>::One(), A, B);
    }
    else if(A.ColDist() == Partial(U) && A.RowDist() == V)
    {
        B.AlignAndResize
        (A.ColAlign(), A.RowAlign(), A.Height(), A.Width(), false, false);
        Zero(B.Matrix());
        AxpyContract(TypeTraits<T>::One(), A, B);
    }
    else if(A.ColDist() == U && A.RowDist() == Collect(V))
    {
        B.AlignColsAndResize
        (A.ColAlign(), A.Height(), A.Width(), false, false);
        Zero(B.Matrix());
        AxpyContract(TypeTraits<T>::One(), A, B);
    }
    else if(A.ColDist() == Collect(U) && A.RowDist() == V)
    {
        B.AlignRowsAndResize
        (A.RowAlign(), A.Height(), A.Width(), false, false);
        Zero(B.Matrix());
        AxpyContract(TypeTraits<T>::One(), A, B);
    }
    else if(A.ColDist() == Collect(U) && A.RowDist() == Collect(V))
    {
        B.Resize(A.Height(), A.Width());
        Zero(B.Matrix());
        AxpyContract(TypeTraits<T>::One(), A, B);
    }
    else
        LogicError("Incompatible distributions");
}

template<typename T, Device D,
         typename=DisableIf<IsDeviceValidType<T,D>>, typename=void>
void ContractDispatch
(ElementalMatrix<T> const& A,
 ElementalMatrix<T>& B)
{
    LogicError("Contract: Bad type/device combo.");
}

template<typename T>
void Contract
(const ElementalMatrix<T>& A,
        ElementalMatrix<T>& B)
{
    EL_DEBUG_CSE
    AssertSameGrids(A, B);
    if (A.GetLocalDevice() != B.GetLocalDevice())
        LogicError("Incompatible device types.");

    switch (A.GetLocalDevice())
    {
    case Device::CPU:
        ContractDispatch<T,Device::CPU>(A,B);
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        ContractDispatch<T,Device::GPU>(A,B);
        break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("Contract: Bad device type.");
    }
}

template<typename T>
void Contract
(const BlockMatrix<T>& A,
        BlockMatrix<T>& B)
{
    EL_DEBUG_CSE
    AssertSameGrids(A, B);
    const Dist U = B.ColDist();
    const Dist V = B.RowDist();
    // TODO: Shorten this implementation?
    if(A.ColDist() == U && A.RowDist() == V)
    {
        Copy(A, B);
    }
    else if(A.ColDist() == U && A.RowDist() == Partial(V))
    {
        B.AlignAndResize
        (A.BlockHeight(), A.BlockWidth(),
          A.ColAlign(), A.RowAlign(), A.ColCut(), A.RowCut(),
          A.Height(), A.Width(), false, false);
        Zero(B.Matrix());
        AxpyContract(TypeTraits<T>::One(), A, B);
    }
    else if(A.ColDist() == Partial(U) && A.RowDist() == V)
    {
        B.AlignAndResize
        (A.BlockHeight(), A.BlockWidth(),
          A.ColAlign(), A.RowAlign(), A.ColCut(), A.RowCut(),
          A.Height(), A.Width(), false, false);
        Zero(B.Matrix());
        AxpyContract(TypeTraits<T>::One(), A, B);
    }
    else if(A.ColDist() == U && A.RowDist() == Collect(V))
    {
        B.AlignColsAndResize
        (A.BlockHeight(), A.ColAlign(), A.ColCut(), A.Height(), A.Width(),
          false, false);
        Zero(B.Matrix());
        AxpyContract(TypeTraits<T>::One(), A, B);
    }
    else if(A.ColDist() == Collect(U) && A.RowDist() == V)
    {
        B.AlignRowsAndResize
        (A.BlockWidth(), A.RowAlign(), A.RowCut(), A.Height(), A.Width(),
          false, false);
        Zero(B.Matrix());
        AxpyContract(TypeTraits<T>::One(), A, B);
    }
    else if(A.ColDist() == Collect(U) && A.RowDist() == Collect(V))
    {
        B.Resize(A.Height(), A.Width());
        Zero(B.Matrix());
        AxpyContract(TypeTraits<T>::One(), A, B);
    }
    else
        LogicError("Incompatible distributions");
}

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void Contract \
  (const ElementalMatrix<T>& A, \
          ElementalMatrix<T>& B); \
  EL_EXTERN template void Contract \
  (const BlockMatrix<T>& A, \
          BlockMatrix<T>& B);

#ifdef HYDROGEN_GPU_USE_FP16
PROTO(gpu_half_type)
#endif // HYDROGEN_GPU_USE_FP16

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#define EL_ENABLE_HALF
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_CONTRACT_HPP
