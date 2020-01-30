#ifndef EL_BLAS_LIKE_LEVEL1_COPYASYNCLOCAL_HPP_
#define EL_BLAS_LIKE_LEVEL1_COPYASYNCLOCAL_HPP_

#include <hydrogen/Device.hpp>

#include "CopyLocal.hpp"

namespace El
{

template <typename T, typename U,
          EnableWhen<And<IsStorageType<T, Device::CPU>,
                         IsStorageType<U, Device::CPU>>, int> = 0>
void CopyAsyncImpl(Matrix<T, Device::CPU> const& src,
                   Matrix<U, Device::CPU>& tgt)
{
    // No optimized asynchronous copy on CPU. Perhaps at some point we
    // will introduce a truly asynchronous Copy by running it on an
    // internal thread. That is not the case yet.
    return Copy(src, tgt);
}

#ifdef HYDROGEN_HAVE_GPU

template <typename T, typename U,
          EnableWhen<And<IsStorageType<T, Device::GPU>,
                         IsStorageType<U, Device::GPU>>, int> = 0>
void CopyAsyncImpl(Matrix<T, Device::GPU> const& src,
                   Matrix<U, Device::GPU>& tgt)
{
    // The (intertype) copy on GPU is already asynchronous with
    // respect to the host. Perhaps at some point we will introduce
    // "emulated asynchrony" by allowing some operations to run on an
    // internal stream; not now.
    return Copy(src, tgt);
}

// These inter-device copy functions are as ASYNCHRONOUS with respect
// to the host as possible. See the CUDA documentation of Memcpy
// synchronization semantics for more information.
template <typename T,
          EnableWhen<And<IsStorageType<T, Device::CPU>,
                         IsStorageType<T, Device::GPU>>, int> = 0>
void CopyAsyncImpl(Matrix<T, Device::CPU> const& A,
                   Matrix<T, Device::GPU>& B)
{
    EL_DEBUG_CSE;
    const Int height = A.Height();
    const Int width = A.Width();
    B.Resize(height, width);
    const Int ldA = A.LDim();
    const Int ldB = B.LDim();
    const T* EL_RESTRICT ABuf = A.LockedBuffer();
    T* EL_RESTRICT BBuf = B.Buffer();

    details::InterdeviceCopy<Device::CPU, Device::GPU>::Copy2DAsync(
        ABuf, ldA, BBuf, ldB, height, width, SyncInfoFromMatrix(B));
}

template <typename T,
          EnableWhen<And<IsStorageType<T, Device::GPU>,
                         IsStorageType<T, Device::CPU>>, int> = 0>
void CopyAsyncImpl(Matrix<T, Device::GPU> const& A,
                   Matrix<T, Device::CPU>& B)
{
    EL_DEBUG_CSE;
    const Int height = A.Height();
    const Int width = A.Width();
    B.Resize(height, width);
    const Int ldA = A.LDim();
    const Int ldB = B.LDim();
    const T* EL_RESTRICT ABuf = A.LockedBuffer();
    T* EL_RESTRICT BBuf = B.Buffer();

    details::InterdeviceCopy<Device::GPU, Device::CPU>::Copy2DAsync(
        ABuf, ldA, BBuf, ldB, height, width, SyncInfoFromMatrix(A));
}
#endif // HYDROGEN_HAVE_GPU

// FIXME: If the source matrix is on the GPU, we should prefer
// changing type before changing device. If the source matrix is on
// the CPU, we should prefer changing the device before changing the
// type. This ensures the most work is done asynchronously with
// respect to the CPU.

// Inter-device and inter-type
template <typename T, typename U, Device D1, Device D2,
          EnableWhen<And<BoolVT<D1!=D2>,
                         IsStorageType<T, D1>,
                         IsStorageType<U, D2>>, int> = 0>
void CopyAsyncImpl(Matrix<T, D1> const& src, Matrix<U, D2>& tgt)
{
    using T_on_D2 = details::CompatibleStorageType<T, D2>;
    Matrix<T_on_D2, D2> tmp;
    CopyAsync(src, tmp); // Change device
    CopyAsync(tmp, tgt); // Change type
}

template <typename T, typename U, Device D1, Device D2,
          EnableWhen<Or<Not<IsStorageType<T,D1>>,
                        Not<IsStorageType<U,D2>>>, int> = 0>
void CopyAsyncImpl(Matrix<T, D1> const&, Matrix<U, D2>&)
{
    LogicError("Cannot dispatch CopyAsync.");
}

template <typename T, typename U, Device D1, Device D2>
void CopyAsync(Matrix<T, D1> const& src, Matrix<U, D2>& tgt)
{
    CopyAsyncImpl(src, tgt);
}

template <typename T, typename U>
void CopyAsync(AbstractMatrix<T> const& Source, AbstractMatrix<U>& Target)
{
    switch (Target.GetDevice())
    {
    case Device::CPU:
        return details::LaunchCopy(
            Source, static_cast<Matrix<U, Device::CPU>&>(Target),
            details::CopyAsyncFunctor{});
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        return details::LaunchCopy(
            Source, static_cast<Matrix<U, Device::GPU>&>(Target),
            details::CopyAsyncFunctor{});
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("Copy: Bad device.");
    }
}

}// namespace El
#endif // EL_BLAS_LIKE_LEVEL1_COPYASYNCLOCAL_HPP_
