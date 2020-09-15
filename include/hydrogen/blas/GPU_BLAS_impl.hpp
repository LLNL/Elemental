#ifndef HYDROGEN_GPU_BLAS_IMPL_HPP_
#define HYDROGEN_GPU_BLAS_IMPL_HPP_

#include <El/hydrogen_config.h>

#include <hydrogen/blas/GPU_BLAS_decl.hpp>

#include <hydrogen/blas/BLAS_Common.hpp>
#include <hydrogen/meta/MetaUtilities.hpp>
#include <hydrogen/meta/TypeTraits.hpp>
#include <hydrogen/Device.hpp>
#include <hydrogen/SyncInfo.hpp>

// Include the kernel decls
#include <hydrogen/blas/gpu/Axpy.hpp>
#include <hydrogen/blas/gpu/Copy.hpp>
#include <hydrogen/blas/gpu/Fill.hpp>
#include <hydrogen/blas/gpu/Scale.hpp>
#include <hydrogen/blas/gpu/Transpose.hpp>

// By assumption, CUDA and ROCm are exclusive: having one means the
// other is prohibited.
#if defined(HYDROGEN_HAVE_CUDA)

// Routines will dispatch to cuBLAS for all applicable operation/type
// combinations. Custom CUDA kernels may or may not be supported for
// the remainder of the operation/type combinations depending on
// needs.

#define GPU_BLAS_USE_CUBLAS
#include <hydrogen/device/gpu/cuda/cuBLAS.hpp>
#include <hydrogen/device/gpu/cuda/cuSOLVER.hpp>

namespace gpu_blas_impl = hydrogen::cublas;
namespace gpu_lapack_impl = hydrogen::cusolver;

#elif defined(HYDROGEN_HAVE_ROCM)

// Routines will dispatch to rocBLAS for all applicable operation/type
// combinations. Custom HIP kernels may or may not be supported for
// the remainder of the operation/type combinations depending on
// needs.

#define GPU_BLAS_USE_ROCBLAS
#include <hydrogen/device/gpu/rocm/rocBLAS.hpp>
#include <hydrogen/device/gpu/rocm/rocSOLVER.hpp>

namespace gpu_blas_impl = hydrogen::rocblas;
namespace gpu_lapack_impl = hydrogen::rocsolver;

#else
#pragma GCC error "LOGIC ERROR: No GPU programming model enabled."
#endif

namespace hydrogen
{
namespace gpu_blas
{
namespace details
{

using gpu_blas_impl::ToSizeT;
using gpu_blas_impl::GetLibraryHandle;
using gpu_blas_impl::IsSupportedType;
using gpu_blas_impl::NativeType;
using gpu_blas_impl::SyncManager;
using gpu_blas_impl::ToNativeDiagType;
using gpu_blas_impl::ToNativeFillMode;
using gpu_blas_impl::ToNativeSideMode;
using gpu_blas_impl::ToNativeTransposeMode;

template <typename T, typename SizeT,
          typename=EnableWhen<IsSupportedType<T,BLAS_Op::AXPY>>>
void AxpyImpl(SizeT size, T const& alpha,
              T const* X, SizeT incx,
              T* Y, SizeT incy,
              SyncInfo<Device::GPU> const& si)
{
    using NTP = MakePointer<NativeType<T>>;
    using CNTP = MakePointerToConst<NativeType<T>>;

    SyncManager mgr(GetLibraryHandle(), si);
    gpu_blas_impl::Axpy(
        GetLibraryHandle(),
        ToSizeT(size), alpha,
        reinterpret_cast<CNTP>(X), ToSizeT(incx),
        reinterpret_cast<NTP>(Y), ToSizeT(incy));
}

template <typename T, typename SizeT,
          typename=EnableWhen<IsSupportedType<T,BLAS_Op::GEAM>>>
void Axpy2DImpl(SizeT nrows, SizeT ncols,
                T const& alpha,
                T const* A, SizeT lda,
                T* B, SizeT ldb,
                SyncInfo<Device::GPU> const& si)
{
    using NTP = MakePointer<NativeType<T>>;
    using CNTP = MakePointerToConst<NativeType<T>>;

    SyncManager mgr(GetLibraryHandle(), si);
    gpu_blas_impl::Geam(
        GetLibraryHandle(),
        ToNativeTransposeMode(TransposeMode::NORMAL),
        ToNativeTransposeMode(TransposeMode::NORMAL),
        nrows, ncols,
        alpha,
        reinterpret_cast<CNTP>(A), lda,
        TypeTraits<T>::One(),
        reinterpret_cast<CNTP>(B), ldb,
        reinterpret_cast<NTP>(B), ldb);
}

template <typename T, typename SizeT,
          typename=EnableWhen<IsSupportedType<T,BLAS_Op::GEAM>>>
void Axpy2DImplTranspose(
    TransposeMode transpA,
    SizeT nrows, SizeT ncols,
    T const& alpha,
    T const* A, SizeT lda,
    T* B, SizeT ldb,
    SyncInfo<Device::GPU> const& si)
{
    using NTP = MakePointer<NativeType<T>>;
    using CNTP = MakePointerToConst<NativeType<T>>;

    SyncManager mgr(GetLibraryHandle(), si);
    gpu_blas_impl::Geam(
        GetLibraryHandle(),
        ToNativeTransposeMode(transpA),
        ToNativeTransposeMode(TransposeMode::NORMAL),
        nrows, ncols,
        alpha,
        reinterpret_cast<CNTP>(A), lda,
        TypeTraits<T>::One(),
        reinterpret_cast<CNTP>(B), ldb,
        reinterpret_cast<NTP>(B), ldb);
}

template <typename T, typename SizeT,
          typename=EnableWhen<IsSupportedType<T,BLAS_Op::COPY>>>
void CopyImpl(SizeT size,
              T const* X, SizeT incx,
              T* Y, SizeT incy,
              SyncInfo<Device::GPU> const& si)
{
    using NTP = MakePointer<NativeType<T>>;
    using CNTP = MakePointerToConst<NativeType<T>>;

    SyncManager mgr(GetLibraryHandle(), si);
    gpu_blas_impl::Copy(
        GetLibraryHandle(),
        size,
        reinterpret_cast<CNTP>(X), incx,
        reinterpret_cast<NTP>(Y), incy);
}

template <typename T, typename SizeT,
          typename=EnableWhen<IsSupportedType<T,BLAS_Op::GEAM>>>
void Copy2DImpl(SizeT nrows, SizeT ncols,
                TransposeMode transA,
                T const* A, SizeT lda,
                T* B, SizeT ldb,
                SyncInfo<Device::GPU> const& si)
{
    using NTP = MakePointer<NativeType<T>>;
    using CNTP = MakePointerToConst<NativeType<T>>;

    SyncManager mgr(GetLibraryHandle(), si);
    gpu_blas_impl::Geam(
        GetLibraryHandle(),
        ToNativeTransposeMode(transA),
        ToNativeTransposeMode(TransposeMode::NORMAL),
        nrows, ncols,
        TypeTraits<T>::One(), reinterpret_cast<CNTP>(A), lda,
        TypeTraits<T>::Zero(), reinterpret_cast<CNTP>(B), ldb,
        reinterpret_cast<NTP>(B), ldb);
}

template <typename T, typename SizeT,
          typename=EnableWhen<IsSupportedType<T,BLAS_Op::DOT>>>
void DotImpl(SizeT num_entries,
             T const* X, SizeT stride_X,
             T const* Y, SizeT stride_Y,
             T* result,
             SyncInfo<Device::GPU> const& si)
{
    using NTP = MakePointer<NativeType<T>>;
    using CNTP = MakePointerToConst<NativeType<T>>;

    SyncManager mgr(GetLibraryHandle(), si);
    gpu_blas_impl::Dot(
        GetLibraryHandle(),
        num_entries,
        reinterpret_cast<CNTP>(X), stride_X,
        reinterpret_cast<CNTP>(Y), stride_Y,
        reinterpret_cast<NTP>(result));
}

template <typename T, typename SizeT,
          typename=EnableWhen<IsSupportedType<T,BLAS_Op::DOT>>>
void Nrm2Impl(SizeT num_entries,
              T const* X, SizeT stride_X,
              T* result,
              SyncInfo<Device::GPU> const& si)
{
    using NTP = MakePointer<NativeType<T>>;
    using CNTP = MakePointerToConst<NativeType<T>>;

    SyncManager mgr(GetLibraryHandle(), si);
    gpu_blas_impl::Nrm2(
        GetLibraryHandle(),
        num_entries,
        reinterpret_cast<CNTP>(X), stride_X,
        reinterpret_cast<NTP>(result));
}

template <typename T, typename SizeT,
          typename=EnableWhen<IsSupportedType<T,BLAS_Op::SCAL>>>
void ScaleImpl(SizeT num_entries,
               T const& alpha,
               T* X, SizeT incx,
               SyncInfo<Device::GPU> const& si)
{
    using NTP = MakePointer<NativeType<T>>;

    SyncManager mgr(GetLibraryHandle(), si);
    gpu_blas_impl::Scale(
        GetLibraryHandle(),
        num_entries,
        alpha, reinterpret_cast<NTP>(X), incx);
}

template <typename T, typename SizeT,
          typename=EnableWhen<IsSupportedType<T,BLAS_Op::GEAM>>>
void Scale2DImpl(SizeT nrows, SizeT ncols,
                 T const& alpha,
                 T* A, SizeT lda,
                 SyncInfo<Device::GPU> const& si)
{
    using NTP = MakePointer<NativeType<T>>;
    using CNTP = MakePointerToConst<NativeType<T>>;

    SyncManager mgr(GetLibraryHandle(), si);
    gpu_blas_impl::Geam(
        GetLibraryHandle(),
        ToNativeTransposeMode(TransposeMode::NORMAL),
        ToNativeTransposeMode(TransposeMode::NORMAL),
        nrows, ncols,
        alpha, reinterpret_cast<CNTP>(A), lda,
        TypeTraits<T>::Zero(), reinterpret_cast<CNTP>(A), lda,
        reinterpret_cast<NTP>(A), lda);
}

template <typename T, typename SizeT,
          typename=EnableWhen<IsSupportedType<T, BLAS_Op::COPY>>>
void Copy2DStridedImpl(
    SizeT nrows, SizeT ncols,
    T const* A, SizeT rowstride_A, SizeT lda,
    T* B, SizeT rowstride_B, SizeT ldb,
    SyncInfo<Device::GPU> const& si)
{
    using NTP = MakePointer<NativeType<T>>;
    using CNTP = MakePointerToConst<NativeType<T>>;

    // This is a "perfect stride" so we can just do it 1D
    if ((rowstride_A * nrows == lda)
        && (rowstride_B * nrows == ldb))
    {
        CopyImpl(nrows*ncols, A, rowstride_A, B, rowstride_B, si);
    }
    else
    {
        // Handle the sync stuff here rather than inside the loop.
        SyncManager mgr(GetLibraryHandle(), si);

        // Do each column as a 1-D copy
        for (SizeT col = 0; col < ncols; ++col)
        {
            gpu_blas_impl::Copy(
                GetLibraryHandle(),
                nrows,
                reinterpret_cast<CNTP>(A), rowstride_A,
                reinterpret_cast<NTP>(B), rowstride_B);
            A += lda;
            B += ldb;
        }

        // FIXME (trb): Experiment with this vs a custom 2-D 2-stride
        // copy kernel.
    }
}

template <typename T, typename SizeT,
          typename=EnableWhen<IsSupportedType<T, BLAS_Op::GEMV>>>
void GemvImpl(
    TransposeMode transA,
    SizeT nrows, SizeT ncols,
    T const& alpha,
    T const* A, SizeT lda,
    T const* x, SizeT incx,
    T const& beta,
    T* y, SizeT incy,
    SyncInfo<Device::GPU> const& si)
{
    using NTP = MakePointer<NativeType<T>>;
    using CNTP = MakePointerToConst<NativeType<T>>;

    SyncManager mgr(GetLibraryHandle(), si);
    gpu_blas_impl::Gemv(
        GetLibraryHandle(),
        ToNativeTransposeMode(transA),
        ToSizeT(nrows), ToSizeT(ncols),
        alpha,
        reinterpret_cast<CNTP>(A), ToSizeT(lda),
        reinterpret_cast<CNTP>(x), ToSizeT(incx),
        beta,
        reinterpret_cast<NTP>(y), ToSizeT(incy));
}


template <typename T, typename SizeT,
          typename=EnableWhen<IsSupportedType<T, BLAS_Op::HERK>>>
void HerkImpl(
    FillMode uplo, TransposeMode trans,
    SizeT n, SizeT k,
    TmpBase<T> const& alpha,
    T const* A, SizeT lda,
    TmpBase<T> const& beta,
    T* C, SizeT ldc,
    SyncInfo<Device::GPU> const& syncinfo)
{
    using NTP = MakePointer<NativeType<T>>;
    using CNTP = MakePointerToConst<NativeType<T>>;

    SyncManager mgr(GetLibraryHandle(), syncinfo);
    gpu_blas_impl::Herk(
        GetLibraryHandle(),
        ToNativeFillMode(uplo), ToNativeTransposeMode(trans),
        ToSizeT(n), ToSizeT(k),
        alpha,
        reinterpret_cast<CNTP>(A), ToSizeT(lda),
        beta,
        reinterpret_cast<NTP>(C), ToSizeT(ldc));
}

template <typename T, typename SizeT,
          typename=EnableWhen<IsSupportedType<T, BLAS_Op::SYRK>>>
void SyrkImpl(
    FillMode uplo, TransposeMode trans,
    SizeT n, SizeT k,
    T const& alpha,
    T const* A, SizeT lda,
    T const& beta,
    T* C, SizeT ldc,
    SyncInfo<Device::GPU> const& syncinfo)
{
    using NTP = MakePointer<NativeType<T>>;
    using CNTP = MakePointerToConst<NativeType<T>>;

    SyncManager mgr(GetLibraryHandle(), syncinfo);
    gpu_blas_impl::Syrk(
        GetLibraryHandle(),
        ToNativeFillMode(uplo), ToNativeTransposeMode(trans),
        ToSizeT(n), ToSizeT(k),
        alpha,
        reinterpret_cast<CNTP>(A), ToSizeT(lda),
        beta,
        reinterpret_cast<NTP>(C), ToSizeT(ldc));
}

template <typename T, typename SizeT,
          typename=EnableWhen<IsSupportedType<T, BLAS_Op::TRSM>>>
void TrsmImpl(
    SideMode side, FillMode uplo,
    TransposeMode trans, DiagType diag,
    SizeT n, SizeT m,
    T const& alpha,
    T const* A, SizeT lda,
    T* B, SizeT ldb,
    SyncInfo<Device::GPU> const& syncinfo)
{
    using NTP = MakePointer<NativeType<T>>;
    using CNTP = MakePointerToConst<NativeType<T>>;

    SyncManager mgr(GetLibraryHandle(), syncinfo);
    gpu_blas_impl::Trsm(
        GetLibraryHandle(),
        ToNativeSideMode(side), ToNativeFillMode(uplo),
        ToNativeTransposeMode(trans), ToNativeDiagType(diag),
        ToSizeT(n), ToSizeT(m),
        alpha,
        reinterpret_cast<CNTP>(A), ToSizeT(lda),
        reinterpret_cast<NTP>(B), ToSizeT(ldb));
}

template <typename T, typename SizeT,
          typename=EnableWhen<IsSupportedType<T, BLAS_Op::GEMM>>>
void GemmImpl(
    TransposeMode transA, TransposeMode transB,
    SizeT m, SizeT n, SizeT k,
    T const& alpha,
    T const* A, SizeT lda,
    T const* B, SizeT ldb,
    T const& beta,
    T* C, SizeT ldc,
    SyncInfo<Device::GPU> const& si)
{
    using NTP = MakePointer<NativeType<T>>;
    using CNTP = MakePointerToConst<NativeType<T>>;

    SyncManager mgr(GetLibraryHandle(), si);
    gpu_blas_impl::Gemm(
        GetLibraryHandle(),
        ToNativeTransposeMode(transA),
        ToNativeTransposeMode(transB),
        ToSizeT(m), ToSizeT(n), ToSizeT(k),
        alpha,
        reinterpret_cast<CNTP>(A), ToSizeT(lda),
        reinterpret_cast<CNTP>(B), ToSizeT(ldb),
        beta,
        reinterpret_cast<NTP>(C), ToSizeT(ldc));
}

template <typename T, typename SizeT, typename StrideT,
          typename=EnableWhen<IsSupportedType<T, BLAS_Op::GEMMSTRIDEDBATCHED>>>
void GemmStridedBatchedImpl(
    TransposeMode transpA, TransposeMode transpB,
    SizeT m, SizeT n, SizeT k,
    T const& alpha,
    T const* A, SizeT lda, StrideT strideA,
    T const* B, SizeT ldb, StrideT strideB,
    T const& beta,
    T* C, SizeT ldc, StrideT strideC,
    SizeT batchCount,
    SyncInfo<Device::GPU> const& si)
{
    using NTP = MakePointer<NativeType<T>>;
    using CNTP = MakePointerToConst<NativeType<T>>;

    SyncManager mgr(GetLibraryHandle(), si);
    gpu_blas_impl::GemmStridedBatched(
        GetLibraryHandle(),
        ToNativeTransposeMode(transpA),
        ToNativeTransposeMode(transpB),
        ToSizeT(m), ToSizeT(n), ToSizeT(k),
        &alpha,
        reinterpret_cast<CNTP>(A), ToSizeT(lda), ToSizeT(strideA),
        reinterpret_cast<CNTP>(B), ToSizeT(ldb), ToSizeT(strideB),
        &beta,
        reinterpret_cast<NTP>(C), ToSizeT(ldc), ToSizeT(strideC),
        ToSizeT(batchCount));
}

template <typename T, typename SizeT,
          typename=EnableWhen<IsSupportedType<T, BLAS_Op::DGMM>>>
void DgmmImpl(SideMode side,
              SizeT m, SizeT n,
              T const* A, SizeT lda,
              T const* X, SizeT incx,
              T* C, SizeT ldc,
              SyncInfo<Device::GPU> const& si)
{
    using NTP = MakePointer<NativeType<T>>;
    using CNTP = MakePointerToConst<NativeType<T>>;

    SyncManager mgr(GetLibraryHandle(), si);
    gpu_blas_impl::Dgmm(
        GetLibraryHandle(),
        ToNativeSideMode(side),
        ToSizeT(m), ToSizeT(n),
        reinterpret_cast<CNTP>(A), ToSizeT(lda),
        reinterpret_cast<CNTP>(X), ToSizeT(incx),
        reinterpret_cast<NTP>(C), ToSizeT(ldc));
}

//
// Custom kernel declarations (impls can't be here because this is a
// C++ header, but dispatch needs to work right)
//

template <typename T, typename SizeT,
          typename=EnableUnless<IsSupportedType<T,BLAS_Op::GEAM>>,
          typename=void>
void Axpy2DImpl(SizeT nrows, SizeT ncols,
                T const& alpha,
                T const* A, SizeT lda,
                T* B, SizeT ldb,
                SyncInfo<Device::GPU> const& si)
{
    Axpy_GPU_impl(nrows, ncols,
                  alpha, A, SizeT(1), lda,
                  B, SizeT(1), ldb, si);
}

template <typename T, typename SizeT,
          typename=EnableUnless<IsSupportedType<T,BLAS_Op::GEAM>>,
          typename=void>
void Axpy2DImplTranspose(TransposeMode transpA,
                         SizeT nrows, SizeT ncols,
                         T const& alpha,
                         T const* A, SizeT lda,
                         T* B, SizeT ldb,
                         SyncInfo<Device::GPU> const& si)
{
    Axpy_GPU_impl(
        transpA, nrows, ncols, alpha, A, lda, B, ldb, si);
}

template <typename T, typename SizeT,
          typename=EnableUnless<IsSupportedType<T,BLAS_Op::COPY>>,
          typename=void>
void CopyImpl(SizeT size,
              T const* X, SizeT incx,
              T* Y, SizeT incy,
              SyncInfo<Device::GPU> const& si)
{
    Copy_GPU_impl(size, X, incx, Y, incy, si);
}

template <typename T, typename SizeT,
          typename=EnableUnless<IsSupportedType<T,BLAS_Op::GEAM>>,
          typename=void>
void Copy2DImpl(SizeT nrows, SizeT ncols,
                TransposeMode transA,
                T const* A, SizeT lda,
                T* B, SizeT ldb,
                SyncInfo<Device::GPU> const& si)
{
    switch (transA)
    {
    case TransposeMode::NORMAL:
        Copy_GPU_impl(nrows, ncols,
                      A, SizeT(1), lda,
                      B, SizeT(1), ldb, si);
        break;
    case TransposeMode::TRANSPOSE:
        // This kernel is a bit funny and takes the dimensions of A,
        // so we must reverse nrows and ncols.
        Transpose_GPU_impl(ncols, nrows, A, lda, B, ldb, si);
        break;
    default:
        throw std::logic_error("Copy2DImpl: TransposeMode not supported!");
    }
}

template <typename T, typename U, typename SizeT>
void Copy2DImpl(SizeT nrows, SizeT ncols,
                TransposeMode transA,
                T const* A, SizeT lda,
                U* B, SizeT ldb,
                SyncInfo<Device::GPU> const& si)
{
    switch (transA)
    {
    case TransposeMode::NORMAL:
        Copy_GPU_impl(nrows, ncols,
                      A, SizeT(1), lda,
                      B, SizeT(1), ldb, si);
        break;
    case TransposeMode::TRANSPOSE:
        throw std::logic_error(
            "Copy2DImpl: Need to implement multitype transpose");
        break;
    default:
        throw std::logic_error("Copy2DImpl: TransposeMode not supported!");
    }
}

template <typename T, typename SizeT,
          typename=EnableUnless<IsSupportedType<T,BLAS_Op::COPY>>,
          typename=void>
void Copy2DStridedImpl(
    SizeT nrows, SizeT ncols,
    T const* A, SizeT rowstride_A, SizeT lda,
    T* B, SizeT rowstride_B, SizeT ldb,
    SyncInfo<Device::GPU> const& si)
{
    Copy_GPU_impl(nrows, ncols,
                  A, rowstride_A, lda,
                  B, rowstride_B, ldb,
                  si);
}

template <typename T, typename SizeT,
          typename=EnableUnless<IsSupportedType<T,BLAS_Op::DOT>>,
          typename=void>
void DotImpl(SizeT, T const*, SizeT, T const*, SizeT, T*,
             SyncInfo<Device::GPU> const&)
{
    std::ostringstream oss;
    oss << "No valid implementation of DOT for T="
        << TypeTraits<T>::Name();
    throw std::logic_error(oss.str());
}

template <typename T, typename SizeT,
          typename=EnableUnless<IsSupportedType<T,BLAS_Op::DOT>>,
          typename=void>
void Nrm2Impl(SizeT, T const*, SizeT, T*, SyncInfo<Device::GPU> const&)
{
    std::ostringstream oss;
    oss << "No valid implementation of NRM2 for T="
        << TypeTraits<T>::Name();
    throw std::logic_error(oss.str());
}

template <typename T, typename SizeT,
          typename=EnableUnless<IsSupportedType<T,BLAS_Op::SCAL>>,
          typename=void>
void ScaleImpl(SizeT size,
               T const& alpha,
               T* X, SizeT incx,
               SyncInfo<Device::GPU> const& si)
{
    Scale_GPU_impl(size, alpha, X, incx, si);
}

template <typename T, typename SizeT,
          typename=EnableUnless<IsSupportedType<T,BLAS_Op::GEAM>>,
          typename=void>
void Scale2DImpl(SizeT nrows, SizeT ncols,
                 T const& alpha,
                 T* A, SizeT lda,
                 SyncInfo<Device::GPU> const& si)
{
    Scale_GPU_impl(nrows, ncols, alpha, A, lda, si);
}

//
// The fallback runtime error versions
//
// FIXME (TRB): These shouldn't exist.
//

template <typename T, typename SizeT,
          typename=EnableUnless<IsSupportedType<T,BLAS_Op::AXPY>>,
          typename=void>
void AxpyImpl(SizeT const&, T const&,
              T const* const&, SizeT const&,
              T const* const&, SizeT const&,
              SyncInfo<Device::GPU> const&)
{
    std::ostringstream oss;
    oss << "No valid implementation of AXPY for T="
        << TypeTraits<T>::Name();
    throw std::logic_error(oss.str());
}

template <typename T, typename SizeT,
          typename=EnableUnless<IsSupportedType<T, BLAS_Op::GEMV>>,
          typename=EnableWhen<IsSupportedType<T, BLAS_Op::GEMM>>>
void GemvImpl(
    TransposeMode transA,
    SizeT nrows, SizeT ncols,
    T const& alpha,
    T const* A, SizeT lda,
    T const* x, SizeT incx,
    T const& beta,
    T* y, SizeT incy,
    SyncInfo<Device::GPU> const& si)
{
    using NTP = MakePointer<NativeType<T>>;
    using CNTP = MakePointerToConst<NativeType<T>>;

    if (incy != SizeT(1))
        throw std::runtime_error("incy must be 1 right now. "
                                 "Let Tom know you've hit this case.");

    auto const ATrans = transA;
    auto const BTrans = (incx == SizeT(1)
                         ? TransposeMode::NORMAL
                         : TransposeMode::TRANSPOSE);
    auto const m = (ATrans == TransposeMode::NORMAL ? nrows : ncols);
    auto const k = (ATrans == TransposeMode::NORMAL ? ncols : nrows);
    auto const n = SizeT(1);
    auto const LDB = (incx == SizeT(1) ? ncols : incx);
    auto const LDC = nrows;

    SyncManager mgr(GetLibraryHandle(), si);
    gpu_blas_impl::Gemm(
        GetLibraryHandle(),
        ToNativeTransposeMode(ATrans), ToNativeTransposeMode(BTrans),
        ToSizeT(m), ToSizeT(n), ToSizeT(k),
        alpha,
        reinterpret_cast<CNTP>(A), ToSizeT(lda),
        reinterpret_cast<CNTP>(x), ToSizeT(LDB),
        beta,
        reinterpret_cast<NTP>(y), ToSizeT(LDC));
}

template <typename T, typename SizeT,
          typename=EnableUnless<IsSupportedType<T,BLAS_Op::GEMV>>,
          typename=EnableUnless<IsSupportedType<T,BLAS_Op::GEMM>>,
          typename=void>
void GemvImpl(
    TransposeMode const&,
    SizeT const&, SizeT const&,
    T const&,
    T const* const&, SizeT const&,
    T const* const&, SizeT const&,
    T const&,
    T const* const&, SizeT const&,
    SyncInfo<Device::GPU> const&)
{
    std::ostringstream oss;
    oss << "No valid implementation of GEMV for T="
        << TypeTraits<T>::Name();
    throw std::logic_error(oss.str());
}

template <typename T, typename SizeT,
          typename=EnableUnless<IsSupportedType<T, BLAS_Op::HERK>>,
          typename=void>
void HerkImpl(
    FillMode const, TransposeMode const,
    SizeT const, SizeT const,
    TmpBase<T> const& ,
    T const* const, SizeT const,
    TmpBase<T> const& ,
    T* const, SizeT const,
    SyncInfo<Device::GPU> const&)
{
    std::ostringstream oss;
    oss << "No valid implementation of HERK for T="
        << TypeTraits<T>::Name();
    throw std::logic_error(oss.str());
}

template <typename T, typename SizeT,
          typename=EnableUnless<IsSupportedType<T, BLAS_Op::SYRK>>,
          typename=void>
void SyrkImpl(
    FillMode const, TransposeMode const,
    SizeT const, SizeT const,
    T const&,
    T const* const, SizeT const,
    T const&,
    T* const, SizeT const,
    SyncInfo<Device::GPU> const&)
{
    std::ostringstream oss;
    oss << "No valid implementation of SYRK for T="
        << TypeTraits<T>::Name();
    throw std::logic_error(oss.str());
}

template <typename T, typename SizeT,
          typename=EnableUnless<IsSupportedType<T, BLAS_Op::TRSM>>,
          typename=void>
void TrsmImpl(
    SideMode const, FillMode const,
    TransposeMode const, DiagType const,
    SizeT const, SizeT const,
    T const&,
    T const* const, SizeT const,
    T* const, SizeT const,
    SyncInfo<Device::GPU> const&)
{
    std::ostringstream oss;
    oss << "No valid implementation of TRSM for T="
        << TypeTraits<T>::Name();
    throw std::logic_error(oss.str());
}

template <typename T, typename SizeT,
          typename=EnableUnless<IsSupportedType<T,BLAS_Op::GEMM>>,
          typename=void>
void GemmImpl(
    TransposeMode const&, TransposeMode const&,
    SizeT const&, SizeT const&, SizeT const&,
    T const&,
    T const* const&, SizeT const&,
    T const* const&, SizeT const&,
    T const&,
    T const* const&, SizeT const&,
    SyncInfo<Device::GPU> const&)
{
    std::ostringstream oss;
    oss << "No valid implementation of GEMM for T="
        << TypeTraits<T>::Name();
    throw std::logic_error(oss.str());
}

template <
    typename T, typename SizeT, typename StrideT,
    typename=EnableUnless<IsSupportedType<T, BLAS_Op::GEMMSTRIDEDBATCHED>>,
    typename=void>
void GemmStridedBatchedImpl(
    TransposeMode, TransposeMode,
    SizeT, SizeT, SizeT,
    T const&,
    T const*, SizeT, StrideT,
    T const*, SizeT, StrideT,
    T const&,
    T*, SizeT, StrideT,
    SizeT,
    SyncInfo<Device::GPU> const&)
{
    std::ostringstream oss;
    oss << "No valid implementation of GEMMSTRIDEDBATCHED for T="
        << TypeTraits<T>::Name();
    throw std::logic_error(oss.str());
}

template <typename T, typename SizeT,
          typename=EnableUnless<IsSupportedType<T,BLAS_Op::DGMM>>,
          typename=void>
void DgmmImpl(SideMode side,
              SizeT const&, SizeT const&,
              T const* const&, SizeT const&,
              T const* const&, SizeT const&,
              T const* const&, SizeT const&,
              SyncInfo<Device::GPU> const&)
{
    std::ostringstream oss;
    oss << "No valid implementation of DGMM for T="
        << TypeTraits<T>::Name();
    throw std::logic_error(oss.str());
}

}// namespace details

template <typename T, typename SizeT>
void Axpy(SizeT size, T const& alpha,
          T const* X, SizeT incx,
          T* Y, SizeT incy,
          SyncInfo<Device::GPU> const& si)
{
    details::AxpyImpl(size, alpha, X, incx, Y, incy, si);
}

template <typename T, typename SizeT>
void Axpy(SizeT num_rows, SizeT num_cols,
          T const& alpha,
          T const* A, SizeT lda,
          T* B, SizeT ldb,
          SyncInfo<Device::GPU> const& si)
{
    details::Axpy2DImpl(
        num_rows, num_cols, alpha, A, lda, B, ldb, si);
}

template <typename T, typename SizeT>
void Axpy(TransposeMode transpA,
          SizeT num_rows, SizeT num_cols,
          T const& alpha,
          T const* A, SizeT lda,
          T* B, SizeT ldb,
          SyncInfo<Device::GPU> const& si)
{
    details::Axpy2DImplTranspose(
        transpA, num_rows, num_cols, alpha, A, lda, B, ldb, si);
}

template <typename T, typename SizeT>
void Axpy(SizeT num_rows, SizeT num_cols,
          T const& alpha,
          T const* A, SizeT row_stride_A, SizeT lda,
          T* B, SizeT row_stride_B, SizeT ldb,
          SyncInfo<Device::GPU> const& si);

template <typename T, typename SizeT>
void Copy(SizeT size,
          T const* X, SizeT incx,
          T* Y, SizeT incy,
          SyncInfo<Device::GPU> const& si)
{
    details::CopyImpl(size, X, incx, Y, incy, si);
}

template <typename T, typename U, typename SizeT>
void Copy(TransposeMode transA,
          SizeT num_rows, SizeT num_cols,
          T const* A, SizeT lda,
          U* B, SizeT ldb,
          SyncInfo<Device::GPU> const& si)
{
    details::Copy2DImpl(num_rows, num_cols, transA,
                        A, lda, B, ldb, si);
}

template <typename T, typename SizeT>
void Copy(SizeT num_rows, SizeT num_cols,
          T const* A, SizeT row_stride_A, SizeT lda,
          T* B, SizeT row_stride_B, SizeT ldb,
          SyncInfo<Device::GPU> const& si)
{
    // We might get better performance by landing on a vendor-supplied
    // routine if the row_strides are 1.
    if (row_stride_A == SizeT{1} && row_stride_B == SizeT{1})
        details::Copy2DImpl(
            TransposeMode::NORMAL,
            num_rows, num_cols,
            A, lda, B, ldb, si);
    else
        details::Copy2DStridedImpl(
            num_rows, num_cols,
            A, row_stride_A, lda,
            B, row_stride_B, ldb, si);
}

template <typename T, typename SizeT>
void Dot(SizeT num_entries,
         T const* X, SizeT stride_X,
         T const* Y, SizeT stride_Y,
         T* result,
         SyncInfo<Device::GPU> const& syncinfo)
{
    details::DotImpl(num_entries, X, stride_X, Y, stride_Y, result, syncinfo);
}

template <typename T, typename SizeT>
void Nrm2(SizeT num_entries,
          T const* X, SizeT stride_X,
          T* result,
          SyncInfo<Device::GPU> const& syncinfo)
{
    details::Nrm2Impl(num_entries, X, stride_X, result, syncinfo);
}

template <typename T, typename SizeT>
void Scale(SizeT size,
           T const& alpha,
           T* X, SizeT incx,
           SyncInfo<Device::GPU> const& si)
{
    details::ScaleImpl(size, alpha, X, incx, si);
}

template <typename T, typename SizeT>
void Scale(SizeT num_rows, SizeT num_cols,
           T const& alpha,
           T* A, SizeT lda,
           SyncInfo<Device::GPU> const& si)
{
    details::Scale2DImpl(num_rows, num_cols, alpha, A, lda, si);
}

//
// BLAS 2 Routines
//

template <typename T, typename SizeT>
void Gemv(
    TransposeMode transA, SizeT m, SizeT n,
    T const& alpha,
    T const* A, SizeT lda,
    T const* x, SizeT incx,
    T const& beta,
    T* y, SizeT incy,
    SyncInfo<Device::GPU> const& si)
{
    details::GemvImpl(transA,
                      m, n,
                      alpha, A, lda, x, incx,
                      beta, y, incy, si);
}

//
// BLAS 3 Routines
//

template <typename T, typename SizeT>
void Herk(
    FillMode uplo, TransposeMode trans,
    SizeT n, SizeT k,
    TmpBase<T> const& alpha,
    T const* A, SizeT lda,
    TmpBase<T> const& beta,
    T* C, SizeT ldc,
    SyncInfo<Device::GPU> const& syncinfo)
{
    details::HerkImpl(uplo, trans,
                      n, k,
                      alpha, A, lda,
                      beta, C, ldc,
                      syncinfo);
}

template <typename T, typename SizeT>
void Syrk(
    FillMode uplo, TransposeMode trans,
    SizeT n, SizeT k,
    T const& alpha,
    T const* A, SizeT lda,
    T const& beta,
    T* C, SizeT ldc,
    SyncInfo<Device::GPU> const& syncinfo)
{
    details::SyrkImpl(uplo, trans,
                      n, k,
                      alpha, A, lda,
                      beta, C, ldc,
                      syncinfo);
}

template <typename T, typename SizeT>
void Trsm(
    SideMode side, FillMode uplo,
    TransposeMode trans, DiagType diag,
    SizeT m, SizeT n,
    T const& alpha,
    T const* A, SizeT lda,
    T* B, SizeT ldb,
    SyncInfo<Device::GPU> const& syncinfo)
{
    details::TrsmImpl(side, uplo, trans, diag, m, n,
                      alpha, A, lda, B, ldb, syncinfo);
}

template <typename T, typename SizeT>
void Gemm(
    TransposeMode transA, TransposeMode transB,
    SizeT m, SizeT n, SizeT k,
    T const& alpha,
    T const* A, SizeT lda,
    T const* B, SizeT ldb,
    T const& beta,
    T* C, SizeT ldc,
    SyncInfo<Device::GPU> const& si)
{
    details::GemmImpl(transA, transB,
                      m, n, k,
                      alpha, A, lda, B, ldb,
                      beta, C, ldc, si);
}

template <typename T, typename SizeT, typename StrideT>
void GemmStridedBatched(
    TransposeMode transpA, TransposeMode transpB,
    SizeT m, SizeT n, SizeT k,
    T const& alpha,
    T const* A, SizeT lda, StrideT strideA,
    T const* B, SizeT ldb, StrideT strideB,
    T const& beta,
    T* C, SizeT ldc, StrideT strideC,
    SizeT batchCount,
    SyncInfo<Device::GPU> const& si)
{
    details::GemmStridedBatchedImpl(transpA, transpB,
                                    m, n, k,
                                    alpha,
                                    A, lda, strideA,
                                    B, ldb, strideB,
                                    beta,
                                    C, ldc, strideC,
                                    batchCount,
                                    si);
}

//
// BLAS-like Extension Routines
//

template <typename T, typename SizeT>
void Dgmm(SideMode side,
          SizeT m, SizeT n,
          T const* A, SizeT lda,
          T const* X, SizeT incx,
          T* C, SizeT ldc,
          SyncInfo<Device::GPU> const& si)
{
    details::DgmmImpl(side, m, n, A, lda, X, incx, C, ldc, si);
}

}// namespace gpu_blas

namespace gpu_lapack
{
namespace details
{
// These might be unique.
using gpu_lapack_impl::ToSizeT;
using gpu_lapack_impl::GetDenseLibraryHandle;
using gpu_lapack_impl::SyncManager;
using gpu_lapack_impl::IsSupportedType;

// These are probably the same.
using gpu_blas_impl::NativeType;
using gpu_blas_impl::ToNativeDiagType;
using gpu_blas_impl::ToNativeFillMode;
using gpu_blas_impl::ToNativeSideMode;
using gpu_blas_impl::ToNativeTransposeMode;

template <typename T, typename SizeT, typename InfoT,
          typename=EnableWhen<IsSupportedType<T,LAPACK_Op::POTRF>>>
void CholeskyFactorizeImpl(FillMode uplo,
                           SizeT n,
                           T* A, SizeT lda,
                           T* workspace, SizeT workspace_size,
                           InfoT* info,
                           SyncInfo<Device::GPU> const& si)
{
    static_assert(IsSame<InfoT, gpu_lapack_impl::InfoT>::value,
                  "Deduced InfoT must match gpu_lapack_impl::InfoT.");

    using NTP = MakePointer<NativeType<T>>;

    SyncManager mgr(GetDenseLibraryHandle(), si);
    gpu_lapack_impl::Potrf(
        GetDenseLibraryHandle(),
        ToNativeFillMode(uplo), ToSizeT(n),
        reinterpret_cast<NTP>(A), ToSizeT(lda),
        reinterpret_cast<NTP>(workspace), ToSizeT(workspace_size),
        info);
}

template <typename T, typename SizeT,
          typename=EnableWhen<IsSupportedType<T,LAPACK_Op::POTRF>>>
void CholeskyFactorizeImpl(FillMode uplo,
                           SizeT n,
                           T* A, SizeT lda,
                           T* workspace, SizeT workspace_size,
                           SyncInfo<Device::GPU> const& si)
{
    simple_buffer<gpu_lapack_impl::InfoT, Device::GPU> info(1, si);
    CholeskyFactorizeImpl(uplo, n, A, lda,
                          workspace, workspace_size, info.data(), si);
#ifndef EL_RELEASE
    gpu_blas_impl::InfoT host_info;
    Copy1DToHost(info.data(), &host_info, 1, si);
    Synchronize(si);
    if (host_info > gpu_blas_impl::InfoT(0))
        throw std::runtime_error("Cholesky: Matrix not HPD.");
    else if (host_info < gpu::blas_impl::InfoT(0))
        throw std::runtime_error("Cholesky: A parameter is bad.");
#endif // EL_RELEASE
}

template <typename T, typename SizeT,
          typename=EnableWhen<IsSupportedType<T,LAPACK_Op::POTRF>>>
void CholeskyFactorizeImpl(FillMode uplo,
                           SizeT n,
                           T* A, SizeT lda,
                           SyncInfo<Device::GPU> const& si)
{
    SyncManager mgr(
        GetDenseLibraryHandle(), si);
    auto const workspace_size =
        gpu_lapack_impl::GetPotrfWorkspaceSize(
            GetDenseLibraryHandle(),
            ToNativeFillMode(uplo),
            ToSizeT(n), A, ToSizeT(lda));
    simple_buffer<T, Device::GPU> workspace(workspace_size, si);
    CholeskyFactorizeImpl(uplo, ToSizeT(n), A, ToSizeT(lda),
                          workspace.data(), ToSizeT(workspace_size),
                          si);
}

template <typename T, typename SizeT,
          typename=EnableUnless<IsSupportedType<T,LAPACK_Op::POTRF>>,
          typename=void>
void CholeskyFactorizeImpl(FillMode const,
                           SizeT const,
                           T const* const, SizeT const,
                           SyncInfo<Device::GPU> const& si)
{
    std::ostringstream oss;
    oss << "No valid implementation of CholeskyFactorize for T="
        << TypeTraits<T>::Name();
    throw std::logic_error(oss.str());
}

template <typename T, typename SizeT,
          typename=EnableUnless<IsSupportedType<T,LAPACK_Op::POTRF>>,
          typename=void>
void CholeskyFactorizeImpl(FillMode const,
                           SizeT const,
                           T const* const, SizeT const,
                           T const* const, SizeT const,
                           SyncInfo<Device::GPU> const&)
{
    std::ostringstream oss;
    oss << "No valid implementation of CholeskyFactorize for T="
        << TypeTraits<T>::Name();
    throw std::logic_error(oss.str());
}

template <typename T, typename SizeT, typename InfoT,
          typename=EnableUnless<IsSupportedType<T,LAPACK_Op::POTRF>>,
          typename=void>
void CholeskyFactorizeImpl(FillMode const,
                           SizeT const,
                           T const* const, SizeT const,
                           T const* const, SizeT const,
                           InfoT const* const,
                           SyncInfo<Device::GPU> const&)
{
    std::ostringstream oss;
    oss << "No valid implementation of CholeskyFactorize for T="
        << TypeTraits<T>::Name();
    throw std::logic_error(oss.str());
}
}// namespace details

template <typename T, typename SizeT>
void CholeskyFactorize(FillMode uplo,
                       SizeT n,
                       T* A, SizeT lda,
                       SyncInfo<Device::GPU> const& si)
{
    details::CholeskyFactorizeImpl(uplo, n, A, lda, si);
}

template <typename T, typename SizeT>
void CholeskyFactorize(FillMode uplo,
                       SizeT n,
                       T* A, SizeT lda,
                       T* workspace, SizeT workspace_size,
                       SyncInfo<Device::GPU> const& si)
{
    details::CholeskyFactorizeImpl(
        uplo, n, A, lda, workspace, workspace_size, si);
}

template <typename T, typename SizeT, typename InfoT>
void CholeskyFactorize(FillMode uplo,
                       SizeT n,
                       T* A, SizeT lda,
                       T* workspace, SizeT workspace_size,
                       InfoT* info,
                       SyncInfo<Device::GPU> const& si)
{
    details::CholeskyFactorizeImpl(
        uplo, n, A, lda, workspace, workspace_size, info, si);
}

}// namespace gpu_lapack
}// namespace hydrogen
#endif // HYDROGEN_GPU_BLAS_IMPL_HPP_
