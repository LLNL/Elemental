/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_COPY_HPP
#define EL_BLAS_COPY_HPP

#ifdef _OPENMP
#include <omp.h>
#endif

#include <El/hydrogen_config.h>

#include <El/core/Grid.hpp>
#include <El/blas_like/level1/Copy/internal_decl.hpp>
#include <El/blas_like/level1/Copy/GeneralPurpose.hpp>
#include <El/blas_like/level1/Copy/util.hpp>

#ifdef HYDROGEN_HAVE_GPU
#include <hydrogen/device/gpu/BasicCopy.hpp>
#endif

#include <hydrogen/meta/MetaUtilities.hpp>

// Introduce some metaprogramming notions.
//
// TODO: Move elsewhere.
namespace El
{

template <bool B>
using BoolVT = std::integral_constant<bool, B>;

namespace details
{

/** @brief A simple metafunction for interoping bitwise-equivalent
 *         types across device interfaces.
 */
template <typename T, Device D>
struct CompatibleStorageTypeT
{
    using type = T;
};

template <typename T, Device D>
using CompatibleStorageType = typename CompatibleStorageTypeT<T, D>::type;

#if defined(HYDROGEN_HAVE_HALF) && defined(HYDROGEN_GPU_USE_FP16)

template <>
struct CompatibleStorageTypeT<cpu_half_type, El::Device::GPU>
{
    using type = gpu_half_type;
};

template <>
struct CompatibleStorageTypeT<gpu_half_type, El::Device::CPU>
{
    using type = cpu_half_type;
};

#endif // defined(HYDROGEN_HAVE_HALF) && defined(HYDROGEN_GPU_USE_FP16)

template <typename T>
using CPUStorageType = CompatibleStorageType<T, Device::CPU>;

#ifdef HYDROGEN_HAVE_GPU
template <typename T>
using GPUStorageType = CompatibleStorageType<T, Device::GPU>;
#endif

// This layer of indirection checks the Tgt types and launches the
// copy if possible.
template <typename CopyFunctor,
          typename T, typename U, Device D1, Device D2,
          EnableWhen<IsStorageType<T, D1>, int> = 0>
void LaunchCopy(Matrix<T, D1> const& src, Matrix<U, D2>& tgt,
                CopyFunctor const& F)
{
   return F(src, tgt);
}

template <typename CopyFunctor,
          typename T, typename U, Device D1, Device D2,
          EnableUnless<IsStorageType<T, D1>, int> = 0>
void LaunchCopy(Matrix<T, D1> const&, Matrix<U, D2>&,
                CopyFunctor const&)
{
    LogicError("The combination U=", TypeTraits<U>::Name(), " "
               "and D=", DeviceName<D2>(), " is not supported.");
}

// This layer of indirection checks the Src types; this overload is
// also useful for some DistMatrix instantiations.
template <typename CopyFunctor,
          typename T, typename U, Device D2,
          EnableWhen<IsStorageType<U, D2>, int> = 0>
void LaunchCopy(AbstractMatrix<T> const& src, Matrix<U, D2>& tgt,
                CopyFunctor const& F)
{
    switch (src.GetDevice())
    {
    case Device::CPU:
        return LaunchCopy(
            static_cast<Matrix<T, Device::CPU> const&>(src), tgt, F);
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        return LaunchCopy(
            static_cast<Matrix<T, Device::GPU> const&>(src), tgt, F);
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("Copy: Bad device.");
    }
}

template <typename CopyFunctor,
          typename T, typename U, Device D2,
          EnableUnless<IsStorageType<U, D2>, int> = 0>
void LaunchCopy(AbstractMatrix<T> const&, Matrix<U, D2>&,
                CopyFunctor const&)
{
    LogicError("The combination U=", TypeTraits<U>::Name(), " "
               "and D=", DeviceName<D2>(), " is not supported.");
}

// The variadic templates allow these functors to be recycled across
// sequential and distributed matrices.

struct CopyFunctor
{
    template <typename... Args>
    void operator()(Args&&... args) const
    {
        return Copy(std::forward<Args>(args)...);
    }
};// CopyFunctor

struct CopyAsyncFunctor
{
    template <typename... Args>
    void operator()(Args&&... args) const
    {
        return CopyAsync(std::forward<Args>(args)...);
    }
};// CopyAsyncFunctor

}// namespace details
}// namespace El

//
// Include all the definitions
//
#include "CopyLocal.hpp"
#include "CopyAsyncLocal.hpp"
#include "CopyDistMatrix.hpp"
#include "CopyAsyncDistMatrix.hpp"
#include "CopyFromRoot.hpp"

namespace El
{

void Copy(BaseDistMatrix const&, BaseDistMatrix&);
void CopyAsync(BaseDistMatrix const&, BaseDistMatrix&);

}// namespace El

#endif // ifndef EL_BLAS_COPY_HPP
