/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_CHOLESKY_LOWER_VARIANT3_HPP
#define EL_CHOLESKY_LOWER_VARIANT3_HPP

namespace El {
namespace cholesky {

#ifdef HYDROGEN_HAVE_GPU
template <typename F>
void LowerVariant3Unblocked(Matrix<F, Device::GPU>& A)
{
    LocalGPUCholesky(FillMode::LOWER_TRIANGLE, A);
}
#endif // HYDROGEN_HAVE_GPU

template <typename F>
void LowerVariant3Unblocked(Matrix<F,Device::CPU>& A)
{
    EL_DEBUG_CSE;
#ifndef EL_RELEASE
    if(A.Height() != A.Width())
        LogicError("Can only compute Cholesky factor of square matrices");
#endif // EL_RELEASE
    using Real = Base<F>;
    Int const n = A.Height();
    Int const ALDim = A.LDim();
    for (Int j=0; j<n; ++j)
    {
        Real alpha11 = RealPart(A(j,j));
        if (alpha11 <= Real(0))
            throw NonHPDMatrixException("A was not numerically HPD");
        alpha11 = Sqrt(alpha11);
        A(j,j) = alpha11;

        const Int a21Height = n-(j+1);
        F* a21 = A.Buffer(j+1,j );
        F* A22 = A.Buffer(j+1,j+1);

        Real alpha11inv = Real(1)/alpha11;
        blas::Scal(a21Height, alpha11inv, a21, 1);
        blas::Her('L', a21Height, -Real(1), a21, 1, A22, ALDim);
    }
}

template <typename F, Device D>
void LowerVariant3Blocked(Matrix<F,D>& A)
{
    EL_DEBUG_CSE;
#ifndef EL_RELEASE
    if(A.Height() != A.Width())
        LogicError("Can only compute Cholesky factor of square matrices");
#endif // EL_RELEASE
    Int const n = A.Height();
    Int const bsize = Blocksize();
    for (Int k=0; k<n; k+=bsize)
    {
        Int const nb = Min(bsize,n-k);
        Range<Int> const ind1(k,    k+nb),
                         ind2(k+nb, n   );

        auto A11 = A(ind1, ind1);
        auto A21 = A(ind2, ind1);
        auto A22 = A(ind2, ind2);

        cholesky::LowerVariant3Unblocked(A11);
        Trsm(RIGHT, LOWER, ADJOINT, NON_UNIT, F(1), A11, A21);
        Herk(LOWER, NORMAL, Base<F>(-1), A21, Base<F>(1), A22);
    }
}

template <typename F>
void LowerVariant3Blocked(AbstractDistMatrix<F>& APre)
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
      if(APre.Height() != APre.Width())
          LogicError("Can only compute Cholesky factor of square matrices");
   )
    const Grid& grid = APre.Grid();

    DistMatrixReadWriteProxy<F,F,MC,MR> AProx(APre);
    auto& A = AProx.Get();

    DistMatrix<F,STAR,STAR> A11_STAR_STAR(grid);
    DistMatrix<F,VC,  STAR> A21_VC_STAR(grid);
    DistMatrix<F,VR,  STAR> A21_VR_STAR(grid);
    DistMatrix<F,STAR,MC  > A21Trans_STAR_MC(grid);
    DistMatrix<F,STAR,MR  > A21Adj_STAR_MR(grid);

    const Int n = A.Height();
    const Int bsize = Blocksize();
    for(Int k=0; k<n; k+=bsize)
    {
        const Int nb = Min(bsize,n-k);

        const Range<Int> ind1(k,    k+nb),
                         ind2(k+nb, n   );

        auto A11 = A(ind1, ind1);
        auto A21 = A(ind2, ind1);
        auto A22 = A(ind2, ind2);

        A11_STAR_STAR = A11;
        Cholesky(LOWER, A11_STAR_STAR);
        A11 = A11_STAR_STAR;

        A21_VC_STAR.AlignWith(A22);
        A21_VC_STAR = A21;
        LocalTrsm
        (RIGHT, LOWER, ADJOINT, NON_UNIT, F(1), A11_STAR_STAR, A21_VC_STAR);

        A21_VR_STAR.AlignWith(A22);
        A21_VR_STAR = A21_VC_STAR;
        A21Trans_STAR_MC.AlignWith(A22);
        A21Adj_STAR_MR.AlignWith(A22);
        Transpose(A21_VC_STAR, A21Trans_STAR_MC);
        Adjoint(A21_VR_STAR, A21Adj_STAR_MR);

        // (A21^T[* ,MC])^T A21^H[* ,MR] = A21[MC,* ] A21^H[* ,MR]
        //                               = (A21 A21^H)[MC,MR]
        LocalTrrk(
            LOWER, TRANSPOSE,
            F(-1), A21Trans_STAR_MC, A21Adj_STAR_MR, F(1), A22);

        Transpose(A21Trans_STAR_MC, A21);
    }
}

} // namespace cholesky
} // namespace El

#endif // ifndef EL_CHOLESKY_LOWER_VARIANT3_HPP
