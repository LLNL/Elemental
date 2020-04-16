/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>
using namespace El;

template<typename T,Dist AColDist,Dist ARowDist,Dist BColDist,Dist BRowDist,
         Device ADevice, Device BDevice>
void
Check(DistMatrix<T,AColDist,ARowDist,ELEMENT,ADevice>& A,
      DistMatrix<T,BColDist,BRowDist,ELEMENT,BDevice>& B, bool print)
{
    EL_DEBUG_ONLY(CallStackEntry cse("Check"))
    const Grid& g = A.Grid();

    const Int height = B.Height();
    const Int width = B.Width();
    SyncInfo<Device::CPU> cpu_si;

    OutputFromRoot
    (g.Comm(),
     "Testing [",DistToString(AColDist),",",DistToString(ARowDist),
     ",",DeviceName<ADevice>(),"]",
     " <- [",DistToString(BColDist),",",DistToString(BRowDist),
     ",",DeviceName<BDevice>(),"]");
    Int colAlign = SampleUniform<Int>(0,A.ColStride());
    Int rowAlign = SampleUniform<Int>(0,A.RowStride());
    mpi::Broadcast(colAlign, 0, g.Comm(), cpu_si);
    mpi::Broadcast(rowAlign, 0, g.Comm(), cpu_si);
    A.Align(colAlign, rowAlign);
    A = B;
    if (A.Height() != B.Height() || A.Width() != B.Width())
        LogicError
        ("A ~ ",A.Height()," x ",A.Width(),", B ~ ",B.Height()," x ",B.Width());

    DistMatrix<T,STAR,STAR,ELEMENT,Device::CPU> A_STAR_STAR(A), B_STAR_STAR(B);
    Int myErrorFlag = 0;
    for (Int j=0; j<width; ++j)
    {
        for (Int i=0; i<height; ++i)
        {
            if (A_STAR_STAR.GetLocal(i,j) != B_STAR_STAR.GetLocal(i,j))
            {
                myErrorFlag = 1;
                break;
            }
        }
        if (myErrorFlag != 0)
            break;
    }

    Int summedErrorFlag;
    mpi::AllReduce(&myErrorFlag, &summedErrorFlag, 1, mpi::SUM, g.Comm(), cpu_si);

    if (summedErrorFlag == 0)
    {
        OutputFromRoot(g.Comm(),"PASSED");
        if (print)
            Print(A, "A");
        if (print)
            Print(B, "B");
    }
    else
    {
        OutputFromRoot(g.Comm(),"FAILED (", summedErrorFlag," ranks failed)");
        if (print)
            Print(A, "A");
        if (print)
            Print(B, "B");
        LogicError("Redistribution test failed");
    }
}


template <Device CopyD, typename T, Dist U, Dist V, Device D,
          typename=EnableIf<IsDeviceValidType<T,CopyD>>>
void CheckAll_device(DistMatrix<T,U,V,ELEMENT,D>& A, bool print)
{
    {
      DistMatrix<T,CIRC,CIRC,ELEMENT,CopyD> A_CIRC_CIRC(A.Grid());
      Check(A_CIRC_CIRC, A, print);
    }

    if (D == Device::GPU)
    {
        hydrogen::break_on_me();
    }

    {
      DistMatrix<T,MC,MR,ELEMENT,CopyD> A_MC_MR(A.Grid());
      Check(A_MC_MR, A, print);
    }

    {
      DistMatrix<T,MC,STAR,ELEMENT,CopyD> A_MC_STAR(A.Grid());
      Check(A_MC_STAR, A, print);
    }

    if (D == Device::CPU)
    {
      DistMatrix<T,MD,STAR,ELEMENT,CopyD> A_MD_STAR(A.Grid());
      Check(A_MD_STAR, A, print);
    }
    else
    {
        OutputFromRoot(
            A.Grid().Comm(), "Skipping (MD,STAR) on device \"",
            DeviceName<D>(), "\"...");
    }

    {
      DistMatrix<T,MR,MC,ELEMENT,CopyD> A_MR_MC(A.Grid());
      Check(A_MR_MC, A, print);
    }

    {
      DistMatrix<T,MR,STAR,ELEMENT,CopyD> A_MR_STAR(A.Grid());
      Check(A_MR_STAR, A, print);
    }

    {
      DistMatrix<T,STAR,MC,ELEMENT,CopyD> A_STAR_MC(A.Grid());
      Check(A_STAR_MC, A, print);
    }

    if (D == Device::CPU)
    {
      DistMatrix<T,STAR,MD,ELEMENT,CopyD> A_STAR_MD(A.Grid());
      Check(A_STAR_MD, A, print);
    }
    else
    {
        OutputFromRoot(
            A.Grid().Comm(), "Skipping (STAR,MD) on device \"",
            DeviceName<D>(), "\"...");
    }

    {
      DistMatrix<T,STAR,MR,ELEMENT,CopyD> A_STAR_MR(A.Grid());
      Check(A_STAR_MR, A, print);
    }

    {
      DistMatrix<T,STAR,STAR,ELEMENT,CopyD> A_STAR_STAR(A.Grid());
      Check(A_STAR_STAR, A, print);
    }

    {
      DistMatrix<T,STAR,VC,ELEMENT,CopyD> A_STAR_VC(A.Grid());
      Check(A_STAR_VC, A, print);
    }

    {
      DistMatrix<T,STAR,VR,ELEMENT,CopyD> A_STAR_VR(A.Grid());
      Check(A_STAR_VR, A, print);
    }

    {
      DistMatrix<T,VC,STAR,ELEMENT,CopyD> A_VC_STAR(A.Grid());
      Check(A_VC_STAR, A, print);
    }

    {
      DistMatrix<T,VR,STAR,ELEMENT,CopyD> A_VR_STAR(A.Grid());
      Check(A_VR_STAR, A, print);
    }
}

template <Device CopyD, typename T, Dist U, Dist V, Device D,
          typename=DisableIf<IsDeviceValidType<T,CopyD>>,
          typename=void>
void CheckAll_device(DistMatrix<T,U,V,ELEMENT,D>& A, bool print)
{
    // Do Nothing
}

template<typename T,Dist U,Dist V,Device D>
void CheckAll(Int m, Int n, const Grid& grid, bool print)
{
    DistMatrix<T,U,V,ELEMENT,D> A(grid);
    SyncInfo<Device::CPU> cpu_si;

    Int colAlign = SampleUniform<Int>(0,A.ColStride());
    Int rowAlign = SampleUniform<Int>(0,A.RowStride());
    mpi::Broadcast(colAlign, 0, grid.Comm(), cpu_si);
    mpi::Broadcast(rowAlign, 0, grid.Comm(), cpu_si);
    A.Align(colAlign, rowAlign);

    const T center = 0;
    const Base<T> radius = 5;
    Uniform(A, m, n, center, radius);

    CheckAll_device<Device::CPU>(A, print);

#ifdef HYDROGEN_HAVE_GPU
    CheckAll_device<Device::GPU>(A, print);
#endif
}

template <typename T, Device D, typename=EnableIf<IsDeviceValidType<T,D>>>
void DistMatrixTest_device(Int m, Int n, Grid const& grid, bool print)
{
    CheckAll<T,CIRC,CIRC,D>(m, n, grid, print);
    CheckAll<T,MC,  MR  ,D>(m, n, grid, print);
    CheckAll<T,MC,  STAR,D>(m, n, grid, print);
    CheckAll<T,MR,  MC  ,D>(m, n, grid, print);
    CheckAll<T,MR,  STAR,D>(m, n, grid, print);
    CheckAll<T,STAR,MC  ,D>(m, n, grid, print);
    CheckAll<T,STAR,MR  ,D>(m, n, grid, print);
    CheckAll<T,STAR,STAR,D>(m, n, grid, print);
    CheckAll<T,STAR,VC  ,D>(m, n, grid, print);
    CheckAll<T,STAR,VR  ,D>(m, n, grid, print);
    CheckAll<T,VC,  STAR,D>(m, n, grid, print);
    CheckAll<T,VR,  STAR,D>(m, n, grid, print);
    if (D == Device::CPU)
    {
        CheckAll<T,MD,  STAR,D>(m, n, grid, print);
        CheckAll<T,STAR,MD  ,D>(m, n, grid, print);
    }
    else
    {
        OutputFromRoot(
            grid.Comm(), "Skipping (MD,STAR) and (STAR,MD) on device \"",
            DeviceName<D>(), "\"...");
    }
}

template <typename T, Device D,
          typename=DisableIf<IsDeviceValidType<T,D>>,
          typename=void>
void DistMatrixTest_device(Int m, Int n, Grid const& grid, bool print)
{
    OutputFromRoot(grid.Comm(),"Skipping type ", TypeName<T>(),
                   " on device ", DeviceName<D>());
}

template<typename T>
void
DistMatrixTest(Int m, Int n, const Grid& grid, bool print)
{
    EL_DEBUG_ONLY(CallStackEntry cse("DistMatrixTest"))
    OutputFromRoot(grid.Comm(),"Testing with ",TypeName<T>());

    DistMatrixTest_device<T,Device::CPU>(m,n,grid,print);

#ifdef HYDROGEN_HAVE_GPU
    DistMatrixTest_device<T,Device::GPU>(m,n,grid,print);
#endif
}

int
main(int argc, char* argv[])
{
    Environment env(argc, argv);
    mpi::Comm comm = mpi::NewWorldComm();

    int gridHeight = Input("--gridHeight","height of process grid",0);
    const bool colMajor = Input("--colMajor","column-major ordering?",true);
    const Int m = Input("--height","height of matrix",50);
    const Int n = Input("--width","width of matrix",50);
    const bool print = Input("--print","print wrong matrices?",false);
    const bool debug = Input("--debug","wait for debugger?",false);
    ProcessInput();
    PrintInputReport();

    if (gridHeight == 0)
        gridHeight = Grid::DefaultHeight(mpi::Size(comm));
    const GridOrder order = colMajor ? COLUMN_MAJOR : ROW_MAJOR;
    const Grid grid(std::move(comm), gridHeight, order);

    if (debug)
    {
        volatile int x = 1;
        while (x) {
            hydrogen::break_on_me();
        };
    }

    DistMatrixTest<Int>(m, n, grid, print);

    DistMatrixTest<float>(m, n, grid, print);
    DistMatrixTest<Complex<float>>(m, n, grid, print);

    DistMatrixTest<double>(m, n, grid, print);
    DistMatrixTest<Complex<double>>(m, n, grid, print);

#ifdef EL_HAVE_QD
    DistMatrixTest<DoubleDouble>(m, n, grid, print);
    DistMatrixTest<QuadDouble>(m, n, grid, print);
#endif

#ifdef EL_HAVE_QUAD
    DistMatrixTest<Quad>(m, n, grid, print);
    DistMatrixTest<Complex<Quad>>(m, n, grid, print);
#endif

#ifdef HYDROGEN_HAVE_HALF
    DistMatrixTest<cpu_half_type>(m, n, grid, print);
#endif

#ifdef EL_HAVE_MPC
    DistMatrixTest<BigInt>(m, n, grid, print);
    OutputFromRoot(g.Comm(),"Setting BigInt precision to 512 bits");
    mpfr::SetMinIntBits(512);
    DistMatrixTest<BigInt>(m, n, grid, print);

    DistMatrixTest<BigFloat>(m, n, grid, print);
    DistMatrixTest<Complex<BigFloat>>(m, n, grid, print);
    OutputFromRoot(g.Comm(),"Setting BigFloat precision to 512 bits");
    mpfr::SetPrecision(512);
    DistMatrixTest<BigFloat>(m, n, grid, print);
    DistMatrixTest<Complex<BigFloat>>(m, n, grid, print);
#endif

    return 0;
}
