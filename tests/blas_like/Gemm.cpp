/*
  Copyright (c) 2009-2016, Jack Poulson
  All rights reserved.

  This file is part of Elemental and is under the BSD 2-Clause License,
  which can be found in the LICENSE file in the root directory, or at
  http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

#include "GemmHelpers/SyncTimer.hpp"

using namespace El;

template<typename T, Device D>
void TestAssociativity
(Orientation orientA, Orientation orientB,
 T alpha,
 DistMatrix<T,MC,MR,ELEMENT,D> const& A,
 DistMatrix<T,MC,MR,ELEMENT,D> const & B,
 T beta,
 DistMatrix<T,MC,MR,ELEMENT,D> const& COrig,
 DistMatrix<T,MC,MR,ELEMENT,D> const& CFinal,
 bool print)
{
    EL_DEBUG_ONLY(CallStackEntry cse("TestAssociativity"));
    InitializeRandom();
    // Test (alpha op(A) op(B) + beta C) X = alpha op(A) (op(B) X) + beta C X
    const Int numRHS = 100;
    const Int n = COrig.Width();
    const Grid& g = A.Grid();
    DistMatrix<T,MC,MR,ELEMENT,D> X(g), Y(g), Z(g);
    Uniform(X, n, numRHS, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    Gemm(orientB, NORMAL, TypeTraits<T>::One(), B, X, Z);
    Gemm(orientA, NORMAL, alpha, A, Z, Y);
    Gemm(NORMAL, NORMAL, beta, COrig, X, TypeTraits<T>::One(), Y);
    const Base<T> YFrobNorm = FrobeniusNorm(Y);
    if (print)
        Print(Y, "Y := alpha op(A) op(B) + beta C");
    T one = TypeTraits<T>::One();
    T neg_one = -one;
    Gemm(NORMAL, NORMAL, neg_one, CFinal, X, one, Y);
    const Base<T> EFrobNorm = FrobeniusNorm(Y);
    if (print)
        Print(Y, "E");
    OutputFromRoot
        (g.Comm(), "|| E ||_F / || Y ||_F = ",
         EFrobNorm, "/", YFrobNorm, "=", EFrobNorm/YFrobNorm);
}

template<typename T, Device D>
void TestGemm
(Orientation orientA,
 Orientation orientB,
 Int m, Int n, Int k,
 T alpha, T beta,
 const Grid& g,
 bool print, bool correctness,
 Int colAlignA=0, Int rowAlignA=0,
 Int colAlignB=0, Int rowAlignB=0,
 Int colAlignC=0, Int rowAlignC=0)
{
  OutputFromRoot(g.Comm(),"Testing with ",TypeName<T>());
    PushIndent();

    double runTime, realGFlops, gFlops;
    DistMatrix<T,MC,MR,ELEMENT,D> A(g), B(g), COrig(g), C(g);

    A.Align(colAlignA, rowAlignA);
    B.Align(colAlignB, rowAlignB);
    C.Align(colAlignC, rowAlignC);

    if (orientA == NORMAL)
        Gaussian(A, m, k);
    else
        Gaussian(A, k, m);
    if (orientB == NORMAL)
        Gaussian(B, k, n);
    else
        Gaussian(B, n, k);
    Gaussian(COrig, m, n);

#ifdef HYDROGEN_HAVE_GPU
    El::gpu::SynchronizeDevice();
#endif // HYDROGEN_HAVE_GPU

    if (print)
    {
        Print(A, "A");
        Print(B, "B");
        Print(COrig, "COrig");
    }

    helpers::SyncTimer<D> timer(SyncInfoFromMatrix(C.LockedMatrix()));
    float cudaTime;

    // Warmup run -- doesn't matter in CPU land
#ifdef HYDROGEN_HAVE_GPU
    if (D == Device::GPU)
    {
        C = COrig;
        Gemm(orientA, orientB, alpha, A, B, beta, C, GEMM_SUMMA_A);
        mpi::Barrier(g.Comm());
    }
#endif

    // Test the variant of Gemm that keeps A stationary
    for (int ii = 0; ii < 6; ++ii)
    {
        C = COrig;
        OutputFromRoot(g.Comm(),"Stationary A algorithm:");
        PushIndent();
        timer.Reset();
        mpi::Barrier(g.Comm());
        timer.Start();
        Gemm(orientA, orientB, alpha, A, B, beta, C, GEMM_SUMMA_A);
        mpi::Barrier(g.Comm());
        timer.Stop();
        runTime = timer.GetTime();
        realGFlops = 2.*double(m)*double(n)*double(k)/(1.e9*runTime);
        gFlops = (IsComplex<T>::value ? 4*realGFlops : realGFlops);
        OutputFromRoot(
            g.Comm(),"Finished in ",runTime," seconds (",gFlops," GFlop/s)");

        flush(std::cout);

        if (print)
            Print(C, BuildString("C := ",alpha," A B + ",beta," C"));
        if (correctness)
            TestAssociativity(orientA, orientB,
                              alpha, A, B, beta, COrig, C,
                              print);
        PopIndent();

        flush(std::cout);
    }

    // Test the variant of Gemm that keeps B stationary
    for (int ii = 0; ii < 6; ++ii)
    {
        C = COrig;
        OutputFromRoot(g.Comm(),"Stationary B Algorithm:");
        PushIndent();
        timer.Reset();
        mpi::Barrier(g.Comm());
        timer.Start();
        Gemm(orientA, orientB, alpha, A, B, beta, C, GEMM_SUMMA_B);
        mpi::Barrier(g.Comm());
        timer.Stop();
        runTime = timer.GetTime();
        realGFlops = 2.*double(m)*double(n)*double(k)/(1.e9*runTime);
        gFlops = (IsComplex<T>::value ? 4*realGFlops : realGFlops);

        OutputFromRoot(
            g.Comm(),"Finished in ",runTime, " seconds (",gFlops," GFlop/s)");

        if (print)
            Print(C, BuildString("C := ",alpha," A B + ",beta," C"));
        if (correctness)
            TestAssociativity(orientA, orientB,
                              alpha, A, B, beta, COrig, C,
                              print);
        PopIndent();

        flush(std::cout);
    }

    // Test the variant of Gemm that keeps C stationary
    for (int ii = 0; ii < 6; ++ii)
    {
        C = COrig;
        OutputFromRoot(g.Comm(),"Stationary C Algorithm:");
        PushIndent();
        timer.Reset();
        mpi::Barrier(g.Comm());
        timer.Start();
        Gemm(orientA, orientB, alpha, A, B, beta, C, GEMM_SUMMA_C);
        mpi::Barrier(g.Comm());
        timer.Stop();
        runTime = timer.GetTime();
        realGFlops = 2.*double(m)*double(n)*double(k)/(1.e9*runTime);
        gFlops = (IsComplex<T>::value ? 4*realGFlops : realGFlops);

        OutputFromRoot(
            g.Comm(),"Finished in ",runTime," seconds (",gFlops," GFlop/s)");

        if (print)
            Print(C, BuildString("C := ",alpha," A B + ",beta," C"));
        if (correctness)
            TestAssociativity
                (orientA, orientB, alpha, A, B, beta, COrig, C, print);
        PopIndent();

        flush(std::cout);
    }

    if (orientA == NORMAL && orientB == NORMAL)
    {
        for (int ii = 0; ii < 0; ++ii)
        {
            // Test the variant of Gemm for panel-panel dot products
            OutputFromRoot(g.Comm(),"Dot Product Algorithm:");
            PushIndent();
            C = COrig;
            timer.Reset();
            mpi::Barrier(g.Comm());
            timer.Start();
            Gemm(NORMAL, NORMAL, alpha, A, B, beta, C, GEMM_SUMMA_DOT);
            mpi::Barrier(g.Comm());
            timer.Stop();
            runTime = timer.GetTime();
            realGFlops = 2.*double(m)*double(n)*double(k)/(1.e9*runTime);
            gFlops = (IsComplex<T>::value ? 4*realGFlops : realGFlops);
            OutputFromRoot(
                g.Comm(),"Finished in ",runTime," seconds (",gFlops,
                " GFlop/s)");

            if (print)
                Print(C, BuildString("C := ",alpha," A B + ",beta," C"));
            if (correctness)
                TestAssociativity
                    (orientA, orientB, alpha, A, B, beta, COrig, C, print);

            PopIndent();
            flush(std::cout);
        }
    }
    PopIndent();

    flush(std::cout);
}

int
main(int argc, char* argv[])
{
    Environment env(argc, argv);
    mpi::Comm comm = mpi::NewWorldComm();

//    try
//    {
        const bool colMajor = Input("--colMajor","column-major ordering?",true);
        int gridHeight = Input("--gridHeight","height of process grid",0);
        const char transA = Input("--transA","orientation of A: N/T/C",'N');
        const char transB = Input("--transB","orientation of B: N/T/C",'N');
        const Int m = Input("--m","height of result",100);
        const Int n = Input("--n","width of result",100);
        const Int k = Input("--k","inner dimension",100);
        const Int nb = Input("--nb","algorithmic blocksize",96);
        const bool print = Input("--print","print matrices?",false);
        const bool correctness = Input("--correctness","correctness?",true);
        const Int colAlignA = Input("--colAlignA","column align of A",0);
        const Int colAlignB = Input("--colAlignB","column align of B",0);
        const Int colAlignC = Input("--colAlignC","column align of C",0);
        const Int rowAlignA = Input("--rowAlignA","row align of A",0);
        const Int rowAlignB = Input("--rowAlignB","row align of B",0);
        const Int rowAlignC = Input("--rowAlignC","row align of C",0);
        const bool testCPU = El::Input("--testCPU", "test CPU gemm?", true);
        const bool testGPU = El::Input("--testGPU", "test GPU gemm?", false);

        ProcessInput();
        PrintInputReport();

        if (gridHeight == 0)
            gridHeight = Grid::DefaultHeight(mpi::Size(comm));
        const GridOrder order = (colMajor ? COLUMN_MAJOR : ROW_MAJOR);
        const Grid g(std::move(comm), gridHeight, order);
        const Orientation orientA = CharToOrientation(transA);
        const Orientation orientB = CharToOrientation(transB);
        SetBlocksize(nb);

        ComplainIfDebug();
        OutputFromRoot(g.Comm(),"Will test Gemm",transA,transB);

#ifdef HYDROGEN_HAVE_GPU
        if (testGPU)
        {
#if defined HYDROGEN_HAVE_HALF && defined HYDROGEN_GPU_USE_FP16
            TestGemm<gpu_half_type,Device::GPU>
                (orientA, orientB,
                 m, n, k,
                 gpu_half_type(3.f), gpu_half_type(4.f),
                 g,
                 print, correctness,
                 colAlignA, rowAlignA,
                 colAlignB, rowAlignB,
                 colAlignC, rowAlignC);
#endif // defined HYDROGEN_HAVE_HALF && defined HYDROGEN_GPU_USE_FP16
            TestGemm<float,Device::GPU>
                (orientA, orientB,
                 m, n, k,
                 float(3), float(4),
                 g,
                 print, correctness,
                 colAlignA, rowAlignA,
                 colAlignB, rowAlignB,
                 colAlignC, rowAlignC);
            TestGemm<double,Device::GPU>
                (orientA, orientB,
                 m, n, k,
                 double(3), double(4),
                 g,
                 print, correctness,
                 colAlignA, rowAlignA,
                 colAlignB, rowAlignB,
                 colAlignC, rowAlignC);
        }
#else
        (void)testGPU;
#endif
        if (testCPU)
        {
            TestGemm<float,Device::CPU>
                (orientA, orientB,
                 m, n, k,
                 float(3), float(4),
                 g,
                 print, correctness,
                 colAlignA, rowAlignA,
                 colAlignB, rowAlignB,
                 colAlignC, rowAlignC);
            TestGemm<Complex<float>,Device::CPU>
                (orientA, orientB,
                 m, n, k,
                 Complex<float>(3), Complex<float>(4),
                 g,
                 print, correctness,
                 colAlignA, rowAlignA,
                 colAlignB, rowAlignB,
                 colAlignC, rowAlignC);

            TestGemm<double,Device::CPU>
                (orientA, orientB,
                 m, n, k,
                 double(3), double(4),
                 g,
                 print, correctness,
                 colAlignA, rowAlignA,
                 colAlignB, rowAlignB,
                 colAlignC, rowAlignC);
            TestGemm<Complex<double>,Device::CPU>
                (orientA, orientB,
                 m, n, k,
                 Complex<double>(3), Complex<double>(4),
                 g,
                 print, correctness,
                 colAlignA, rowAlignA,
                 colAlignB, rowAlignB,
                 colAlignC, rowAlignC);

#ifdef EL_HAVE_QD
            TestGemm<DoubleDouble,Device::CPU>
                (orientA, orientB,
                 m, n, k,
                 DoubleDouble(3), DoubleDouble(4),
                 g,
                 print, correctness,
                 colAlignA, rowAlignA,
                 colAlignB, rowAlignB,
                 colAlignC, rowAlignC);
            TestGemm<QuadDouble,Device::CPU>
                (orientA, orientB,
                 m, n, k,
                 QuadDouble(3), QuadDouble(4),
                 g,
                 print, correctness,
                 colAlignA, rowAlignA,
                 colAlignB, rowAlignB,
                 colAlignC, rowAlignC);

            TestGemm<Complex<DoubleDouble>,Device::CPU>
                (orientA, orientB,
                 m, n, k,
                 Complex<DoubleDouble>(3), Complex<DoubleDouble>(4),
                 g,
                 print, correctness,
                 colAlignA, rowAlignA,
                 colAlignB, rowAlignB,
                 colAlignC, rowAlignC);
            TestGemm<Complex<QuadDouble>,Device::CPU>
                (orientA, orientB,
                 m, n, k,
                 Complex<QuadDouble>(3), Complex<QuadDouble>(4),
                 g,
                 print, correctness,
                 colAlignA, rowAlignA,
                 colAlignB, rowAlignB,
                 colAlignC, rowAlignC);
#endif

#ifdef HYDROGEN_HAVE_HALF
            TestGemm<cpu_half_type,Device::CPU>
                (orientA, orientB,
                 m, n, k,
                 cpu_half_type(3), cpu_half_type(4),
                 g,
                 print, correctness,
                 colAlignA, rowAlignA,
                 colAlignB, rowAlignB,
                 colAlignC, rowAlignC);
#endif

#ifdef EL_HAVE_QUAD
            TestGemm<Quad,Device::CPU>
                (orientA, orientB,
                 m, n, k,
                 Quad(3), Quad(4),
                 g,
                 print, correctness,
                 colAlignA, rowAlignA,
                 colAlignB, rowAlignB,
                 colAlignC, rowAlignC);
            TestGemm<Complex<Quad>,Device::CPU>
                (orientA, orientB,
                 m, n, k,
                 Complex<Quad>(3), Complex<Quad>(4),
                 g,
                 print, correctness,
                 colAlignA, rowAlignA,
                 colAlignB, rowAlignB,
                 colAlignC, rowAlignC);
#endif

#ifdef EL_HAVE_MPC
            TestGemm<BigFloat,Device::CPU>
                (orientA, orientB,
                 m, n, k,
                 BigFloat(3), BigFloat(4),
                 g,
                 print, correctness,
                 colAlignA, rowAlignA,
                 colAlignB, rowAlignB,
                 colAlignC, rowAlignC);
            TestGemm<Complex<BigFloat>,Device::CPU>
                (orientA, orientB,
                 m, n, k,
                 Complex<BigFloat>(3), Complex<BigFloat>(4),
                 g,
                 print, correctness,
                 colAlignA, rowAlignA,
                 colAlignB, rowAlignB,
                 colAlignC, rowAlignC);
#endif
        }
        //}
    //catch(exception& e) { ReportException(e); }

    return 0;
}
