/*
This is largely borrowed from tests/blas_like/Gemm.cpp in Elemental,
for which the original license statement reads:

  Copyright (c) 2009-2016, Jack Poulson
  All rights reserved.

  This file is part of Elemental and is under the BSD 2-Clause License,
  which can be found in the LICENSE file in the root directory, or at
  http://opensource.org/licenses/BSD-2-Clause

The borrowed portion is the TestAssociativity function and the setup
portion of TestGemm.
*/
#include <El.hpp>
#include "GemmHelpers/SyncTimer.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <vector>

#ifdef HYDROGEN_HAVE_CUDA
#include <cuda_profiler_api.h>
#endif

using namespace El;

// The basic idea of this file is to run a suite of GEMM experiments
// without reinitializing MPI/CUDA. To provide some notion of
// isolation, individual tests will be bookended by device syncs and
// barriers. Within each test, the kernel under test will be run a few
// times without timing (probably twice), with checks for correctness
// after each warmup. Then the kernel will be run several more times
// in a tight loop (probably 10 times); a timer will record each call
// to give min, max, mean, stddev stats for each kernel.

// A flag type to indicate things are broken.
struct UndefinedType {};

// Helper infrastructure
enum class FloatType {
#ifdef HYDROGEN_HAVE_HALF
    HALF,
#endif // HYDROGEN_HAVE_HALF
    FLOAT,
    DOUBLE,
};// enum class FloatType

Device StringToDevice(std::string const&);
std::string DeviceToString(Device);

FloatType StringToFloatType(std::string const&);
std::string FloatTypeToString(FloatType);

Orientation StringToOrientation(std::string const&);
std::string OrientationToString(Orientation);

GemmAlgorithm StringToGemmAlgorithm(std::string const&);
std::string GemmAlgorithmToString(GemmAlgorithm);

struct Experiment
{
    Experiment(std::string const& dev, std::string const& type,
               std::string const& transA, std::string const& transB,
               std::string const& alg,
               std::string const& m, std::string const& n,
               std::string const& k, std::string const& blk);
    Device device;
    FloatType type;
    Orientation orient_A;
    Orientation orient_B;
    GemmAlgorithm alg;
    size_t m, n, k, nb;
};

using ExperimentSuite = std::vector<Experiment>;
using ExperimentResult = std::vector<long double>;
using ExperimentResults = std::vector<ExperimentResult>;

ExperimentSuite ParseExperimentFile(
    std::string const& filename, mpi::Comm const& comm);

ExperimentResults RunExperiments(ExperimentSuite const&, Grid const&);

void OutputResults(ExperimentSuite const&, ExperimentResults const&,
                   std::string const& output_file, mpi::Comm const&);

template<typename T, Device D>
void TestAssociativity(
    Orientation orientA, Orientation orientB,
    T alpha,
    DistMatrix<T,MC,MR,ELEMENT,D> const& A,
    DistMatrix<T,MC,MR,ELEMENT,D> const & B,
    T beta,
    DistMatrix<T,MC,MR,ELEMENT,D> const& COrig,
    DistMatrix<T,MC,MR,ELEMENT,D> const& CFinal)
{
    EL_DEBUG_ONLY(CallStackEntry cse("TestAssociativity"));

    InitializeRandom(); // Always want the same answer. This isn't a
                        // performance-critical section, so we don't
                        // care about any overhead of this.

    // Test (alpha op(A) op(B) + beta C) X = alpha op(A) (op(B) X) + beta C X
    const Int numRHS = 100;
    const Int n = COrig.Width();
    const Grid& g = A.Grid();

    // Compute Y = alpha op(A) (op(B) X) + beta C X
    DistMatrix<T,MC,MR,ELEMENT,D> X(g), Y(g), Z(g);
    Uniform(X, n, numRHS, T(-0.25f), T(0.25f));
    Gemm(orientB, NORMAL, TypeTraits<T>::One(), B, X, Z);
    Gemm(orientA, NORMAL, alpha, A, Z, Y);
    Gemm(NORMAL, NORMAL, beta, COrig, X, TypeTraits<T>::One(), Y);
    const Base<T> YFrobNorm = FrobeniusNorm(Y);

    // Compute Y = Y - CFinal * X
    T one = TypeTraits<T>::One();
    T neg_one = -one;
    Gemm(NORMAL, NORMAL, neg_one, CFinal, X, one, Y);
    const Base<T> EFrobNorm = FrobeniusNorm(Y);

    PushIndent();
    OutputFromRoot(
        g.Comm(), "|| E ||_F / || Y ||_F = ",
        EFrobNorm, "/", YFrobNorm, " = ", EFrobNorm/YFrobNorm);
    PopIndent();
    flush(std::cout);
}

// Returns the result from a single experiment, which is a vector of
// timers. The warmup runs are not timed; they do perform correctness
// checks. Correctness checks are not done for the timed runs.
template<typename T, Device D>
ExperimentResult TestGemm(
    Orientation orientA, Orientation orientB,
    Int m, Int n, Int k, Int block_size,
    GemmAlgorithm alg,
    const Grid& g)
{
    OutputFromRoot(
        g.Comm(),
        "Testing Gemm",
        OrientationToChar(orientA), OrientationToChar(orientB),
        "_", GemmAlgorithmToString(alg),
        " with ", TypeName<T>(), " on ", DeviceName<D>());
    PushIndent();
    OutputFromRoot(g.Comm(), "M=", m, " N=", n, " K=", k, " NB=", block_size);
    flush(std::cout);

    constexpr size_t num_warmup_runs = 5UL;
    constexpr size_t num_timed_runs = 10UL;

    SetBlocksize(block_size);

    T alpha = T(0.5f), beta = T(-0.5f);

    // Create the matrices
    DistMatrix<T,MC,MR,ELEMENT,D> A(g), B(g), COrig(g), C(g);

    // Decide upon size:
    Int A_rows = (orientA == NORMAL ? m : k);
    Int A_cols = (orientA == NORMAL ? k : m);
    Int B_rows = (orientB == NORMAL ? k : n);
    Int B_cols = (orientB == NORMAL ? n : k);

    // Setup matrices:
    Uniform(A, A_rows, A_cols, T(-0.1f), T(0.1f));
    Uniform(B, B_rows, B_cols, T(-0.1f), T(0.1f));
    Uniform(COrig, m, n, T(-0.1f), T(0.1f));

    // Wait for everything to be all setup.
    mpi::Barrier(g.Comm());
#ifdef HYDROGEN_HAVE_CUDA
    H_CHECK_CUDA(cudaDeviceSynchronize());
#endif // HYDROGEN_HAVE_CUDA

    // Setup the timers
    auto si = SyncInfoFromMatrix(C.LockedMatrix());
    std::vector<helpers::SyncTimer<D>> timers;
    timers.reserve(num_timed_runs);
    for (size_t ii = 0; ii < num_timed_runs; ++ii)
        timers.emplace_back(si);

    // Warmup runs:
    PushIndent();
    OutputFromRoot(g.Comm(), "Correctness tests:");
    for (size_t ii = 0; ii < num_warmup_runs; ++ii)
    {
        C = COrig;
        Gemm(orientA, orientB, alpha, A, B, beta, C, alg);
        TestAssociativity(orientA, orientB, alpha, A, B, beta, COrig, C);
    }
    flush(std::cout);
    PopIndent();

    // Reset some state
    C = COrig;
    mpi::Barrier(g.Comm());
#ifdef HYDROGEN_HAVE_CUDA
    H_CHECK_CUDA(cudaDeviceSynchronize());
#endif // HYDROGEN_HAVE_CUDA

    // Timed runs:
    size_t constexpr num_skips = 2;
    for (size_t ii = 0; ii < num_timed_runs; ++ii)
    {
#pragma unroll
        for (size_t skip_run = 0; skip_run < num_skips; ++skip_run)
            Gemm(orientA, orientB, alpha, A, B, beta, C, alg);

        auto& timer = timers[ii];
        timer.Start();
        Gemm(orientA, orientB, alpha, A, B, beta, C, alg);
        timer.Stop();
    }

    // IDK if I need this, per se.
    Synchronize(si);

    long double mean = 0.f, stddev = 0.f;
    std::vector<long double> times;
    times.reserve(timers.size());
    size_t count = 0;
    for (auto const& t : timers)
    {
        times.push_back(t.GetTime());
        mean += times.back();
    }

    mean = mean / static_cast<long double>(times.size());
    for (auto const& t : times)
        stddev += (t - mean) * (t - mean) / (times.size() - 1);
    stddev = std::sqrt(stddev);

    mpi::Barrier(g.Comm());
#ifdef HYDROGEN_HAVE_CUDA
    H_CHECK_CUDA(cudaDeviceSynchronize());
#endif // HYDROGEN_HAVE_CUDA

    OutputFromRoot(g.Comm(), "Mean: ", mean, "s, StdDev: ", stddev);
    PopIndent();
    OutputFromRoot(g.Comm(), "Finshed.\n");
    flush(std::cout);

    return times;
}

template <>
ExperimentResult TestGemm<UndefinedType, El::Device::CPU>(
    Orientation orientA, Orientation orientB,
    Int m, Int n, Int k, Int block_size,
    GemmAlgorithm alg,
    const Grid& g)
{
  RuntimeError("Invalid type detected.");
  return ExperimentResult{};// silence compiler warning
}

#ifdef HYDROGEN_HAVE_GPU
template <>
ExperimentResult TestGemm<UndefinedType, El::Device::GPU>(
    Orientation orientA, Orientation orientB,
    Int m, Int n, Int k, Int block_size,
    GemmAlgorithm alg,
    const Grid& g)
{
  RuntimeError("Invalid type detected.");
  return ExperimentResult{};// silence compiler warning
}
#endif // HYDROGEN_HAVE_GPU

// Experiment file format:
/*
   DEVICE:TYPE:ORIENT:ORIENT:ALG:M:N:K:BLK_SIZE
*/

int main(int argc, char* argv[])
{
    Environment env(argc, argv);
    mpi::Comm world_comm = mpi::NewWorldComm();

    const std::string input_file = Input("--f", "Experiment config file",
                                         std::string("not_a_thing.ext"));
    const std::string output_file = Input("--o", "Output file",
                                          std::string("also_not_a_thing.ext"));
    int gridHeight = Input("--gridHeight","height of process grid",0);
    volatile int wait = Input("--waitDebug","wait for debugger",0);

    while (wait)
    {
    }

    ProcessInput();
    PrintInputReport();

    // Setup the grid
    if (gridHeight == 0)
        gridHeight = Grid::DefaultHeight(mpi::Size(world_comm));
    const Grid g(std::move(world_comm), gridHeight);

    OutputFromRoot(g.Comm(), "Grid: ", g.Height(), "x", g.Width(),"\n");

    // A communicator we can use here
    mpi::Comm const& comm = g.Comm();

    ExperimentSuite suite = ParseExperimentFile(input_file, comm);

#ifdef HYDROGEN_HAVE_CUDA
    H_CHECK_CUDA(cudaProfilerStart());
#endif

    ExperimentResults results = RunExperiments(suite, g);

#ifdef HYDROGEN_HAVE_CUDA
    H_CHECK_CUDA(cudaProfilerStop());
#endif

    mpi::Barrier(comm);
#ifdef HYDROGEN_HAVE_CUDA
    H_CHECK_CUDA(cudaDeviceSynchronize());
#endif // HYDROGEN_HAVE_CUDA

    OutputResults(suite, results, output_file, comm);

    mpi::Barrier(comm);
#ifdef HYDROGEN_HAVE_CUDA
    H_CHECK_CUDA(cudaDeviceSynchronize());
#endif // HYDROGEN_HAVE_CUDA

    return 0;
}

//
// Helper function definitions
//

Device StringToDevice(std::string const& str)
{
    unsigned char const match_char =
        static_cast<unsigned char>(
            std::toupper(static_cast<unsigned char>(str[0])));
    switch (match_char)
    {
    case 'C':
        return Device::CPU;
#ifdef HYDROGEN_HAVE_GPU
    case 'G':
        return Device::GPU;
#endif // HYDROGEN_HAVE_GPU
    default:
        throw std::runtime_error("Bad device string");
    }
}

std::string DeviceToString(Device D)
{
    switch (D)
    {
    case Device::CPU:
        return "CPU";
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        return "GPU";
#endif // HYDROGEN_HAVE_GPU
    }
    throw std::runtime_error("unrecognized device");
}

FloatType StringToFloatType(std::string const& str)
{
    unsigned char const match_char =
        static_cast<unsigned char>(
            std::toupper(static_cast<unsigned char>(str[0])));
    switch (match_char)
    {
#ifdef HYDROGEN_HAVE_HALF
    case 'H':
        return FloatType::HALF;
#endif // HYDROGEN_HAVE_HALF
    case 'F':
        return FloatType::FLOAT;
    case 'D':
        return FloatType::DOUBLE;
    default:
        throw std::runtime_error("Bad FloatType string");
    }
}

std::string FloatTypeToString(FloatType F)
{
    switch (F)
    {
#ifdef HYDROGEN_HAVE_HALF
    case FloatType::HALF:
        return "half";
#endif // HYDROGEN_HAVE_HALF
    case FloatType::FLOAT:
        return "float";
    case FloatType::DOUBLE:
        return "double";
    }
    return "unknown type"; // silence compiler warning
}

Orientation StringToOrientation(std::string const& str)
{
    char const match_char =
        static_cast<unsigned char>(
            std::toupper(static_cast<unsigned char>(str[0])));
    return CharToOrientation(match_char);
}

std::string OrientationToString(Orientation O)
{
    switch (O)
    {
    case Orientation::NORMAL:
        return "Normal";
    case Orientation::TRANSPOSE:
        return "Transpose";
    case Orientation::ADJOINT:
        return "Adjoint";
    }
    return "Unknown orientation";
}

GemmAlgorithm StringToGemmAlgorithm(std::string const& str)
{
    if (str == "DEFAULT")
        return GEMM_DEFAULT;
    if (str == "SUMMA_A_MS")
        return GEMM_SUMMA_A_MS;
    if (str == "SUMMA_A")
        return GEMM_SUMMA_A;
    if (str == "SUMMA_B_MS")
        return GEMM_SUMMA_B_MS;
    if (str == "SUMMA_B")
        return GEMM_SUMMA_B;
    if (str == "SUMMA_C_MS")
        return GEMM_SUMMA_C_MS;
    if (str == "SUMMA_C")
        return GEMM_SUMMA_C;
    if (str == "SUMMA_DOT")
        return GEMM_SUMMA_DOT;
    if (str == "CANNON")
        return GEMM_CANNON;
    //if (str == "COSMA")
    //    return GEMM_COSMA;

    throw std::runtime_error("Bad Gemm algorithm string");
}

std::string GemmAlgorithmToString(GemmAlgorithm alg)
{
    switch (alg)
    {
    case GEMM_DEFAULT:      return "DEFAULT";
    case GEMM_SUMMA_A_MS:   return "SUMMA_A_MS";
    case GEMM_SUMMA_A:      return "SUMMA_A";
    case GEMM_SUMMA_B_MS:   return "SUMMA_B_MS";
    case GEMM_SUMMA_B:      return "SUMMA_B";
    case GEMM_SUMMA_C_MS:   return "SUMMA_C_MS";
    case GEMM_SUMMA_C:      return "SUMMA_C";
    case GEMM_SUMMA_DOT:    return "SUMMA_DOT";
    case GEMM_CANNON:       return "CANNON";
    //case GEMM_COSMA:       return "COSMA";
    }
    return "Unknown GEMM Algorithm";// silence compiler warning
}

Experiment::Experiment(std::string const& dev,
                       std::string const& flt,
                       std::string const& transA,
                       std::string const& transB,
                       std::string const& algorithm,
                       std::string const& mm,
                       std::string const& nn,
                       std::string const& kk,
                       std::string const& blk)
    : device{StringToDevice(dev)},
      type{StringToFloatType(flt)},
      orient_A{StringToOrientation(transA)},
      orient_B{StringToOrientation(transB)},
      alg{StringToGemmAlgorithm(algorithm)},
      m{std::stoul(mm)},
      n{std::stoul(nn)},
      k{std::stoul(kk)},
      nb{std::stoul(blk)}
{}

ExperimentSuite ParseExperimentFile(
    std::istream& ifs, mpi::Comm const& comm)
{
    enum Fields { DEVICE=1, TYPE=2,
                  ORIENTA=3, ORIENTB=4, ALG=5,
                  M=6, N=7, K=8, NB=9 };
    if (false)//!ifs.good())
    {
        throw std::runtime_error("Bad experiment file");
    }

    std::string line;
    std::smatch match;
    std::regex experiment("([[:alpha:]]+):"
                          "([[:alpha:]]+):"
                          "([[:alpha:]]+):"
                          "([[:alpha:]]+):"
                          "([A-Z_]+):"
                          "([[:alnum:]]+):"
                          "([[:alnum:]]+):"
                          "([[:alnum:]]+):"
                          "([[:alnum:]]+)", std::regex::extended);

    ExperimentSuite experiments;
    while (getline(ifs, line))
    {
        if (regex_search(line, match, experiment))
        {
            experiments.emplace_back(
                match[DEVICE], match[TYPE],
                match[ORIENTA], match[ORIENTB], match[ALG],
                match[M], match[N], match[K], match[NB]);
        }
    }
    return experiments;
}

ExperimentSuite ParseExperimentFile(
    std::string const& filename, mpi::Comm const& comm)
{
    std::ifstream ifs(filename);
    if (!ifs)
        std::cout << "Can't open file \"" << filename << "\""
                  << std::endl;
    return ParseExperimentFile(ifs, comm);
}

#ifdef HYDROGEN_HAVE_HALF
template <Device D>
struct HalfTypeT;

template <>
struct HalfTypeT<Device::CPU>
{
    using type = cpu_half_type;
};

#ifdef HYDROGEN_GPU_USE_FP16
template <>
struct HalfTypeT<Device::GPU>
{
    using type = gpu_half_type;
};
#else
template <>
struct HalfTypeT<Device::GPU>
{
    using type = UndefinedType;
};
#endif // HYDROGEN_HAVE_GPU
#endif // HYDROGEN_HAVE_HALF

template <Device D>
ExperimentResult RunExperiment(Experiment const& exp, Grid const& grid)
{
    switch (exp.type)
    {
#ifdef HYDROGEN_HAVE_HALF
    case FloatType::HALF:
    {
        using half_type = typename HalfTypeT<D>::type;
        return TestGemm<half_type,D>(
            exp.orient_A, exp.orient_B,
            exp.m, exp.n, exp.k, exp.nb,
            exp.alg, grid);
    }
#endif // HYDROGEN_HAVE_HALF
    case FloatType::FLOAT:
        return TestGemm<float,D>(
            exp.orient_A, exp.orient_B,
            exp.m, exp.n, exp.k, exp.nb,
            exp.alg, grid);
    case FloatType::DOUBLE:
        return TestGemm<double,D>(
            exp.orient_A, exp.orient_B,
            exp.m, exp.n, exp.k, exp.nb,
            exp.alg, grid);
    }
    return ExperimentResult{};// silence compiler warning
}

ExperimentResults RunExperiments(
    ExperimentSuite const& suite, Grid const& grid)
{
    ExperimentResults results;
    results.reserve(suite.size());

    for (auto const& exp : suite)
    {
        switch (exp.device)
        {
        case Device::CPU:
            results.emplace_back(RunExperiment<Device::CPU>(exp, grid));
            break;
#ifdef HYDROGEN_HAVE_GPU
        case Device::GPU:
            results.emplace_back(RunExperiment<Device::GPU>(exp, grid));
#endif // HYDROGEN_HAVE_GPU
        }
    }
    return results;
}

void OutputResults(
    ExperimentSuite const& suite, ExperimentResults const& results,
    std::ostream& ofs, mpi::Comm const&)
{
    // FIXME: Gather all results to root and do some sort of reductions.

    // The fields are:
    //   [DEV,TYP,ORA,ORB,ALG,M,N,K,NB,RESULT]

    auto exp_it = suite.cbegin();
    auto res_it = results.cbegin();

    auto exp_end = suite.cend();
    auto res_end = results.cend();

    char sep = ':';
    for ( ; exp_it != exp_end; ++exp_it, ++res_it)
    {
        // Output experiment metadata
        auto const& exp = *exp_it;
        auto const& res = *res_it;

        ofs << DeviceToString(exp.device) << sep
            << FloatTypeToString(exp.type) << sep
            << OrientationToString(exp.orient_A) << sep
            << OrientationToString(exp.orient_B) << sep
            << GemmAlgorithmToString(exp.alg) << sep
            << exp.m << sep
            << exp.n << sep
            << exp.k << sep
            << exp.nb;

        // Output times
        for (auto const& r : res)
            ofs << sep << r;

        // End of record
        ofs << "\n";
    }
    flush(ofs);
}

void OutputResults(
    ExperimentSuite const& suite, ExperimentResults const& results,
    std::string const& output_file, mpi::Comm const& comm)
{
    if (comm.Rank() == 0)
    {
        std::ofstream ofs(output_file);
        if (!ofs)
            throw std::runtime_error("Bad news: " + output_file);
        return OutputResults(suite, results, ofs, comm);
    }
}
