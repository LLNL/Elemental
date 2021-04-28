#include <catch2/catch.hpp>

#include "El.hpp"
#include "hydrogen/blas/gpu/EntrywiseMapImpl.hpp"

namespace
{
using namespace El;

template <typename S, typename T>
void Abs(Matrix<S, Device::CPU> const& A,
         Matrix<T, Device::CPU>& B)
{
    EntrywiseMap(A, B,
                 std::function<T(S const&)>(
                     [](S const& a)
                     {
                         return El::Abs(a);
                     }));
}

template <typename S, typename T>
void Abs(Matrix<S, Device::GPU> const& A,
         Matrix<T, Device::GPU>& B)
{
    EntrywiseMap(A, B,
                 [] __device__ (S const& a)
                 {
                     return a < S(0) ? -a : a;
                 });
}

template <typename T>
void Abs(Matrix<El::Complex<T>, Device::GPU> const& A,
         Matrix<T, Device::GPU>& B)
{
    EntrywiseMap(A, B,
                 [] __device__ (thrust::complex<T> const& a)
                 {
                     return thrust::abs(a);
                 });
}

}

// Just for our own clarity of what the CHECK is checking.
#define CHECK_HOST(...) CHECK(__VA_ARGS__)
#define CHECK_DEVICE(...) CHECK(__VA_ARGS__)

TEMPLATE_TEST_CASE("Testing hydrogen::EntrywiseMap -- Real.",
                   "[blas][utils][gpu]",
                   float, double)
{
    using T = TestType;
    using MatrixType = El::Matrix<T, El::Device::GPU>;
    using CPUMatrixType = El::Matrix<T, El::Device::CPU>;

    MatrixType A(137, 171, 151), B;
    El::Fill(A, T(-5));

    REQUIRE_NOTHROW(Abs(A, B));

    // Verify the resize happened
    CHECK_HOST(B.Height() == A.Height());
    CHECK_HOST(B.Width() == A.Width());

    // Verify the operation executed correctly.
    CPUMatrixType Bcpu;
    El::Copy(B, Bcpu);

    std::vector<std::tuple<El::Int, El::Int, T>> errors;
    for (El::Int col = 0; col < Bcpu.Width(); ++col)
        for (El::Int row = 0; row < Bcpu.Height(); ++row)
        {
            if (Bcpu.CRef(row, col) != T(5))
            {
                errors.emplace_back(row, col, Bcpu(row, col));
            }
        }

    auto const num_bad_entries = errors.size();
    REQUIRE(num_bad_entries == 0ULL);
}

TEMPLATE_TEST_CASE("Testing hydrogen::EntrywiseMap -- Complex.",
                   "[blas][utils][gpu]",
                   float, double)
{
    using T = TestType;
    using ComplexT = El::Complex<T>;
    using RealMatrixType = El::Matrix<T, El::Device::GPU>;
    using ComplexMatrixType = El::Matrix<ComplexT, El::Device::GPU>;
    using CPUMatrixType = El::Matrix<T, El::Device::CPU>;

    ComplexMatrixType A(137, 171, 151);
    RealMatrixType B;
    El::Fill(A, ComplexT(T(3), T(4)));

    REQUIRE_NOTHROW(Abs(A, B));

    // Verify the resize happened
    CHECK_HOST(B.Height() == A.Height());
    CHECK_HOST(B.Width() == A.Width());

    // Verify the operation executed correctly.
    CPUMatrixType Bcpu;
    El::Copy(B, Bcpu);

    std::vector<std::tuple<El::Int, El::Int, T>> errors;
    for (El::Int col = 0; col < Bcpu.Width(); ++col)
        for (El::Int row = 0; row < Bcpu.Height(); ++row)
        {
            if (Bcpu.CRef(row, col) != Approx(T(5)))
            {
                errors.emplace_back(row, col, Bcpu(row, col));
                std::cerr << "Problem with row=" << row
                          << ",col=" << col << ": " << Bcpu(row, col)
                          << std::endl;
            }
        }

    auto const num_bad_entries = errors.size();
    REQUIRE(num_bad_entries == 0ULL);
}
