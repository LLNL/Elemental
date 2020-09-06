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
                     [](S const& a){ return El::Abs(a); }));
}

template <typename S, typename T>
void Abs(Matrix<S, Device::GPU> const& A,
         Matrix<T, Device::GPU>& B)
{
    EntrywiseMap(A, B,
                 [] __device__ (S const& a){
                     return a < S(0) ? -a : a;
                 });
}

}

// Just for our own clarity of what the CHECK is checking.
#define CHECK_HOST(...) CHECK(__VA_ARGS__)
#define CHECK_DEVICE(...) CHECK(__VA_ARGS__)

TEMPLATE_TEST_CASE("Testing hydrogen::EntrywiseMap.",
                   "[blas][utils][gpu]",
                   float, double)
{
    using T = TestType;
    using MatrixType = El::Matrix<T, El::Device::GPU>;
    using CPUMatrixType = El::Matrix<T, El::Device::CPU>;

    MatrixType A, B;
    El::Uniform(A, 16, 16, T(-2), T(1));

    Abs(A, B);

    CHECK_HOST(B.Height() == A.Height());
    CHECK_HOST(B.Width() == A.Width());

    CPUMatrixType Acpu, Bcpu;
    El::Copy(A, Acpu);
    El::Copy(B, Bcpu);

    std::vector<std::tuple<El::Int, El::Int, T, T>> errors;
    for (El::Int col = 0; col < Acpu.Width(); ++col)
        for (El::Int row = 0; row < Acpu.Width(); ++row)
        {
            if (Bcpu.CRef(row, col) != El::Abs(Acpu.CRef(row, col)))
            {
                errors.emplace_back(row, col, Acpu(row, col), Bcpu(row, col));
            }
        }

    auto const num_bad_entries = errors.size();
    REQUIRE(num_bad_entries == 0ULL);
}
