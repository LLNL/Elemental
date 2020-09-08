#include <catch2/catch.hpp>

#include "El.hpp"
#include "hydrogen/blas/gpu/CombineImpl.hpp"

namespace {
using namespace El;

template <typename S, typename T>
void TestAxpy(S alpha,
              Matrix<S, Device::GPU> const& A,
              Matrix<T, Device::GPU>& B)
{
    Combine(A, B,
            [alpha] __device__ (S const& a, T const& b)
            {
                return T(alpha*a) + b;
            });
}
}// namespace <anon>

TEMPLATE_TEST_CASE("Testing hydrogen::Combine.",
                   "[blas][utils][gpu]",
                   float, double)
{
    using T = TestType;
    using MatrixType = El::Matrix<T, El::Device::GPU>;
    using CPUMatrixType = El::Matrix<T, El::Device::CPU>;

    El::Int m = 128, n = 161;
    MatrixType A(m, n, m+3), B(m, n, m+7);
    Fill(A, T(1));
    Fill(B, T(2));

    REQUIRE_NOTHROW(TestAxpy(T(5), A, B));

    CPUMatrixType Bcpu;
    El::Copy(B, Bcpu);

    std::vector<std::tuple<El::Int, El::Int, T>> errors;
    for (El::Int col = 0; col < Bcpu.Width(); ++col)
        for (El::Int row = 0; row < Bcpu.Height(); ++row)
        {
            if (Bcpu.CRef(row, col) != T(7))
            {
                errors.emplace_back(row, col, Bcpu(row, col));
            }
        }

    auto const num_bad_entries = errors.size();
    REQUIRE(num_bad_entries == 0ULL);
}
