#include <catch2/catch.hpp>

// Other includes
#include <El.hpp>
#include <hydrogen/meta/TypeTraits.hpp>

TEMPLATE_TEST_CASE(
  "Testing Interdevice Matrix Copy",
  "[seq][matrix][blas][copy]",
  float, double)
{
    using value_type = TestType;
    using cpu_matrix_type = El::Matrix<value_type, El::Device::CPU>;
    using gpu_matrix_type = El::Matrix<value_type, El::Device::GPU>;
    using size_type = typename cpu_matrix_type::size_type;

    auto const zero_size = hydrogen::TypeTraits<size_type>::Zero();

    SECTION ("Copy host to device.")
    {
        cpu_matrix_type cpu_src(7, 11);
        gpu_matrix_type gpu_tgt;

        CHECK(cpu_src.Height() == 7);
        CHECK(cpu_src.Width() == 11);
        CHECK(gpu_tgt.Height() == zero_size);
        CHECK(gpu_tgt.Width() == zero_size);

        // Set some values
        MakeUniform(cpu_src);

        // Do the copy
        Copy(cpu_src, gpu_tgt);

        CHECK(gpu_tgt.Height() == cpu_src.Height());
        CHECK(gpu_tgt.Width() == cpu_src.Width());

        // This takes a different codepath, for better or worse
        cpu_matrix_type cpu_tgt(gpu_tgt);
        for (size_type col = zero_size; col < cpu_src.Width(); ++col)
            for (size_type row = zero_size; row < cpu_src.Height(); ++row)
            {
                INFO("Row = " << row << "; Col = " << col);
                CHECK(cpu_src(row, col) == cpu_tgt(row, col));
            }
    }

    SECTION ("Copy device to host.")
    {
        gpu_matrix_type gpu_src(13, 11, 17);
        cpu_matrix_type cpu_tgt;

        CHECK(gpu_src.Height() == 13);
        CHECK(gpu_src.Width() == 11);
        CHECK(cpu_tgt.Height() == zero_size);
        CHECK(cpu_tgt.Width() == zero_size);

        // Set some values
        MakeUniform(gpu_src);

        // Do the copy
        Copy(gpu_src, cpu_tgt);

        CHECK(cpu_tgt.Height() == gpu_src.Height());
        CHECK(cpu_tgt.Width() == gpu_src.Width());

        cpu_matrix_type cpu_src(gpu_src);
        for (size_type col = zero_size; col < cpu_src.Width(); ++col)
            for (size_type row = zero_size; row < cpu_src.Height(); ++row)
            {
                INFO("Row = " << row << "; Col = " << col);
                CHECK(cpu_src(row, col) == cpu_tgt(row, col));
            }
    }
}
