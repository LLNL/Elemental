#include <hydrogen/blas/gpu/Copy.hpp>

#include <El/hydrogen_config.h>
#include <hydrogen/meta/TypeTraits.hpp>

#include <hydrogen/device/gpu/GPURuntime.hpp>

namespace hydrogen
{
namespace
{
template <typename SrcT, typename DestT, typename SizeT>
__global__ void copy_1d_kernel(
    SizeT num_entries,
    SrcT const* __restrict__ src, SizeT src_stride,
    DestT* __restrict__ dest, SizeT dest_stride)
{
    SizeT const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_entries)
        dest[idx*dest_stride] = src[idx*src_stride];
}

// FIXME (trb): It's not clear to me that this is the "right" kernel
// for row_stride > 1. Cache performance likely gets trashed such that
// "row_stride" times as many cache misses occur.
template <int TILE_SIZE, int BLK_COLS,
          typename SrcT, typename DestT, typename SizeT>
__global__ void copy_2d_kernel(
    SizeT m, SizeT n,
    SrcT const* __restrict__ src, SizeT src_row_stride, SizeT src_col_stride,
    DestT* __restrict__ dest, SizeT dest_row_stride, SizeT dest_col_stride)
{
    using StorageType = GPUStaticStorageType<SrcT>;
    __shared__ StorageType tile_shared[TILE_SIZE][TILE_SIZE+1];
    auto tile = reinterpret_cast<SrcT(*)[TILE_SIZE+1]>(tile_shared);

    SizeT const start_row = blockIdx.x * TILE_SIZE + threadIdx.x;
    SizeT const start_col = blockIdx.y * TILE_SIZE + threadIdx.y;

    src += start_row*src_row_stride + start_col*src_col_stride;
    dest += start_row*dest_row_stride + start_col*dest_col_stride;
    if (start_row < m && start_col < n)
    {
        if (start_col + TILE_SIZE < n)
        {
            // Load the data
            #pragma unroll
            for (int ii = 0; ii < TILE_SIZE; ii += BLK_COLS)
                tile[threadIdx.y+ii][threadIdx.x] = src[ii*src_col_stride];

            // Store the data
            #pragma unroll
            for (int ii = 0; ii < TILE_SIZE; ii += BLK_COLS)
                dest[ii*dest_col_stride] = tile[threadIdx.y+ii][threadIdx.x];
        }
        else
        {
            // Load the data
            for (int ii = 0;
                 ii < TILE_SIZE && start_col + ii < n; ii += BLK_COLS)
            {
                tile[threadIdx.y+ii][threadIdx.x] = src[ii*src_col_stride];
            }

            // Store the data
            for (int ii = 0;
                 ii < TILE_SIZE && start_col + ii < n; ii += BLK_COLS)
            {
                dest[ii*dest_col_stride] = tile[threadIdx.y+ii][threadIdx.x];
            }
        }
    }
}

template <typename SrcT, typename DestT>
struct CanCopy : std::true_type {};

template <typename SrcRealT, typename DestT>
struct CanCopy<El::Complex<SrcRealT>, DestT>
  : std::false_type
{};

template <typename SrcT, typename DestRealT>
struct CanCopy<SrcT, El::Complex<DestRealT>>
  : std::true_type
{};

template <typename SrcRealT, typename DestRealT>
struct CanCopy<El::Complex<SrcRealT>, El::Complex<DestRealT>>
  : std::true_type
{};

template <typename SrcT, typename DestT, typename SizeT,
          EnableWhen<CanCopy<SrcT, DestT>, int> = 0>
void do_gpu_copy(
    SizeT num_entries,
    SrcT const* src, SizeT src_stride,
    DestT * dest, SizeT dest_stride,
    SyncInfo<Device::GPU> const& sync_info)
{
    if (num_entries <= TypeTraits<SizeT>::Zero())
        return;

#ifdef HYDROGEN_DO_BOUNDS_CHECKING
    // The kernel parameters are __restrict__-ed. This helps ensure
    // that's not a lie.
    void const* max_src = src + src_stride*num_entries;
    void const* max_dest = dest + dest_stride*num_entries;
    if ((dest < max_src) && (src < max_dest))
        throw std::logic_error(
            "Overlapping memory regions are not allowed.");
#endif // HYDROGEN_DO_BOUNDS_CHECKING

    constexpr size_t threads_per_block = 128;
    auto blocks = (num_entries + threads_per_block - 1)/ threads_per_block;

    gpu::LaunchKernel(
        copy_1d_kernel<NativeGPUType<SrcT>, NativeGPUType<DestT>, SizeT>,
        blocks, threads_per_block,
        0, sync_info,
        num_entries,
        AsNativeGPUType(src), src_stride,
        AsNativeGPUType(dest), dest_stride);
}

template <typename SrcT, typename DestT, typename SizeT,
          EnableUnless<CanCopy<SrcT, DestT>, int> = 1>
void do_gpu_copy(SizeT const&, SrcT const* const, SizeT const&,
                 DestT * const, SizeT const&,
                 SyncInfo<Device::GPU> const&)
{
    throw std::logic_error("Cannot copy SrcT to DestT.");
}

template <typename SrcT, typename DestT, typename SizeT,
          EnableWhen<CanCopy<SrcT, DestT>, int> = 0>
void do_gpu_copy(
    SizeT num_rows, SizeT num_cols,
    SrcT const* src, SizeT src_row_stride, SizeT src_col_stride,
    DestT* dest, SizeT dest_row_stride, SizeT dest_col_stride,
    SyncInfo<Device::GPU> const& sync_info)
{
    if (num_rows == 0 || num_cols == 0)
        return;

#ifdef HYDROGEN_DO_BOUNDS_CHECKING
    // The kernel parameters are __restrict__-ed. This helps ensure
    // that's not a lie.
    void const* max_src = src + src_col_stride*num_cols;
    void const* max_dest = dest + dest_col_stride*num_cols;
    if ((dest < max_src) && (src < max_dest))
        throw std::logic_error(
            "Overlapping memory regions are not allowed.");
#endif // HYDROGEN_DO_BOUNDS_CHECKING

    constexpr int TILE_SIZE = 32;
    constexpr int BLK_COLS = 8;

    dim3 blks((num_rows + TILE_SIZE - 1)/TILE_SIZE,
              (num_cols + TILE_SIZE - 1)/TILE_SIZE, 1);
    dim3 thds(TILE_SIZE, BLK_COLS, 1);

    gpu::LaunchKernel(
        copy_2d_kernel<TILE_SIZE, BLK_COLS,
                       NativeGPUType<SrcT>, NativeGPUType<DestT>, SizeT>,
        blks, thds, 0, sync_info,
        num_rows, num_cols,
        AsNativeGPUType(src), src_row_stride, src_col_stride,
        AsNativeGPUType(dest), dest_row_stride, dest_col_stride);
}

template <typename SrcT, typename DestT, typename SizeT,
          EnableUnless<CanCopy<SrcT, DestT>, int> = 1>
void do_gpu_copy(SizeT const&, SizeT const&,
                 SrcT const* const, SizeT const&, SizeT const&,
                 DestT * const, SizeT const&, SizeT const&,
                 SyncInfo<Device::GPU> const&)
{
    throw std::logic_error("Cannot copy SrcT to DestT.");
}

}// namespace <anon>

template <typename SrcT, typename DestT, typename SizeT, typename>
void Copy_GPU_impl(
    SizeT num_entries,
    SrcT const* src, SizeT src_stride,
    DestT * dest, SizeT dest_stride,
    SyncInfo<Device::GPU> const& sync_info)
{
  do_gpu_copy(num_entries, src, src_stride, dest, dest_stride, sync_info);
}

template <typename SrcT, typename DestT, typename SizeT, typename>
void Copy_GPU_impl(
    SizeT num_rows, SizeT num_cols,
    SrcT const* src, SizeT src_row_stride, SizeT src_col_stride,
    DestT* dest, SizeT dest_row_stride, SizeT dest_col_stride,
    SyncInfo<Device::GPU> const& sync_info)
{
    do_gpu_copy(num_rows, num_cols,
                src, src_row_stride, src_col_stride,
                dest, dest_row_stride, dest_col_stride,
                sync_info);
}

#define ETI(SourceType, DestType, SizeType)                          \
    template void Copy_GPU_impl(                                     \
        SizeType, SourceType const*, SizeType,                       \
        DestType*, SizeType, SyncInfo<Device::GPU> const&);          \
    template void Copy_GPU_impl(                                     \
        SizeType, SizeType,                                          \
        SourceType const*, SizeType, SizeType,                       \
        DestType*, SizeType, SizeType, SyncInfo<Device::GPU> const&)

#define ETI_ALL_SIZE_TYPES(SourceType, DestType)        \
    ETI(SourceType, DestType, int);                     \
    ETI(SourceType, DestType, long);                    \
    ETI(SourceType, DestType, long long);               \
    ETI(SourceType, DestType, unsigned);                \
    ETI(SourceType, DestType, size_t)

#ifdef HYDROGEN_GPU_USE_FP16
#define ETI_ALL_TARGETS(SourceType) \
  ETI_ALL_SIZE_TYPES(SourceType, gpu_half_type);        \
  ETI_ALL_SIZE_TYPES(SourceType, float);                \
  ETI_ALL_SIZE_TYPES(SourceType, double);               \
  ETI_ALL_SIZE_TYPES(SourceType, El::Complex<float>);   \
  ETI_ALL_SIZE_TYPES(SourceType, El::Complex<double>)
#else
#define ETI_ALL_TARGETS(SourceType) \
  ETI_ALL_SIZE_TYPES(SourceType, float);                \
  ETI_ALL_SIZE_TYPES(SourceType, double);               \
  ETI_ALL_SIZE_TYPES(SourceType, El::Complex<float>);   \
  ETI_ALL_SIZE_TYPES(SourceType, El::Complex<double>)
#endif

#ifdef HYDROGEN_GPU_USE_FP16
ETI_ALL_TARGETS(gpu_half_type);
#endif
ETI_ALL_TARGETS(float);
ETI_ALL_TARGETS(double);
ETI_ALL_TARGETS(El::Complex<float>);
ETI_ALL_TARGETS(El::Complex<double>);


}// namespace hydrogen
