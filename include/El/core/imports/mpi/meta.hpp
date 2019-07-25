#pragma once
#ifndef EL_IMPORTS_MPI_META_HPP_
#define EL_IMPORTS_MPI_META_HPP_

#include <El/config.h>
#include <hydrogen/Device.hpp>
#include <hydrogen/SyncInfo.hpp>

#include <type_traits>
#include <utility>

namespace El
{
namespace mpi
{

/** @class IsMpiDeviceValidType
 *  @brief Indicate whether a type is a valid MPI type on a given device.
 *
 *  This predicate defaults to match whether the type is a valid
 *  storage/compute type on the device.
 *
 *  @todo This should be true for valid "storage" types but maybe not
 *        valid for all "compute" types. When the distinction is made
 *        explicit in the code, this should default to the former.
 *
 *  @tparam T The type to test
 *  @tparam D The device to test
 */
template <typename T, Device D>
struct IsMpiDeviceValidType : IsDeviceValidType<T,D> {};

#ifdef HYDROGEN_HAVE_GPU
// Signed integer types
template <>
struct IsMpiDeviceValidType<char, Device::GPU> : std::true_type {};
template <>
struct IsMpiDeviceValidType<signed char, Device::GPU> : std::true_type {};
template <>
struct IsMpiDeviceValidType<int, Device::GPU> : std::true_type {};
template <>
struct IsMpiDeviceValidType<long int, Device::GPU> : std::true_type {};
template <>
struct IsMpiDeviceValidType<short int, Device::GPU> : std::true_type {};
template <>
struct IsMpiDeviceValidType<long long int, Device::GPU> : std::true_type {};

// Unsigned types
template <>
struct IsMpiDeviceValidType<unsigned char, Device::GPU> : std::true_type {};
template <>
struct IsMpiDeviceValidType<unsigned short, Device::GPU> : std::true_type {};
template <>
struct IsMpiDeviceValidType<unsigned int, Device::GPU> : std::true_type {};
template <>
struct IsMpiDeviceValidType<unsigned long int, Device::GPU> : std::true_type {};
template <>
struct IsMpiDeviceValidType<unsigned long long int, Device::GPU>
    : std::true_type {};
#endif // HYDROGEN_HAVE_GPU

#ifdef HYDROGEN_HAVE_ALUMINUM
namespace internal
{

#if __cplusplus < 201402L

/** @class IntegerSequence
 *  @brief A compile time list of integers.
 *
 *  This is provided since we are not fully C++14 compatible.
 *
 *  @tparam IntT The integer type
 *  @tparam Is The integers
 */
template <typename IntT, IntT... Is>
struct IntegerSequence
{
    /** @brief Get the number of elements in @c Is.
     *
     *  This is equivalent to @c sizeof...(Is) and is provided for
     *  compatibility with the STL.
     *
     *  @return The number of elements in @c Is.
     */
    static constexpr size_t size() noexcept
    {
        return sizeof...(Is);
    }
};

/** @brief A convenience alias for IntegerSequence<size_t, Is> */
template <size_t... Is>
using IndexSequence = IntegerSequence<size_t, Is...>;

/** @class MergeIndexSequences
 *  @brief Join two IndexSequences
 *
 *  If @c Seq1 and @c Seq2 are both 1,2,3, the resulting sequence is
 *  1,2,3,4,5,6.
 *
 *  @tparam Seq1 The first sequence
 *  @tparam Seq2 The second sequence
 */
template <typename Seq1, typename Seq2>
struct MergeIndexSequences;

template <size_t... Is1, size_t... Is2>
struct MergeIndexSequences<IndexSequence<Is1...>, IndexSequence<Is2...>>
{
  using type = IndexSequence<Is1..., (Is2 + sizeof...(Is1))...>;
};

/** @class GenerateIndexSequence
 *  @brief Create the sequence from 0 to N, exclusive of N.
 *
 *  @tparam N The upper-bound of the sequence (excluded).
 */
template <size_t N>
struct GenerateIndexSequence
    : MergeIndexSequences<typename GenerateIndexSequence<N/2>::type,
                          typename GenerateIndexSequence<N-N/2>::type>
{};

template <>
struct GenerateIndexSequence<1>
{
  using type = IndexSequence<0>;
};

/** @brief A convenience wrapper around @c GenerateIndexSequence */
template <size_t N>
using MakeIndexSequence = typename GenerateIndexSequence<N>::type;

#else

// Use the C++14 STL features explicitly. Note that we only care about
// index_sequence. If general integer_sequence support is needed, this
// should not be a huge problem.

/** @brief A compile time list of @c size_t.
 *
 *  @tparam Is The list of @c size_ts.
 */
template <size_t ... Is>
using IndexSequence = std::index_sequence<Is...>;

/** @brief Generate a sequence from 0 to N (exclusive of N).
 *
 *  @tparam N The (excluded) upper bound of the sequence.
 */
template <size_t N>
using MakeIndexSequence = std::make_index_sequence<N>;

#endif

/** @class GetIthTypeT
 *  @brief Get a type out of a typelist by index.
 *
 *  @tparam N The index of the type.
 *  @tparam List The typelist.
 */
template <size_t N, typename List>
struct GetIthTypeT
    : GetIthTypeT<N-1, Tail<List>>
{};

template <typename List>
struct GetIthTypeT<0,List>
    : HeadT<List>
{};

/** @brief Convenience function for @c GetIthTypeT. */
template <size_t I, typename List>
using GetIthType = typename GetIthTypeT<I,List>::type;

/** @brief A convenience typedef for representing zero as
 *         a constexpr @c size_t.
 */
using Zero = std::integral_constant<size_t, 0>;

/** @brief A metafunction to add one to a constexpr @c size_t value. */
template <typename V>
struct PlusOne
{
    static constexpr size_t value = 1 + V::value;
};

template <typename T>
struct WrapIntegerConstant
{
    using type = T;
};

template <typename T, typename List>
struct IndexInTypeList
{
    using TrueValue = Zero;
    using FalseValue = PlusOne<IndexInTypeList<T, Tail<List>>>;
    using Type = Select<IsSame<T, Head<List>>,
                        TrueValue,
                        FalseValue>;
    static constexpr size_t value = Type::value;
};

#if 0
/** @class IndexInTypeList
 *  @brief Metafunction to compute the location of a type in
 *         a given typelist.
 *
 *  Error if type is not in the list.
 *
 *  @tparam T the type to find
 *  @tparam List the list of types
 */
template <typename T, typename List>
struct IndexInTypeList
    : IndexInTypeListT<T,List>::type
{};
#endif // 0

}// namespace internal
#endif // HYDROGEN_HAVE_ALUMINUM

}// namespace mpi
}// namespace El
#endif /* EL_IMPORTS_MPI_META_HPP */
