#ifndef HYDROGEN_META_METAUTILITIES_HPP_
#define HYDROGEN_META_METAUTILITIES_HPP_

/** @file
 *
 *  Implementation of some basic metaprogramming utilities that are
 *  not tied to specific functionality within the library and may see
 *  more general use.
 *
 *  @todo Add unit tests for this stuff.
 *  @todo Finish documentation
 */

/** @defgroup meta_utils Metaprogramming Utilities
 *
 *  A collection of useful metaprogramming utilities that are likely
 *  to see general use throughout the metaprogramming needs of the
 *  library.
 */
#include <type_traits>

namespace hydrogen
{

/** @name Predicate manipulation */
///@{

/** @struct Not
 *  @brief A predicate that inverts the value of the input predicate.
 */
template <typename Predicate>
struct Not
{
    static constexpr bool value = !Predicate::value;
};

// Metafunction for "And"
template <typename... Ts> struct And;
template <> struct And<> : std::true_type {};
template <typename T, typename... Ts>
struct And<T,Ts...>
{
    static constexpr bool value = T::value && And<Ts...>::value;
};

// Metafunction for "Or"
template <typename... Ts> struct Or;
template <> struct Or<> : std::false_type {};
template <typename T, typename... Ts>
struct Or<T,Ts...>
{
    static constexpr bool value = T::value || Or<Ts...>::value;
};

///@}
/** @name SFINAE utilities */
///@{

/** @struct SubstitutionFailure
 *  @brief A type that indicates a substitution has failed.
 */
struct SubstitutionFailure {};

/** @struct SubstitutionSuccess
 *  @brief A type that indicates a substitution was successful.
 */
template <typename T>
struct SubstitutionSuccess : std::true_type {};

/** @struct SubstitutionSuccess<SubstitutionFailure>
 *  @brief A specialization to catch SubstitutionFailure.
 *
 *  This is not always needed but can be a useful mechanism for
 *  certain types of SFINAE-based metaprogramming. See _The C++
 *  Programming Language_ by Stroustrup for examples.
 *
 *  @todo (TRB) Fix the above reference
 */
template <> struct SubstitutionSuccess<SubstitutionFailure>
    : std::false_type
{};

/** @brief Lisp-like "when" half-conditional for std::enable_if SFINAE
 *
 *  "When" clauses only have a "true" branch. That makes this a more
 *  natural name for this behavior.
 */
template <typename T, typename TrueT=void>
using EnableWhen = typename std::enable_if<T::value, TrueT>::type;

/** @brief Lisp-like "unless" half-conditional for std::enable_if SFINAE
 *
 *  "Unless" clauses only have a "false" branch. That makes this a more
 *  natural name for this behavior.
 */
template <typename T, typename FalseT=void>
using EnableUnless = EnableWhen<Not<T>,FalseT>;

///@}
/** @name STL-like convenience functions */
///@{

/** @brief Convenience metafunction for std::add_const. */
template <typename T>
using MakeConst = typename std::add_const<T>::type;

/** @brief Convenience metafunction for std::add_pointer. */
template <typename T>
using MakePointer = typename std::add_pointer<T>::type;

/** @brief Convenience metafunction to make a pointer to const. */
template <typename T>
using MakePointerToConst = MakePointer<MakeConst<T>>;

/** @brief Convenience type predicate to check if two types are the same. */
template <typename T, typename U>
using IsSame = std::is_same<T,U>;

// Wrapper around std::conditional
template <typename B, typename T, typename U>
using Select = typename std::conditional<B::value, T, U>::type;

///@}
/** @brief Metaprogramming with Enums */
///@{

// Metafunction for constexpr enum equality
template <typename EnumT, EnumT A, EnumT B>
struct EnumSame : std::false_type {};
template <typename EnumT, EnumT A>
struct EnumSame<EnumT,A,A> : std::true_type {};

///@}
/** @brief Miscellaneous metafunctions */
///@{

/** @brief Metafunction for determining whether all types match a
 *         given type.
 */
template <typename... Ts> struct AllMatch : std::false_type {};

// Base
template <typename T>
struct AllMatch<T> : std::true_type {};

// Recursive step
template <typename T, typename U, typename... Us>
struct AllMatch<T,U,Us...>
    : And<IsSame<T,U>, AllMatch<T,Us...>>
{};

///@}
}// namespace hydrogen
#endif // HYDROGEN_META_METAUTILITIES_HPP_
