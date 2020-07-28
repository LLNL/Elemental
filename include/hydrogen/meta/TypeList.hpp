#ifndef HYDROGEN_META_TYPELIST_HPP_
#define HYDROGEN_META_TYPELIST_HPP_

/** @file
 *
 *  Implementation of a typelist. This can be a helpful utility for
 *  some metaprogramming functionality.
 *
 *  @todo Make all SFINAE-friendly
 *  @todo Finish documentation
 *
 *  @ingroup meta_utils
 */
#include "MetaUtilities.hpp"

namespace hydrogen
{

/** @name Typelists */
///@{

// Basic typelist implementation
template <typename... Ts> struct TypeList {};

/** @struct HeadT
 *  @brief A metafunction that returns the first element in the list.
 *
 *  Lisp would call this the "car".
 */
template <typename T>
struct HeadT;

template <typename T, typename... Ts>
struct HeadT<TypeList<T,Ts...>>
{
    using type = T;
};

template <typename T> using Head = typename HeadT<T>::type;
template <typename T> using Car = Head<T>;

/** @struct TailT
 *  @brief A metafunction that returns the tail of a list
 *
 *  The tail of a list is the sublist of the list without the
 *  head. Lisp would call this the "cdr".
 */
template <typename T>
struct TailT;

template <typename T, typename... Ts>
struct TailT<TypeList<T, Ts...>>
{
    using type = TypeList<Ts...>;
};

template <typename T> using Tail = typename TailT<T>::type;
template <typename T> using Cdr = Tail<T>;

/** @struct JoinT
 *  @brief A metafunction that joins two lists into one list.
 */
template <typename List1, typename List2>
struct JoinT;

template <typename... T1s, typename... T2s>
struct JoinT<TypeList<T1s...>,TypeList<T2s...>>
{
    using type = TypeList<T1s..., T2s...>;
};

template <typename List1, typename List2>
using Join = typename JoinT<List1, List2>::type;

/** @class SelectFirstMatch
 *  @brief Metafunction that returns the first match in the list.
 *
 *  This function tests every type in the list against the test type
 *  using the given binary predicate. It returns the first type in the
 *  list that matches. The list is traversed front to back.
 *
 *  @tparam List Must be a typelist of candidate types.
 *  @tparam U The test type.
 *  @tparam Pred Is the binary predicate class that takes Head<List>
 *          and U as arguments.
 */
template <typename List, typename U, template <class,class> class Pred>
struct SelectFirstMatch
    : Select<Pred<U,Head<List>>, HeadT<List>,
             SelectFirstMatch<Tail<List>,U,Pred>>
{};

// Predicate that returns true if Pred<T, X> is true_type for any X in List.
template <typename List, typename T, template <class, class> class Pred>
struct IsTrueForAny;

template <typename T, template <class, class> class Pred>
struct IsTrueForAny<TypeList<>, T, Pred> : std::false_type {};

template <typename List, typename T, template <class, class> class Pred>
struct IsTrueForAny
    : Or<Pred<T,Head<List>>, IsTrueForAny<Tail<List>,T,Pred>>
{};

// Predicate that returns true if Pred<T, X> is true_type for all X in List.
template <typename List, typename T, template <class, class> class Pred>
struct IsTrueForAll;

template <typename T, template <class, class> class Pred>
struct IsTrueForAll<TypeList<>, T, Pred> : std::true_type {};

template <typename List, typename T, template <class, class> class Pred>
struct IsTrueForAll
    : And<Pred<T,Head<List>>, IsTrueForAll<Tail<List>,T,Pred>>
{};

// Add a new element to the front of a typelist
template <typename List, typename T>
struct ConsT;

template <typename List, typename T>
using Cons = typename ConsT<List,T>::type;

// Cons
template <typename T, typename... Ts>
struct ConsT<T, TypeList<Ts...>>
{
    using type = TypeList<T, Ts...>;
};

// Remove all instances of T from the list.
template <typename List, typename T>
struct RemoveAllT;

template <typename List, typename T>
using RemoveAll = typename RemoveAllT<List,T>::type;

// Base Case
template <typename T>
struct RemoveAllT<TypeList<>, T>
{
    using type = TypeList<>;
};

// Match case
template <typename T, typename... Ts>
struct RemoveAllT<TypeList<T, Ts...>, T>
    : RemoveAllT<TypeList<Ts...>, T>
{};

// Recursive call
template <typename S, typename... Ts, typename T>
struct RemoveAllT<TypeList<S, Ts...>, T>
    : ConsT<S, RemoveAll<TypeList<Ts...>, T>>
{};

///@}

}// namespace hydrogen
#endif // HYDROGEN_META_TYPELIST_HPP_
