#ifndef HYDROGEN_UTILS_NUMERICTYPECONVERSION_HPP_
#define HYDROGEN_UTILS_NUMERICTYPECONVERSION_HPP_

#include <typeinfo>

/**
 *  @file
 *
 *  Utilities for converting between numeric types (float, double,
 *  int, half).  This is non-trivial when using the half-precision
 *  type.
 *
 *  Allows conversions from type F to type T using To<T>(F const &),
 *  for example To<half>(0) returns a half value of zero.
*/

#ifdef HYDROGEN_HAVE_HALF
#include <half.hpp>
#endif // HYDROGEN_HAVE_HALF

namespace hydrogen
{

#ifdef HYDROGEN_HAVE_HALF

template <typename F, typename T>
struct Caster
{
    static T Cast(F const& x)
    {
        return static_cast<T>(x);
    }
};

template <typename F>
struct Caster<F, half_float::half>
{
    static half_float::half Cast(F const& x)
    {
        return half_float::half_cast<half_float::half>(x);
    }
};

#ifdef HYDROGEN_GPU_USE_FP16
template <typename F>
struct Caster<F, __half>
{
    template <typename DF,
              EnableWhen<std::is_integral<DF>, int> = 0>
    static __half Cast(DF const& x)
    {
        return __half(float(x));
    }

    template <typename DF,
              EnableUnless<std::is_integral<DF>, int> = 0>
    static __half Cast(DF const& x)
    {
        return static_cast<__half>(x);
    }
};

#ifdef HYDROGEN_HAVE_ROCM
template <>
struct Caster<__half, double>
{

    static double Cast(__half const& x)
    {
        return float(x);
    }
};

#endif // HYDROGEN_HAVE_ROCM
#endif // HYDROGEN_GPU_USE_FP16

template <typename T, typename F>
T To(F const& x)
{
    return Caster<F, T>::Cast(x);
}

#else

template <typename T, typename F>
T To(F const& x)
{
    return static_cast<T>(x);
}

#endif

/** @brief Exception class that indicates a narrow_cast failed.
 */
class bad_narrow_cast : public std::bad_cast
{
public:
    const char* what() const noexcept override
    {
        return "bad narrow_cast";
    }
};// class bad_narrow_cast

/** @brief A cast that asserts a safe narrowing conversion.
 *
 *  Throws an exception if the "to-type" can not exactly represent the
 *  given number.
 *
 *  @tparam ToType The type to which the number is converted.
 *  @tparam FromType The type of the input number.
 *
 *  @param[in] value The value to convert.
 *
 *  @throws bad_narrow_cast If the conversion is not exactly represented.
 */
template <typename ToType, typename FromType>
ToType narrow_cast(FromType const& value)
{
    auto result = static_cast<ToType>(value);
    if (static_cast<FromType>(result) != value)
        throw bad_narrow_cast();
    return result;
}
}// namespace hydrogen
#endif // ifndef HYDROGEN_UTILS_NUMERICTYPECONVERSION_HPP_
