#ifndef HYDROGEN_ERROR_HPP_
#define HYDROGEN_ERROR_HPP_

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

// "Basic exceptions" are those that are constructible with their
// "what string", similar to std::runtime_error and std::logic_error.

#define H_THROW_BASIC_ASSERT_EXCEPTION(cond,excptn, msg)        \
    do                                                          \
    {                                                           \
        std::ostringstream tbe_oss__;                           \
        tbe_oss__ << "Assertion\n\n"                            \
                  << "    " #cond << "\n\n"                     \
                  << "in function\n\n"                          \
                  << "    " << H_PRETTY_FUNCTION << "\n\n"      \
                  << "failed!\n\n"                              \
                  << "{\n"                                      \
                  << "    File: " << __FILE__ << "\n"           \
                  << "    Line: " << __LINE__ << "\n"           \
                  << "    Mesg: " << msg << "\n"                \
                  << "}\n";                                     \
        ::hydrogen::break_on_me();                              \
        throw excptn(tbe_oss__.str());                          \
    } while (false)

#define H_REPORT_DTOR_EXCEPTION_AND_TERMINATE(excptn)                   \
    do                                                                  \
    {                                                                   \
        std::ostringstream dtor_excpt_oss;                              \
        dtor_excpt_oss << "An exception was detected in a destructor!\n\n" \
                       << "File: " << __FILE__ << "\n"                  \
                       << "Line: " << __LINE__ << "\n"                  \
                       << "Function: " << H_PRETTY_FUNCTION << "\n"     \
                       << "Exception:\n\n" << excptn.what() << "\n\n"   \
                       << "Now calling std::terminate(). Good bye.\n";  \
        std::cerr << dtor_excpt_oss.str() << std::endl;                 \
        ::hydrogen::break_on_me();                                      \
    } while (false)

//
// ASSERTIONS
//

#define H_ASSERT(cond, excptn, msg)                             \
    if (!(cond))                                                \
        H_THROW_BASIC_ASSERT_EXCEPTION(cond, excptn, msg)

#define H_ASSERT_FALSE(cond, excptn, msg)               \
    if (cond)                                           \
        H_THROW_BASIC_ASSERT_EXCEPTION(!(cond), excptn, msg)


//
// Exception classes
//

// Really, "basic exceptions" are just those that have no data and
// forward all their arguments to their parent.
#define H_ADD_BASIC_EXCEPTION_CLASS(name, parent)       \
    struct name : parent                                \
    {                                                   \
        template <typename... Ts>                       \
        name(Ts&&... args)                              \
            : parent(std::forward<Ts>(args)...)         \
        {}                                              \
    }

namespace hydrogen
{

/** @class RuntimeError
 *  @brief The base exception for runtime errors thrown by Hydrogen.
 *
 *  Runtime errors are those that are due to factors external to the
 *  program.
 */
//H_ADD_BASIC_EXCEPTION_CLASS(RuntimeError, std::runtime_error);

/** @class LogicError
 *  @brief The base exception for logic errors thrown by Hydrogen.
 *
 *  Logic errors are those due to factors internal to the program and
 *  are more likely to be preventable than RuntimeErrors.
 */
//H_ADD_BASIC_EXCEPTION_CLASS(LogicError, std::logic_error);

/** @brief A no-op that can be set as a predictable breakpoint in a
 *         debugger.
 */
void break_on_me();

}// namespace hydrogen
#endif /* HYDROGEN_ERROR_HPP_ */
