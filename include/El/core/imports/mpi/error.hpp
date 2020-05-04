#pragma once
#ifndef EL_CORE_IMPORTS_MPI_UTILS_HPP_
#define EL_CORE_IMPORTS_MPI_UTILS_HPP_

#include <El/config.h>

// Note (trb): I don't want to include the file that defines
// EL_FUNCTION (include/El/core/environment/decl.hpp) because that's a
// big include for one #define...

#ifdef EL_HAVE_PRETTY_FUNCTION
#define EL_MPI_FUNCTION __PRETTY_FUNCTION__
#else
#define EL_MPI_FUNCTION __func__
#endif

#ifndef EL_RELEASE
#define EL_CHECK_MPI_CALL(cmd)                                  \
    do                                                          \
    {                                                           \
        int error = (cmd);                                      \
        if (error != MPI_SUCCESS)                               \
        {                                                       \
            El::mpi::MpiError(                                  \
                error, EL_MPI_FUNCTION, __FILE__, __LINE__);    \
        }                                                       \
    } while (false)
#else
#define EL_CHECK_MPI_CALL(cmd) cmd
#endif

namespace El
{
namespace mpi
{

/** @brief Retrieve the error description from the MPI implementation.
 *
 *  @param mpi_error_code The error code returned from an MPI call.
 *
 *  @return An @c std::string description of the MPI error.
 */
inline std::string GetErrorString(int mpi_error_code)
{
    char errorString[MPI_MAX_ERROR_STRING];
    int lenghtOfErrorString;
    MPI_Error_string(mpi_error_code, errorString, &lenghtOfErrorString);
    return errorString;
}


/** @class MpiException
 *  @brief An error from an MPI call
 */
class MpiException : public std::runtime_error
{
public:

    /** @brief Construct an MPI exception given the salient pieces of data.
     *
     *  @param mpi_error_code The return-value from an MPI call
     *  @param function The name of the function from which the error has
     *         occurred.
     *  @param file The name of the file in which the error has occurred.
     *  @param line The line number in the file at which the error has
     *         occurred.
     */
    MpiException(int mpi_error_code,
                 std::string const& function,
                 std::string const& file,
                 int line)
        : std::runtime_error{build_string_(
            mpi_error_code, function, file, line)},
          mpi_error_code_{mpi_error_code}
    {}

    int GetMpiErrorCode() const noexcept { return mpi_error_code_; }
private:

    std::string build_string_(
        int error_code, std::string const& func, std::string const& file,
        int line)
    {
        std::ostringstream oss;
        oss << "Detected MPI error:\n"
            << "  Function: " << func
            << "      File: " << file
            << "      Line: " << line
            << "   Message: " << GetErrorString(error_code)
            << "\n";
        return oss.str();
    }

    int mpi_error_code_;
};// class MpiError


/** @brief Indicate an MPI error has occurred.
 *
 *  The main value of this function is providing a hook for debuggers
 *  and an explicit stack frame.
 *
 *  @param mpi_error_code The return-value from an MPI call
 *  @param function The name of the function from which the error has
 *         occurred.
 *  @param file The name of the file in which the error has occurred.
 *  @param line The line number in the file at which the error has
 *         occurred.
 */
inline void MpiError(
    int mpi_error_code, std::string const& function,
    std::string const& file, int line)
{
    El::break_on_me();
    throw MpiException{mpi_error_code, function, file, line};
}

}// namespace mpi
}// namespace El
#endif /* EL_CORE_IMPORTS_MPI_UTILS_HPP_ */
