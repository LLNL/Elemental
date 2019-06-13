#ifndef HYDROGEN_DEVICE_GPUERROR_HPP_
#define HYDROGEN_DEVICE_GPUERROR_HPP_

#include <stdexcept>

#include <hydrogen/Error.hpp>

namespace hydrogen
{

/** @name ErrorHandling */
///@{

H_ADD_BASIC_EXCEPTION_CLASS(GPUError, std::runtime_error);

///@}
}// namespace
#endif // HYDROGEN_DEVICE_GPUERROR_HPP_
