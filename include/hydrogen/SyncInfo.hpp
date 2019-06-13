#ifndef HYDROGEN_SYNCINFO_HPP_
#define HYDROGEN_SYNCINFO_HPP_

#include <El/hydrogen_config.h>

#include "SyncInfoBase.hpp"

#ifdef HYDROGEN_HAVE_GPU
#include "device/gpu/SyncInfo.hpp"
#endif // HYDROGEN_HAVE_GPU

#endif // HYDROGEN_SYNCINFO_HPP_
