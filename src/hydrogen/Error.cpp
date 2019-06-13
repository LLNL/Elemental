#include <hydrogen/Error.hpp>

namespace hydrogen
{
namespace
{
volatile size_t break_on_me_called_ = 0UL;
}

void break_on_me()
{
    break_on_me_called_ += 1UL;
}

}// namespace hydrogen
