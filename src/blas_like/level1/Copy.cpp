#include "El/core.hpp"
#include "El/blas_like/level1/Copy.hpp"

static_assert(std::is_integral<El::Int>::value,
              "El::Int should be integral!");

namespace El
{
namespace
{

// For now, I just want to generate the tensor-product of
// {MatrixTypes}^2 for Copy. This will cover most of the usecases in
// LBANN. AFAIK, the integer matrices are always converted to a
// floating-point type before they interact "virtually"; all
// operations on those matrices are dispatched statically.

using MatrixTypes = TypeList<
    float, double, Complex<float>, Complex<double>
#ifdef HYDROGEN_HAVE_HALF
    , cpu_half_type
#endif // HYDROGEN_HAVE_HALF
#ifdef HYDROGEN_GPU_USE_FP16
    , gpu_half_type
#endif // HYDROGEN_GPU_USE_FP16
    >;

template <template <typename> class X, typename... Ts>
using Expand = TypeList<X<Ts>...>;

template <template <typename> class X, typename List>
struct ExpandTLT {};

template <template <typename> class X, typename... Ts>
struct ExpandTLT<X, TypeList<Ts...>>
{
    using type = Expand<X, Ts...>;
};

template <template <typename> class X, typename List>
using ExpandTL = typename ExpandTLT<X, List>::type;

// This is replaced by a generic multiple dispatch engine in
// DiHydrogen; this is a one-off use-case for now, so there's no need
// to backport a robust implementation.
template <typename FunctorT, typename LHSList, typename RHSList>
struct CopyDispatcher
{
    static void Do(FunctorT f,
                   BaseDistMatrix const& src, BaseDistMatrix& tgt)
    {
        using LHead = Head<LHSList>;
        using LTail = Tail<LHSList>;
        if (auto const* ptr = dynamic_cast<LHead const*>(&src))
            return CopyDispatcher<FunctorT, LHSList, RHSList>::DoRHS(
                f, *ptr, tgt);
        else
            return CopyDispatcher<FunctorT, LTail, RHSList>::Do(f, src, tgt);
    }

    template <typename LHSType>
    static void DoRHS(FunctorT f, LHSType const& src, BaseDistMatrix& tgt)
    {
        using RHead = Head<RHSList>;
        using RTail = Tail<RHSList>;
        if (auto* ptr = dynamic_cast<RHead*>(&tgt))
            return f(src, *ptr);
        else
            return CopyDispatcher<FunctorT, LHSList, RTail>::DoRHS(f, src, tgt);
    }
};// struct CopyDispatcher

template <typename FunctorT, typename RHSList>
struct CopyDispatcher<FunctorT, TypeList<>, RHSList>
{
    static void Do(FunctorT const&,
                   BaseDistMatrix const&, BaseDistMatrix const&)
    {
        LogicError("Source matrix type not found.");
    }
};

template <typename FunctorT, typename LHSList>
struct CopyDispatcher<FunctorT, LHSList, TypeList<>>
{
    static void DoRHS(FunctorT const&,
                      BaseDistMatrix const&, BaseDistMatrix const&)
    {
        LogicError("Target matrix type not found.");
    }
};

}// namespace <anon>

void Copy(BaseDistMatrix const& Source, BaseDistMatrix& Target)
{
    using FunctorT = details::CopyFunctor;
    using MatrixTs = ExpandTL<AbstractDistMatrix, MatrixTypes>;
    using Dispatcher = CopyDispatcher<FunctorT, MatrixTs, MatrixTs>;
    FunctorT f;
    return Dispatcher::Do(f, Source, Target);
}

void CopyAsync(BaseDistMatrix const& Source, BaseDistMatrix& Target)
{
    using FunctorT = details::CopyAsyncFunctor;
    using MatrixTs = ExpandTL<AbstractDistMatrix, MatrixTypes>;
    using Dispatcher = CopyDispatcher<FunctorT, MatrixTs, MatrixTs>;
    FunctorT f;
    return Dispatcher::Do(f, Source, Target);
}

}// namespace El
