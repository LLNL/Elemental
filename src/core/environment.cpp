/*
   Copyright (c) 2009-2016, Jack Poulson
                      2013, Jeff Hammond
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>

#include <El/hydrogen_config.h>

#ifdef HYDROGEN_HAVE_GPU
#include <hydrogen/device/GPU.hpp>
#endif // HYDROGEN_HAVE_GPU

#include <algorithm>
#include <set>

namespace {

El::Int numElemInits = 0;
bool elemInitializedMpi = false;

El::Args* args = 0;

}

namespace El {

void break_on_me() {}

void PrintVersion( ostream& os )
{
    os << "Elemental version information:\n"
       << "  Git revision: " << EL_GIT_SHA1 << "\n"
       << "  Version:      " << EL_VERSION_MAJOR << "."
                             << EL_VERSION_MINOR << "\n"
       << "  Build type:   " << EL_CMAKE_BUILD_TYPE << "\n"
       << endl;
}

void PrintConfig( ostream& os )
{
    os <<
      "Elemental configuration:\n" <<
      "  Math libraries:               " << EL_MATH_LIBS << "\n"
#ifdef EL_HAVE_FLA_BSVD
      "  Have FLAME bidiagonal SVD:    YES\n"
#else
      "  Have FLAME bidiagonal SVD:    NO\n"
#endif
#ifdef EL_HYBRID
      "  Hybrid mode:                  YES\n"
#else
      "  Hybrid mode:                  NO\n"
#endif
#ifdef EL_HAVE_QT5
      "  Have Qt5:                     YES\n"
#else
      "  Have Qt5:                     NO\n"
#endif
#ifdef EL_AVOID_COMPLEX_MPI
      "  Avoiding complex MPI:         YES\n"
#else
      "  Avoiding complex MPI:         NO\n"
#endif
#ifdef EL_USE_BYTE_ALLGATHERS
      "  Use byte AllGathers:          YES\n"
#else
      "  Use byte AllGathers:          NO\n"
#endif
       << endl;
}

void PrintCCompilerInfo( ostream& os )
{
    os << "Elemental's C compiler info:\n"
       << "  EL_CMAKE_C_COMPILER:    " << EL_CMAKE_C_COMPILER << "\n"
       << "  EL_MPI_C_COMPILER:      " << EL_MPI_C_COMPILER << "\n"
       << "  EL_MPI_C_INCLUDE_PATH:  " << EL_MPI_C_INCLUDE_PATH << "\n"
       << "  EL_MPI_C_COMPILE_FLAGS: " << EL_MPI_C_COMPILE_FLAGS << "\n"
       << "  EL_MPI_C_LINK_FLAGS:    " << EL_MPI_C_LINK_FLAGS << "\n"
       << "  EL_MPI_C_LIBRARIES:     " << EL_MPI_C_LIBRARIES << "\n"
       << endl;
}

void PrintCxxCompilerInfo( ostream& os )
{
    os << "Elemental's C++ compiler info:\n"
       << "  EL_CMAKE_CXX_COMPILER:    " << EL_CMAKE_CXX_COMPILER << "\n"
       << "  EL_CXX_FLAGS:             " << EL_CXX_FLAGS << "\n"
       << "  EL_MPI_CXX_COMPILER:      " << EL_MPI_CXX_COMPILER << "\n"
       << "  EL_MPI_CXX_INCLUDE_PATH:  " << EL_MPI_CXX_INCLUDE_PATH << "\n"
       << "  EL_MPI_CXX_COMPILE_FLAGS: " << EL_MPI_CXX_COMPILE_FLAGS << "\n"
       << "  EL_MPI_CXX_LINK_FLAGS:    " << EL_MPI_CXX_LINK_FLAGS << "\n"
       << "  EL_MPI_CXX_LIBRARIES:     " << EL_MPI_CXX_LIBRARIES << "\n"
       << endl;
}

bool Using64BitInt()
{
#ifdef EL_USE_64BIT_INTS
    return true;
#else
    return false;
#endif
}

bool Using64BitBlasInt()
{
#ifdef EL_USE_64BIT_BLAS_INTS
    return true;
#else
    return false;
#endif
}

bool Initialized()
{ return ::numElemInits > 0; }

void Initialize()
{
    int argc=0;
    char** argv=NULL;
    Initialize( argc, argv );
}

#ifdef HYDROGEN_GPU_USE_FP16
namespace
{
// FIXME (trb): move this somewhere better

void GPUHalfSumFunc(void * a, void * b, int * len, MPI_Datatype *) EL_NO_EXCEPT
{
    auto in = static_cast<gpu_half_type const*>(a);
    auto out = static_cast<gpu_half_type*>(b);
    auto const size = *len;
    for (auto ii = decltype(size){0}; ii < size; ++ii)
        out[ii] = float(in[ii]) + float(out[ii]);
}
void GPUHalfProductFunc(
    void * a, void * b, int * len, MPI_Datatype *) EL_NO_EXCEPT
{
    auto in = static_cast<gpu_half_type const*>(a);
    auto out = static_cast<gpu_half_type*>(b);
    auto const size = *len;
    for (auto ii = decltype(size){0}; ii < size; ++ii)
        out[ii] = float(in[ii]) * float(out[ii]);
}
void GPUHalfMaxFunc(void * a, void * b, int * len, MPI_Datatype *) EL_NO_EXCEPT
{
    auto in = static_cast<gpu_half_type const*>(a);
    auto out = static_cast<gpu_half_type*>(b);
    auto const size = *len;
    for (auto ii = decltype(size){0}; ii < size; ++ii)
        if (float(in[ii]) > float(out[ii]))
            out[ii] = in[ii];
}
void GPUHalfMinFunc(void * a, void * b, int * len, MPI_Datatype *) EL_NO_EXCEPT
{
    auto in = static_cast<gpu_half_type const*>(a);
    auto out = static_cast<gpu_half_type*>(b);
    auto const size = *len;
    for (auto ii = decltype(size){0}; ii < size; ++ii)
        if (float(in[ii]) < float(out[ii]))
            out[ii] = in[ii];
}
}// namespace <anon>
#endif // HYDROGEN_GPU_USE_FP16

#ifdef HYDROGEN_HAVE_HALF
namespace
{
// FIXME (trb): move this somewhere better

void HalfSumFunc(void * a, void * b, int * len, MPI_Datatype *) EL_NO_EXCEPT
{
    auto in = static_cast<cpu_half_type const*>(a);
    auto out = static_cast<cpu_half_type*>(b);
    auto const size = *len;
    for (auto ii = decltype(size){0}; ii < size; ++ii)
        out[ii] += in[ii];
}
void HalfProductFunc(void * a, void * b, int * len, MPI_Datatype *) EL_NO_EXCEPT
{
    auto in = static_cast<cpu_half_type const*>(a);
    auto out = static_cast<cpu_half_type*>(b);
    auto const size = *len;
    for (auto ii = decltype(size){0}; ii < size; ++ii)
        out[ii] *= in[ii];
}
void HalfMaxFunc(void * a, void * b, int * len, MPI_Datatype *) EL_NO_EXCEPT
{
    auto in = static_cast<cpu_half_type const*>(a);
    auto out = static_cast<cpu_half_type*>(b);
    auto const size = *len;
    for (auto ii = decltype(size){0}; ii < size; ++ii)
        if (in[ii] > out[ii])
            out[ii] = in[ii];
}
void HalfMinFunc(void * a, void * b, int * len, MPI_Datatype *) EL_NO_EXCEPT
{
    auto in = static_cast<cpu_half_type const*>(a);
    auto out = static_cast<cpu_half_type*>(b);
    auto const size = *len;
    for (auto ii = decltype(size){0}; ii < size; ++ii)
        if (in[ii] < out[ii])
            out[ii] = in[ii];
}
}// namespace <anon>
#endif // HYDROGEN_HAVE_HALF

void Initialize( int& argc, char**& argv )
{
    if( ::numElemInits > 0 )
    {
        ++::numElemInits;
        return;
    }

    ::args = new Args( argc, argv, mpi::COMM_WORLD, std::cerr );

#ifdef HYDROGEN_HAVE_GPU
    gpu::Initialize();
#endif // HYDROGEN_HAVE_GPU

    ::numElemInits = 1;
    if( !mpi::Initialized() )
    {
        if( mpi::Finalized() )
        {
            LogicError
            ("Cannot initialize elemental after finalizing MPI");
        }

        const Int provided =
            mpi::InitializeThread
            ( argc, argv, mpi::THREAD_MULTIPLE );
        const int commRank = mpi::Rank( mpi::COMM_WORLD );
        if( provided != mpi::THREAD_MULTIPLE && commRank == 0 )
        {
            cerr << "WARNING: Could not achieve THREAD_MULTIPLE support."
                 << endl;
        }
        ::elemInitializedMpi = true;
    }
    else
    {
        const Int provided = mpi::QueryThread();
        if( provided != mpi::THREAD_MULTIPLE )
        {
            throw std::runtime_error
            ("MPI initialized with inadequate thread support for Elemental");
        }
    }

#ifdef HYDROGEN_GPU_USE_FP16
    {
        mpi::Types<gpu_half_type>::type = MPI_SHORT;
        mpi::Types<gpu_half_type>::createdType = false;

        bool const commutes = true;
        MPI_Op_create((mpi::UserFunction*)GPUHalfSumFunc, commutes,
                      &mpi::Types<gpu_half_type>::sumOp.op);
        mpi::Types<gpu_half_type>::createdSumOp = true;
        MPI_Op_create((mpi::UserFunction*)GPUHalfProductFunc, commutes,
                      &mpi::Types<gpu_half_type>::prodOp.op);
        mpi::Types<gpu_half_type>::createdProdOp = true;
        MPI_Op_create((mpi::UserFunction*)GPUHalfMaxFunc, commutes,
                      &mpi::Types<gpu_half_type>::maxOp.op);
        mpi::Types<gpu_half_type>::createdMaxOp = true;
        MPI_Op_create((mpi::UserFunction*)GPUHalfMinFunc, commutes,
                      &mpi::Types<gpu_half_type>::minOp.op);
        mpi::Types<gpu_half_type>::createdMinOp = true;
    }
#endif // HYDROGEN_GPU_USE_FP16

#ifdef HYDROGEN_HAVE_HALF
    // FIXME (trb): move this somewhere better
    {
        mpi::Types<cpu_half_type>::type = MPI_SHORT;
        mpi::Types<cpu_half_type>::createdType = false;

        bool const commutes = true;
        MPI_Op_create((mpi::UserFunction*)HalfSumFunc, commutes,
                      &mpi::Types<cpu_half_type>::sumOp.op);
        mpi::Types<cpu_half_type>::createdSumOp = true;
        MPI_Op_create((mpi::UserFunction*)HalfProductFunc, commutes,
                      &mpi::Types<cpu_half_type>::prodOp.op);
        mpi::Types<cpu_half_type>::createdProdOp = true;
        MPI_Op_create((mpi::UserFunction*)HalfMaxFunc, commutes,
                      &mpi::Types<cpu_half_type>::maxOp.op);
        mpi::Types<cpu_half_type>::createdMaxOp = true;
        MPI_Op_create((mpi::UserFunction*)HalfMinFunc, commutes,
                      &mpi::Types<cpu_half_type>::minOp.op);
        mpi::Types<cpu_half_type>::createdMinOp = true;
    }
#endif

#ifdef HYDROGEN_HAVE_CUDA
    cublas::Initialize();
#endif
#ifdef HYDROGEN_HAVE_ROCM
    hydrogen::rocblas::Initialize();
#endif

#ifdef EL_HAVE_QT5
    InitializeQt5( argc, argv );
#endif

    // Queue a default algorithmic blocksize
    EmptyBlocksizeStack();
    PushBlocksizeStack( 128 );

    // Build the default grid
    Grid::InitializeDefault();
    Grid::InitializeTrivial();

#ifdef HYDROGEN_HAVE_QD
    InitializeQD();
#endif

    InitializeRandom();

    // Create the types and ops.
    // mpfr::SetPrecision within InitializeRandom created the BigFloat types
    mpi::CreateCustom();
}

void Finalize()
{
    EL_DEBUG_CSE
    if( ::numElemInits <= 0 )
    {
        cerr << "Finalized Elemental more times than initialized" << endl;
        return;
    }
    --::numElemInits;

    if( mpi::Finalized() )
        cerr << "Warning: MPI was finalized before Elemental." << endl;

    // Reset the global comms
    const_cast<El::mpi::Comm&>(El::mpi::COMM_SELF).Reset();
    const_cast<El::mpi::Comm&>(El::mpi::COMM_WORLD).Reset();

    if( ::numElemInits == 0 )
    {
        delete ::args;
        ::args = 0;

        Grid::FinalizeDefault();
        Grid::FinalizeTrivial();

        // Destroy the types and ops
        mpi::DestroyCustom();

#ifdef EL_HAVE_QT5
        FinalizeQt5();
#endif
        if( ::elemInitializedMpi )
            mpi::Finalize();

        EmptyBlocksizeStack();

#ifdef HYDROGEN_HAVE_QD
        FinalizeQD();
#endif

        FinalizeRandom();
    }

#ifdef HYDROGEN_HAVE_GPU
    gpu::Finalize();
#endif

    EL_DEBUG_ONLY( CloseLog() )
#ifdef HYDROGEN_HAVE_MPC
    if( EL_RUNNING_ON_VALGRIND )
        mpfr_free_cache();
#endif
}

Args& GetArgs()
{
    if( args == 0 )
        throw std::runtime_error("No available instance of Args");
    return *::args;
}

void Args::HandleVersion( ostream& os ) const
{
    string version = "--version";
    char** arg = std::find( argv_, argv_+argc_, version );
    const bool foundVersion = ( arg != argv_+argc_ );
    if( foundVersion )
    {
        if( mpi::Rank() == 0 )
            PrintVersion();
        throw ArgException();
    }
}

void Args::HandleBuild( ostream& os ) const
{
    string build = "--build";
    char** arg = std::find( argv_, argv_+argc_, build );
    const bool foundBuild = ( arg != argv_+argc_ );
    if( foundBuild )
    {
        if( mpi::Rank() == 0 )
        {
            PrintVersion();
            PrintConfig();
            PrintCCompilerInfo();
            PrintCxxCompilerInfo();
        }
        throw ArgException();
    }
}

void ReportException( const exception& e, ostream& os )
{
    try
    {
        const ArgException& argExcept = dynamic_cast<const ArgException&>(e);
        if( string(argExcept.what()) != "" )
            os << argExcept.what() << endl;
        EL_DEBUG_ONLY(DumpCallStack(os))
    }
    catch( UnrecoverableException& recovExcept )
    {
        if( string(e.what()) != "" )
        {
            os << "Process " << mpi::Rank()
               << " caught an unrecoverable exception with message:\n"
               << e.what() << endl;
        }
        EL_DEBUG_ONLY(DumpCallStack(os))
        mpi::Abort( mpi::COMM_WORLD, 1 );
    }
    catch( exception& castExcept )
    {
        if( string(e.what()) != "" )
        {
            os << "Process " << mpi::Rank() << " caught error message:\n"
               << e.what() << endl;
        }
        EL_DEBUG_ONLY(DumpCallStack(os))
    }
}

void ComplainIfDebug()
{
    EL_DEBUG_ONLY(
        if( mpi::Rank() == 0 )
        {
            Output("=======================================================");
            Output(" In debug mode! Do not expect competitive performance! ");
            Output("=======================================================");
        }
    )
}

template<typename T>
bool IsSorted( const vector<T>& x )
{
    const Int vecLength = x.size();
    for( Int i=1; i<vecLength; ++i )
    {
        if( x[i] < x[i-1] )
            return false;
    }
    return true;
}

// While is_strictly_sorted exists in Boost, it does not exist in the STL (yet)
template<typename T>
bool IsStrictlySorted( const vector<T>& x )
{
    const Int vecLength = x.size();
    for( Int i=1; i<vecLength; ++i )
    {
        if( x[i] <= x[i-1] )
            return false;
    }
    return true;
}

void Union
( vector<Int>& both, const vector<Int>& first, const vector<Int>& second )
{
    both.resize( first.size()+second.size() );
    auto it = std::set_union
      ( first.cbegin(),  first.cend(),
        second.cbegin(), second.cend(),
        both.begin() );
    both.resize( Int(it-both.begin()) );
}

vector<Int>
Union( const vector<Int>& first, const vector<Int>& second )
{
    vector<Int> both;
    Union( both, first, second );
    return both;
}

void RelativeIndices
( vector<Int>& relInds, const vector<Int>& sub, const vector<Int>& full )
{
    const Int numSub = sub.size();
    relInds.resize( numSub );
    auto it = full.cbegin();
    for( Int i=0; i<numSub; ++i )
    {
        const Int index = sub[i];
        it = std::lower_bound( it, full.cend(), index );
        EL_DEBUG_ONLY(
          if( it == full.cend() )
              LogicError("Index was not found");
        )
        relInds[i] = Int(it-full.cbegin());
    }
}

vector<Int> RelativeIndices( const vector<Int>& sub, const vector<Int>& full )
{
    vector<Int> relInds;
    RelativeIndices( relInds, sub, full );
    return relInds;
}

Int Find( const vector<Int>& sortedInds, Int index )
{
    EL_DEBUG_CSE
    auto it = std::lower_bound( sortedInds.cbegin(), sortedInds.cend(), index );
    EL_DEBUG_ONLY(
      if( it == sortedInds.cend() )
          LogicError("All indices were smaller");
      if( *it != index )
          LogicError("Could not find index");
    )
    return it - sortedInds.cbegin();
}

#define EL_NO_COMPLEX_PROTO
#define PROTO(T) \
  template bool IsSorted( const vector<T>& x ); \
  template bool IsStrictlySorted( const vector<T>& x );
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#define EL_ENABLE_HALF
#include <El/macros/Instantiate.h>

} // namespace El
