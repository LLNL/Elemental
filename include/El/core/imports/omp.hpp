/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_IMPORTS_OMP_HPP
#define EL_IMPORTS_OMP_HPP
#ifdef EL_HYBRID
# include <omp.h>
# if defined(HYDROGEN_HAVE_OMP_TASKLOOP)
#   define EL_PARALLEL_FOR _Pragma("omp taskloop default(shared)")
# else
#   define EL_PARALLEL_FOR _Pragma("omp parallel for")
# endif
# ifdef EL_HAVE_OMP_COLLAPSE
#   if defined(HYDROGEN_HAVE_OMP_TASKLOOP)
#     define EL_PARALLEL_FOR_COLLAPSE2 _Pragma("omp taskloop collapse(2) default(shared)")
#   else
#     define EL_PARALLEL_FOR_COLLAPSE2 _Pragma("omp parallel for collapse(2)")
#   endif
# else
#  define EL_PARALLEL_FOR_COLLAPSE2 EL_PARALLEL_FOR
# endif
# ifdef EL_HAVE_OMP_SIMD
#  define EL_SIMD _Pragma("omp simd")
#  define EL_SIMD_COLLAPSE2 _Pragma("omp simd collapse(2)")
# else
#  define EL_SIMD
#  define EL_SIMD_COLLAPSE2
# endif
#else
# define EL_PARALLEL_FOR
# define EL_PARALLEL_FOR_COLLAPSE2
# define EL_SIMD
#endif

#ifdef EL_PARALLELIZE_INNER_LOOPS
# define EL_INNER_PARALLEL_FOR           EL_PARALLEL_FOR
# define EL_INNER_PARALLEL_FOR_COLLAPSE2 EL_PARALLEL_FOR_COLLAPSE2
# define EL_OUTER_PARALLEL_FOR
# define EL_OUTER_PARALLEL_FOR_COLLAPSE2
#else
# define EL_INNER_PARALLEL_FOR
# define EL_INNER_PARALLEL_FOR_COLLAPSE2
# define EL_OUTER_PARALLEL_FOR           EL_PARALLEL_FOR
# define EL_OUTER_PARALLEL_FOR_COLLAPSE2 EL_PARALLEL_FOR_COLLAPSE2
#endif

#endif // ifndef EL_IMPORTS_OMP_HPP
