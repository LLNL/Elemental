/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

#ifdef EL_HAVE_QT5
# include "El/io/DisplayWindow-premoc.hpp"
# include "El/io/ComplexDisplayWindow-premoc.hpp"
# include <QApplication>
#endif

namespace El {

void ProcessEvents( int numMsecs )
{
#ifdef EL_HAVE_QT5
    QCoreApplication::instance()->processEvents
    ( QEventLoop::AllEvents, numMsecs );
#endif
}

template <typename Real>
void Display(AbstractMatrix<Real> const& A, std::string title)
{
        switch (A.GetDevice())
    {
    case Device::CPU:
        Display(static_cast<Matrix<Real,Device::CPU> const&>(A), title);
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
    {
        // Copy to the CPU
        Matrix<Real,Device::CPU> A_CPU{
            static_cast<Matrix<Real,Device::GPU> const&>(A)};
        Display(A_CPU, title);
    }
    break;
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("Display: Bad Device type.");
    }
}

template<typename Real>
void Display( const Matrix<Real>& A, string title )
{
    EL_DEBUG_CSE
#ifdef EL_HAVE_QT5
    if( GuiDisabled() )
    {
        Print( A, title );
        return;
    }

    // Convert A to double-precision since Qt's MOC does not support templates
    const Int m = A.Height();
    const Int n = A.Width();
    Matrix<double>* ADouble = new Matrix<double>( m, n );
    for( Int j=0; j<n; ++j )
        for( Int i=0; i<m; ++i )
            ADouble->Set( i, j, double(A.Get(i,j)) );

    QString qTitle = QString::fromStdString( title );
    DisplayWindow* displayWindow = new DisplayWindow;
    displayWindow->Display( ADouble, qTitle );
    displayWindow->show();

    // Spend at most 200 milliseconds rendering
    ProcessEvents( 200 );
#else
    Print( A, title );
#endif
}

template<typename Real>
void Display( const Matrix<Complex<Real>>& A, string title )
{
    EL_DEBUG_CSE
#ifdef EL_HAVE_QT5
    if( GuiDisabled() )
    {
        Print( A, title );
        return;
    }

    // Convert A to double-precision since Qt's MOC does not support templates
    const Int m = A.Height();
    const Int n = A.Width();
    Matrix<Complex<double>>* ADouble = new Matrix<Complex<double>>( m, n );
    for( Int j=0; j<n; ++j )
    {
        for( Int i=0; i<m; ++i )
        {
            const Complex<Real> alpha = A.Get(i,j);
            const Complex<double> alphaDouble =
                Complex<double>(alpha.real(),alpha.imag());
            ADouble->Set( i, j, alphaDouble );
        }
    }

    QString qTitle = QString::fromStdString( title );
    ComplexDisplayWindow* displayWindow = new ComplexDisplayWindow;
    displayWindow->Display( ADouble, qTitle );
    displayWindow->show();

    // Spend at most 200 milliseconds rendering
    ProcessEvents( 200 );
#else
    Print( A, title );
#endif
}

template<typename T>
void Display( const AbstractDistMatrix<T>& A, string title )
{
    EL_DEBUG_CSE
    if( A.ColStride() == 1 && A.RowStride() == 1 )
    {
        if( A.CrossRank() == A.Root() && A.RedundantRank() == 0 )
            Display( A.LockedMatrix(), title );
    }
    else
    {
        DistMatrix<T,CIRC,CIRC> A_CIRC_CIRC( A );
        if( A_CIRC_CIRC.CrossRank() == A_CIRC_CIRC.Root() )
            Display( A_CIRC_CIRC.Matrix(), title );
    }
}


#define PROTO(T) \
  template void Display( const Matrix<T>& A, string title ); \
  template void Display( const AbstractDistMatrix<T>& A, string title );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#define EL_ENABLE_HALF
#include <El/macros/Instantiate.h>

} // namespace El
