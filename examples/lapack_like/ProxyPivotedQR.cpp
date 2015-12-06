#include "El.hpp"
using namespace El;

// NOTE: This definition is nearly identical to StandardProxy but is
//       meant to demonstrate how to manually build and use a pivot proxy
template<typename F>
class Proxy
{
private:
    Int numPower_, numOversample_;

public:
    Proxy( Int numPower=1, Int numOversample=10 )
    : numPower_(numPower), numOversample_(numOversample)
    { }

    void operator()
    ( const Matrix<F>& A,
            Matrix<Int>& perm,
            Int numPivots,
            bool smallestFirst=false ) const
    {
        const Int m = A.Height();

        // Generate a Gaussian random matrix
        Matrix<F> Omega;
        Gaussian( Omega, numPivots+numOversample_, m );

        // Form Omega (A A^H)^q A = Omega A (A^H A)^2
        Matrix<F> Y, Z;
        Gemm( NORMAL, NORMAL, F(1), Omega, A, Y );
        for( Int powerIter=0; powerIter<numPower_; ++powerIter )
        {
            Gemm( NORMAL, ADJOINT, F(1), Y, A, Z );
            Gemm( NORMAL, NORMAL, F(1), Z, A, Y );
        }

        QRCtrl<Base<F>> ctrl;
        ctrl.boundRank = true;
        ctrl.maxRank = numPivots;
        ctrl.smallestFirst = smallestFirst;
        Matrix<F> t, d;
        QR( Y, t, d, perm, ctrl );
    }

    void operator()
    ( const ElementalMatrix<F>& APre,
            ElementalMatrix<Int>& perm,
            Int numPivots,
            bool smallestFirst=false ) const
    {
        const Int m = APre.Height();
        const Grid& g = APre.Grid();

        DistMatrixReadProxy<F,F,MC,MR> AProxy( APre );
        auto& A = AProxy.GetLocked();

        // Generate a Gaussian random matrix
        DistMatrix<F> Omega(g);
        Gaussian( Omega, numPivots+numOversample_, m );

        // Form Omega (A A^H)^q A = Omega A (A^H A)^2
        DistMatrix<F> Y(g), Z(g);
        Gemm( NORMAL, NORMAL, F(1), Omega, A, Y );
        for( Int powerIter=0; powerIter<numPower_; ++powerIter )
        {
            Gemm( NORMAL, ADJOINT, F(1), Y, A, Z );
            Gemm( NORMAL, NORMAL, F(1), Z, A, Y );
        }

        QRCtrl<Base<F>> ctrl;
        ctrl.boundRank = true;
        ctrl.maxRank = numPivots;
        ctrl.smallestFirst = smallestFirst;
        DistMatrix<F,MD,STAR> t(g), d(g);
        QR( Y, t, d, perm, ctrl );
    }
};

int main( int argc, char* argv[] )
{
    Environment env( argc, argv );
    const int commRank = mpi::Rank(mpi::COMM_WORLD);

    try
    {
        const Int m = Input("--m","matrix height",300);
        const Int n = Input("--n","matrix width",300);
        const Int nb = Input("--nb","blocksize",32);
        const bool panelPiv = Input("--panelPiv","panel pivoting?",false);
        const Int oversample = Input("--oversample","oversample factor",10);
        const Int numPower = Input("--numPower","# of power iterations",1);
        const bool smallestFirst =
          Input("--smallestFirst","smallest norms first?",false);
        const bool print = Input("--print","print matrices?",false);
        ProcessInput();
        PrintInputReport();

        SetBlocksize( nb );

        DistMatrix<double> A;
        Uniform( A, m, n );
        auto ACopy = A;
        if( print )
            Print( A, "A" );
        Timer timer;

        if( commRank == 0 )
            timer.Start();
        DistMatrix<double> t, d;
        DistMatrix<Int,VC,STAR> p;
        Proxy<double> prox(numPower,oversample);
        qr::ProxyHouseholder( A, t, d, p, prox, panelPiv, smallestFirst ); 
        if( commRank == 0 ) 
            Output("Proxy QR time: ",timer.Stop()," seconds");
        if( print )
        {
            Print( A, "QR" );
            Print( t, "t" );
            Print( d, "d" );
            Print( p, "p" );
        }
        DistMatrix<double,MD,STAR> diagR;
        GetDiagonal( A, diagR );
        Print( diagR, "diag(R)" );

        A = ACopy;
        if( commRank == 0 )
            timer.Start();
        QRCtrl<double> ctrl;
        ctrl.smallestFirst = smallestFirst;
        QR( A, t, d, p, ctrl ); 
        if( commRank == 0 ) 
            Output("Businger-Golub time: ",timer.Stop()," seconds");
        if( print )
        {
            Print( A, "QR" );
            Print( t, "t" );
            Print( d, "d" );
            Print( p, "p" );
        }
        GetDiagonal( A, diagR );
        Print( diagR, "diag(R)" );

        A = ACopy;
        if( commRank == 0 )
            timer.Start();
        QR( A, t, d ); 
        if( commRank == 0 ) 
            Output("Standard QR time: ",timer.Stop()," seconds");
        if( print )
        {
            Print( A, "QR" );
            Print( t, "t" );
            Print( d, "d" );
        }
        GetDiagonal( A, diagR );
        Print( diagR, "diag(R)" );
    }
    catch( std::exception& e ) { ReportException(e); }

    return 0;
}
