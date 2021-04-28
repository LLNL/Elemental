/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/

/*
  Testing interaction of 2 matrices with different distributions.
  Not sure why it's call "DifferentGrids".  Maybe the partitioning
  is based on a grid.
*/
#include <El.hpp>
#include <chrono>
using namespace El;

int
main( int argc, char* argv[] )
{
    Environment env( argc, argv );
    mpi::Comm comm = mpi::NewWorldComm();
    mpi::Comm comm_sqrt, comm_sqrt_sec;


    const Int commSize = mpi::Size( comm );
    const Int rank = mpi::Rank(comm);
    printf("Rank is %d\n", rank);

    
   
    

    try
    {
        const bool colMajor = Input("--colMajor","column-major ordering?",true);
        const bool colMajorSqrt = Input("--colMajorSqrt","colMajor sqrt?",true);
        const Int m = Input("--height","height of matrix",50);
        const Int n = Input("--width","width of matrix",100);
        const bool print = Input("--print","print matrices?",false);
        const Int iters = Input("--iters","Iterations (default:100)?",100);
        const Int grid1_width = Input("--g1Width","width of grid 1?",1);
        const Int grid1_height = Input("--g1Height","height of grid 1?",2);
        const Int warmup = Input("--warmup","warmup iterations?",10);
        const Int numSubGrids = Input("--numSubGrids","number of subgrids?",2);
        const Int numMatrix = Input("--numMatrix","number of matrices per subgrid?",2);
        const Int slice_dim = Input("--dim","dimension of slice dim?",2);

        ProcessInput();
        PrintInputReport();

        //ff turing off or on GPU please update the GPU variable manually 
        const auto D = Device::GPU;
        const bool GPU = true;

        const GridOrder order = ( colMajor ? COLUMN_MAJOR : ROW_MAJOR );
        const GridOrder orderGrid = ( colMajorSqrt ? COLUMN_MAJOR : ROW_MAJOR );

        // size of grids.
        const Int grid1Size = grid1_height * grid1_width;



        std::vector<std::vector<int>> gridsRanks(numSubGrids);

        int temp_count = 0;
        for(Int i =0; i< numSubGrids;++i)
        {
            for(Int j=0; j<grid1Size; ++j)
            {
                gridsRanks[i].push_back(temp_count);
                temp_count++;
            }
        }
        

        mpi::Group group, grid1Group,grid2Group;

        std::vector<mpi::Group> groupsVector(numSubGrids);
        mpi::CommGroup( comm, group );

        for(Int i=0; i < numSubGrids; ++i)
        {
            mpi::Incl( group, gridsRanks[i].size(), gridsRanks[i].data(), groupsVector[i] );
        }

        


        std::vector<std::unique_ptr<El::Grid>> gridsVector;


        const Grid grid( std::move(comm), group, commSize, orderGrid );

        for(int i =0 ;i<numSubGrids;++i)
        {
            
            gridsVector.push_back(std::unique_ptr<El::Grid>(new El::Grid(mpi::NewWorldComm(),groupsVector[i], grid1Size, orderGrid )));
            
            
            
        }


        
        DistMatrix<double,STAR,VC,ELEMENT,D> A(grid),  A_temp(grid);


        std::vector<std::unique_ptr<AbstractDistMatrix<double>>> B_vector;
        B_vector.resize(numSubGrids*numMatrix);
        temp_count = 0;
        int counter = 0;
        for(auto& B: B_vector)
        {
            B = std::make_unique<DistMatrix<double,STAR,VC,ELEMENT,D>>(*gridsVector[temp_count], 0);
            counter++;
            if(counter%numMatrix==0)
            {
                temp_count++;
            }
            
        }



        Int indexB = -1;
        Int posInSubGrid = -1;

        for(Int i = 0; i<numSubGrids*numMatrix; ++i)
        {            

            if(B_vector[i]->Participating())
            {
                indexB = i;
                posInSubGrid = B_vector[i]->Grid().VCRank();
                break;
            }

        }
        Identity(A, m, n );

        double val = 0;
        for(Int j=0; j<n;j++)
        {
            for(Int i = 0; i<m;i++)
            {
                A.Set(i,j,val);
                val++;
            }
        }

        
        mpi::Comm allreduceComm;

        mpi::Split(mpi::NewWorldComm(), posInSubGrid, rank, allreduceComm);
        SyncInfo<D> syncGeneral = SyncInfo<D>();

            
        auto duration_all =0;
        



        duration_all =0;
        for(Int i=0 ;i< iters; ++i){

            auto start = std::chrono::high_resolution_clock::now();
            
            El::copy::TranslateBetweenGridsScatter<double,D,D>(A,B_vector,slice_dim,allreduceComm,syncGeneral);

            if(GPU)
            {
                cudaDeviceSynchronize();
            }

            auto end = std::chrono::high_resolution_clock::now();



            auto duration = duration_cast<std::chrono::microseconds>(end - start); 

            
            if(i>warmup)
            {
                duration_all = duration_all+duration.count();
            }

        }

        
        std::cout << "Rank:"<<rank<< " Total Time taken Slice AllGather Comm OPt(A<-B):" << duration_all/(iters-warmup) << endl; 

        duration_all =0;
        for(Int i=0 ;i< iters; ++i){

            auto start = std::chrono::high_resolution_clock::now();
            
            El::copy::TranslateBetweenGridsScatter<double,D,D>(A,B_vector,slice_dim,allreduceComm,syncGeneral,2);
            
            if(GPU)
            {
                cudaDeviceSynchronize();
            }

            auto end = std::chrono::high_resolution_clock::now();



            auto duration = duration_cast<std::chrono::microseconds>(end - start); 

            
            if(i>warmup)
            {
                duration_all = duration_all+duration.count();
            }

        }

        
        std::cout << "Rank:"<<rank<< " Total Time taken Slice Scatter Comm OPt(A<-B):" << duration_all/(iters-warmup) << endl; 



        duration_all =0;
        for(Int i=0 ;i< iters; ++i){

            auto start = std::chrono::high_resolution_clock::now();
            
            El::copy::TranslateBetweenGridsScatter<double,D,D>(A,B_vector,slice_dim,allreduceComm,syncGeneral,1);

            if(GPU)
            {
                cudaDeviceSynchronize();
            }
            auto end = std::chrono::high_resolution_clock::now();



            auto duration = duration_cast<std::chrono::microseconds>(end - start); 

            
            if(i>warmup)
            {
                duration_all = duration_all+duration.count();
            }

        }

        
        std::cout << "Rank:"<<rank<< " Total Time taken Scatter Comm (A<-B):" << duration_all/(iters-warmup) << endl; 
        temp_count=0;
        for(auto& B: B_vector)
        {
            if( B->Participating() )
            {
                auto p = "B_vector"+std::to_string(temp_count);
                if( print )
                    Print( *B, p );

            }

            temp_count++;
            
        }


        
        
        if( B_vector[0]->Participating() )
        {
            
            if( print )
                Print( *B_vector[0], "B_vector[0]" );
        }

        if( print &&  B_vector[1]->Participating() )
            Print( *B_vector[1], "B_vector[1]" );



        if( A.Participating() && print)
        {
            printf("Height: %d and Width: %d \n",A.LocalHeight(), A.LocalWidth());
            Print( A, "A" );
        }

        
    }
    catch( std::exception& e ) { ReportException(e); }

    return 0;
}
