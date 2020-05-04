namespace El {
namespace gemm {

template <typename T, typename=EnableIf<IsDeviceValidType<T,Device::GPU>>>
void SUMMA_NNA_impl_multistream(
    T alpha,
    AbstractDistMatrix<T> const& APre,
    AbstractDistMatrix<T> const& BPre,
    AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE;

    constexpr auto D = Device::GPU;
    auto SyncInfo_C = SyncInfoFromMatrix(
        static_cast<Matrix<T,D> const&>(CPre.LockedMatrix()));

    AUTO_PROFILE_REGION(
        "SUMMA.NNA.multistream",
        SyncInfo_C);

    const Int n = CPre.Width();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();
    auto const num_blocks = (n + bsize - 1) / bsize;

    // Setup proxies to assert data layout
    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> AProx(APre);
    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> BProx(BPre);
    DistMatrixReadWriteProxy<T,T,MC,MR,ELEMENT,D> CProx(CPre);
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();
    auto& C = CProx.Get();

    auto SyncManager = MakeMultiSync(
        SyncInfo_C,
        SyncInfoFromMatrix(A.LockedMatrix()),
        SyncInfoFromMatrix(B.LockedMatrix()));

    // Get the sync pool.
    auto const& stream_pool = GetSyncInfoPool(C.Grid());
    auto const num_streams =
        stream_pool.Size() == 1UL
        ? 1UL
        : std::min(stream_pool.Size(), size_t(num_blocks));

    // Setup the temporary matrices. One of each per "stream team".
    std::vector<DistMatrix<T,VR,STAR,ELEMENT,D>> B1_VR_STAR;
    std::vector<DistMatrix<T,STAR,MR,ELEMENT,D>> B1Trans_STAR_MR;
    std::vector<DistMatrix<T,MC,STAR,ELEMENT,D>> D1_MC_STAR;

    B1_VR_STAR.reserve(num_streams);
    B1Trans_STAR_MR.reserve(num_streams);
    D1_MC_STAR.reserve(num_streams);

    // Basic setup functions for the temp matrices; also, assign data
    // to streams in a round-robin fashion.
    for (auto id = 0UL; id < num_streams; ++id)
    {
        auto B1 = B1_VR_STAR.emplace(B1_VR_STAR.end(), g);
        auto B1T = B1Trans_STAR_MR.emplace(B1Trans_STAR_MR.end(), g);
        auto D1 = D1_MC_STAR.emplace(D1_MC_STAR.end(), g);

        auto const& the_stream = stream_pool.Next();

        // A and B are logically const; these just need to have the
        // right alignment and "stream affinity".
        B1->AlignWith(A);
        B1T->AlignWith(A);
        D1->AlignWith(A);

        SetSyncInfo(B1->Matrix(), the_stream);
        SetSyncInfo(B1T->Matrix(), the_stream);
        SetSyncInfo(D1->Matrix(), the_stream);

        AddSynchronizationPoint(SyncInfo_C, the_stream);
    }

    Int k = 0;
    while (k < n)
    {
        Int const k_start = k;

        // Launch everything up through the GEMM
        for (size_t blk = 0UL; blk < num_streams && k < n; ++blk, k+=bsize)
        {
            DistMatrix<T,MC,MR,ELEMENT,D> A1(g);
            LockedView(A1, A);

            const Int nb = Min(bsize,n-k);

            auto B1 = B(ALL, IR(k,k+nb));
            auto& BVRSTAR = B1_VR_STAR[blk];
            auto& BTSTARMR = B1Trans_STAR_MR[blk];
            auto& DMCSTAR = D1_MC_STAR[blk];
            auto const& the_stream =
                SyncInfoFromMatrix(BVRSTAR.LockedMatrix());

            SetSyncInfo(A1.Matrix(), the_stream);
            SetSyncInfo(B1.Matrix(), the_stream);

            // D1[MC,*] := alpha A[MC,MR] B1[MR,*]
            BVRSTAR = B1;
            Transpose(BVRSTAR, BTSTARMR);
            LocalGemm(NORMAL, TRANSPOSE,
                      alpha, A1, BTSTARMR, DMCSTAR);
        }

        k = k_start;

        // Launch the final communications
        for (size_t blk = 0UL; blk < num_streams && k < n; ++blk, k+=bsize)
        {
            const Int nb = Min(bsize,n-k);
            auto C1 = C(ALL, IR(k,k+nb));
            auto& DMCSTAR = D1_MC_STAR[blk];
            SetSyncInfo(C1.Matrix(),
                        SyncInfoFromMatrix(DMCSTAR.LockedMatrix()));

            // C1[MC,MR] += scattered result of D1[MC,*] summed over grid rows
            AxpyContract(TypeTraits<T>::One(), DMCSTAR, C1);
        }
    }

    // Have C wait on all streams
    for (auto const& mat : D1_MC_STAR)
        AddSynchronizationPoint(
            SyncInfoFromMatrix(mat.LockedMatrix()), SyncInfo_C);
}

template <typename T,
          typename=DisableIf<IsDeviceValidType<T,Device::GPU>>, typename=void>
void SUMMA_NNA_impl_multistream(T alpha,
               AbstractDistMatrix<T> const& APre,
               AbstractDistMatrix<T> const& BPre,
               AbstractDistMatrix<T>& CPre)
{
    LogicError("SUMMA_NNA_impl_multistream type-device combo not supported.");
}

// Normal Normal Gemm that avoids communicating the matrix B
template <typename T, typename=EnableIf<IsDeviceValidType<T,Device::GPU>>>
void SUMMA_NNB_impl_multistream(
    T alpha,
    const AbstractDistMatrix<T>& APre,
    const AbstractDistMatrix<T>& BPre,
    AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE;

    constexpr Device D = Device::GPU;

    AUTO_PROFILE_REGION(
        "SUMMA.NNB.multistream",
        SyncInfoFromMatrix(
            static_cast<Matrix<T,D> const&>(CPre.LockedMatrix())));

    Int const m = CPre.Height();
    Int const bsize = Blocksize();
    Grid const& g = APre.Grid();
    auto const num_blocks = (m + bsize - 1) / bsize;

    // Setup proxies to assert data layout
    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> AProx(APre);
    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> BProx(BPre);
    DistMatrixReadWriteProxy<T,T,MC,MR,ELEMENT,D> CProx(CPre);
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();
    auto& C = CProx.Get();

    auto const SyncInfo_C = SyncInfoFromMatrix(C.LockedMatrix());
    auto SyncManager = MakeMultiSync(
        SyncInfo_C,
        SyncInfoFromMatrix(A.LockedMatrix()),
        SyncInfoFromMatrix(B.LockedMatrix()));

    // Get the sync pool.
    auto const& stream_pool = GetSyncInfoPool(C.Grid());
    auto const num_streams =
        stream_pool.Size() == 1UL
        ? 1UL
        : std::min(stream_pool.Size(), size_t(num_blocks));

    // Temporary distributions
    std::vector<DistMatrix<T,STAR,MC,ELEMENT,D>> A1_STAR_MC;
    std::vector<DistMatrix<T,MR,STAR,ELEMENT,D>> D1Trans_MR_STAR;

    A1_STAR_MC.reserve(num_streams);
    D1Trans_MR_STAR.reserve(num_streams);

    for (auto id = 0UL; id < num_streams; ++id)
    {
        auto A1 = A1_STAR_MC.emplace(A1_STAR_MC.end(), g);
        auto D1 = D1Trans_MR_STAR.emplace(D1Trans_MR_STAR.end(), g);

        auto const& the_stream = stream_pool.Next();

        A1->AlignWith(B);
        D1->AlignWith(B);

        SetSyncInfo(A1->Matrix(), the_stream);
        SetSyncInfo(D1->Matrix(), the_stream);

        AddSynchronizationPoint(SyncInfo_C, the_stream);
    }

    Int k = 0;
    while (k < m)
    {
        Int const k_start = k;
        for (size_t blk = 0UL; blk < num_streams && k < m; ++blk, k+=bsize)
        {
            DistMatrix<T,MC,MR,ELEMENT,D> B1(g);
            LockedView(B1, B);

            const Int nb = Min(bsize,m-k);
            auto A1 = A(IR(k,k+nb), ALL);

            auto& ASTARMC = A1_STAR_MC[blk];
            auto& DTMRSTAR = D1Trans_MR_STAR[blk];

            auto const& the_stream =
                SyncInfoFromMatrix(DTMRSTAR.Matrix());

            SetSyncInfo(A1.Matrix(), the_stream);
            SetSyncInfo(B1.Matrix(), the_stream);

            // D1^T[MR,* ] := alpha B^T[MR,MC] A1^T[MC,* ]
            ASTARMC = A1;
            LocalGemm(
                TRANSPOSE, TRANSPOSE, alpha, B1, ASTARMC, DTMRSTAR);
        }

        k = k_start;
        for (size_t blk = 0UL; blk < num_streams && k < m; ++blk, k+=bsize)
        {
            const Int nb = Min(bsize,m-k);
            auto C1 = C(IR(k,k+nb), ALL);
            auto& DTMRSTAR = D1Trans_MR_STAR[blk];
            SetSyncInfo(C1.Matrix(), SyncInfoFromMatrix(DTMRSTAR.Matrix()));

            TransposeAxpyContract(TypeTraits<T>::One(), DTMRSTAR, C1);
        }
    }

    // Syncronize against C
    for (auto const& mat : D1Trans_MR_STAR)
        AddSynchronizationPoint(SyncInfoFromMatrix(mat.LockedMatrix()),
                                SyncInfo_C);
}

template <typename T,
          typename=DisableIf<IsDeviceValidType<T,Device::GPU>>, typename=void>
void SUMMA_NNB_impl_multistream(T alpha,
                                AbstractDistMatrix<T> const& APre,
                                AbstractDistMatrix<T> const& BPre,
                                AbstractDistMatrix<T>& CPre)
{
    LogicError("SUMMA_NNB_impl_multistream type-device combo not supported.");
}

template <typename T, typename=EnableIf<IsDeviceValidType<T,Device::GPU>>>
void SUMMA_NNC_impl_multistream(
    T alpha,
    AbstractDistMatrix<T> const& APre,
    AbstractDistMatrix<T> const& BPre,
    AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE;

    constexpr auto D = Device::GPU;

    AUTO_PROFILE_REGION(
        "SUMMA.NNC.multistream",
        SyncInfoFromMatrix(
            static_cast<Matrix<T,D> const&>(CPre.LockedMatrix())));

    const Int sumDim = APre.Width();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();
    auto const num_blocks = (sumDim + bsize - 1) / bsize;

    // Setup proxies to assert data layout
    // (TRB 10/09/2019): Is this really necessary?
    //   --> For C yes, for A and B, probably not. But having it
    //       guarantees the description "allgather-allgather-compute".
    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> AProx(APre);
    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> BProx(BPre);
    DistMatrixReadWriteProxy<T,T,MC,MR,ELEMENT,D> CProx(CPre);
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();
    auto& C = CProx.Get();

    auto const SyncInfo_C = SyncInfoFromMatrix(C.LockedMatrix());
    auto SyncManager = MakeMultiSync(
        SyncInfoFromMatrix(C.LockedMatrix()),
        SyncInfoFromMatrix(A.LockedMatrix()),
        SyncInfoFromMatrix(B.LockedMatrix()));

    // Get the sync pool.
    auto const& stream_pool = GetSyncInfoPool(C.Grid());
    auto const num_streams =
        stream_pool.Size() == 1UL
        ? 1UL
        : std::min(stream_pool.Size(), size_t(num_blocks));

    // Setup the temporary matrices. One of each per "stream team".
    std::vector<DistMatrix<T,MC,STAR,ELEMENT,D>> A1_MC_STAR;
    std::vector<DistMatrix<T,MR,STAR,ELEMENT,D>> B1Trans_MR_STAR;
    std::vector<DistMatrix<T,MC,MR,ELEMENT,D>> C_TMP;

    A1_MC_STAR.reserve(num_streams);
    B1Trans_MR_STAR.reserve(num_streams);
    C_TMP.reserve(num_streams);

    // Basic setup functions for the temp matrices; also, assign data
    // to streams in a round-robin fashion.
    for (auto id = 0UL; id < num_streams; ++id)
    {
        auto const& the_stream = stream_pool.Next();

        auto A1 = A1_MC_STAR.emplace(A1_MC_STAR.end(), g);
        auto B1 = B1Trans_MR_STAR.emplace(
            B1Trans_MR_STAR.end(), g);
        auto C1 = C_TMP.emplace(C_TMP.end(), g);

        // A and B are logically const; these just need to have the
        // right alignment and "stream affinity".
        SetSyncInfo(A1->Matrix(), the_stream);
        SetSyncInfo(B1->Matrix(), the_stream);

        A1->Resize(A.Height(), bsize);
        B1->Resize(B.Width(), bsize);

        A1->AlignWith(C);
        B1->AlignWith(C);

        AddSynchronizationPoint(SyncInfo_C, the_stream);
        AddSynchronizationPoint(SyncInfo_C, the_stream);

        // The copies of C should be initialized to zero so the
        // accumulation is correct.
        if (id == 0UL)
        {
            View(*C1, C);
            SetSyncInfo(C1->Matrix(), the_stream);
        }
        else
        {
            SetSyncInfo(C1->Matrix(), the_stream);
            C1->Resize(C.Height(), C.Width());
            C1->AlignWith(C);

            // Zero things out.
            Zero(*C1);
        }
    }

    // From this point on, we don't use the stream pool explicitly;
    // matrices are directly queried for their stream
    // information. This seemed less prone to error.

    // Compute the rank-k updates. This is Allgather-Allgather, Compute.
    size_t team_id = 0;
    for (Int k = 0; k < sumDim; k += bsize)
    {
        // Ultimately, these are "locked views" of ranges of
        // columns/rows of A/B, resp.
        DistMatrix<T,MC,MR,ELEMENT,D> A1(g), B1(g);
        Int const nb = Min(bsize, sumDim-k);

        auto& AMCSTAR = A1_MC_STAR[team_id];
        auto& BTMRSTAR = B1Trans_MR_STAR[team_id];

        // Set the streams for the views so operations with them are
        // correctly synchronized and ordered.
        auto const& the_stream =
            SyncInfoFromMatrix(AMCSTAR.Matrix());

        SetSyncInfo(A1.Matrix(), the_stream);
        SetSyncInfo(B1.Matrix(), the_stream);

        // Select the block of columns/rows of A/B.
        A1 = A(ALL,        IR(k,k+nb));
        B1 = B(IR(k,k+nb), ALL       );

        // Perform data movement (allgathers).
        AMCSTAR = A1;
        Transpose(B1, BTMRSTAR);

        // Compute the local portion of the rank-k update. This is
        // stored in the "team-local" storage. This assures no data
        // race in updates to C (since C isn't being updated yet).
        LocalGemm(NORMAL, TRANSPOSE,
                  alpha,
                  AMCSTAR,
                  BTMRSTAR,
                  TypeTraits<T>::One(), C_TMP[team_id]);

        // Bookkeeping.
        team_id = (team_id + 1) % num_streams;
    }

    AddSynchronizationPoint(SyncInfoFromMatrix(C_TMP.front().LockedMatrix()),
                            SyncInfoFromMatrix(C.LockedMatrix()));

    // Compute the reduction into the "real C". This work will
    // serialize on C's stream, so there is no race here.
    for (size_t ii = 1; ii < C_TMP.size(); ++ii)
    {
        Axpy(TypeTraits<T>::One(), C_TMP[ii], C);
    }
}

template <typename T,
          typename=DisableIf<IsDeviceValidType<T,Device::GPU>>, typename=void>
void SUMMA_NNC_impl_multistream(T alpha,
               AbstractDistMatrix<T> const& APre,
               AbstractDistMatrix<T> const& BPre,
               AbstractDistMatrix<T>& CPre)
{
    LogicError("SUMMA_NNC_impl_multistream type-device combo not supported.");
}

} // namespace gemm
} // namespace El
