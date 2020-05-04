namespace El {
namespace gemm {

template <typename T, typename=EnableIf<IsDeviceValidType<T,Device::GPU>>>
void SUMMA_NTA_impl_multistream(
    Orientation orientB,
    T alpha,
    AbstractDistMatrix<T> const& APre,
    AbstractDistMatrix<T> const& BPre,
    AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE;

    constexpr Device D = Device::GPU;

    auto SyncInfo_C = SyncInfoFromMatrix(
        static_cast<Matrix<T,D> const&>(CPre.LockedMatrix()));

    AUTO_PROFILE_REGION(
        "SUMMA.NTA.multistream",
        SyncInfo_C);

    Int const n = CPre.Width();
    Int const bsize = Blocksize();
    Grid const& g = APre.Grid();
    bool const conjugate = (orientB == ADJOINT);
    auto const num_blocks = (n + bsize - 1) / bsize;

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

    // Temporary distributions
    std::vector<DistMatrix<T,MR,STAR,ELEMENT,D>> B1Trans_MR_STAR;
    std::vector<DistMatrix<T,MC,STAR,ELEMENT,D>> D1_MC_STAR;

    B1Trans_MR_STAR.reserve(num_streams);
    D1_MC_STAR.reserve(num_streams);

    // Setup temporaries
    for (auto id = 0UL; id < num_streams; ++id)
    {
        auto B1T = B1Trans_MR_STAR.emplace(B1Trans_MR_STAR.end(), g);
        auto D1 = D1_MC_STAR.emplace(D1_MC_STAR.end(), g);

        auto const& the_stream = stream_pool.Next();

        B1T->AlignWith(A);
        D1->AlignWith(A);

        SetSyncInfo(B1T->Matrix(), the_stream);
        SetSyncInfo(D1->Matrix(), the_stream);

        AddSynchronizationPoint(SyncInfo_C, the_stream);
    }

    Int k = 0;
    while (k < n)
    {
        Int const k_start = k;

        k = k_start;
        for (size_t blk = 0UL; blk < num_streams && k < n; ++blk, k+=bsize)
        {
            Int const nb = Min(bsize,n-k);
            auto B1 = B(IR(k,k+nb), ALL       );
            auto& BTMRSTAR = B1Trans_MR_STAR[blk];
            SetSyncInfo(B1.Matrix(), SyncInfoFromMatrix(BTMRSTAR.Matrix()));

            Transpose(B1, BTMRSTAR, conjugate);
        }

        k = k_start;

        // Launch the final communications
        for (size_t blk = 0UL; blk < num_streams && k < n; ++blk, k+=bsize)
        {
            DistMatrix<T,MC,MR,ELEMENT,D> A1(g);
            LockedView(A1, A);

            Int const nb = Min(bsize,n-k);
            auto C1 = C(ALL,        IR(k,k+nb));
            auto& BTMRSTAR = B1Trans_MR_STAR[blk];
            auto& DMCSTAR = D1_MC_STAR[blk];
            SetSyncInfo(A1.Matrix(), SyncInfoFromMatrix(DMCSTAR.Matrix()));
            SetSyncInfo(C1.Matrix(), SyncInfoFromMatrix(DMCSTAR.Matrix()));

            // C1[MC,*] := alpha A[MC,MR] (B1^[T/H])[MR,*]
            LocalGemm(NORMAL, NORMAL, alpha, A1, BTMRSTAR, DMCSTAR);

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
void SUMMA_NTA_impl_multistream(Orientation orientB,
                                T alpha,
                                AbstractDistMatrix<T> const& APre,
                                AbstractDistMatrix<T> const& BPre,
                                AbstractDistMatrix<T>& CPre)
{
    LogicError("SUMMA_NTA_impl_multistream type-device combo not supported.");
}

template <typename T, typename=EnableIf<IsDeviceValidType<T,Device::GPU>>>
void SUMMA_NTB_impl_multistream(
    Orientation orientB,
    T alpha,
    AbstractDistMatrix<T> const& APre,
    AbstractDistMatrix<T> const& BPre,
    AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE;

    constexpr Device D = Device::GPU;

    auto SyncInfo_C = SyncInfoFromMatrix(
        static_cast<Matrix<T,D> const&>(CPre.LockedMatrix()));

    AUTO_PROFILE_REGION(
        "SUMMA.NTB.multistream",
        SyncInfo_C);

    const Int m = CPre.Height();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();
    auto const num_blocks = (m + bsize - 1) / bsize;

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

    // Temporary distributions
    std::vector<DistMatrix<T,MR,STAR,ELEMENT,D>> A1Trans_MR_STAR;
    std::vector<DistMatrix<T,STAR,MC,ELEMENT,D>> D1_STAR_MC;
    std::vector<DistMatrix<T,MR,MC,ELEMENT,D>> D1_MR_MC;

    A1Trans_MR_STAR.reserve(num_streams);
    D1_STAR_MC.reserve(num_streams);
    D1_MR_MC.reserve(num_streams);

    for (auto id = 0UL; id < num_streams; ++id)
    {
        A1Trans_MR_STAR.emplace_back(g);
        D1_STAR_MC.emplace_back(g);
        D1_MR_MC.emplace_back(g);

        auto& A1 = A1Trans_MR_STAR.back();
        auto& D1 = D1_STAR_MC.back();
        auto& DMRMC = D1_MR_MC.back();

        auto const& the_stream = stream_pool.Next();

        A1.AlignWith(B);
        D1.AlignWith(B);

        SetSyncInfo(A1.Matrix(), the_stream);
        SetSyncInfo(D1.Matrix(), the_stream);
        SetSyncInfo(DMRMC.Matrix(), the_stream);

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

            auto& ATMRSTAR = A1Trans_MR_STAR[blk];
            auto& DSTARMC = D1_STAR_MC[blk];

            auto const& the_stream =
                SyncInfoFromMatrix(DSTARMC.LockedMatrix());

            SetSyncInfo(A1.Matrix(), the_stream);
            SetSyncInfo(B1.Matrix(), the_stream);

            // D1[*,MC] := alpha A1[*,MR] (B[MC,MR])^T
            //           = alpha (A1^T)[MR,*] (B^T)[MR,MC]
            Transpose(A1, ATMRSTAR);
            LocalGemm(TRANSPOSE, orientB, alpha, ATMRSTAR, B1, DSTARMC);
        }

        k = k_start;
        for (size_t blk = 0UL; blk < num_streams && k < m; ++blk, k+=bsize)
        {
            const Int nb = Min(bsize,m-k);
            auto C1 = C(IR(k,k+nb), ALL);
            auto& DSTARMC = D1_STAR_MC[blk];
            auto& DMRMC = D1_MR_MC[blk];
            SetSyncInfo(C1.Matrix(),
                        SyncInfoFromMatrix(DSTARMC.LockedMatrix()));

            // C1[MC,MR] += scattered & transposed D1[*,MC] summed over grid rows
            Contract(DSTARMC, DMRMC);
            Axpy(TypeTraits<T>::One(), DMRMC, C1);
        }
    }

    for (auto const& mat : D1_MR_MC)
        AddSynchronizationPoint(
            SyncInfoFromMatrix(mat.LockedMatrix()), SyncInfo_C);
}

template <typename T,
          typename=DisableIf<IsDeviceValidType<T,Device::GPU>>, typename=void>
void SUMMA_NTB_impl_multistream(Orientation orientB,
                                T alpha,
                                AbstractDistMatrix<T> const& APre,
                                AbstractDistMatrix<T> const& BPre,
                                AbstractDistMatrix<T>& CPre)
{
    LogicError("SUMMA_NTB_impl_multistream type-device combo not supported.");
}

template <typename T, typename=EnableIf<IsDeviceValidType<T,Device::GPU>>>
void SUMMA_NTC_impl_multistream(
    Orientation orientB,
    T alpha,
    AbstractDistMatrix<T> const& APre,
    AbstractDistMatrix<T> const& BPre,
    AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE;
    constexpr Device D = Device::GPU;

    AUTO_PROFILE_REGION(
        "SUMMA.NTC.multistream",
        SyncInfoFromMatrix(
            static_cast<Matrix<T,D> const&>(CPre.LockedMatrix())));


    const Int sumDim = APre.Width();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();
    const bool conjugate = (orientB == ADJOINT);
    auto const num_blocks = (sumDim + bsize - 1) / bsize;

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

    // Temporary distributions
    std::vector<DistMatrix<T,MC,STAR,ELEMENT,D>> A1_MC_STAR;
    std::vector<DistMatrix<T,VR,STAR,ELEMENT,D>> B1_VR_STAR;
    std::vector<DistMatrix<T,STAR,MR,ELEMENT,D>> B1Trans_STAR_MR;
    std::vector<DistMatrix<T,MC,MR,ELEMENT,D>> C_TMP;

        // Get the sync pool.
    auto const& stream_pool = GetSyncInfoPool(C.Grid());
    auto const num_streams =
        stream_pool.Size() == 1UL
        ? 1UL
        : std::min(stream_pool.Size(), size_t(num_blocks));

    A1_MC_STAR.reserve(num_streams);
    B1_VR_STAR.reserve(num_streams);
    B1Trans_STAR_MR.reserve(num_streams);
    C_TMP.reserve(num_streams);

    for (auto id = 0UL; id < num_streams; ++id)
    {
        auto A1 = A1_MC_STAR.emplace(A1_MC_STAR.end(), g);
        auto B1 = B1_VR_STAR.emplace(B1_VR_STAR.end(), g);
        auto B1T = B1Trans_STAR_MR.emplace(B1Trans_STAR_MR.end(), g);
        auto C1 = C_TMP.emplace(C_TMP.end(), g);

        auto const& the_stream = stream_pool.Next();

        A1->AlignWith(C);
        B1->AlignWith(C);
        B1T->AlignWith(C);

        SetSyncInfo(A1->Matrix(), the_stream);
        SetSyncInfo(B1->Matrix(), the_stream);
        SetSyncInfo(B1T->Matrix(), the_stream);

        AddSynchronizationPoint(SyncInfo_C, the_stream);
        AddSynchronizationPoint(SyncInfo_C, the_stream);

        if (id == 0UL)
        {
            View(*C1, C);
            SetSyncInfo(C1->Matrix(), the_stream);
        }
        else
        {
            C1->AlignWith(C);
            SetSyncInfo(C1->Matrix(), the_stream);
            Zeros(*C1, C.Height(), C.Width());
        }
    }

    size_t team_id = 0;
    for (Int k=0; k<sumDim; k+=bsize)
    {
        const Int nb = Min(bsize,sumDim-k);
        auto A1 = A(ALL, IR(k,k+nb));
        auto B1 = B(ALL, IR(k,k+nb));

        auto& AMCSTAR = A1_MC_STAR[team_id];
        auto& BVRSTAR = B1_VR_STAR[team_id];
        auto& BTSTARMR = B1Trans_STAR_MR[team_id];
        auto& CView = C_TMP[team_id];

        auto const& the_stream =
            SyncInfoFromMatrix(AMCSTAR.Matrix());

        SetSyncInfo(A1.Matrix(), the_stream);
        SetSyncInfo(B1.Matrix(), the_stream);

        AMCSTAR = A1;
        BVRSTAR = B1;
        Transpose(BVRSTAR, BTSTARMR, conjugate);

        // C[MC,MR] += alpha A1[MC,*] (B1[MR,*])^T
        LocalGemm(
            NORMAL, NORMAL,
            alpha, AMCSTAR, BTSTARMR,
            TypeTraits<T>::One(), CView);

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
void SUMMA_NTC_impl_multistream(Orientation orientB,
                                T alpha,
                                AbstractDistMatrix<T> const& APre,
                                AbstractDistMatrix<T> const& BPre,
                                AbstractDistMatrix<T>& CPre)
{
    LogicError("SUMMA_NTA_impl_multistream type-device combo not supported.");
}

} // namespace gemm
} // namespace El
