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

    const Int n = CPre.Width();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();
    const bool conjugate = (orientB == ADJOINT);
    auto const num_blocks = (n + bsize - 1) / bsize;

    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> AProx(APre);
    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> BProx(BPre);
    DistMatrixReadWriteProxy<T,T,MC,MR,ELEMENT,D> CProx(CPre);
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();
    auto& C = CProx.Get();

    // Get the sync pool.
    auto const& stream_pool = GetSyncInfoPool(C.Grid());
    auto const num_stream_teams =
        stream_pool.Size() == 1UL
        ? 1UL
        : std::min(stream_pool.Size() / 2, size_t(num_blocks));

    // Temporary distributions
    std::vector<DistMatrix<T,MR,STAR,ELEMENT,D>> B1Trans_MR_STAR;
    std::vector<DistMatrix<T,MC,STAR,ELEMENT,D>> D1_MC_STAR;

    B1Trans_MR_STAR.reserve(num_stream_teams);
    D1_MC_STAR.reserve(num_stream_teams);

    // Setup temporaries
    for (auto id = 0UL; id < num_stream_teams; ++id)
    {
        auto B1T = B1Trans_MR_STAR.emplace(B1Trans_MR_STAR.end(), g);
        auto D1 = D1_MC_STAR.emplace(D1_MC_STAR.end(), g);

        auto const& stream_one = stream_pool.Next();
        auto const& stream_two = stream_pool.Next();

        B1T->AlignWith(A);
        D1->AlignWith(A);

        SetSyncInfo(B1T->Matrix(), stream_one);
        SetSyncInfo(D1->Matrix(), stream_two);
    }

    size_t team_id = 0UL;
    for(Int k=0; k<n; k+=bsize)
    {
        DistMatrix<T,MC,MR,ELEMENT,D> A1(g);
        LockedView(A1, A);

        const Int nb = Min(bsize,n-k);

        auto B1 = B(IR(k,k+nb), ALL       );
        auto C1 = C(ALL,        IR(k,k+nb));

        auto& BTMRSTAR = B1Trans_MR_STAR[team_id];
        auto& DMCSTAR = D1_MC_STAR[team_id];

        auto const& stream_one =
            SyncInfoFromMatrix(BTMRSTAR.Matrix());
        auto const& stream_two =
            SyncInfoFromMatrix(DMCSTAR.Matrix());

        SetSyncInfo(A1.Matrix(), stream_one);
        SetSyncInfo(B1.Matrix(), stream_one);
        SetSyncInfo(C1.Matrix(), stream_two);

        // C1[MC,*] := alpha A[MC,MR] (B1^[T/H])[MR,*]
        Transpose(B1, BTMRSTAR, conjugate);
        LocalGemm(NORMAL, NORMAL, alpha, A1, BTMRSTAR, DMCSTAR);

        // C1[MC,MR] += scattered result of D1[MC,*] summed over grid rows
        AxpyContract(TypeTraits<T>::One(), DMCSTAR, C1);

        // Bookkeeping.
        team_id = (team_id + 1) % num_stream_teams;

        // Have C wait on all streams
        AddSynchronizationPoint(stream_two, SyncInfo_C);
    }
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

    // Get the sync pool.
    auto const& stream_pool = GetSyncInfoPool(C.Grid());
    auto const num_stream_teams =
        stream_pool.Size() == 1UL
        ? 1UL
        : std::min(stream_pool.Size() / 2, size_t(num_blocks));

    // Temporary distributions
    std::vector<DistMatrix<T,MR,STAR,ELEMENT,D>> A1Trans_MR_STAR;
    std::vector<DistMatrix<T,STAR,MC,ELEMENT,D>> D1_STAR_MC;
    std::vector<DistMatrix<T,MR,MC,ELEMENT,D>> D1_MR_MC;

    A1Trans_MR_STAR.reserve(num_stream_teams);
    D1_STAR_MC.reserve(num_stream_teams);
    D1_MR_MC.reserve(num_stream_teams);

    for (auto id = 0UL; id < num_stream_teams; ++id)
    {
        A1Trans_MR_STAR.emplace_back(g);
        D1_STAR_MC.emplace_back(g);
        D1_MR_MC.emplace_back(g);

        auto& A1 = A1Trans_MR_STAR.back();
        auto& D1 = D1_STAR_MC.back();
        auto& DMRMC = D1_MR_MC.back();

        auto const& stream_one = stream_pool.Next();
        auto const& stream_two = stream_pool.Next();

        A1.AlignWith(B);
        D1.AlignWith(B);

        SetSyncInfo(A1.Matrix(), stream_one);
        SetSyncInfo(D1.Matrix(), stream_one);
        SetSyncInfo(DMRMC.Matrix(), stream_two);
    }

    size_t team_id = 0UL;
    for(Int k=0; k<m; k+=bsize)
    {
        DistMatrix<T,MC,MR,ELEMENT,D> B1(g);
        LockedView(B1, B);

        const Int nb = Min(bsize,m-k);
        auto A1 = A(IR(k,k+nb), ALL);
        auto C1 = C(IR(k,k+nb), ALL);

        auto& ATMRSTAR = A1Trans_MR_STAR[team_id];
        auto& DSTARMC = D1_STAR_MC[team_id];
        auto& DMRMC = D1_MR_MC[team_id];

        auto const& stream_one =
            SyncInfoFromMatrix(ATMRSTAR.LockedMatrix());
        auto const& stream_two =
            SyncInfoFromMatrix(DMRMC.LockedMatrix());

        SetSyncInfo(A1.Matrix(), stream_one);
        SetSyncInfo(B1.Matrix(), stream_one);
        SetSyncInfo(C1.Matrix(), stream_two);

        // D1[*,MC] := alpha A1[*,MR] (B[MC,MR])^T
        //           = alpha (A1^T)[MR,*] (B^T)[MR,MC]
        Transpose(A1, ATMRSTAR);
        LocalGemm(TRANSPOSE, orientB, alpha, ATMRSTAR, B1, DSTARMC);

        // C1[MC,MR] += scattered & transposed D1[*,MC] summed over grid rows
        Contract(DSTARMC, DMRMC);
        Axpy(TypeTraits<T>::One(), DMRMC, C1);

        // Bookkeeping.
        team_id = (team_id + 1) % num_stream_teams;

        // Syncronize against C
        AddSynchronizationPoint(stream_two, SyncInfo_C);
    }
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

    // Temporary distributions
    std::vector<DistMatrix<T,MC,STAR,ELEMENT,D>> A1_MC_STAR;
    std::vector<DistMatrix<T,VR,STAR,ELEMENT,D>> B1_VR_STAR;
    std::vector<DistMatrix<T,STAR,MR,ELEMENT,D>> B1Trans_STAR_MR;
    std::vector<DistMatrix<T,MC,MR,ELEMENT,D>> C_TMP;

        // Get the sync pool.
    auto const& stream_pool = GetSyncInfoPool(C.Grid());
    auto const num_stream_teams =
        stream_pool.Size() == 1UL
        ? 1UL
        : std::min(stream_pool.Size() / 2, size_t(num_blocks));

    A1_MC_STAR.reserve(num_stream_teams);
    B1_VR_STAR.reserve(num_stream_teams);
    B1Trans_STAR_MR.reserve(num_stream_teams);
    C_TMP.reserve(num_stream_teams);

    for (auto id = 0UL; id < num_stream_teams; ++id)
    {
        auto A1 = A1_MC_STAR.emplace(A1_MC_STAR.end(), g);
        auto B1 = B1_VR_STAR.emplace(B1_VR_STAR.end(), g);
        auto B1T = B1Trans_STAR_MR.emplace(B1Trans_STAR_MR.end(), g);
        auto C1 = C_TMP.emplace(C_TMP.end(), g);

        auto const& stream_one = stream_pool.Next();
        auto const& stream_two = stream_pool.Next();

        A1->AlignWith(C);
        B1->AlignWith(C);
        B1T->AlignWith(C);

        SetSyncInfo(A1->Matrix(), stream_one);
        SetSyncInfo(B1->Matrix(), stream_two);
        SetSyncInfo(B1T->Matrix(), stream_two);

        if (id == 0UL)
        {
            View(*C1, C);
            SetSyncInfo(C1->Matrix(), stream_two);
        }
        else
        {
            C1->AlignWith(C);
            SetSyncInfo(C1->Matrix(), stream_two);
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

        auto const& stream_one =
            SyncInfoFromMatrix(AMCSTAR.Matrix());
        auto const& stream_two =
            SyncInfoFromMatrix(CView.Matrix());

        SetSyncInfo(A1.Matrix(), stream_one);
        SetSyncInfo(B1.Matrix(), stream_two);

        AMCSTAR = A1;
        BVRSTAR = B1;
        Transpose(BVRSTAR, BTSTARMR, conjugate);

        // C[MC,MR] += alpha A1[MC,*] (B1[MR,*])^T
        LocalGemm(
            NORMAL, NORMAL,
            alpha, AMCSTAR, BTSTARMR,
            TypeTraits<T>::One(), CView);

        // Bookkeeping.
        team_id = (team_id + 1) % num_stream_teams;
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
