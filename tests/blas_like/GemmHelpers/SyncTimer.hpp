#include <El.hpp>

#include <thread>

namespace helpers
{

// A timer class that measures increments relative to a SyncInfo object.
template <El::Device D>
class SyncTimer;

template <>
class SyncTimer<El::Device::CPU>
{
public:
    SyncTimer(El::SyncInfo<El::Device::CPU> const&) {}

    ~SyncTimer() = default;

    void Start()
    {
        timer_.Start();
    }

    void Stop()
    {
        time_ = timer_.Stop();
    }

    /** @brief Get elapsed time in seconds. */
    long double GetTime() const
    {
        return time_;
    }

    void Reset()
    {
        timer_.Reset();
    }

private:
    El::Timer timer_;
    double time_ = 0.0;
};// class CPUTimer

#ifdef HYDROGEN_HAVE_CUDA
template <>
class SyncTimer<El::Device::GPU>
{
public:
    SyncTimer(El::SyncInfo<El::Device::GPU> const& si)
        : si_ {si},
          started_ {false},
          stopped_ {false}
    {
        H_CHECK_CUDA(cudaEventCreate(&start_));
        H_CHECK_CUDA(cudaEventCreate(&stop_));
    }

    ~SyncTimer()
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void Start()
    {
        if (started_ || stopped_)
            throw std::runtime_error("Start(): Bad timer state.");

        H_CHECK_CUDA(cudaEventRecord(start_, si_.Stream()));
        started_ = true;
    }

    void Stop()
    {
        if (stopped_ || !started_)
            throw std::runtime_error("Stop(): Bad timer state.");

        H_CHECK_CUDA(cudaEventRecord(stop_, si_.Stream()));
        stopped_ = true;
    }

    /** @brief Get elapsed time in seconds. */
    long double GetTime() const
    {
        if (!(started_ && stopped_))
            throw std::runtime_error("GetTime(): Bad timer state.");

        float elapsed_time_ms;
        H_CHECK_CUDA(cudaEventSynchronize(stop_));
        H_CHECK_CUDA(cudaEventElapsedTime(&elapsed_time_ms, start_, stop_));
        return elapsed_time_ms / 1000.l;
    }

    void Reset()
    {
        started_ = stopped_ = false;
    }

private:
    El::SyncInfo<El::Device::GPU> si_;
    cudaEvent_t start_, stop_;
    bool started_, stopped_;
};// class SyncTimer<GPU>

#elif defined(HYDROGEN_HAVE_ROCM)

template <>
class SyncTimer<El::Device::GPU>
{
public:
    SyncTimer(El::SyncInfo<El::Device::GPU> const& si)
        : si_ {si},
          started_ {false},
          stopped_ {false}
    {
        H_CHECK_HIP(hipEventCreate(&start_));
        H_CHECK_HIP(hipEventCreate(&stop_));
    }

    ~SyncTimer()
    {
        auto statusignore = hipEventDestroy(start_);
        statusignore = hipEventDestroy(stop_);
        (void) statusignore;
    }

    void Start()
    {
        if (started_ || stopped_)
            throw std::runtime_error("Start(): Bad timer state.");

        H_CHECK_HIP(hipEventRecord(start_, si_.Stream()));
        started_ = true;
    }

    void Stop()
    {
        if (stopped_ || !started_)
            throw std::runtime_error("Stop(): Bad timer state.");

        H_CHECK_HIP(hipEventRecord(stop_, si_.Stream()));
        stopped_ = true;
    }

    /** @brief Get elapsed time in seconds. */
    long double GetTime() const
    {
        if (!(started_ && stopped_))
            throw std::runtime_error("GetTime(): Bad timer state.");

        float elapsed_time_ms;
        H_CHECK_HIP(hipEventSynchronize(stop_));
        H_CHECK_HIP(hipEventElapsedTime(&elapsed_time_ms, start_, stop_));
        return elapsed_time_ms / 1000.l;
    }

    void Reset()
    {
        started_ = stopped_ = false;
    }

private:
    El::SyncInfo<El::Device::GPU> si_;
    hipEvent_t start_, stop_;
    bool started_, stopped_;
};// class SyncTimer<GPU>

#endif // HYDROGEN_HAVE_CUDA

template <El::Device D>
SyncTimer<D> MakeSyncTimer(El::SyncInfo<D> const& si)
{
    return SyncTimer<D>{si};
}

}// namespace helpers
