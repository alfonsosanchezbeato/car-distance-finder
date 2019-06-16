#include <mutex>
#include <thread>
#include <condition_variable>

template <typename In, typename Out>
struct MonoProcessor {
    MonoProcessor(void);
    virtual ~MonoProcessor(void);

    // If the thread is busy, it does nothing. Otherwise, it pushes new input
    // data to the thread and refreshes "out" with the new output data.
    void process(const In& in, Out& out);

protected:
    virtual void setNextGetLast(const In& in, Out& out) = 0;
    virtual void update(void) = 0;

private:
    std::mutex dataMtx_;
    std::condition_variable dataCond_;
    bool finish_;
    // Keep this last as it uses the other members
    std::thread processThread_;

    void threadMethod(void);
};

template <typename In, typename Out>
MonoProcessor<In, Out>::MonoProcessor(void) :
    finish_{false},
    processThread_{&MonoProcessor::threadMethod, this}
{
}

template <typename In, typename Out>
MonoProcessor<In, Out>::~MonoProcessor(void)
{
    {
        std::unique_lock<std::mutex> lock(dataMtx_);
        finish_ = true;
        dataCond_.notify_one();
    }

    processThread_.join();
}

template <typename In, typename Out>
void MonoProcessor<In, Out>::process(const In& in, Out& out)
{
    std::unique_lock<std::mutex> lock(dataMtx_, std::defer_lock_t());
    if (lock.try_lock()) {
        setNextGetLast(in, out);
        dataCond_.notify_one();
    }
}

template <typename In, typename Out>
void MonoProcessor<In, Out>::threadMethod(void)
{
    while (true) {
        std::unique_lock<std::mutex> lock(dataMtx_);
        dataCond_.wait(lock);

        if (finish_)
            break;

        update();
    }
}
