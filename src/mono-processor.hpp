#include <mutex>
#include <thread>
#include <condition_variable>

// MonoProcessor contains a thread and handles tasks, one at a time.  It is the
// equivalent of a "single processor CPU". The idea behind is that we want to
// spend only one core in these tasks, and that it is not an isue if some of
// them are not performed. We use it to handle real-time processing of frames,
// for things like tracking an object, where we cannot stop because we did not
// have time to analyze one frame. Users of the class call the transact() method
// to both get the result of the latest task and to push the data for the next
// one.
template <typename In, typename Out>
struct MonoProcessor {
    MonoProcessor(void);
    virtual ~MonoProcessor(void);

    // If the thread is busy, it does nothing. Otherwise, it pushes new input
    // data to the thread and refreshes "out" with the new output data.  Returns
    // true if the transaction has happened, false othewise.
    bool transact(const In& in, Out& out);

protected:
    // Called from transact() inside mutex context so the child class can safely
    // perform the transaction
    virtual void transactSafe(const In& in, Out& out) = 0;
    // Does the processing of one task - called from the MonoProcessor thread
    // inside mutex context
    virtual void process(void) = 0;

    // Make sure to stop the thread from the child destructor, we cannot do
    // that safely from the parent destructor as at that point the child objects
    // will have been already destroyed, which is an issue if they were being
    // used by the thread.
    void stop(void);

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
}

template <typename In, typename Out>
void MonoProcessor<In, Out>::stop(void)
{
    {
        std::unique_lock<std::mutex> lock(dataMtx_);
        finish_ = true;
        dataCond_.notify_one();
    }

    processThread_.join();
}

template <typename In, typename Out>
bool MonoProcessor<In, Out>::transact(const In& in, Out& out)
{
    std::unique_lock<std::mutex> lock(dataMtx_, std::defer_lock_t());
    if (lock.try_lock()) {
        transactSafe(in, out);
        dataCond_.notify_one();
        return true;
    }
    return false;
}

template <typename In, typename Out>
void MonoProcessor<In, Out>::threadMethod(void)
{
    while (true) {
        std::unique_lock<std::mutex> lock(dataMtx_);
        dataCond_.wait(lock);

        if (finish_)
            break;

        process();
    }
}
