/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details:
 *
 * Copyright (C) 2019 Alfonso Sanchez-Beato
 */
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
//
// The template parameters are In, the type for the input to be processed, Out,
// the type of the results of the processing, and Task, which encodes how the
// processing is performed.
//
// The Task type must implement the following methods:
// 1. void transactSafe(const In& in, Out& out);
//    Called from transact() inside mutex context so the child class can safely
//    perform the transaction
// 2. void process(void);
//    Does the processing of one task - called from the MonoProcessor thread
//    inside mutex context
template <typename In, typename Out, typename Task>
struct MonoProcessor {
    // We force the task to be moved to the processor object
    MonoProcessor(Task&& task);
    virtual ~MonoProcessor(void);

    // If the thread is busy, it does nothing. Otherwise, it pushes new input
    // data to the thread and refreshes "out" with the new output data.  Returns
    // true if the transaction has happened, false othewise.
    bool transact(const In& in, Out& out);

    // Returns true if the thread is busy, false otherwise
    bool processorIdle(void);

private:
    Task task_;
    std::mutex dataMtx_;
    std::condition_variable dataCond_;
    bool finish_, wakeThread_;
    // Keep this last as it uses the other members
    std::thread processThread_;

    void threadMethod(void);
};

template <typename In, typename Out, typename Task>
MonoProcessor<In, Out, Task>::MonoProcessor(Task&& task) :
    task_{std::move(task)},
    finish_{false},
    wakeThread_{false},
    processThread_{&MonoProcessor::threadMethod, this}
{
}

template <typename In, typename Out, typename Task>
MonoProcessor<In, Out, Task>::~MonoProcessor(void)
{
    {
        std::lock_guard<std::mutex> lock(dataMtx_);
        finish_ = true;
        wakeThread_ = true;
        dataCond_.notify_one();
    }

    processThread_.join();
}

template <typename In, typename Out, typename Task>
bool MonoProcessor<In, Out, Task>::transact(const In& in, Out& out)
{
    std::unique_lock<std::mutex> lock(dataMtx_, std::defer_lock_t());
    if (lock.try_lock()) {
        task_.transactSafe(in, out);
        wakeThread_ = true;
        dataCond_.notify_one();
        return true;
    }
    return false;
}

template <typename In, typename Out, typename Task>
void MonoProcessor<In, Out, Task>::threadMethod(void)
{
    std::unique_lock<std::mutex> lock(dataMtx_);
    while (true) {
        dataCond_.wait(lock, [this]{ return wakeThread_; });
        wakeThread_ = false;

        if (finish_)
            break;

        task_.process();
    }
}

template <typename In, typename Out, typename Task>
bool MonoProcessor<In, Out, Task>::processorIdle(void)
{
    std::unique_lock<std::mutex> lock(dataMtx_, std::defer_lock_t());
    if (lock.try_lock())
        return true;

    return false;
}
