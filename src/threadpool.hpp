#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <condition_variable>
#include <atomic>


class ThreadPool {
    public:
        ThreadPool(size_t num_threads, bool verbose=true);
        ~ThreadPool();
        void push(std::vector<std::function<void()>> tasks);

    private:
        int num_threads;
        bool stop;
        bool verbose;
        std::vector<std::thread> workers;
        std::queue<std::function<void()>> queue;  // TODO: 굳이 queue를 써야 하는지?
        std::mutex queue_mutex;
        std::condition_variable condition;
        std::atomic<int> active_tasks;
        std::atomic<int> finished_tasks;

        void _worker(int thread_idx);
};
