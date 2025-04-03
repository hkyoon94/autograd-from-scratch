#include <atomic>
#include <iostream>

#include "threadpool.hpp"


ThreadPool::ThreadPool(size_t num_threads, bool verbose)
: num_threads(num_threads), stop(false), verbose(verbose) {
    if (verbose) {
        std::cout << "Creating " << num_threads << " worker threads..." << std::endl;
    }
    for (size_t i = 0; i < num_threads; ++i) {
        // worker thread 생성
        workers.emplace_back(
            [this, i] { this->_worker(i); }
        );
    }
    if (verbose) {
        std::cout << "Created all worker threads" << std::endl;
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers) {
        worker.join();
    }
}

void ThreadPool::_worker(int thread_idx) {
    // Thread가 깨어나도 되는 조건
    auto wakeup_condition = [this] { return stop || !queue.empty(); };

    // 이 곳에 사전작업 추가 가능
    // ...

    while (true) {
        std::function<void()> task;
        {   // queue 접근을 위해 thread간 lock 활성화
            std::unique_lock<std::mutex> lock(queue_mutex);
            
            // lock을 해제한 상태로 대기 -> notify시 wakeup_condition을 검사한 뒤 lock을 획득하고 진행
            // Spurious wakeups를 방지하기 위해 wakeup_condition인 lambda function을 사용
            // 그렇지 않으면, notify_all() / notify_one()이 호출되지 않았는데도 진행할 수 있음
            // 따라서, predicate는 스레드가 깨어나지 말아야 할 상황을 반드시 검사해야 하고,
            // 깨어나도 되는 상황일 때만 notify를 호출해야 함 
            condition.wait(lock, wakeup_condition);
            if (this->stop) {
                return;  // stop condition이 주어지면 thread 종료
            }
            // stop condition이 아닌 경우
            active_tasks += 1;  // lock이 걸려있으므로 안전
            // queue의 task를 가져온 후 제거
            task = std::move(queue.front());
            queue.pop();
        }
        task();  // task 실행

        active_tasks.fetch_sub(1, std::memory_order_relaxed);  // atomic subtract
        finished_tasks.fetch_add(1, std::memory_order_relaxed);  // atomic increment
    }
}

void ThreadPool::push(std::vector<std::function<void()>> tasks) {
    int tasks_size = tasks.size();
    finished_tasks.store(0);
    active_tasks.store(0);
    int task_ct = 0;

    while (true) {
        if (active_tasks.load() < num_threads && task_ct < tasks_size) {
            {   // queue에 task를 추가하기 위해 lock 활성화
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue.emplace(std::move(tasks[task_ct]));
            }
            // lock 해제 후 대기중인 worker thread에게 알림
            condition.notify_one();
            task_ct++;
        }
        if (finished_tasks.load() == tasks_size) {
            if (verbose) {
                std::cout << "Finished all tasks" << std::endl;
            }
            return;
        }
    }
}
