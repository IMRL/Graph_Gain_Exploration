#ifndef _MULTI_EXECUTOR_H_
#define _MULTI_EXECUTOR_H_

#include <functional>
#include <vector>
#include <array>
#include <thread>

inline void worker(const std::function<void()> *tasks, size_t N)
{
    for (size_t i = 0; i < N; i++)
    {
        tasks[i]();
    }
}

template <size_t N>
void multi_execute(const std::vector<std::function<void()>> &tasks)
{
    std::array<size_t, N> task_split;
    size_t K = ((tasks.size() + N - 1) / N);
    for (size_t i = 0; i < N; i++)
    {
        if ((i + 1) * K <= tasks.size())
            task_split[i] = K;
        else if (i * K < tasks.size())
            task_split[i] = tasks.size() - i * K;
        else
            task_split[i] = 0;
    }
    std::vector<std::thread> threads;
    size_t total = 0;
    for (size_t i = 0; i < N; i++)
    {
        threads.emplace_back(worker, &tasks[total], task_split[i]);
        total += task_split[i];
        if (total == tasks.size())
            break;
    }
    for (auto &thread : threads)
    {
        thread.join();
    }
}

template <>
inline void multi_execute<1>(const std::vector<std::function<void()>> &tasks)
{
    worker(&tasks[0], tasks.size());
}

#endif
