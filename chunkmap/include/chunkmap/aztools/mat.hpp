#ifndef _AZURITY_TOOLS_MAT_HPP_
#define _AZURITY_TOOLS_MAT_HPP_

#include <opencv2/opencv.hpp>
#include "bitset.hpp"

namespace aztools::bitsmat
{
    template <typename T>
    struct wrap
    {
        using type = void;
    };

    template <size_t N, typename Impl>
    struct wrap<bitset<N, Impl>>
    {
    private:
        template <unsigned int M, unsigned int P = 0>
        static constexpr unsigned int Log2()
        {
            if constexpr (M <= 1u)
            {
                return P;
            }
            else
            {
                return Log2<M / 2, P + 1>();
            }
        }
        template <unsigned int M, unsigned int P = 1>
        static constexpr unsigned int Exp2()
        {
            if constexpr (M <= 0u)
            {
                return P;
            }
            else
            {
                return Exp2<M - 1, P * 2>();
            }
        }

        using T = bitset<N, Impl>;
        static constexpr size_t less_cap = Exp2<Log2<N>()>();
        static constexpr size_t cap = (less_cap < N ? less_cap * 2 : less_cap) >> 3;
        static_assert(cap <= CV_CN_MAX);

    public:
        using type = cv::Vec<uint8_t, cap>;

        T &operator()(type &value) const
        {
            return *reinterpret_cast<T *>(&value);
        }
        const T &operator()(const type &value) const
        {
            return *reinterpret_cast<T const *>(&value);
        }
    };

    template <typename T>
    using Mat_ = cv::Mat_<typename wrap<T>::type>;
}

#endif
