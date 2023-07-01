#ifndef _AZURITY_TOOLS_BITSMAT_BITSET_HPP_
#define _AZURITY_TOOLS_BITSMAT_BITSET_HPP_

#include <utility>
#include <vector>
#include <cassert>

namespace aztools::bitsmat
{

    struct bitset_t
    {
        template <typename Content>
        struct Accessor
        {
            Content *const content;
            size_t start;
            size_t size;

            Accessor(Content *const content, size_t start, size_t size) : content{content}, start{start}, size{size} {}
            Accessor(const Accessor &) = delete;
            Accessor(Accessor &&) = delete;

            bool get(size_t pos) const { return content->test(start + pos); }

            template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
            Accessor &operator=(const T &value)
            {
                const size_t value_len = sizeof(T) * 8;
                const size_t set_len = std::min(value_len, size);
                for (size_t offset = 0; offset < set_len; offset++)
                    content->set(start + offset, value & (T(1) << offset));
                for (size_t offset = set_len; offset < size; offset++)
                    content->reset(start + offset);
                return *this;
            }
            template <typename VT>
            Accessor &operator=(const Accessor<VT> &value)
            {
                assert(value.size == size && (value.content != content || (value.start + value.size <= start || start + size <= value.start)));
                for (size_t offset = 0; offset < size; offset++)
                    content->set(start + offset, value.get(offset));
                return *this;
            }
            template <typename VT>
            Accessor &operator=(Accessor<VT> &&value)
            {
                assert(value.size == size && (value.content != content || (value.start + value.size <= start || start + size <= value.start)));
                for (size_t offset = 0; offset < size; offset++)
                    content->set(start + offset, value.get(offset));
                return *this;
            }

            operator uint8_t() const
            {
                uint8_t value = 0;
                for (int offset = size - 1; offset >= 0; offset--)
                {
                    value <<= 1;
                    value |= content->test(start + offset) ? 1 : 0;
                }
                return value;
            }
            operator uint16_t() const
            {
                uint16_t value = 0;
                for (int offset = size - 1; offset >= 0; offset--)
                {
                    value <<= 1;
                    value |= content->test(start + offset) ? 1 : 0;
                }
                return value;
            }
            operator uint32_t() const
            {
                uint32_t value = 0;
                for (int offset = size - 1; offset >= 0; offset--)
                {
                    value <<= 1;
                    value |= content->test(start + offset) ? 1 : 0;
                }
                return value;
            }
            operator uint64_t() const
            {
                uint64_t value = 0;
                for (int offset = size - 1; offset >= 0; offset--)
                {
                    value <<= 1;
                    value |= content->test(start + offset) ? 1 : 0;
                }
                return value;
            }
        };
    };

    template <size_t N, typename Impl>
    struct bitset : bitset_t
    {
    private:
        using _Type = bitset<N, Impl>;
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

        static constexpr size_t BitsOffset = Log2<sizeof(Impl)>() + 3;
        static constexpr size_t BitsMax = sizeof(Impl) * 8 - 1;

        Impl content[(N + BitsMax) >> BitsOffset];
        static constexpr size_t cap = (N + BitsMax) >> BitsOffset;

    public:
        static constexpr size_t MAX_BITS = BitsMax;

        constexpr bitset() : content()
        {
            for (size_t i = 0; i < cap; i++)
            {
                content[i] = 0;
            }
        }

        constexpr bitset(std::initializer_list<Impl> args) : content()
        {
            size_t i = 0;
            for (auto it = args.begin(); it != args.end(); it++)
            {
                content[i++] = *it;
                if (i >= cap)
                    break;
            }
            for (int j = i; j < cap; j++)
            {
                content[j] = 0;
            }
        }

        template <typename It>
        bitset(It first, It last) : content()
        {
            size_t i = 0;
            for (auto it = first; it != last; it++)
            {
                content[i++] = *it;
                if (i >= cap)
                    break;
            }
            for (int j = i; j < cap; j++)
            {
                content[j] = 0;
            }
        }

        constexpr size_t size() const
        {
            return N;
        }

        bool test(size_t pos) const
        {
            return content[pos >> BitsOffset] & (Impl(1) << (pos & BitsMax));
        }

        constexpr bool all() const
        {
            Impl result = ~Impl(0);
            for (int i = 0; i < cap - 1; i++)
            {
                result &= content[i];
            }
            if (N & BitsMax)
                result &= (content[cap - 1] | ~((1ul << (N & BitsMax)) - 1));
            else
                result &= content[cap - 1];
            constexpr Impl mask = ~Impl(0);
            return result == mask;
        }

        constexpr bool any() const
        {
            uint64_t result = 0;
            for (int i = 0; i < cap - 1; i++)
            {
                result |= content[i];
            }
            if (N & BitsMax)
                result |= (content[cap - 1] & ((1ul << (N & BitsMax)) - 1));
            else
                result |= content[cap - 1];
            return result != 0;
        }

        constexpr bool none() const { return !any(); }

        constexpr size_t count() const
        {
            size_t result = 0;
            for (size_t i = 0; i < N; i++)
            {
                if (content[i >> BitsOffset] & (1ull << (i & BitsMax)))
                {
                    result++;
                }
            }
            return result;
        }

        constexpr bool operator==(const _Type &bit) const
        {
            bool result = true;
            for (size_t i = 0; i < cap; i++)
            {
                result &= content[i] == bit.content[i];
            }
            return result;
        }
        constexpr bool operator!=(const _Type &bit) const { return !operator==(bit); }

        constexpr _Type operator&(const _Type &bit) const
        {
            _Type result;
            for (size_t i = 0; i < cap; i++)
            {
                result.content[i] = content[i] & bit.content[i];
            }
            return result;
        }
        constexpr _Type operator|(const _Type &bit) const
        {
            _Type result;
            for (size_t i = 0; i < cap; i++)
            {
                result.content[i] = content[i] | bit.content[i];
            }
            return result;
        }
        constexpr _Type operator^(const _Type &bit) const
        {
            _Type result;
            for (size_t i = 0; i < cap; i++)
            {
                result.content[i] = content[i] ^ bit.content[i];
            }
            return result;
        }
        constexpr _Type operator~() const
        {
            _Type result;
            for (size_t i = 0; i < cap; i++)
            {
                result.content[i] = ~content[i];
            }
            return result;
        }

        _Type &operator&=(const _Type &bit)
        {
            for (size_t i = 0; i < cap; i++)
            {
                content[i] &= bit.content[i];
            }
            return *this;
        }
        _Type &operator|=(const _Type &bit)
        {
            for (size_t i = 0; i < cap; i++)
            {
                content[i] |= bit.content[i];
            }
            return *this;
        }
        _Type &operator^=(const _Type &bit)
        {
            for (size_t i = 0; i < cap; i++)
            {
                content[i] ^= bit.content[i];
            }
            return *this;
        }

        constexpr _Type operator<<(size_t n) const
        {
            _Type result;
            size_t base = (n >> BitsOffset);
            size_t delta = (n & BitsMax);
            uint64_t old = 0;
            for (int i = base; i < cap; i++)
            {
                result.content[i] = (old >> (BitsMax + 1 - delta)) | (content[i - base] << delta);
                old = content[i - base];
            }
            return result;
        }
        constexpr _Type operator>>(size_t n) const
        {
            _Type result;
            size_t base = (n >> 6);
            size_t delta = (n & BitsMax);
            uint64_t old = 0;
            for (size_t i = cap; i > base; i--)
            {
                result.content[i - base - 1] = (old << (BitsMax + 1 - delta)) | (content[i - 1] >> delta);
                old = content[i - 1];
            }
            return result;
        }

        _Type &set(size_t pos, bool val = true)
        {
            assert(pos < N);
            if (val)
            {
                content[pos >> BitsOffset] |= (Impl(1) << (pos & BitsMax));
            }
            else
            {
                content[pos >> BitsOffset] &= ~(Impl(1) << (pos & BitsMax));
            }
            return *this;
        }
        _Type &reset(size_t pos)
        {
            assert(pos < N);
            content[pos >> BitsOffset] &= ~(Impl(1) << (pos & BitsMax));
            return *this;
        }
        _Type &flip(size_t pos)
        {
            assert(pos < N);
            content[pos >> BitsOffset] ^= (Impl(1) << (pos & BitsMax));
            return *this;
        }

        Accessor<_Type> operator[](size_t pos)
        {
            return operator()(pos, 1);
        }

        Accessor<_Type> operator()(size_t start = 0, int n = -1)
        {
            if (n < 0)
                n = N - start;
            assert(start + n <= N);
            return Accessor<_Type>(this, start, n);
        }

        const Accessor<const _Type> operator()(size_t start = 0, int n = -1) const
        {
            if (n < 0)
                n = N - start;
            assert(start + n <= N);
            return Accessor<const _Type>(this, start, n);
        }
    };
}

#endif
