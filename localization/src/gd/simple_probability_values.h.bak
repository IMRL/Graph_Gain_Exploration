#ifndef SIMPLE_PROB_VALUES_H_
#define SIMPLE_PROB_VALUES_H_

#include <utility>
#include <cmath>

constexpr size_t PROB_N_MAX = 12;

template <size_t N, typename = std::enable_if_t<(N > 0 && N < PROB_N_MAX)>>
class simple_prob_values
{
    static constexpr uint32_t value_bound = uint32_t(1) << N;
    static constexpr float prob_min = 0.1;
    static constexpr float prob_max = 1.0 - prob_min;

    static float ToOdds(float prob) { return prob / (1 - prob); }
    static float ToProb(float odds) { return odds / (1 + odds); }
    static uint16_t Bound(float prob) { return std::round((std::min(std::max(prob, prob_min), prob_max) - prob_min) / (prob_max - prob_min) * (max_value - min_value) + min_value); }
    static float Unbound(uint16_t bound) { return (bound - min_value) / float(max_value - min_value) * (prob_max - prob_min) + prob_min; }

public:
    static constexpr uint16_t init_value = uint16_t(1) << (N - 1);
    static constexpr uint16_t min_value = uint16_t(1);
    static constexpr uint16_t max_value = value_bound - 1;
    static inline std::array<std::array<uint16_t, value_bound>, value_bound> generate_value_mat()
    {
        std::array<std::array<uint16_t, value_bound>, value_bound> prob_mat{};
        for (size_t r = 1; r < value_bound; r++)
            for (size_t c = 1; c < value_bound; c++)
            {
                prob_mat[r][c] = Bound(ToProb(ToOdds(Unbound(r)) * ToOdds(Unbound(c))));
            }
        for(size_t i=1;i<value_bound;i++)
        {
            prob_mat[0][i] = prob_mat[init_value][i];
            prob_mat[i][0] = prob_mat[i][init_value];
        }
        return prob_mat;
    }
    static inline std::array<uint16_t, value_bound> generate_step_table(const float step)
    {
        std::array<uint16_t, value_bound> step_table{};
        for (size_t i = 1; i < value_bound; i++)
        {
            step_table[i] = Bound(ToProb(ToOdds(Unbound(i)) * ToOdds(step)));
        }
        step_table[0] = step_table[init_value];
        return step_table;
    }
};

#endif
