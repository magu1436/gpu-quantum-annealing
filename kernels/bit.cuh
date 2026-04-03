#include <bit>

__device__ inline bool is_diff_by_one_bit(unsigned int i, unsigned int j) {
    return std::popcount(i ^ j) == 1;
}