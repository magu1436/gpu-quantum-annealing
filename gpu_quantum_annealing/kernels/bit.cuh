#pragma once

__device__ __forceinline__ bool is_diff_by_one_bit(unsigned int i, unsigned int j) {
    return __popc(i ^ j) == 1;
}