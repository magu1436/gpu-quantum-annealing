extern "C" __global__
void integer_partition(const int* nums, unsigned int bit_count, double* result) {
    unsigned int state = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int state_count = 1u << bit_count;
    if (state >= state_count) return;

    double sum = 0.0;
    for (unsigned int i = 0; i < bit_count; i++) {
        for (unsigned int j = i + 1; j < bit_count; j++) {
            int a = 2 * ((state >> (bit_count - 1 - i)) & 1) - 1;
            int b = 2 * ((state >> (bit_count - 1 - j)) & 1) - 1;
            sum += (double)a * (double)b * (double)nums[i] * (double)nums[j];
        }
    }
    result[state] = sum;
}