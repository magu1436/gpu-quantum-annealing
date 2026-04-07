#include "complex.cuh"
#include "complex_ops.cuh"

extern "C" __global__
void update_f0(Complex64* f0, double* sum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        f0[idx] = rscale(1.0 / sum[0], f0[idx]);
    }
}