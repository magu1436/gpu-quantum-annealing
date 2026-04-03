#include "complex.cuh"
#include "complex_ops.cuh"

extern "C" __global__
void update_f0(Complex64* f_temp, double* sum, int n, Complex64* f0) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * n + x;

    if (x < n && y < n) {
        f0[idx] = rscale(1.0 / sum[0], f_temp[idx]);
    }
}