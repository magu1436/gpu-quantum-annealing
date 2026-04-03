#include "complex.cuh"
#include "complex_ops.cuh"

extern "C" __global__
void develop_time(Complex64* t, Complex64* f0, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * n + x;
    if (y == 0 && idx < n * n) {
        Complex64 sum;
        sum.re = 0.0;
        sum.im = 0.0;
        for (int i = 0; i < n; i++) {
            sum.re += t[idx + i].re + f0[i].re;
            sum.im = t[idx + i].im + f0[i].im;
        }
        fo[idx] = sum;
    }
}