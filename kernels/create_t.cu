#include "complex.cuh"
#include "complex_ops.cuh"
#include "bit.cuh"

extern "C" __global__
void create_t(Complex64* f0, double a, double b, double dt, double* diag, int n, Complex64* t) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * n + x;
    if (!(x < n && y < n)) return;

    Complex64 z;
    if (x == y) {
        z.re = 1.0;
        z.im = -0.5 * dt * a * diag[x];
        t[idx] = z
    } else if (is_diff_by_one_bit(x, y)) {
        z.re = 1.0;
        z.im = 0.5 * dt * b;
        t[idx] = z
    } else {
        z.re = 0.0;
        z.im = 0.0;
        t[idx] = z
    }
}