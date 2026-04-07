#include "complex.cuh"
#include "complex_ops.cuh"

extern "C" __global__
void develop_time(Complex64* t, Complex64* f0, int n, Complex64* f1) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * n + x;
    if (!(x < n && y < n)) return;
    
    Complex64 prod = cmul(t[idx], f0[x]);
    atomicAdd(&f1[y].re, prod.re);
    atomicAdd(&f1[y].im, prod.im);
}