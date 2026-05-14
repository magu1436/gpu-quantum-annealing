#include "complex.cuh"
#include "complex_ops.cuh"

extern "C" __global__
void quadratic_develop_time_warp(
    double a,
    double b,
    double dt,
    const double* diag,
    const Complex64* f_current,
    const Complex64* f_prev,
    int n,
    int bit_count,
    Complex64* f_developed
) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warp_per_block + warp_in_block;

    if (!(row < n)) return;

    Complex64 sum;
    sum.re = 0.0;
    sum.im = 0.0;

    // lane 0 は対角項の計算も実施
    if (lane == 0) {
        Complex64 diag_elem;
        diag_elem.re = 0.0;
        diag_elem.im = dt * a * diag[row];  // 2 * 0.5 * dt * a * diag[row];
        sum = cmul(diag_elem, f_current[row]);
    }

    // 非対角成分
    Complex64 non_diag_elem;
    non_diag_elem.re = 0.0;
    non_diag_elem.im = dt * b; // -2 * 0.5 * dt * b;

    for (int bit = lane; bit < bit_count; bit += 32) {
        unsigned int col = row ^ (1u << bit);
        sum = cadd(sum, cmul(non_diag_elem, f_current[col]));
    }

    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum.re += __shfl_xor_sync(mask, sum.re, offset);
        sum.im += __shfl_xor_sync(mask, sum.im, offset);
    }
    if (lane == 0) {
        f_developed[row] = csub(f_prev[row], sum);
    }
}