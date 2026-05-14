#include "complex.cuh"
#include "complex_ops.cuh"

extern "C" __global__
void develop_time_warp(
    double a,
    double b,
    double dt,
    const double* diag,
    const Complex64* f0,
    int n,
    int bit_count,
    Complex64* f1
) {
    // warp内における lane 番号を取り出す
    // 二進数列のうち、下5桁のみを取り出している
    const int lane = threadIdx.x & 31;

    // 所属 block における warp 番号
    const int warp_in_block = threadIdx.x >> 5;

    // 1 block あたりの warp 数
    // blockDim.x は block 内の thread 数 を示す
    // blockDim.x >> 5 は blockDim.x / 32 とほぼ同じ (端数切捨て)
    // 1234 を 10 で割ると 123.4 となり, 小数点以下を切り捨てると 123 となる
    const int warp_per_block = blockDim.x >> 5;

    // 全体 warp としたときのインデックス
    const int row = blockIdx.x * warp_per_block + warp_in_block;

    if (!(row < n)) return;

    Complex64 sum;
    sum.re = 0.0;
    sum.im = 0.0;

    // lane 0 は対角項の計算も実施
    if (lane == 0) {
        Complex64 t_diag;
        t_diag.re = 1.0;
        t_diag.im = -0.5 * dt * a * diag[row];
        sum = cmul(t_diag, f0[row]);
    }

    Complex64 t_off;
    t_off.re = 0.0;
    t_off.im = -0.5 * dt * b;

    // 非対角項の計算
    // lane, lane + 32, lane + 64, ... の総和をとる
    for (int bit = lane; bit < bit_count; bit += 32) {
        unsigned col = row ^ (1u << bit);
        // sum_re += 0.0;
        sum = cadd(sum, cmul(t_off, f0[col]));
    }

    // 総和に参加する lane を示す
    // 例えば, 11010 なら, lane 1, 3, 4 を参加させる
    // 今回は全 lane を参加させる
    unsigned mask = 0xffffffffu;
    // 同一 warp 内の総和を取得する
    for (int offset = 16; offset > 0; offset >>= 1) {
        // 自身の lane 番号に offset だけ足した lane の値を取得し, 加算する
        sum.re += __shfl_xor_sync(mask, sum.re, offset);
        sum.im += __shfl_xor_sync(mask, sum.im, offset);
    }
    if (lane == 0) {
        f1[row] = sum;
    }
}