#pragma once
# include "complex.cuh"

__device__ inline Complex64 cadd(Complex64 a, Complex64 b) {
    Complex64 z;
    z.re = a.re + b.re;
    z.im = a.im + b.im;
    return z;
}

__device__ inline Complex64 csub(Complex64 a, Complex64 b) {
    Complex64 z;
    z.re = a.re - b.re;
    z.im = a.im - b.im;
    return z;
}

__device__ inline Complex64 cmul(Complex64 a, Complex64 b) {
    Complex64 z;
    z.re = a.re * b.re - a.im * b.im;
    z.im = a.re * b.im + a.im * b.re;
    return z;
}

__device__ inline Complex64 rscale(double s, Complex64 z) {
    Complex64 result;
    result.re = z.re * s;
    result.im = z.im * s;
    return result;
}