# include "complex.cuh"
# include "complex_ops.cuh"

extern "C" __global__
void add_to_calc_norm(const Complex64* array, double* sum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        double val = array[idx].re * array[idx].re + array[idx].im * array[idx].im;
        atomicAdd(sum, val);
    }
}