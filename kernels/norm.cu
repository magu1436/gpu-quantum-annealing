# include "complex.cuh"
# include "complex_ops.cuh"

extern "C" __global__
void norm(const Complex64* array, float* sum, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * n + x;
    if (x < n && y < n) {
        double val = array[idx].re * array[idx].re + array[idx].im * array[idx].im;
        atmomicAdd(sum, val);
    }
}