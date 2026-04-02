# include "complex.cuh"
# include "complex_ops.cuh"

extern "C" __global__
void scalar_multiple(const float scalar, const float* array, float* result, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * n + x;
    if (x < n && y < n) {
        result[idx] = scalar * array[idx];
    }
}

extern "C" __global__
void scalar_multiple_complex(const float scalar, const Complex64* array, Complex64* result, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * n + x;
    if (x < n && y < n) {
        result[idx] = rscale(scalar, array[idx]);
    }
}