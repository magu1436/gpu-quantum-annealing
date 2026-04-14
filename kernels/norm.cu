# include "complex.cuh"
# include "complex_ops.cuh"

extern "C" __global__
void add_to_calc_norm(const Complex64* array, double* sum, int n) {
    extern __shared__ double sdata[];

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) *sum = 0.0;

    double val = 0.0;

    if (idx < n) val = array[idx].re * array[idx].re + array[idx].im * array[idx].im;
    sdata[threadIdx.x] = val;
    __syncthreads();

    // 0 ~ 31 番までのスレッドが, shared memory を通して 自身 + stride までの値を足し合わせる
    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // 0 ~ 31 番までのスレッド(1 warp) は shuffle を用いて足し合わせる
    // shered memory 経由より高速
    double warp_sum = sdata[threadIdx.x];
    if (threadIdx.x < 32) {
        unsigned mask = 0xffffffffu;
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(mask, warp_sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(sum, warp_sum);
        }
    }
}