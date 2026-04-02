extern "C" __global__
void scalar_multiple(const int scalar, const float* array, float* result, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.x * blockDim.y + threadIdx.z;

    int idx = y * n + x;
    if (x < n && y < n) {
        result[idx] = array[idx] * scalar;
    }
}