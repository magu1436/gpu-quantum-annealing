#include "complex.cuh"
#include "complex_ops.cuh"
#include "bit.cuh"

extern "C" __global__
void develop_time_latest(
    double a,
    double b,
    double dt,
    const double* diag,
    const Complex64* f0,
    int n,
    int bit_count,
    Complex64* f1
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (!(x < n)) return;

    Complex64 z;
    z.re = 0.0;
    z.im = 0.0;

    // 対角項だけ先に加算
    Complex64 t_diag;
    t_diag.re = 1.0;
    t_diag.im = -0.5 * dt * (b + a * diag[x]);
    z = cmul(t_diag, f0[x]);

    // 非対角項
    Complex64 t_off;
    t_off.re = 0.0;
    t_off.im = 0.5 * dt * b;

    for (int bit = 0; bit < bit_count; bit++) {
        // 一箇所だけ 1 でそれ以外が 0 である数値を作って x を反転させて, 
        // ハミング距離が 1 の y を作成
        // 1u は unsigned int 1 (符号なし整数) を表す
        int y = x ^ (1u << bit);

        // 念のためメモリアクセス範囲外でないかチェック
        if (!(y < n)) continue;
        
        // 非対角項の加算
        z = cadd(z, cmul(t_off, f0[y]));
    }
    f1[x] = z;
}