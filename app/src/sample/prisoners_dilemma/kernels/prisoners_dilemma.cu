#include "util.cuh"
#define MAX_PLAYERS 32
#define MAX_PEN 128

// スラック変数の設定などに拡張性なし
// TODO: スラック変数の精度を表す変数を追加し, スラック変数の設定を行う仕様を実装
extern "C" __global__
void create_prisoners_dilemma_diag(
    int num_player,
    int num_pen,
    int num_slack,
    int start_slack,
    int* hyper_params,
    double* hll,
    int n,
    int hyper_params_num,
    int bit_count
){
    int state_number = blockIdx.x * blockDim.x + threadIdx.x;

    if (state_number >= n) return;

    int i, j, binary;
    int pen[MAX_PEN];
    init(pen, 0, num_pen);
    int h = 0;

    int actions[MAX_PLAYERS];
    for(j = 0; j < num_player; j++){
        actions[j] = (state_number >> (bit_count - j - 1)) & 1;
    }

    // ハミルトニアン目的関数値
    for (i = 0; i < num_player; i++) {
        h += -payoff_of_n_prisoners_dilemma(i, actions, num_player);
    }

    // ペナルティ項の作成
    for (i = 0; i < num_player; i++) {
        for (binary = 0; binary < 2; binary++) {
            int temp[MAX_PLAYERS];
            for (j = 0; j < num_player; j++) {
                temp[j] = actions[j];
            }
            temp[i] = binary;
            pen[i * 2 + binary] = payoff_of_n_prisoners_dilemma(i, temp, num_player);
        }
    }
    for (i = 0; i < num_pen; i++) {
        for (j = 0; j < num_slack; j++) {
            const int slack_index = i * num_slack + start_slack + j;    // j なかった
            pen[i] += (1 << j) * ((state_number >> (bit_count - 1 - slack_index)) & 1);
        }
    }
    for (i = 0; i < num_player; i++) {
        for (j = 0; j < 2; j++) {
            pen[i * 2 + j] -= hyper_params[i];
        }
    }
    for (i = 0; i < num_pen; i++) {
        h += pen[i] * pen[i];
    }
    for (i = 0; i < hyper_params_num; i++) {
        h += hyper_params[i];
    }
    hll[state_number] = (double)h;
}