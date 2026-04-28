#pragma once

__device__ __forceinline__ void init(int* arr, int value, int length) {
    for (int i = 0; i < length; i++) {
        arr[i] = value;
    }
}

__device__ __forceinline__ int payoff_of_n_prisoners_dilemma(int p, int* prisoners, int num_player){
    // 全員自白
    bool flag = true;
    for (int i = 0; i < num_player; i++){
        if (prisoners[i] != 1){
            flag = false;
            break;
        }
    }
    if (flag) return -5;

    // 全員黙秘
    flag = true;
    for (int i = 0; i < num_player; i++){
        if (prisoners[i] != 0){
            flag = false;
            break;
        }
    }
    if (flag) return -3;
    
    // 自分が自白、ほか黙秘
    if (prisoners[p] == 1) return -1;

    // 自分が黙秘、ほか自白
    return -10;
}