#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <math.h>
#include "util.cuh"

#define N 21
#define Nums 2^21

// int payoff_of_3_prisoners_dilemma(int p, int i, int j, int k)
// {
//     /*1. 全員自白*/
//     if (i && j && k)
//         return -5;
//     /*2. 全員黙秘*/
//     if (!i && !j && !k)
//         return -3;

//     /*pで戦略を参照できるように strategy配列としてまとめる*/
//     int strategy[3] = {i, j, k};

//     /*3. 自分が自白、ほか黙秘あり*/
//     if (strategy[p] == 1)
//         return -1;
//     /*4. 自分が黙秘、ほか自白あり*/
//     return -10;
// }



int main(){
// 1. CPUメモリにおける変数宣言
    int i,j,k;
    // 定数宣言
    double H[Nums] = {0.0};
    cuDoubleComplex h_f0[Nums];
    for (i=0;i<Nums;i++){
        h_f0[i] = make_cuDoubleComplex(1.0 / sqrt(Nums), 0.0);
    }

    /*improve : */
    /*定数宣言*/
    /*0 : 黙秘 , 1 : 自白　とする*/
    const int num_player = 3;
    int strategies[num_player][2];
    for(int i = 0; i < num_player; i++){
        strategies[i] = {0, 1};
    }

    /*各プレイヤーの戦略数を取得*/
    int num_stg[num_player];
    int sum_stg = 0;
    for (i = 0; i < num_player; i++) {
        const int stg = len(strategies[i]);
        num_stg[i] = stg;
        sum_stg += stg;
    }

    // int x, y, z;                           /*戦略のバイナリ変数*/
    // int s1, s2, s3;                        /*ペナルティ項毎のスラック変数用*/
    int prisoners[num_player];
    init(prisoners, 0);
    int slacks[num_player];
    init(slacks, 0);
    int H0 = 0;                            /*目的関数部分のハミルトニアン関数 H0(q0,q1,q2)（演算子ではない。）*/
    int alpha = -5, beta = -5, gamma = -5; /*戦略の期待値を抑えるハイパーパラメータ*/
    int hyper_params[] = {alpha, beta, gamma};
    const int num_pen = num_stg[0] * num_player;  /*ペナルティ項の個数.*/
    const int num_slack = 3;               /*各ペナルティ項におけるスラック変数の個数*/  // 要修正
    const int start_slack = num_player;             /*スラックが始まるインデックス番号*/  // 要修正
    int Pen[num_pen];                      /*各ペナルティ項*/

     /*qubit数 : 戦略3つ + スラック3個*6行=18個 の計21個*/
    for (i = 0; i < Nums; i++)
    {
        // ペナルティ項の初期化
        init(Pen, 0);

        /*x,y,z は　iの２進数表記における左から 0,1,2個目のキュビット*/
        for(j = 0; j < num_player; j++){
            prisoners[j] = (i >> (N - j - 1)) & 1;
        }

        /*ハミルトニアンの目的関数値*/
        //H[i] += (-1) * payoff_sum[prisoners[0]][prisoners[1]][prisoners[2]];
        // 事前に payoff_sum に計算しておくのではなく, 逐次計算して与える仕様に変更
        for (j = 0; j < num_player; j++){
            H[i] += paypff_of_n_prisoners_dilemma(j, prisoners);
        }

        /*ペナルティ項の作成*/
        /*improve : Pen[index] の indexの形に拡張性がない。*/
        for (j = 0; j < num_player; j++){
            for (binary = 0; binary < 2; binary++){
                int* temp[num_player];
                memcpy(temp, prisoners, sizeof(prisoners));
                temp[j] = binary;
                Pen[(int)(pow(2, j) + binary)] = paypff_of_n_prisoners_dilemma(j, temp);
            }
        }

        for (j = 0; j < num_pen; j++)
        {
            for (k = 0; k < num_slack; k++) {
                Pen[j] += (int)pow(2, k) * ((i >> (N - 1 - j * num_slack - start_slack)) & 1);
            }
        }
        /*ペナルティ項にスラックと alpha,beta,gammaを加える*/
        for (j = 0; j < num_player; j++){
            for (k = 0; k < 2; k++){
                Pen[(int)(pow(2, j) + k)] -= hyper_params[j];
            }
        }

        for (j = 0; j < num_pen; j++)
        {
            Pen[j] = Pen[j] * Pen[j];
            H[i] += Pen[j];
        }

        for (j = 0; j < len(hyper_params); j++){
            H[i] += hyper_params[j];
        }
    }
}