#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 2000  // Tamanho da grade
#define T 1000  // Quantidade de iterações
#define D 0.1   // Coeficiente de coesão
#define DELTA_T 0.01
#define DELTA_X 1.0

void diff_eq(double **C, double **C_new) {
    for (int t = 0; t < T; t++) {
        // Calculo da equação de difusão para toda a matrix
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                C_new[i][j] = C[i][j] + D * DELTA_T * (
                    (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4 * C[i][j]) / (DELTA_X * DELTA_X)
                );
            }
        }

        // Atualiza a matriz para a próxima iteração
        double difmedio = 0.;
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                difmedio += fabs(C_new[i][j] - C[i][j]);    // fabs = pega o valor absoluto
                C[i][j] = C_new[i][j];
            }
        }
        if ((t % 100) == 0)
            printf("Iteração %d - diferença média=%g\n", t, difmedio / ((N - 2) * (N - 2)));
    }
}

int main() {
    // ------- Concentração Inicial -------
    // Cria a matriz C de tamanho N
    double **C = (double **)malloc(N * sizeof(double *));
    if (C == NULL) {
        fprintf(stderr, "Falha na alocação de memória\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        C[i] = (double *)malloc(N * sizeof(double));
        if (C[i] == NULL) {
            fprintf(stderr, "Falha na alocação de memória\n");
            return 1;
        }
    }

    // Inicializa a matriz C
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
        }
    }

    double **C_new = (double **)malloc(N * sizeof(double *));
    if (C_new == NULL) {
        fprintf(stderr, "Falha na alocação de memória\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        C_new[i] = (double *)malloc(N * sizeof(double));
        if (C_new[i] == NULL) {
            fprintf(stderr, "Falha na alocação de memória\n");
            return 1;
        }
    }

    // Inicializa a matriz C_new
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C_new[i][j] = 0.0;
        }
    }

    // Inicializa a concentração no centro
    C[N / 2][N / 2] = 1.0;

    // Executa o processo da equação de difusão
    diff_eq(C, C_new);

    // Exibe os resultados
    printf("\nConcentração final no centro: %f\n", C[N / 2][N / 2]);

    // // Salva a matrix no arquivo de planilha
    // FILE *fp = fopen("/content/matriz_sequencial_output.txt", "w");

    // // Salvando matrix no aqruivo txt
    // if(fp == NULL) {
    //   printf("Erro ao abrir arquivo .csv");
    // }
    // else {
    //   for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //       if(C[i][j] >= 0.0001)
    //         fprintf(fp, "i:%d j:%d Matriz:%f ", i, j, C[i][j]);
    //     }
    //     fprintf(fp, "\n");
    //   }
    //   fclose(fp);
    // }

    // Libera memória alocada
    for (int i = 0; i < N; i++) {
        free(C[i]);
        free(C_new[i]);
    }
    free(C);
    free(C_new);

    return 0;
}