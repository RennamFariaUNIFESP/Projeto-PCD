
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 2000  // Tamanho da grade
#define T 1000  // Quantidade de iterações
#define D 0.1   // Coeficiente de coesão
#define DELTA_T 0.01
#define DELTA_X 1.0

void diff_eq(double **C, double **C_new) { 
    int i, j;
    
    for (int t = 0; t < T; t++) {
        // Paralelização da equação de difusão
        #pragma omp parallel for collapse(2) shared(C_new, C) private(i, j)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                C_new[i][j] = C[i][j] + D * DELTA_T * (
                    (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4 * C[i][j]) / (DELTA_X * DELTA_X)
                );
            }
        }

        // Atualiza a matriz para a próxima iteração
        double difmedio = 0.;
        
        // Paralelização da diferença média
        #pragma omp parallel for collapse(2) shared(C_new, C) private(i, j) reduction(+:difmedio) 
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                difmedio += fabs(C_new[i][j] - C[i][j]);    // fabs = pega o valor absoluto
                C[i][j] = C_new[i][j];
            }
        }

        // // Checar a quantidade de threads rodando
        // if (t == 0) {
        //     #pragma omp parallel
        //     {
        //         if (omp_get_thread_num() == 0) {
        //             printf("Número total de threads sendo usadas: %d\n", omp_get_num_threads());
        //         }
        //     }
        // }
        
        if ((t % 100) == 0)
          printf("Iteração %d - diferença média=%g\n", t, difmedio / ((N - 2) * (N - 2)));
    }
}

int main() {
    double start_time, elapsed_time;

    // Inicializa o tempo do código
    start_time = omp_get_wtime();

    int i, j;

    // int numb_threads = 2;
    // int numb_threads = 3;
    int numb_threads = 4;
    // int numb_threads = 6;
    // int numb_threads = 8;

    // Configuração do número de threads do OpenMP
    omp_set_num_threads(numb_threads);

    // ------- Concentração Inicial -------
    // Cria a matriz C de tamanho N
    double **C = (double **)malloc(N * sizeof(double *));

    // Verifica se a matriz foi criada corretamente
    if (C == NULL) {
      fprintf(stderr, "Falha na alocação de memória\n");
      return 1;
    }

    // Cria o restante da matriz C de tamanho N*N
    for (int i = 0; i < N; i++) {
      C[i] = (double *)malloc(N * sizeof(double));
      if (C[i] == NULL) {
        fprintf(stderr, "Falha na alocação de memória\n");
        return 1;
      }
    }

    // Limpa a matriz C
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        C[i][j] = 0.;
      }
    }

    // ------- Concentração para a próxima iteração -------
    // Cria a matriz C_new de tamanho N
    double **C_new = (double **)malloc(N * sizeof(double *));

    // Verifica se a matriz foi criada corretamente
    if (C_new == NULL) {
      fprintf(stderr, "Falha na alocação de memória\n");
      return 1;
    }

    // Cria o restante da matriz C_new de tamanho N*N
    for (int i = 0; i < N; i++) {
      C_new[i] = (double *)malloc(N * sizeof(double));
      if (C_new[i] == NULL) {
        fprintf(stderr, "Falha na alocação de memória\n");
        return 1;
      }
    }

    // Limpa a matriz C_new
    #pragma omp parallel for collapse(2) shared(C_new) private(i, j)
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        C_new[i][j] = 0.;
      }
    }

    // Inicializa a concentração no centro da matriz C
    C[N / 2][N / 2] = 1.0;

    // Executa o processo da equação de difusão com as matrizes
    diff_eq(C, C_new);

    // Exibe os resultados
    printf("\nConcentração final no centro: %f\n", C[N / 2][N / 2]);

    // // Salva a matrix no arquivo de planilha
    //FILE *fp = fopen("/content/openMP.txt", "w");

    // // Salvando matrix no aqruivo txt
    // if(fp == NULL) {
    //   printf("Erro ao abrir arquivo .txt\n");
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

    // Liberar memória alocada
    for (int i = 0; i < N; i++) {
        free(C[i]);
        free(C_new[i]);
    }
    free(C);
    free(C_new);

    // Finaliza o tempo do processo da equação
    elapsed_time = (omp_get_wtime() - start_time);

    printf("Tempo final do código: %f\n", elapsed_time);

    return 0;
}