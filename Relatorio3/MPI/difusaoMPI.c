#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 2000 // Tamanho da grade
#define T 1000  // Quantidade de iterações
#define D 0.1   // Coeficiente de coesão
#define DELTA_T 0.01
#define DELTA_X 1.0

// Função para verificar se um número é potência de 2
int is_power_of_two(int x) {
    return (x > 0) && ((x & (x - 1)) == 0);
}

void diff_eq(double **C, double **C_new, int localRows, int localCols, int rowStart, int colStart, int rows, int cols) {
    MPI_Status status;
    int myId;
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);

    int row = myId / cols;
    int col = myId % cols;

    int up    = (row > 0) ? myId - cols : MPI_PROC_NULL;
    int down  = (row < rows - 1) ? myId + cols : MPI_PROC_NULL;
    int left  = (col > 0) ? myId - 1 : MPI_PROC_NULL;
    int right = (col < cols - 1) ? myId + 1 : MPI_PROC_NULL;

    for (int t = 0; t < T; t++) {
        MPI_Barrier(MPI_COMM_WORLD);

        // Troca de informações verticais
        if (up != MPI_PROC_NULL) {
            MPI_Sendrecv(C[1], localCols, MPI_DOUBLE, up, 0,
                        C[0], localCols, MPI_DOUBLE, up, 1,
                        MPI_COMM_WORLD, &status);
        }
        if (down != MPI_PROC_NULL) {
            MPI_Sendrecv(C[localRows-2], localCols, MPI_DOUBLE, down, 1,
                        C[localRows-1], localCols, MPI_DOUBLE, down, 0,
                        MPI_COMM_WORLD, &status);
        }

        double *send_col = (double *)malloc(localRows * sizeof(double));
        double *recv_col = (double *)malloc(localRows * sizeof(double));

        // Troca de informações horizontais
        if (left != MPI_PROC_NULL || right != MPI_PROC_NULL) {
            for (int i = 0; i < localRows; i++) {
                send_col[i] = C[i][1];
            }
        }
        if (left != MPI_PROC_NULL) {
            MPI_Sendrecv(send_col, localRows, MPI_DOUBLE, left, 2,
                        recv_col, localRows, MPI_DOUBLE, left, 3,
                        MPI_COMM_WORLD, &status);
            for (int i = 0; i < localRows; i++) {
                C[i][0] = recv_col[i];
            }
        }
        if (right != MPI_PROC_NULL) {
            for (int i = 0; i < localRows; i++) {
                send_col[i] = C[i][localCols-2];
            }
            MPI_Sendrecv(send_col, localRows, MPI_DOUBLE, right, 3,
                        recv_col, localRows, MPI_DOUBLE, right, 2,
                        MPI_COMM_WORLD, &status);
            for (int i = 0; i < localRows; i++) {
                C[i][localCols-1] = recv_col[i];
            }
        }
        free(send_col);
        free(recv_col);

        // Equação diferencial
        double difmedio = 0.0;
        for (int i = 1; i < localRows-1; i++) {
            for (int j = 1; j < localCols-1; j++) {
                C_new[i][j] = C[i][j] + D * DELTA_T * (
                    (C[i-1][j] + C[i+1][j] + C[i][j-1] + C[i][j+1] - 4.0 * C[i][j])
                    / (DELTA_X * DELTA_X)
                );
                difmedio += fabs(C_new[i][j] - C[i][j]);
                C[i][j] = C_new[i][j];
            }
        }

        if (t % 100 == 0) {
            printf("Process %d - Iteração %d - diferença média=%g\n", myId, t, difmedio / ((localRows - 2) * (localCols - 2)));
            fflush(stdout);
        }
    }
}

int save_results(double **C, int localRows, int localCols, int rows, int cols, int myId, int numProcs) {
    double *localData = NULL;
    double *globalData = NULL;
    int localSize = (localRows - 2) * (localCols - 2);  // Tamanho sem as bordas

    // Aloca memória para dados globais no processo 0
    if (myId == 0) {
        globalData = (double *)malloc(numProcs * localSize * sizeof(double));
        if (globalData == NULL) {
            fprintf(stderr, "Failed to allocate global data array\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    // Aloca memória para dados locais em todos os processos
    localData = (double *)malloc(localSize * sizeof(double));
    if (localData == NULL) {
        fprintf(stderr, "Failed to allocate local data array\n");
        if (myId == 0) free(globalData);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    // Copia dados locais para o array linear
    int idx = 0;
    for (int i = 1; i < localRows-1; i++) {
        for (int j = 1; j < localCols-1; j++) {
            localData[idx++] = C[i][j];
        }
    }

    // Coleta todos os dados no processo 0
    MPI_Gather(localData, localSize, MPI_DOUBLE,
               globalData, localSize, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    // Processo 0 salva os dados no arquivo
    if (myId == 0) {
        FILE *fp = fopen("matriz_MPI_output.txt", "w");
        if (fp == NULL) {
            printf("Erro ao abrir arquivo.txt\n");
            free(globalData);
            free(localData);
            return 1;
        }

        for (int p = 0; p < numProcs; p++) {
            int pRow = p / cols;
            int pCol = p % cols;
            int baseI = pRow * (N / rows);
            int baseJ = pCol * (N / cols);
            int offset = p * localSize;

            for (int i = 0; i < localRows-2; i++) {
                for (int j = 0; j < localCols-2; j++) {
                    double val = globalData[offset + i * (localCols-2) + j];
                    if (val >= 0.0001) {
                        fprintf(fp, "i:%d j:%d Matriz:%f\n", 
                                baseI + i, baseJ + j, val);
                    }
                }
            }
        }
        fclose(fp);
    }

    // Libera memória
    free(localData);
    if (myId == 0) {
        free(globalData);
    }

    return 0;
}

int main(int argc, char *argv[]) {
    int myId, numProcs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    if (!is_power_of_two(numProcs)) {
        if (myId == 0) {
            printf("Erro: O número de processos deve ser potência de 2 (1,2,4,8,16,...).\n");
        }
        MPI_Finalize();
        return 1;
    }

    int rows, cols;
    // Find the best (rows, cols) such that rows * cols = numProcs
    for (int i = sqrt(numProcs); i >= 1; i--) {
        if (numProcs % i == 0) {
            rows = i;
            cols = numProcs / i;
            break;
        }
    }

    if (myId == 0) {
        printf("Using grid: %d x %d\n", rows, cols);
    }

    int localRows = (N / rows) + 2;
    int localCols = (N / cols) + 2;
    int rowStart = (myId / cols) * (N / rows);
    int colStart = (myId % cols) * (N / cols);

    double **C = (double **)malloc(localRows * sizeof(double *));
    double **C_new = (double **)malloc(localRows * sizeof(double *));
    for (int i = 0; i < localRows; i++) {
        C[i] = (double *)calloc(localCols, sizeof(double));
        C_new[i] = (double *)calloc(localCols, sizeof(double));
    }

    int centerGlobal = N / 2;
    if ((centerGlobal >= rowStart && centerGlobal < rowStart + localRows - 2) &&
        (centerGlobal >= colStart && centerGlobal < colStart + localCols - 2)) {
        int localI = (centerGlobal - rowStart) + 1;
        int localJ = (centerGlobal - colStart) + 1;
        C[localI][localJ] = 1.0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    diff_eq(C, C_new, localRows, localCols, rowStart, colStart, rows, cols);
    MPI_Barrier(MPI_COMM_WORLD);

    // Salva os resultados, separado para uma Função
    // save_results(C, localRows, localCols, rows, cols, myId, numProcs);

    // Libera memória
    for (int i = 0; i < localRows; i++) {
        free(C[i]);
        free(C_new[i]);
    }
    free(C);
    free(C_new);

    MPI_Finalize();
    return 0;
}