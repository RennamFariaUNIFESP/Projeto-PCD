//%%writefile difusaoMPI.c

# include <mpi.h>
# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <time.h>

# define N 2000 // Tamanho da grade
# define T 1000 // Quantidade de iterações
# define D 0.1  // Coeficiente de coesão
# define DELTA_T 0.01
# define DELTA_X 1.0

void diff_eq(double **C, double **C_new, int localN, int lineStart, int colStart) {
    MPI_Status status;
    int myId;

    MPI_Comm_rank(MPI_COMM_WORLD, &myId);

    // Verifica se é necessário enviar informação/receber dos vizinhos
    int up = (myId >= 2) ? myId - 2 : MPI_PROC_NULL;
    int down = (myId < 2) ? myId + 2 : MPI_PROC_NULL;
    int left = (myId % 2 == 1) ? myId - 1 : MPI_PROC_NULL;
    int right = (myId % 2 == 0) ? myId + 1 : MPI_PROC_NULL;

    for (int t = 0; t < T; t++) {
        MPI_Barrier(MPI_COMM_WORLD);

        // Troca de informações com o vizinho
        // Verticais
        if (up != MPI_PROC_NULL) {
            MPI_Sendrecv(C[1], localN, MPI_DOUBLE, up, 0,
                        C[0], localN, MPI_DOUBLE, up, 1,
                        MPI_COMM_WORLD, &status);
        }

        if (down != MPI_PROC_NULL) {
            MPI_Sendrecv(C[localN-2], localN, MPI_DOUBLE, down, 1,
                        C[localN-1], localN, MPI_DOUBLE, down, 0,
                        MPI_COMM_WORLD, &status);
        }

        double *send_col = (double *)malloc(localN * sizeof(double));
        double *recv_col = (double *)malloc(localN * sizeof(double));

        // Horizontais
        if (left != MPI_PROC_NULL || right != MPI_PROC_NULL) {
            for (int i = 0; i < localN; i++) {
                send_col[i] = C[i][1];  // First real column for left
            }
        }

        if (left != MPI_PROC_NULL) {
            MPI_Sendrecv(send_col, localN, MPI_DOUBLE, left, 2,
                        recv_col, localN, MPI_DOUBLE, left, 3,
                        MPI_COMM_WORLD, &status);
            for (int i = 0; i < localN; i++) {
                C[i][0] = recv_col[i];
            }
        }

        if (right != MPI_PROC_NULL) {
            for (int i = 0; i < localN; i++) {
                send_col[i] = C[i][localN-2];  // Last real column for right
            }
            MPI_Sendrecv(send_col, localN, MPI_DOUBLE, right, 3,
                        recv_col, localN, MPI_DOUBLE, right, 2,
                        MPI_COMM_WORLD, &status);
            for (int i = 0; i < localN; i++) {
                C[i][localN-1] = recv_col[i];
            }
        }

        free(send_col);
        free(recv_col);

        // Equação
        double difmedio = 0.0;
        for (int i = 1; i < localN-1; i++) {
            for (int j = 1; j < localN-1; j++) {
                C_new[i][j] = C[i][j] + D * DELTA_T * (
                    (C[i-1][j] + C[i+1][j] + C[i][j-1] + C[i][j+1] - 4.0 * C[i][j])
                    / (DELTA_X * DELTA_X)
                );
                difmedio += fabs(C_new[i][j] - C[i][j]);
                C[i][j] = C_new[i][j];
            }
        }

        if (t % 100 == 0) {
            printf("Process %d - Iteração %d - diferença média=%g\n", myId, t, difmedio / ((localN - 2) * (localN - 2)));
            fflush(stdout);
        }
    }
}

int main(int argc, char *argv[]) {
    int myId, numProcs;
    int centerGlobal = N/2;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // Execução para apenas 4 processos
    if (numProcs != 4) {
        if (myId == 0) {
            printf("É posssível rodar apenas com 4 processos\n");
        }
        MPI_Finalize();
        return 1;
    }

    int localN = (N / 2) + 2;     // Com extremidades adicionais
    int lineStart = (myId / 2) * (N / 2);
    int colStart = (myId % 2) * (N / 2);

    printf("Process %d started - Line/Col Start: %d/%d, LocalN: %d\n",
           myId, lineStart, colStart, localN);
    fflush(stdout);

    // ------- Inicializando matrizes -------
    double **C = (double **)malloc(localN * sizeof(double *));
    if (C == NULL) {
        fprintf(stderr, "Falha na alocação de memória para C\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    double **C_new = (double **)malloc(localN * sizeof(double *));
    if (C_new == NULL) {
        fprintf(stderr, "Falha na alocação de memória para C_new\n");
        free(C);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    for (int i = 0; i < localN; i++) {
        C[i] = (double *)calloc(localN, sizeof(double));
        C_new[i] = (double *)calloc(localN, sizeof(double));

        if (C[i] == NULL || C_new[i] == NULL) {
            fprintf(stderr, "Falha na alocação de memória\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        // Inicializa os valores
        for (int j = 0; j < localN; j++) {
            C[i][j] = 0.0;
            C_new[i][j] = 0.0;
        }
    }

    // Encontra o processo que posssui o centro da matriz geral e seta o valor inicial
    if ((centerGlobal >= lineStart && centerGlobal < lineStart + localN - 2) &&
        (centerGlobal >= colStart && centerGlobal < colStart + localN - 2)) {
        int localI = (centerGlobal - lineStart) + 1;
        int localJ = (centerGlobal - colStart) + 1;
        C[localI][localJ] = 1.0;
        printf("Processo %d inicializa a concentração no centro, na posicao [%d][%d]\n\n", myId, localI, localJ);
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Equação
    diff_eq(C, C_new, localN, lineStart, colStart);

    MPI_Barrier(MPI_COMM_WORLD);

    double *localData = NULL;
    double *globalData = NULL;
    int localSize = (N/2) * (N/2);

    // Matrizes auxiliares
    if (myId == 0) {
        globalData = (double *)malloc(N * N * sizeof(double));
        if (globalData == NULL) {
            fprintf(stderr, "Failed to allocate global data array\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    localData = (double *)malloc(localSize * sizeof(double));
    if (localData == NULL) {
        fprintf(stderr, "Failed to allocate local data array\n");
        if (myId == 0) free(globalData);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    // Salva dados sem suas extremidades adicionais
    int idx = 0;
    for (int i = 1; i < localN-1; i++) {
        for (int j = 1; j < localN-1; j++) {
            localData[idx++] = C[i][j];
        }
    }

    // Une todos tresultados em uma matrix
    MPI_Gather(localData, localSize, MPI_DOUBLE,
               globalData, localSize, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    // Processo 0 salavando matrix no arquivo txt
    if (myId == 0) {
        // FILE *fp = fopen("/content/matriz_MPI_output.txt", "w");
        // if (fp == NULL) {
        //     printf("Erro ao abrir arquivo.txt");
        // }
        // else {
        //     for (int p = 0; p < 4; p++) {
        //         int baseI = (p / 2) * (N/2);
        //         int baseJ = (p % 2) * (N/2);
        //         int offset = p * localSize;

        //         for (int i = 0; i < N/2; i++) {
        //             for (int j = 0; j < N/2; j++) {
        //                 double val = globalData[offset + i * (N/2) + j];
        //                 if (val >= 0.0001) {
        //                     fprintf(fp, "i:%d j:%d Matriz:%f ", baseI + i, baseJ + j, val);
        //                 }
        //             }
        //         }
        //     }
        //     fclose(fp);
            
        //     printf("\nConcentração final no centro: %.6f\n", 
        //         globalData[3 * localSize]);
        // }
        printf("\nConcentração final no centro: %.6f\n", 
                globalData[3 * localSize]);
    }

    // Libera memória alocada
    free(localData);
    if (myId == 0) {
        free(globalData);
    }

    for (int i = 0; i < localN; i++) {
        free(C[i]);
        free(C_new[i]);
    }
    free(C);
    free(C_new);

    MPI_Finalize();
    return 0;
}