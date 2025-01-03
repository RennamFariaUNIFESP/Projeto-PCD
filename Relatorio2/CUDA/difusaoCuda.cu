#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cuda.h>

//#define N 10
#define N 2000  // Tamanho da grade
#define T 1000  // Quantidade de iterações
#define D 0.1   // Coeficiente de coesão
#define DELTA_T 0.01
#define DELTA_X 1.0

#define RADIUS 1
#define BLOCK_SIZE 16

#define cudaCheck(error) \
  if (error != cudaSuccess) { \
    printf("Fatal error: %s at %s:%d\n", \
        cudaGetErrorString(error), \
        __FILE__, __LINE__); \
    exit(1); \
}

__global__ void diff_eq(const double *input, double *output, int width, int height) {
    // Calculate thread indices - simplified for clarity
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int globalX = blockIdx.x * blockDim.x + tx;
    int globalY = blockIdx.y * blockDim.y + ty;

    // Shared memory for 2D block
    __shared__ double sharedMem[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    // Local coordinates (including halo)
    int localX = tx + 1;
    int localY = ty + 1;

    // Initialize shared memory
    sharedMem[localY][localX] = 0.0;
    __syncthreads();

    // Load data into shared memory
    if (globalX < width && globalY < height) {
        sharedMem[localY][localX] = input[globalY * width + globalX];
    }

    // Load halo cells
    if (tx == 0 && globalX > 0) {
        sharedMem[localY][0] = input[globalY * width + (globalX - 1)];
    }
    if (tx == blockDim.x - 1 && globalX < width - 1) {
        sharedMem[localY][localX + 1] = input[globalY * width + (globalX + 1)];
    }
    if (ty == 0 && globalY > 0) {
        sharedMem[0][localX] = input[(globalY - 1) * width + globalX];
    }
    if (ty == blockDim.y - 1 && globalY < height - 1) {
        sharedMem[localY + 1][localX] = input[(globalY + 1) * width + globalX];
    }

    __syncthreads();

    // Compute stencil only for valid points
    if (globalX > 0 && globalX < width-1 && globalY > 0 && globalY < height-1) {
        double center = sharedMem[localY][localX];
        double north = sharedMem[localY-1][localX];
        double south = sharedMem[localY+1][localX];
        double west = sharedMem[localY][localX-1];
        double east = sharedMem[localY][localX+1];

        double newValue = center + D * (
            north + south + west + east - 4.0 * center
        ) * (DELTA_T / (DELTA_X * DELTA_X));

        output[globalY * width + globalX] = newValue; 
    }
}

__global__ void calculate_diffmedio(const double *input, const double *output, float *difmedio, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) { 
        float diff = fabs(output[y * width + x] - input[y * width + x]);
        atomicAdd(difmedio, diff);
    }
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Tamanho da matriz N*N (em bytes)
    size_t size = N * N * sizeof(double);

    // =========== Host setup =============

    // Alocar e inicializar a matriz no host
    double *host_C = (double *)malloc(size);
    double *host_C_output = (double *)malloc(size);

    // Verifica se a matriz foi criada corretamente
    if (host_C == NULL || host_C_output == NULL) {
      fprintf(stderr, "Falha na alocação de memória\n");
      return 1;
    }
    // Limpa a matrizes
    for (int i = 0; i < (N * N); i++) {
        host_C[i] = 0;
        host_C_output[i] = 0;
    }

    // Inicializa a concsentração no centro da matriz C
    host_C[((N/2) * N) + N/2] = 1.0;

    // ========== Device setup ============

    // Alocar e inicializar a matriz no device
    double *dev_C, *dev_C_output;

    cudaCheck(cudaMalloc(&dev_C, size));
    cudaCheck(cudaMalloc(&dev_C_output, size));

    // Verifica se a matriz foi criada corretamente
    if (dev_C == NULL || dev_C_output == NULL) {
        fprintf(stderr, "Failed to allocate device memory\n");
        free(host_C);
        free(host_C_output);
        return 1;
    }

    cudaCheck(cudaMemcpy(dev_C, host_C, size, cudaMemcpyHostToDevice));

    // Configura block size e threads
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);

    // Inicializa o tempo do código
    cudaEventRecord(start); 

    // Executa o processo da equação de difusão com as matrizes
    for (int t = 0; t < T; t++) {
        diff_eq<<<gridSize, blockSize>>>(dev_C, dev_C_output, N, N);
        cudaCheck(cudaDeviceSynchronize());

        // Calculate difference mean every 100 iterations
        if ((t % 100) == 0) {
          // Initialize difmedio on host and device
          float host_difmedio;
          float *dev_difmedio;

          cudaMalloc(&dev_difmedio, sizeof(float));
          cudaMemset(dev_difmedio, 0, sizeof(float)); 

          // Launch kernel to calculate difmedio
          calculate_diffmedio<<<gridSize, blockSize>>>(dev_C, dev_C_output, dev_difmedio, N, N);
          cudaCheck(cudaDeviceSynchronize());

          // Copy difmedio from device to host
          cudaMemcpy(&host_difmedio, dev_difmedio, sizeof(float), cudaMemcpyDeviceToHost);

          // Calculate and print average difmedio
          host_difmedio /= ((N - 2) * (N - 2)); 
          printf("Iteração %d - diferença média=%g\n", t, host_difmedio);

          // Free device memory
          cudaFree(dev_difmedio);
        }

        // Swap buffers
        double *temp = dev_C;
        dev_C = dev_C_output;
        dev_C_output = temp;
    }

    // Finaliza o tempo do processo da equação e salva-o
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Finaliza o tempo do processo da equação
    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    // Synchronize and check for errors again
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    // Retornada dados ao hosst
    cudaCheck(cudaMemcpy(host_C_output, dev_C_output, size, cudaMemcpyDeviceToHost));

    printf("\nConcentração final no centro: %f\n", host_C_output[((N/2) * N) + N/2]);

    // Salvando matrix no aqruivo txt
    FILE *fp = fopen("/content/matriz_Cuda_output.txt", "w");

    if (fp == NULL) {
      printf("Erro ao abrir arquivo.txt\n");
    } else {
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          if (host_C_output[i * N + j] >= 0.0001) {
            fprintf(fp, "i:%d j:%d Matriz:%f ", i, j, host_C_output[i * N + j]);
          }
        }
        fprintf(fp, "\n");
      }
      fclose(fp);
    }

    // Liberar memória alocada
    free(host_C);
    free(host_C_output);
    cudaFree(dev_C);
    cudaFree(dev_C_output);


    // Printa o tempo final
    printf("Tempo final do código: %f\n", elapsed_time / 1000.0);

    return 0;
}
