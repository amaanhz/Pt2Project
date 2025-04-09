#include <cuda_runtime.h>
#include <stdio.h>
#include "BMFord.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__host__ __device__ inline void printArr(const int* arr, const int* mask, int size) {
    for (int i = 0; i < size; i++) {
        printf("arr[%d] = %d, prev[%d] = %d\n", i, arr[i], i, mask[i]);
    }
}

__global__ void Relax(int* dev_graph, int* dev_dist, int* dev_prev) {

}

Result** cuda_BMFord(GraphMatrix& graph) {
    int graphSize = graph.GetSize(); int matSize = graphSize * graphSize;
    int* dev_dist; int* dev_prev; int* dev_graph;
    GraphMatrix dist = GraphMatrix(graph, INT_MAX);
    GraphMatrix prev = GraphMatrix(graph, -1);

    gpuErrchk(cudaMalloc(&dev_dist, sizeof(int) * matSize));
    gpuErrchk(cudaMalloc(&dev_prev, sizeof(int) * matSize));
    gpuErrchk(cudaMalloc(&dev_graph, sizeof(int) * matSize));

    gpuErrchk(cudaMemcpy(dev_dist, &dist, sizeof(int) * matSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_prev, &prev, sizeof(int) * matSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_graph, &graph, sizeof(int) * matSize, cudaMemcpyHostToDevice));


}
