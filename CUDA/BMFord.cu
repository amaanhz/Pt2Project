#include <cuda_runtime.h>
#include <stdio.h>
<<<<<<< HEAD
#include <cooperative_groups.h>
=======
>>>>>>> 3391f487b3f81d24d1a12d471d4c29f6c6b01b98
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

<<<<<<< HEAD
__global__ void Relax(int* dev_graph, int* dev_dist, int* dev_prev, int graph_size) {

}

Result** cuda_BMFord(GraphMatrix& graph, int block_length) {
=======
__global__ void Relax(int* dev_graph, int* dev_dist, int* dev_prev) {

}

Result** cuda_BMFord(GraphMatrix& graph) {
>>>>>>> 3391f487b3f81d24d1a12d471d4c29f6c6b01b98
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

<<<<<<< HEAD
    // BMFord works by iterating V times
    // And on each iteration every vertex relaxes

    // blockX: block for each source vertex
    dim3 grid_dim(graphSize, graphSize / block_length + ((graphSize % block_length) > 0));
=======
>>>>>>> 3391f487b3f81d24d1a12d471d4c29f6c6b01b98

}
