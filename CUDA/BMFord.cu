#include <cuda_runtime.h>
#include <stdio.h>
#include <cooperative_groups.h>
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

__global__ void Relax_alt(int* dev_graph, int* dev_dist, int* dev_prev, int graphSize, int v) {
    if (blockIdx.x * blockDim.x + threadIdx.x > graphSize - 1 || blockIdx.y * blockDim.y + threadIdx.y > graphSize - 1) {
        return;
    }
    int tidx = blockIdx.x * blockDim.x * graphSize + threadIdx.x * graphSize + v;

    int w = dev_graph[tidx];
    if (w < INT_MAX) {
        int to_u = blockIdx.y * blockDim.y * graphSize + threadIdx.y * graphSize + blockIdx.x * blockDim.x + threadIdx.x;
        int to_v = blockIdx.y * blockDim.y * graphSize + threadIdx.y * graphSize + v;
        //printf("to_v = %d * %d + %d * %d + %d = %d\n", blockIdx.z, graphSize, blockIdx.y, blockDim.y, threadIdx.y, to_v);
        int d_u = dev_dist[to_u]; int d_v = dev_dist[to_v];

        if (d_u != INT_MAX) {
            if (d_u + w < d_v) {
                //printf("tidx = %d, to_u = %d, to_v = %d, d_u = %d, d_v = %d, prev_v = %d, blockIdx.y = %d, "
                //                   "blockDim.y = %d\n", tidx, to_u, to_v, d_u, d_v, prev_v, blockIdx.y, blockDim.y);
                //printf("d_u + w = %d, %d < d_v (%d), setting prev_v to %d, source vertex = %d, v = %d\n",
                //    d_u + w, d_u + w, d_v, blockIdx.x * blockDim.x + threadIdx.x, blockIdx.z, blockIdx.y * blockDim.y + threadIdx.y);
                d_v = d_u + w;
                int prev_v = blockIdx.x * blockDim.x + threadIdx.x; // to_u -> u
                dev_dist[to_v] = d_v; dev_prev[to_v] = prev_v;
            }
        }
    }
}

__global__ void Relax(int* dev_graph, int* dev_dist, int* dev_prev, int graphSize) {
    if (blockIdx.x * blockDim.x + threadIdx.x > graphSize - 1 || blockIdx.y * blockDim.y + threadIdx.y > graphSize - 1) {
        //printf("returning with u: %d and v: %d\n", blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
        return;
    }
    int tidx = blockIdx.x * blockDim.x * graphSize + threadIdx.x * graphSize + blockIdx.y * blockDim.y + threadIdx.y;
    //printf("tidx: %d\n", tidx);
    // in graph, this is an edge
    // in dist, this is the shortest distance from u to v
    // in prev, this is the previous node visited on the shortest path from u to v
    // need to keep these contexts in mind

    // need to grab distance[u] and distance[v] for the source vertex in question
    // u: x
    // v: y
    // src: z

    int w = dev_graph[tidx];
    if (w < INT_MAX) {
        int to_u = blockIdx.z * graphSize + blockIdx.x * blockDim.x + threadIdx.x;
        int to_v = blockIdx.z * graphSize + blockIdx.y * blockDim.y + threadIdx.y;
        //printf("to_v = %d * %d + %d * %d + %d = %d\n", blockIdx.z, graphSize, blockIdx.y, blockDim.y, threadIdx.y, to_v);
        int d_u = dev_dist[to_u]; int d_v = dev_dist[to_v];

        //printf("d_u (%d) + w (%d) = %d, %d < d_v (%d), prev_v to %d, source vertex = %d, u = %d, v = %d, we go in? %d\n",
        //            d_u, w, d_u + w, d_u + w, d_v, blockIdx.x * blockDim.x + threadIdx.x, blockIdx.z, blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, d_u + w < d_v);

        if (d_u != INT_MAX) {
            if (d_u + w < d_v) {
                d_v = d_u + w;
                int prev_v = blockIdx.x * blockDim.x + threadIdx.x; // to_u -> u
                dev_dist[to_v] = d_v; dev_prev[to_v] = prev_v;
            }
        }
    }
}

Result** cuda_BMFord(GraphMatrix& graph, int block_length) {
    int graphSize = graph.GetSize(); int matSize = graphSize * graphSize;
    int* dev_dist; int* dev_prev; int* dev_graph;
    GraphMatrix dist = GraphMatrix(graph, INT_MAX);
    for (int i = 0; i < graphSize; i++) {
        dist[graphSize * i + i] = 0;
    }
    GraphMatrix prev = GraphMatrix(graph, -1);

    gpuErrchk(cudaMalloc(&dev_dist, sizeof(int) * matSize));
    gpuErrchk(cudaMalloc(&dev_prev, sizeof(int) * matSize));
    gpuErrchk(cudaMalloc(&dev_graph, sizeof(int) * matSize));

    gpuErrchk(cudaMemcpy(dev_dist, &dist[0], sizeof(int) * matSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_prev, &prev[0], sizeof(int) * matSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_graph, &graph[0], sizeof(int) * matSize, cudaMemcpyHostToDevice));

    // BMFord works by iterating V times
    // And on each iteration every vertex relaxes

    // split graph into blocks so we can cover all vertex pairs
    // have a z dimension for dealing with each source vertex

    int num_blocks = graphSize / block_length + (graphSize % block_length != 0);

    dim3 grid_dim(num_blocks, num_blocks, graphSize);
    dim3 block_dim(block_length, block_length);

    /*for (int i = 0; i < graphSize; i++) {
        Relax<<<grid_dim, block_dim>>>(dev_graph, dev_dist, dev_prev, graphSize);
    }*/
    dim3 grid_dim2(num_blocks, num_blocks);
    for (int i = 0; i < graphSize; i++) {
        for (int v = 0; v < graphSize; v++) {
            Relax_alt<<<grid_dim2, block_dim>>>(dev_graph, dev_dist, dev_prev, graphSize, v);
        }
    }
    cudaDeviceSynchronize();
    cudaPeekAtLastError();

    cudaMemcpy(&dist[0], dev_dist, sizeof(int) * matSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(&prev[0], dev_prev, sizeof(int) * matSize, cudaMemcpyDeviceToHost);
    cudaFree(dev_dist); cudaFree(dev_prev); cudaFree(dev_graph);

    Result** results = new Result*[graphSize];

    for (int i = 0; i < graphSize; i++ ) {
        results[i] = new Result;
        results[i]->dist = &dist[graphSize * i];
        results[i]->prev = &prev[graphSize * i];
    }

    return results;
}
