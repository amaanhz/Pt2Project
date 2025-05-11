#include <stdio.h>
#include <limits.h>
#include <queue>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include "Dijkstra.cuh"

#include <thread>

#include "GraphParse.h"
#include "GraphMatrix.h"

#define BLOCK_SIZE 1024

using namespace std;

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

__global__ void dev_process(const int* edges, int* dist, int* prev, int* queues,
    int dim, const int* min_array) {
    int src = blockIdx.x;
    int u = min_array[src];

    int intoSrcEdges = blockIdx.y * blockDim.x + threadIdx.x; // -> (a, v) for any a

    if (intoSrcEdges >= dim) { return; }

    int uIndex = src * dim + u;
    int myIndex = u * dim + intoSrcEdges; // w(u, v) in graph
    int sdtidx = src * dim + intoSrcEdges;

    queues[uIndex] = 0;

    //printf("tidx >= dim = %d, tidx = %d\n", tidx >= dim, tidx);


    //printf("tidx = %d, src = %d, u = %d, dim = %d dist[tidx]: %d, node = %d, edges[myIndex] = %d, dist[uIndex] = %d, queues = %d\n",
    //    tidx, src, u, dim, dist[tidx], src * dim + u, edges[myIndex], dist[uIndex], queues[src*dim + tidx]);
    if (!queues[sdtidx] || edges[myIndex] == INT_MAX || dist[uIndex] == INT_MAX) { return; }



    //if (tidx == 1) { printArr(edges, queues, dim*dim); }
    int alt = dist[uIndex] + edges[myIndex]; // dist[u] + Graph.Edges(u, v)
    //printf("alt: %d, dist[%d] = %d (edges[%d] = %d) (src = %d, u = %d)\n", alt, tidx, dist[tidx], myIndex,
    //    edges[myIndex], src, u);
    if (alt < dist[sdtidx]) {
        //printf("Found a shorter path for tidx %d: setting dist[tidx] = %d and prev[tidx] = %d\n", tidx, alt, u);
        dist[sdtidx] = alt;
        prev[sdtidx] = u;
    }
}

__device__ void fold_mins(int* masked, int* indexes, int size) {
    int fold_size = size > 1 ? (size >> 1) + 1 : 0;
    /*if (blockIdx.x == 1 && threadIdx.x == 0) {
        printf("Before folds\n");
        printArr(masked, indexes, size);
        printf("\n");
    }*/
    while (fold_size > 0) {
        if (threadIdx.x <= fold_size) {
            int otherIndex = fold_size + threadIdx.x;
            //if (blockIdx.x == 1) printf("threadIdx.x = %d, otherIndex = %d\n", threadIdx.x, otherIndex);
            int minval = masked[threadIdx.x];
            int mindex = indexes[threadIdx.x];
            //if (blockIdx.x == 1) { printf("masked[%d] = %d, masked[%d] = %d, %d\n", threadIdx.x, minval, otherIndex, masked[otherIndex], masked[otherIndex] < minval); }
            if (otherIndex < blockDim.x) {
                if (masked[otherIndex] < minval) {
                    minval = masked[otherIndex];
                    mindex = indexes[otherIndex];
                }
            }
            masked[threadIdx.x] = minval;
            indexes[threadIdx.x] = mindex;
        }
        /*if (blockIdx.x == 1 && threadIdx.x == 0) {
            printf("fold_size = %d\n", fold_size);
            printArr(masked, indexes, size);
            printf("\n");
        }*/
        fold_size = fold_size > 2 ? (fold_size >> 1) + 1 : fold_size >> 1;
        __syncthreads();
    }
}

__global__ void get_mins_rnd2(const int* arr, const int* mindx_acc, int* out_minid, int size, int graph_size) {
    extern __shared__ int minvals[];
    int* mindxs = minvals + blockDim.x;

    mindxs[threadIdx.x] = mindx_acc[blockIdx.x * size + threadIdx.x];
    if (mindxs[threadIdx.x] > -1) {
        minvals[threadIdx.x] = arr[blockIdx.x * graph_size + mindxs[threadIdx.x]];
    }
    else {
        minvals[threadIdx.x] = INT_MAX;
    }

    if (threadIdx.x >= size) { // overflowing
        mindxs[threadIdx.x] = -1;
        minvals[threadIdx.x] = INT_MAX;
    }

    __syncthreads();

    /*if (blockIdx.x == 1 && threadIdx.x == 0) {
        printArr(minvals, mindxs, size);
        printf("\n");
    }*/

    fold_mins(minvals, mindxs, size);

    if (threadIdx.x == 0) {
        out_minid[blockIdx.x] = mindxs[0];
        //if (blockIdx.x == 1) printf("2nd round min found = %d, actual min = %d\n\n", minvals[0], *min_element(minvals, minvals + blockDim.x));
    }
}

__global__ void get_mins(const int* arr, const int* queues, int* out_minid, int size, int* mindx_acc) {

    int tidx = blockIdx.x * size + blockIdx.y * blockDim.x + threadIdx.x; // our edge


    // initialise array with queues to mask
    extern __shared__ int masked[];
    int* indexes = masked + blockDim.x;
    masked[threadIdx.x] = queues[tidx] ? arr[tidx] : INT_MAX;
    indexes[threadIdx.x] = blockIdx.y * blockDim.x + threadIdx.x;

    if (blockIdx.y * blockDim.x + threadIdx.x >= size) { // overflowing
        masked[threadIdx.x] = INT_MAX;
        indexes[threadIdx.x] = -1;
        return;
    }

    __syncthreads();

    fold_mins(masked, indexes, size);

    __syncthreads();

    if (threadIdx.x == 0) {
        if (masked[1] < masked[0]) { masked[0] = masked[1]; indexes[0] = indexes[1]; }
        //if (blockIdx.x == 1) printf("min found for block %d = %d, indexes[0] = %d, actual min = %d\n", blockIdx.y, masked[0], indexes[0], *min_element(masked, masked + blockDim.x));
        mindx_acc[blockIdx.x * gridDim.y + blockIdx.y] = masked[0] != INT_MAX ? indexes[0] : -1;
        //out_minid[blockIdx.x] = indexes[0];
    }


    /*if (threadIdx.x == 0) {
        int* miniter = min_element(masked, masked + BLOCK_SIZE);
        //printf("Found min %d at index %d (source: %d)\n", *miniter, miniter - masked, blockIdx.x);
        min_acc[blockIdx.x * gridDim.y + blockIdx.y] = miniter - masked + blockIdx.y * blockDim.y;

        /*if (gridDim.y == 1) {
            out_minid[blockIdx.x] = min_acc[blockIdx.x * gridDim.y + blockIdx.y];
            printf("Setting out_minid[%d] = %d\n", blockIdx.x, min_acc[blockIdx.x * gridDim.y + blockIdx.y]);
        }#1#
    }

    __syncthreads();*/

}

Result** cuda_DijkstraAPSP(GraphMatrix& graph) {
    int dim = graph.GetSize();
    Result** results = new Result*[dim];

    GraphMatrix dist = GraphMatrix(graph, INT_MAX);
    GraphMatrix prev = GraphMatrix(graph, -1);
    GraphMatrix queues = GraphMatrix(graph, 1);


    for (int i = 0; i < dim; i++) {
        dist[dim * i + i] = 0;
    }

    int total = dim*dim;

    int* dev_dist; int* dev_queues;
    gpuErrchk(cudaMalloc(&dev_dist, total*sizeof(int)));
    gpuErrchk(cudaMemcpy(dev_dist, dist.GetMatrix(), total*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&dev_queues, total*sizeof(int)));
    gpuErrchk(cudaMemcpy(dev_queues, queues.GetMatrix(), total*sizeof(int), cudaMemcpyHostToDevice));

    int* out_minid;
    gpuErrchk(cudaMalloc(&out_minid, sizeof(int) * dim));

    //printArr(graph.GetMatrix(), queues.GetMatrix(), total);

    int* dev_graph; int* dev_prev;
    gpuErrchk(cudaMalloc(&dev_graph, total * sizeof(int)));
    gpuErrchk(cudaMalloc(&dev_prev, total * sizeof(int)));
    gpuErrchk(cudaMemcpy(dev_graph, graph.GetMatrix(), total * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_prev, prev.GetMatrix(), total * sizeof(int), cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();

    cudaStream_t streams[2];
    gpuErrchk(cudaStreamCreate(streams));
    gpuErrchk(cudaStreamCreate(streams + 1));

    cudaDeviceSynchronize();

    int grid_dim = dim / BLOCK_SIZE + (dim % BLOCK_SIZE > 0);
    dim3 process_grid(dim, grid_dim);

    int* min_accumulator;
    gpuErrchk(cudaMalloc(&min_accumulator, sizeof(int) * grid_dim * dim));

    size_t free, totalmem;
    for (int n = 0; n < dim; n++) {
        get_mins<<<process_grid, BLOCK_SIZE, sizeof(int) * BLOCK_SIZE * 2, *streams>>>(dev_dist, dev_queues, out_minid, dim, min_accumulator);
        get_mins_rnd2<<<dim, BLOCK_SIZE, sizeof(int) * BLOCK_SIZE * 2, *streams>>>(dev_dist, min_accumulator, out_minid, grid_dim, dim);
        dev_process<<<process_grid, BLOCK_SIZE, 0, *streams>>>(dev_graph, dev_dist, dev_prev, dev_queues, dim, out_minid);
        cudaDeviceSynchronize();
    }

    cudaDeviceSynchronize();
    cudaPeekAtLastError();

    cudaMemcpy(&dist[0], dev_dist, total*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&prev[0], dev_prev, total*sizeof(int), cudaMemcpyDeviceToHost);

    //printArr(&dist[0], &prev[0], total);

    for (int i = 0; i < dim; i++ ) {
        results[i] = new Result;
        results[i]->dist = &dist[dim * i];
        results[i]->prev = &prev[dim * i];
    }

    return results;
}
