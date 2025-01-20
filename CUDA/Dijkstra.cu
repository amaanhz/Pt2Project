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

__global__ void dev_min(const int* arr, const int* idxs, const int* mask, int size, int* out_vals, int* out_idxs) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x; // how far into the array we index
    bool idxs_exist = *idxs != -1;
    int split = size >> 1; // array is split into two
    // we will compare pairs from each half

    extern __shared__ int minvals[]; // shared in the block
    int* argmins = (int*)&minvals[blockDim.x]; // arrays are just next to eachother

    if (tidx > split) { return; }

    int min = arr[tidx];
    int minid = tidx;
    int otherid = split + tidx;

    if ((otherid < size && arr[otherid] < min || !mask[tidx]) && mask[otherid]) {
        // if arr[tidx] is not in the queue, default to arr[otherid] even if its not smaller
        min = arr[otherid];
        minid = otherid;
    }
    else { // both nodes are not in the queue
        min = INT_MAX;
    }


    minvals[threadIdx.x] = min; // highest sharing we can do here is block-wide
    argmins[threadIdx.x] = minid;

    // should have minimum between pairs in first and second half of array in each block's work set
    // now need to find minimum of all these
    // so lets the find the min within each block, since we are shared here
    // keep splitting, like we did for the full array

    for (int bsplit = (int)(size < blockDim.x ? size : blockDim.x) >> 1; bsplit > 0; bsplit >>= 1) {
        if (threadIdx.x > bsplit) { return; } // dump any threads right of the split
        otherid = bsplit + threadIdx.x;
        if ((otherid + blockIdx.x * blockDim.x) * 2 > size) { return; }

        __syncthreads();

        if (otherid < blockDim.x && minvals[otherid] < min) {
            min = minvals[otherid];
            minid = argmins[otherid];
        }
        minvals[threadIdx.x] = min;
        argmins[threadIdx.x] = minid;
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        out_vals[blockIdx.x] = minvals[0];
        if (!idxs_exist) {
            out_idxs[blockIdx.x] = argmins[0];
        }
        else {
            out_idxs[blockIdx.x] = idxs[argmins[0]];
        }
    }
}

int fastmin(int* arr, int* queues, int size) {
    int oldsize = size;
    int* d_arr; int* mask;

    gpuErrchk(cudaMalloc(&d_arr, size*sizeof(int)));
    gpuErrchk(cudaMemcpy(d_arr, arr, size*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&mask, size*sizeof(int)));
    gpuErrchk(cudaMemcpy(mask, queues, size*sizeof(int), cudaMemcpyHostToDevice));

    int* idxs; int t[1] = {-1};
    gpuErrchk(cudaMalloc(&idxs, size*sizeof(int)));
    gpuErrchk(cudaMemcpy(idxs, t, sizeof(int), cudaMemcpyHostToDevice));

    int* out_vals; int* out_idxs;
    while (size > 1) {
        int grid_size = ceil((size / (double) BLOCK_SIZE) / 2);
        int mem_size = BLOCK_SIZE * (sizeof(int) * 2);

        gpuErrchk(cudaMalloc(&out_vals, grid_size*sizeof(int)));
        gpuErrchk(cudaMalloc(&out_idxs, grid_size*sizeof(int)));


        dev_min<<<grid_size, BLOCK_SIZE, mem_size>>>(d_arr, idxs, mask, size, out_vals, out_idxs);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        size = grid_size;
        idxs = out_idxs;
        d_arr = out_vals;
    }


    printf("\n\n");

    int min;
    gpuErrchk(cudaMemcpy(&min, d_arr, sizeof(int), cudaMemcpyDeviceToHost));
    int argmin;
    gpuErrchk(cudaMemcpy(&argmin, idxs, sizeof(int), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_arr));
    gpuErrchk(cudaFree(idxs));
    gpuErrchk(cudaFree(mask));

    //printf("Min = %d at index %d\n", min, argmin);
    /*for (int i = 0; i < 501; i++) {
        arr[501 * i + i] = INT_MAX;
    }
    int* actualiter = min_element(arr, arr + oldsize);
    int actual = *actualiter; long int actualidx = actualiter - arr;
    printf("Actual min = %d at index %ld\n", actual, actualidx);*/

    return argmin;
}


__global__ void dev_process(const int* src_edges, const int* u_edges, int* dist, int* prev, const int* queues, int size,
    int u) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= size) { return; }
    if (!queues[tidx] || u_edges[tidx] == INT_MAX) { return; }

    //printf("tidx = %d, u = %d, size = %d test: %d\n", tidx, u, size, dist[tidx]);

    int alt = dist[u] + u_edges[tidx]; // dist[u] + Graph.Edges(u, v)
    if (alt < dist[tidx]) {
        dist[tidx] = alt;
        prev[tidx] = u;
    }
}

void placeOnDevice(int* ptr, int size, int* src) {
    gpuErrchk(cudaMalloc(&ptr, size * sizeof(int)));
    gpuErrchk(cudaMemcpy(ptr, src, size * sizeof(int), cudaMemcpyHostToDevice));
}

void process_node(GraphMatrix& graph, GraphMatrix& dist, GraphMatrix& prev, GraphMatrix& queues, int node, int dim) {
    int row = node / dim; int col = node % dim;
    // row is the source node, col is the shortest distance node from source, u
    int& u = col;
    queues[dim * u] = 0;

    // dist[u] == node
    // Graph.Edges(u, v) which are neighbours of u == &graph[u]

    int grid_size = ceil(dim / (double) BLOCK_SIZE);

    int indexIn = dim * row;
    int* src_graph; int* src_dist; int* src_prev; int* src_queues;
    int* u_edges;
    //placeOnDevice(src_graph, dim, &graph[indexIn]);

    gpuErrchk(cudaMalloc(&u_edges, dim * sizeof(int)));
    gpuErrchk(cudaMemcpy(u_edges, &graph[dim * u], dim * sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&src_dist, dim * sizeof(int)));
    gpuErrchk(cudaMemcpy(src_dist, &dist[indexIn], dim * sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&src_prev, dim * sizeof(int)));
    gpuErrchk(cudaMemcpy(src_prev, &prev[indexIn], dim * sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&src_queues, dim * sizeof(int)));
    gpuErrchk(cudaMemcpy(src_queues, &queues[indexIn], dim * sizeof(int), cudaMemcpyHostToDevice));

    dev_process<<<grid_size, BLOCK_SIZE>>>(src_graph, u_edges, src_dist, src_prev, src_queues, dim, u);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(&dist[indexIn], src_dist, dim*sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&prev[indexIn], src_prev, dim*sizeof(int), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(u_edges)); gpuErrchk(cudaFree(src_dist)); gpuErrchk(cudaFree(src_prev));
    gpuErrchk(cudaFree(src_queues));
}


Result** cuda_DijkstraAPSP(GraphMatrix& graph) {
    int dim = graph.GetSize();
    Result** results = new Result*[dim];

    GraphMatrix dist = GraphMatrix(graph, INT_MAX);
    GraphMatrix prev = GraphMatrix(graph, -1);
    GraphMatrix queues = GraphMatrix(graph, 1);
    for (int i = 0; i < dim; i++) {
        dist[dim * i + i] = 0;
        queues[dim * i + i] = 0;
    }

    int remaining = dim*dim - dim;
    //graph.printGraph();

    size_t free, total;
    while (remaining > 0) {
        //int next_node = fastmin(dist.GetMatrix(), queues.GetMatrix(), dim*dim);
        int next_node = 0;
        //printf("next_node = %d == [%d][%d]\n", next_node, row, col);

        process_node(graph, dist, prev, queues, next_node, dim);
        remaining--;
        cudaMemGetInfo(&free, &total);
        printf("Memory Available: %ld/%ld\n", free, total);
    }

    return results;
}
