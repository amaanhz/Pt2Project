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

__device__ inline void printArr(const int* arr, const int* mask, int size) {
    for (int i = 0; i < size; i++) {
        printf("arr[%d] = %d, mask[%d] = %d\n", i, arr[i], i, mask[i]);
    }
}

__global__ void dev_min(const int* arr, const int* idxs, int* mask, int size, int* out_vals, int* out_idxs) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x; // how far into the array we index
    bool idxs_exist = *idxs != -1;
    int split = size >> 1; // array is split into two
    // we will compare pairs from each half

    extern __shared__ int minvals[]; // shared in the block
    int* argmins = (int*)&minvals[blockDim.x]; // arrays are just next to eachother

    if (tidx > split) { return; }


    if (tidx == 0) { printArr(arr, mask, size); }

    int min = arr[tidx];
    int minid = tidx;
    int otherid = split + tidx;

    if (mask[otherid] && ((otherid < size && arr[otherid] < min) || !mask[tidx])) {
        // if arr[tidx] is not in the queue, default to arr[otherid] even if its not smaller
        //printf("choosing %d at index %d over %d at index %d\n", arr[otherid], otherid, arr[tidx], tidx);
        min = arr[otherid];
        minid = otherid;
    }
    if (!mask[tidx] && !mask[otherid]) { // both nodes are not in the queue
        min = INT_MAX;
    }


    minvals[threadIdx.x] = min; // highest sharing we can do here is block-wide
    argmins[threadIdx.x] = minid;

    //printf("argmins[%d] = %d\n", threadIdx.x, argmins[0]);

    // should have minimum between pairs in first and second half of array in each block's work set
    // now need to find minimum of all these
    // so lets the find the min within each block, since we are shared here
    // keep splitting, like we did for the full array

    for (int bsplit = (int)(size < blockDim.x ? size >> 1 : blockDim.x) >> 1; bsplit >= 0; bsplit >>= 1) {
        int threshold = (bsplit & 1 ? bsplit + 1 : bsplit);
        if (threadIdx.x > threshold) {
            //printf("tidx %d is killing itself!\n", tidx);
            return;
        } // dump any threads right of the split
        otherid = threshold + threadIdx.x; // compare against corresponding past the split
        if (threadIdx.x == 0 && bsplit == 0) { otherid = 1; }
        if (otherid > (size >> 1)) { return; }
        int oidx = otherid + blockIdx.x * blockDim.x;
        if (oidx > size) { printf("tidx %d is killing itself! (oidx : %d)\n", tidx, oidx); return; }
        if (tidx == 0) { printf("otherid = %d, argmins[otherid] = %d, minvals[otherid] = %d, threshold = %d\n",
            otherid, argmins[otherid], minvals[otherid], threshold); }


        if ( mask[argmins[otherid]] && (otherid < blockDim.x && minvals[otherid] < min) || !mask[minid]) {
            printf("tidx %d -> choosing %d at index %d (mask: %d) over %d at index %d (mask: %d, threshold: %d)\n",
                tidx, minvals[otherid], otherid, mask[argmins[otherid]], min, minid, mask[minid], threshold);
            min = minvals[otherid];
            minid = argmins[otherid];
        }
        minvals[threadIdx.x] = min;
        argmins[threadIdx.x] = minid;
        if (threadIdx.x == 0 && bsplit == 0) { break; }
        __syncthreads();
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
        mask[blockIdx.x] = 1; // if there is another iteration, make sure blockID masks are 1 so they dont get ignored
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

    int* out_vals; int* out_idxs; int* out_mask;
    while (size > 1) {
        int grid_size = ceil((size / (double) BLOCK_SIZE) / 2);
        int mem_size = BLOCK_SIZE * (sizeof(int) * 2);

        gpuErrchk(cudaMalloc(&out_vals, grid_size*sizeof(int)));
        gpuErrchk(cudaMalloc(&out_idxs, grid_size*sizeof(int)));

        dev_min<<<grid_size, BLOCK_SIZE, mem_size>>>(d_arr, idxs, mask, size, out_vals, out_idxs);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        cudaFree(d_arr); cudaFree(idxs);

        size = grid_size;
        idxs = out_idxs;
        d_arr = out_vals;
    }


    //printf("\n\n");

    int min;
    gpuErrchk(cudaMemcpy(&min, d_arr, sizeof(int), cudaMemcpyDeviceToHost));
    int argmin;
    gpuErrchk(cudaMemcpy(&argmin, idxs, sizeof(int), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_arr));
    gpuErrchk(cudaFree(idxs));
    gpuErrchk(cudaFree(mask));

    printf("Min = %d at index %d\n", min, argmin);
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
    //printf("alt: %d, dist[%d] = %d\n", alt, tidx, dist[tidx]);
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
    queues[node] = 0;

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

    printf("u is %d and:\n", u);
    //dist.printGraph();
    dev_process<<<grid_size, BLOCK_SIZE>>>(src_graph, u_edges, src_dist, src_prev, src_queues, dim, u);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(&dist[indexIn], src_dist, dim*sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&prev[indexIn], src_prev, dim*sizeof(int), cudaMemcpyDeviceToHost));

    //dist.printGraph();
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
        //dist.printGraph();
        int next_node = fastmin(dist.GetMatrix(), queues.GetMatrix(), dim*dim);
        //printf("next_node = %d == [%d][%d]\n", next_node, row, col);

        process_node(graph, dist, prev, queues, next_node, dim);
        remaining--;
        //printf("remaining = %d\n", remaining);
        //cudaMemGetInfo(&free, &total);
        //printf("Memory Available: %ld/%ld\n", free, total);
    }

    for (int i = 0; i < dim; i++ ) {
        results[i] = new Result;
        for (int j = 0; j < dim; j++) {
            results[i]->dist = &dist[dim * i + j];
            results[i]->prev = &prev[dim * i + j];
        }
    }

    return results;
}
