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

Result** cuda_DijkstraAPSP(const GraphMatrix& graph) {
    Result** results = new Result*[graph.GetSize()];
    queue<int> q;

    GraphMatrix dist = GraphMatrix(graph, INT_MAX);
    GraphMatrix prev = GraphMatrix(graph, -1);



    return results;
}

__global__ void dev_min(int* arr, int size, int* out_vals, int* out_idxs) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x; // how far into the array we index
    int split = size >> 1; // array is split into two
    // we will compare pairs from each half

    extern __shared__ int minvals[]; // shared in the block
    int* argmins = (int*)&minvals[blockDim.x]; // arrays are just next to eachother

    if (tidx > split) { return; }

    int min = arr[tidx];
    int minid = tidx;
    int otherid = split + tidx;

    //printf("Comparing index %d with index %d\n", tidx, otherid);
    if (otherid < size && arr[otherid] < min) {
        //printf("Index %d was smaller! (arr[%d] = %d, arr[%d] = %d\n\n", otherid, minid, min, otherid, arr[otherid]);
        min = arr[otherid];
        minid = otherid;
    }

    minvals[threadIdx.x] = min; // highest sharing we can do here is block-wide
    argmins[threadIdx.x] = minid;

    // should have minimum between pairs in first and second half of array in each block's
    // work set

    // now need to find minimum of all these

    // so lets the find the min within each block, since we are shared here
    // keep splitting, like we did for the full array

    for (int bsplit = blockDim.x >> 1; bsplit > 0; bsplit >>= 1) {
        if (threadIdx.x > bsplit) { return; } // dump any threads right of the split
        otherid = bsplit + threadIdx.x;
        if ((otherid + blockIdx.x * blockDim.x) * 2 > size) { return; }
        __syncthreads();

        if (otherid < blockDim.x && minvals[otherid] < min) {
            min = minvals[otherid];
            minid = argmins[otherid];
        }
        if (blockIdx.x == 1 && min == 0 && bsplit == 512) {
            printf("tid = %d, otherid = %d, oidx = %d\n", threadIdx.x, otherid,
                otherid + blockIdx.x * blockDim.x);
        }
        minvals[threadIdx.x] = min;
        argmins[threadIdx.x] = minid;
        //if (blockIdx.x == 1) {
        //    printf("min for tid %d: %d ( bsplit = %d ) ( tidx = %d ) \n", threadIdx.x, min, bsplit, tidx);
        //    printf("minid for tid %d: %d ( bsplit = %d ) ( tidx = %d ) \n", threadIdx.x, minid, bsplit, tidx);
        //}
        //if (blockIdx.x == 1) {
        //    printf("minvals[%d]: %d\n", threadIdx.x, minvals[threadIdx.x]);
        //    printf("argmins[%d]: %d\n", threadIdx.x, argmins[threadIdx.x]);
        //}
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        printf("Reached here\n");
        //printf("minvals[0]: %d\n", minvals[0]);
        //printf("argmins[0]: %d\n", argmins[0]);
        out_vals[blockIdx.x] = minvals[0];
        out_idxs[blockIdx.x] = argmins[0];
    }
}

int fastmin(int* arr, int size) {
    int* d_arr;

    gpuErrchk(cudaMalloc(&d_arr, size*sizeof(int)));

    gpuErrchk(cudaMemcpy(d_arr, arr, size*sizeof(int), cudaMemcpyHostToDevice));

    int grid_size = ceil((size / (double) BLOCK_SIZE) / 2);
    int mem_size = BLOCK_SIZE * (sizeof(int) * 2);

    int* out_vals;
    gpuErrchk(cudaMalloc(&out_vals, grid_size*sizeof(int)));
    int* out_idxs;
    gpuErrchk(cudaMalloc(&out_idxs, grid_size*sizeof(int)));

    dev_min<<<grid_size, BLOCK_SIZE, mem_size>>>(d_arr, size, out_vals, out_idxs);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    printf("\n\n");
    int* blockResults = new int[grid_size];
    gpuErrchk(cudaMemcpy(blockResults, out_vals, grid_size*sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid_size; i++) {
        printf("%d \n", blockResults[i]);
    }

    printf("Min found: %d\n", *min_element(blockResults, blockResults + grid_size));
    printf("Actual min is %d\n\n", *min_element(arr, arr + size));
    delete[] blockResults;
    return 0;
}

