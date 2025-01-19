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

__global__ void dev_min(const int* arr, const int* idxs, int size, int* out_vals, int* out_idxs) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x; // how far into the array we index
    int split = size >> 1; // array is split into two
    // we will compare pairs from each half

    extern __shared__ int minvals[]; // shared in the block
    int* argmins = (int*)&minvals[blockDim.x]; // arrays are just next to eachother

    if (tidx > split) { return; }

    if (size == 2) { printf("idxs[0] = %d, idxs[1] = %d\n", idxs[0], idxs[1]); }
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

    for (int bsplit = (size < blockDim.x ? size : blockDim.x) >> 1; bsplit > 0; bsplit >>= 1) {
        if (threadIdx.x > bsplit) { return; } // dump any threads right of the split
        otherid = bsplit + threadIdx.x;
        if (size == 2) { printf("Reached here: otherid = %d, blockIdx.x = %d, blockDim.x = %d, bsplit = %d\n",
            otherid, blockIdx.x, blockDim.x, bsplit); }

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
        //printf("Reached here\n");
        printf("minvals[0]: %d\n", minvals[0]);
        printf("argmins[0]: %d\n", argmins[0]);
        out_vals[blockIdx.x] = minvals[0];
        if (*idxs == -1) {
            out_idxs[blockIdx.x] = argmins[0];
            //printf("\nout_idxs[%d] = %d\n", blockIdx.x, out_idxs[blockIdx.x]);
        }
        else {
            //printf("argmins[0] = %d\n", argmins[0]);
            out_idxs[blockIdx.x] = idxs[argmins[0]];
        }
    }
}

/*__global__ void minArgMin(int* vals, int* idxs, int size, int* out_vals, int* out_idxs) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int split = size >> 1;

    extern __shared__ int minvals[];
    int* argmins = (int*)&minvals[blockDim.x];

    if (tidx > split) { return; }

    int min = vals[tidx];
    int minid = tidx;
    int otherid = split + tidx;

    if (otherid < size && vals[otherid] < min) {
        min = vals[otherid];
        minid = otherid;
    }

    minvals[threadIdx.x] = min; // highest sharing we can do here is block-wide
    argmins[threadIdx.x] = minid;

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
    }
}*/

int fastmin(int* arr, int size) {
    int oldsize = size;
    int* d_arr;

    gpuErrchk(cudaMalloc(&d_arr, size*sizeof(int)));
    gpuErrchk(cudaMemcpy(d_arr, arr, size*sizeof(int), cudaMemcpyHostToDevice));

    int* idxs; int t[1] = {-1};
    gpuErrchk(cudaMalloc(&idxs, size*sizeof(int)));
    gpuErrchk(cudaMemcpy(idxs, t, sizeof(int), cudaMemcpyHostToDevice));

    while (size > 1) {
        int grid_size = ceil((size / (double) BLOCK_SIZE) / 2);
        int mem_size = BLOCK_SIZE * (sizeof(int) * 2);

        int* out_vals;
        gpuErrchk(cudaMalloc(&out_vals, grid_size*sizeof(int)));
        int* out_idxs;
        gpuErrchk(cudaMalloc(&out_idxs, grid_size*sizeof(int)));


        dev_min<<<grid_size, BLOCK_SIZE, mem_size>>>(d_arr, idxs, size, out_vals, out_idxs);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        size = grid_size;
        idxs = out_idxs;
        d_arr = out_vals;
    }


    printf("\n\n");

    int min; int argmin;
    gpuErrchk(cudaMemcpy(&min, d_arr, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&argmin, idxs, sizeof(int), cudaMemcpyDeviceToHost));

    printf("Min = %d at index %d\n", min, argmin);

    int* actualiter = min_element(arr, arr + oldsize);
    int actual = *actualiter; long int actualidx = actualiter - arr;

    printf("Actual min = %d at index %ld\n", actual, actualidx);

    return 0;
}

