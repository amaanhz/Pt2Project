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

#define BLOCK_LENGTH 32
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

__global__ void dev_min(const int* arr, const int* idxs, const int* mask, int size,
    int* out_vals, int* out_idxs, int* out_min, int* out_minid, bool idxs_exist) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x; // how far into the array we index
    //bool idxs_exist = *idxs != -1;
    //if (tidx == 0) {
    //    printf("*idxs == %d\n", *idxs);
    //}
    int split = size >> 1; // array is split into two
    // we will compare pairs from each half

    extern __shared__ int minvals[]; // shared in the block
    int* argmins = (int*)&minvals[blockDim.x]; // arrays are just next to eachother

    if (tidx > split) { return; }


    //if (tidx == 0) { printArr(arr, mask, size); }

    int min = arr[tidx];
    int minid = tidx;
    int otherid = split + tidx;

    //if (otherid < size) {
        if (mask[otherid] && ( arr[otherid] < min || !mask[tidx])) {
            // if arr[tidx] is not in the queue, default to arr[otherid] even if its not smaller
            //printf("choosing %d at index %d over %d at index %d\n", arr[otherid], otherid, arr[tidx], tidx);
            min = arr[otherid];
            minid = otherid;
        }
        if (!mask[tidx] && !mask[otherid]) { // both nodes are not in the queue
            min = INT_MAX;
        }
    //}


    minvals[threadIdx.x] = min; // highest sharing we can do here is block-wide
    argmins[threadIdx.x] = minid;

    //printf("argmins[%d] = %d\n", threadIdx.x, argmins[threadIdx.x]);
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
        if (oidx > size) {
            //printf("tidx %d is killing itself! (oidx : %d)\n", tidx, oidx);
            return;
        }
        //if (tidx == 0) { printf("otherid = %d, argmins[otherid] = %d, minvals[otherid] = %d, threshold = %d\n",
        //    otherid, argmins[otherid], minvals[otherid], threshold); }

        //__syncthreads();
        if ( mask[argmins[otherid]] && (otherid < blockDim.x && minvals[otherid] < min) || !mask[minid]) {
            //printf("tidx %d -> choosing %d at index %d (mask: %d) over %d at index %d (mask: %d, threshold: %d)\n",
            //    tidx, minvals[otherid], otherid, mask[argmins[otherid]], min, minid, mask[minid], threshold);
            min = minvals[otherid];
            minid = argmins[otherid];
        }
        minvals[threadIdx.x] = min;
        argmins[threadIdx.x] = minid;
        if (threadIdx.x == 0 && bsplit == 0) { break; }
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
        if (gridDim.x == 1) {
            *out_min = out_vals[0];
            *out_minid = out_idxs[0];
        }
    }
}

void fastmin(const int* arr, const int* queues, int* in_idxs, int size, int* out_vals, int* out_idxs,
    int* block_id_masks, int grid_size, int* out_min, int* out_minid, cudaStream_t* stream) {
    int oldsize = size;
    const int* d_arr = arr; const int* mask = queues; int* idxs = in_idxs;

    //printf("grid size = %d\n", grid_size);
    /*while (size > 1) {
        grid_size = ceil((size / (double) BLOCK_SIZE) / 2);
        int mem_size = BLOCK_SIZE * (sizeof(int) * 2);

        int block_size = BLOCK_SIZE;


        dev_min<<<grid_size, block_size, mem_size, *stream>>>(d_arr, idxs, mask, size, out_vals, out_idxs, out_min,
            out_minid, !(size==oldsize));

        size = grid_size;
        idxs = out_idxs;
        d_arr = out_vals;
        mask = block_id_masks;
    }*/


    //printf("\n");
    //int resetIdxs[1] = {-1};
    //gpuErrchk(cudaMemcpy(in_idxs, resetIdxs, sizeof(int), cudaMemcpyHostToDevice)) // set *idxs -> -1

    //int min; cudaMemcpy(&min, out_vals, sizeof(int), cudaMemcpyDeviceToHost);
    //int argmin; cudaMemcpy(&argmin, out_idxs, sizeof(int), cudaMemcpyDeviceToHost);
    int temp_arr[oldsize]; int temp_mask[oldsize];
    cudaMemcpy(temp_arr, arr, sizeof(int) * oldsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_mask, queues, sizeof(int) * oldsize, cudaMemcpyDeviceToHost);
    for (int i = 0; i < oldsize; i++) {
        if (temp_mask[i] == 0) { temp_arr[i] = INT_MAX;}
    }
    //printf("Min = %d at index %d\n", min, argmin);
    const int* actualiter = min_element(temp_arr, temp_arr + oldsize);
    int actual = *actualiter; long int actualidx = actualiter - temp_arr;
    printf("Actual min = %d at index %ld\n", actual, actualidx);
    cudaMemcpy(out_min, &actual, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(out_minid, &actualidx, sizeof(int), cudaMemcpyHostToDevice);
}



__global__ void dev_process(const int* edges, int* dist, int* prev, int* queues,
    int dim, const int* min_array) {
    int src = blockIdx.x;
    int u = min_array[src];

    int tidx = blockIdx.y * blockDim.y + threadIdx.x;

    if (tidx >= dim) { return; }

    int uIndex = src * dim + u;
    int myIndex = u * dim + tidx; // w(u, v) in graph
    int sdtidx = src * dim + tidx;

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

__global__ void get_mins(const int* arr, const int* queues, int* out_minid, int size, int* min_acc) {

    int tidx = blockIdx.x * size + blockIdx.y * blockDim.y + threadIdx.x; // our edge


    // initialise array with queues to mask
    extern __shared__ int masked[];
    int* indexes = masked + BLOCK_SIZE;
    masked[threadIdx.x] = queues[tidx] ? arr[tidx] : INT_MAX;
    indexes[threadIdx.x] = blockIdx.y * blockDim.y + threadIdx.x;

    if (blockIdx.y * blockDim.y + threadIdx.x >= size) { // overflowing
        masked[threadIdx.x] = INT_MAX;
        indexes[threadIdx.x] = -1;
        return;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        int* miniter = min_element(masked, masked + BLOCK_SIZE);
        printf("Found min %d at index %d (source: %d)\n", *miniter, miniter - masked, blockIdx.x);
        min_acc[blockIdx.x * gridDim.y + blockIdx.y] = miniter - masked + blockIdx.y * blockDim.y;

        if (gridDim.y == 1) {
            out_minid[blockIdx.x] = min_acc[blockIdx.x * gridDim.y + blockIdx.y];
            printf("Setting out_minid[%d] = %d\n", blockIdx.x, min_acc[blockIdx.x * gridDim.y + blockIdx.y]);
        }
    }

    __syncthreads();

}

Result** cuda_DijkstraAPSP(GraphMatrix& graph) {
    int dim = graph.GetSize();
    Result** results = new Result*[dim];

    GraphMatrix dist = GraphMatrix(graph, INT_MAX);
    GraphMatrix prev = GraphMatrix(graph, -1);
    GraphMatrix queues = GraphMatrix(graph, 1);


    for (int i = 0; i < dim; i++) {
        dist[dim * i + i] = 0;
        //queues[dim * i + i] = 0;
    }

    int total = dim*dim;

    //printArr(graph.GetMatrix(), prev.GetMatrix(), total);

    //graph.printGraph();

    int grid_size = ceil((total / (double) BLOCK_SIZE) / 2);
    int* dev_dist; int* dev_queues;
    gpuErrchk(cudaMalloc(&dev_dist, total*sizeof(int)));
    gpuErrchk(cudaMemcpy(dev_dist, dist.GetMatrix(), total*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&dev_queues, total*sizeof(int)));
    gpuErrchk(cudaMemcpy(dev_queues, queues.GetMatrix(), total*sizeof(int), cudaMemcpyHostToDevice));

    int* out_min;
    gpuErrchk(cudaMalloc(&out_min, sizeof(int) * dim));
    int* out_minid;
    gpuErrchk(cudaMalloc(&out_minid, sizeof(int) * dim));

    int* dev_idxs; int t[1] = {-1};
    gpuErrchk(cudaMalloc(&dev_idxs, total*sizeof(int)));


    int* out_vals; int* out_idxs;
    gpuErrchk(cudaMalloc(&out_vals, dim*grid_size*sizeof(int)));
    gpuErrchk(cudaMalloc(&out_idxs, dim*grid_size*sizeof(int)));

    int* blockmasks = new int[grid_size];
    int* block_id_masks;
    for (int i = 0; i < grid_size; i++) { blockmasks[i] = 1; }
    //printArr(graph.GetMatrix(), queues.GetMatrix(), total);
    gpuErrchk(cudaMalloc(&block_id_masks, grid_size*sizeof(int)));
    gpuErrchk(cudaMemcpy(block_id_masks, blockmasks, grid_size*sizeof(int), cudaMemcpyHostToDevice));

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
        /*for (int m = 0; m < dim; m++) {
            int indexIn = m * dim;
            fastmin(dev_dist + indexIn, dev_queues + indexIn, dev_idxs + indexIn, dim, out_vals + m * grid_size,
                out_idxs + m * grid_size, block_id_masks, grid_size, out_min + m, out_minid + m,
                streams
                );
        }*/
        get_mins<<<process_grid, BLOCK_SIZE, sizeof(int) * BLOCK_SIZE * 2, *streams>>>(dev_dist, dev_queues, out_minid, dim, min_accumulator);
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
