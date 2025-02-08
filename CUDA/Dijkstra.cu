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

__global__ void dev_min(const int* arr, const int* idxs, const int* mask, int size, int* out_vals, int* out_idxs,
    int* out_min, int* out_minid) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x; // how far into the array we index
    bool idxs_exist = *idxs != -1;
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
    while (size > 1) {
        grid_size = ceil((size / (double) BLOCK_SIZE) / 2);
        int mem_size = BLOCK_SIZE * (sizeof(int) * 2);

        int block_size = BLOCK_SIZE;


        dev_min<<<grid_size, block_size, mem_size, *stream>>>(d_arr, idxs, mask, size, out_vals, out_idxs, out_min, out_minid);

        size = grid_size;
        idxs = out_idxs;
        d_arr = out_vals;
        mask = block_id_masks;
    }


    //printf("\n\n");
    //int resetIdxs[1] = {-1};
    //gpuErrchk(cudaMemcpy(in_idxs, resetIdxs, sizeof(int), cudaMemcpyHostToDevice)) // set *idxs -> -1

    //int min; cudaMemcpy(&min, out_vals, sizeof(int), cudaMemcpyDeviceToHost);
    //int argmin; cudaMemcpy(&argmin, out_idxs, sizeof(int), cudaMemcpyDeviceToHost);
    //printf("Min = %d at index %d\n", min, argmin);
    //const int* actualiter = min_element(arr, arr + oldsize);
    //int actual = *actualiter; long int actualidx = actualiter - arr;
    //printf("Actual min = %d at index %ld\n", actual, actualidx);
}


__global__ void dev_process(const int* edges, int* dist, int* prev, int* queues, int dim, int* node_p, int src) {
    int u = *node_p;

    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    queues[src * dim + u] = 0;

    if (tidx >= dim) { return; }


    int myIndex = u * dim + tidx; // "v"
    int uIndex = src * dim + u;

    //printf("tidx = %d, src = %d, u = %d, dim = %d dist[tidx]: %d, node = %d, edges[myIndex] = %d, dist[uIndex] = %d, queues = %d\n",
        //tidx, src, u, dim, dist[tidx], src * dim + u, edges[myIndex], dist[uIndex], queues[src*dim + tidx]);
    if (!queues[src * dim + tidx] || edges[myIndex] == INT_MAX || dist[uIndex] == INT_MAX) { return; }



    //if (tidx == 1) { printArr(edges, queues, dim*dim); }
    int alt = dist[uIndex] + edges[myIndex]; // dist[u] + Graph.Edges(u, v)
    //printf("alt: %d, dist[%d] = %d (edges[%d] = %d) (src = %d, u = %d)\n", alt, tidx, dist[tidx], myIndex,
        //edges[myIndex], src, u);
    if (alt < dist[src * dim + tidx]) {
        //printf("Found a shorter path for tidx %d: setting dist[tidx] = %d and prev[tidx] = %d\n", tidx, alt, u);
        dist[src * dim + tidx] = alt;
        prev[src * dim + tidx] = u;
    }
}

void process_node(int* graph, int* dist, int* prev, int* queues, int* node, int dim, int grid_size, cudaStream_t* stream,
    int src) {

    // row is the source node, col is the shortest distance node from source, u

    // dist[u] == node
    // Graph.Edges(u, v) which are neighbours of u == &graph[u]

    //placeOnDevice(src_graph, dim, &graph[indexIn]);


    //printf("u is %d and:\n", u);
    //dist.printGraph();

    dev_process<<<grid_size, BLOCK_SIZE, 0, *stream>>>(graph, dist, prev, queues, dim, node, src);
    //dist.printGraph();
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

    //graph.printGraph();

    int grid_size = ceil((total / (double) BLOCK_SIZE) / 2);
    int* dev_dist; int* dev_queues;
    gpuErrchk(cudaMalloc(&dev_dist, total*sizeof(int)));
    gpuErrchk(cudaMemcpy(dev_dist, dist.GetMatrix(), total*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&dev_queues, total*sizeof(int)));
    gpuErrchk(cudaMemcpy(dev_queues, queues.GetMatrix(), total*sizeof(int), cudaMemcpyHostToDevice));

    int* out_min; int* out_minid;
    gpuErrchk(cudaMalloc(&out_min, sizeof(int) * dim));
    gpuErrchk(cudaMalloc(&out_minid, sizeof(int) * dim));

    int* dev_idxs; int t[1] = {-1};
    gpuErrchk(cudaMalloc(&dev_idxs, total*sizeof(int)));


    int* out_vals; int* out_idxs;
    gpuErrchk(cudaMalloc(&out_vals, dim*grid_size*sizeof(int)));
    gpuErrchk(cudaMalloc(&out_idxs, dim*grid_size*sizeof(int)));

    int* blockmasks = new int[grid_size];
    int* block_id_masks;
    for (int i = 0; i < grid_size; i++) { blockmasks[i] = 1; }
    gpuErrchk(cudaMalloc(&block_id_masks, grid_size*sizeof(int)));
    gpuErrchk(cudaMemcpy(block_id_masks, blockmasks, grid_size*sizeof(int), cudaMemcpyHostToDevice));

    int* dev_graph; int* dev_prev;
    gpuErrchk(cudaMalloc(&dev_graph, total * sizeof(int)));
    gpuErrchk(cudaMalloc(&dev_prev, total * sizeof(int)));
    gpuErrchk(cudaMemcpy(dev_graph, graph.GetMatrix(), total * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_prev, prev.GetMatrix(), total * sizeof(int), cudaMemcpyHostToDevice));

    cudaStream_t streams[dim];

    size_t free, totalmem;
    for (int n = 0; n < dim; n++) {
        cudaStreamCreateWithFlags(streams + n, cudaStreamNonBlocking);
        int indexIn = n * dim;

        gpuErrchk(cudaMemcpy(dev_idxs + indexIn, t, sizeof(int), cudaMemcpyHostToDevice));

        for (int m = 0; m < dim; m++) {
            //dist.printGraph();
            fastmin(dev_dist + indexIn, dev_queues + indexIn, dev_idxs + indexIn, dim, out_vals + n * grid_size,
                out_idxs + n * grid_size, block_id_masks, grid_size, out_min + n, out_minid + n,
                streams + n
                );
            //printf("next_node = %d == [%d][%d]\n", next_node, row, col);

            process_node(dev_graph, dev_dist, dev_prev, dev_queues, out_minid + n, dim, grid_size, streams + n, n);
            //printf("remaining = %d\n", total - n);
            cudaMemGetInfo(&free, &totalmem);
            printf("Memory Available: %ld/%ld\n", free, totalmem);
        }
    }

    cudaDeviceSynchronize();
    cudaPeekAtLastError();

    cudaMemcpy(&dist[0], dev_dist, total*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&prev[0], dev_prev, total*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < dim; i++ ) {
        results[i] = new Result;
        results[i]->dist = &dist[dim * i];
        results[i]->prev = &prev[dim * i];
    }

    return results;
}
