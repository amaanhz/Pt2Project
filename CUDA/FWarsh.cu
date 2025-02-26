#include <cuda_runtime.h>
#include <stdio.h>
#include "FWarsh.cuh"

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

Vec2::Vec2(int x, int y) : x(x), y(y) {}

Triple::Triple(Vec2 p1, Vec2 p2, Vec2 p3) : p1(p1), p2(p2), p3(p3) { }


__global__ void dep_block (int b, int num_blocks, int bl, int rem, int* dev_dist, int* dev_prev) {
    // B[b, b], B[b, b], B[b, b]

    if ( threadIdx.x >= bl || threadIdx.y >= bl ) return;
    int blockIndex = bl * b;
    int maxIndex = bl;
    if (b == num_blocks - 1) {
        maxIndex = rem;
        if (threadIdx.x > rem || threadIdx.y > rem) return;
    }

    int rowIndex = threadIdx.x * bl;
    int cell = rowIndex + threadIdx.y;

    // copy block data to shared memory
    extern __shared__ int dist[];
    int* prev = dist + bl;
    dist[cell] = dev_dist[cell]; prev[cell] = dev_prev[cell];

    __syncthreads(); // make sure shared memory is fully initialised

    for (int k = 0; k < maxIndex; k++) {
        int kRow = k * bl;
        if (dist[rowIndex + k] != INT_MAX && dist[kRow + threadIdx.y] != INT_MAX) {
            int t = dist[rowIndex + k] + dist[kRow + threadIdx.y];
            __syncthreads();
            if (t < dist[cell]) {
                dist[cell] = t;
                prev[cell] = prev[kRow + threadIdx.y];
            }
        }
        __syncthreads(); // block must iterate in sync
    }

    // write-back
    dev_dist[cell] = dist[cell];
    dev_prev[cell] = prev[cell];
}

__device__ void block_loop (int* b1, int* b2, int* b3, int rowIndex, int maxK) {
    for (int k = 0; k < maxK; k++) {

    }
}

__global__ void pdep_blocks (int b, int num_blocks, int bl, int rem, int* dev_dist, int* dev_prev) {
    // B[b, i], B[b, b], B[b, i]
    // B[i, b], B[i, b], B[b, b]

    if ( threadIdx.x >= bl || threadIdx.y >= bl ) return;

    // find out which block we are
    int blockX; int blockY;
    if ( blockIdx.x >= b ) {
        blockX = blockIdx.x + 1;
        if (blockIdx.y == b) { blockY = b; }
        else { return; }
    }
    else if ( blockIdx.y >= b ) {
        blockY = blockIdx.y + 1;
        if (blockIdx.x == b) { blockX = b; }
        else { return; }
    }

    // need to fetch our block, and block B[b, b]
    extern __shared__ int dist[];
    int* dist_i = dist + bl;
    int* prev = dist_i + bl;
    int* prev_i = prev + bl;

    int blockIndex = b * bl;
    int rowIndex = threadIdx.x * bl;

    int indexIn = blockIndex + rowIndex + threadIdx.y;
    dist[indexIn] = dev_dist[indexIn];
    prev[indexIn] = dev_prev[indexIn];


    int* dist_1; int* dist_2; int* dist_3;
    int* prev_1; int* prev_2; int* prev_3;

    if (blockX == b) { // B[b, i], B[b, b], B[b, i]
        indexIn = blockIndex + rowIndex + blockY + threadIdx.y;
        dist_i[indexIn] = dev_dist[indexIn];
        prev_i[indexIn] = prev[indexIn];

        dist_1 = dist_i; dist_2 = dist; dist_3 = dist_i;
        prev_1 = prev_i; prev_2 = prev; prev_3 = prev_i;
    }
    else if ( blockY == b ) { // B[i, b], B[i, b], B[b, b]
        indexIn = blockX * bl + rowIndex + b + threadIdx.y;
        dist_i[indexIn] = dev_dist[indexIn];
        prev_i[indexIn] = prev[indexIn];

        dist_1 = dist_i; dist_2 = dist_i; dist_3 = dist;
        prev_1 = prev_i; prev_2 = prev_i; prev_3 = prev;
    }

    __syncthreads(); // ensure all initialised




}

__global__ void indep_blocks (int b, int num_blocks, int bl, int rem, int* dev_dist, int* dev_prev) {
    // B[i, j], B[i, b], B[b, j]

    if ( threadIdx.x >= bl ) return;
}

Result** cuda_FWarsh(GraphMatrix& graph, int block_length) {
    int graphSize = graph.GetSize();
    int matSize = graphSize * graphSize;

    int* dev_dist; cudaMalloc(&dev_dist, sizeof(int) * matSize);
    cudaMemcpy(dev_dist, graph.GetMatrix(), sizeof(int) * matSize, cudaMemcpyHostToDevice);

    int* dev_prev; cudaMalloc(&dev_prev, sizeof(int) * matSize);
    GraphMatrix prev = GraphMatrix(graph, -1);
    for (int r = 0; r < graphSize; r++) {
        int rowIndex = r * graphSize;
        for (int c = 0; c < graphSize; c++) {
            if (graph[rowIndex + c] != INT_MAX) { prev[rowIndex + c] = r; } // set previous as in graph
        }
    }


    int num_blocks = (graph.GetSize() + block_length - 1) / block_length; // ceiling the value (1 axis)
    int rem = graph.GetSize() % block_length;
    if (rem == 0) { rem = block_length; }

    dim3 block_threads(block_length, block_length);

    dim3 pdep_dim(num_blocks - 1, num_blocks - 1);

    int indep_count = (num_blocks * num_blocks) - pdep_count - 1;
    dim3 indep_dim(indep_count / 2, indep_count / 2);

    size_t memsize = sizeof(int) * block_length * 2;

    for (int block = 0; block < num_blocks; block++) {
        dep_block<<<1, block_threads, memsize>>>(block, num_blocks, block_length, rem, dev_dist, dev_prev);
        pdep_blocks<<<pdep_dim, block_threads, memsize * 2>>>(block, num_blocks, block_length, rem, dev_dist, dev_prev);
        indep_blocks<<<indep_dim, block_threads, memsize * 2>>>(block, num_blocks, block_length, rem, dev_dist, dev_prev);
    }
}
