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

__host__ __device__ inline void printArr(const int* arr, const int* mask, int size) {
    for (int i = 0; i < size; i++) {
        printf("arr[%d] = %d, prev[%d] = %d\n", i, arr[i], i, mask[i]);
    }
}

Vec2::Vec2(int x, int y) : x(x), y(y) {}

Triple::Triple(Vec2 p1, Vec2 p2, Vec2 p3) : p1(p1), p2(p2), p3(p3) { }

__device__ void block_loop (int* d1, int* p1, int* d2, int* d3, int* p3, int rowIndex,
    int maxIndex, int cell) {
    // all pointers are pointers to shared block memory
    // only contains block values, not whole matrix
    for (int k = 0; k < maxIndex; k++) {
        int kRow = k * maxIndex;
        //printf("kRow = %d, rowIndex = %d, rowIndex + k = %d, kRow + threadIdx.y = %d\n", kRow, rowIndex,
        //rowIndex + k, kRow + threadIdx.y);
        int t1 = d2[rowIndex + k]; int t2 = d3[kRow + threadIdx.y];
        __syncthreads();
        if (t1 != INT_MAX && t2 != INT_MAX) {
            int t = t1 + t2;
            if (t < d1[cell]) {
                //printf("t = %d, t < d1[%d] = %d\n", t, cell, d1[cell]);
                d1[cell] = t;
                //printf("Setting p1[%d] = p3[%d] = %d\n\n", cell, kRow + threadIdx.y, p3[kRow + threadIdx.y]);
                p1[cell] = p3[kRow + threadIdx.y];
            }
        }
        __syncthreads();
    }
}

__device__ void block_loop_alt(int* d1, int* p1, int* d2, int* d3, int* p3, int rowIndex,
    int maxIndex, int cell) {
    for (int k = 0; k < maxIndex; k++) {
        int kRow = k * maxIndex;
        for (int i = 0; i < maxIndex; i++) {
            int iRow = i * maxIndex;
            for (int j = 0; j < maxIndex; j++ ) {
                if (d2[iRow + k] != INT_MAX && d3[kRow + j] != INT_MAX) {
                    int t = d2[iRow + k] + d3[kRow + j];
                    printArr(d1, p1, maxIndex * maxIndex);
                    if (t < d1[iRow + j]) {
                        printf("t = %d, d1[%d + %d] = %d, p1[%d] = %d, p3[%d] = %d\n", t, iRow, j, d1[iRow + j],
                            iRow + j, p1[iRow + j], kRow + j, p3[kRow + j]);
                        printf("\n");
                        //printf("Changing things\n");
                        d1[iRow + j] = t;
                        p1[iRow + j] = p3[kRow + j];
                    }
                }
            }
        }
    }
}

__global__ void dep_block (int b, int num_blocks, int bl, int graphLength, int rem, int* dev_dist, int* dev_prev) {
    // B[b, b], B[b, b], B[b, b]


    if ( threadIdx.x >= bl || threadIdx.y >= bl ) return;


    int maxIndex = bl;
    if (b == num_blocks - 1) {
        maxIndex = rem;
        if (threadIdx.x > rem - 1 || threadIdx.y > rem - 1) return;
    }
    //printf("hello\n");
    int blockSize = bl * bl;

    int rowsDown = graphLength * num_blocks;


    int rowIndex = threadIdx.x * maxIndex;
    int cell = rowIndex + threadIdx.y;

    // copy block data to shared memory
    extern __shared__ int dist[];
    int* prev = dist + blockSize;

    int intoDevBlock = graphLength * bl * b + graphLength * threadIdx.x + bl * b + threadIdx.y;
    //int intoDevBlock = cell

    //printf("threadIdx.x = %d, threadIdx.y = %d, cell = %d, intoDevBlock = %d\n", threadIdx.x, threadIdx.y, cell, intoDevBlock);

    //printf("intoDevBlock for %d, %d = %d, dev_dist[%d] = %d, dev_prev[%d] = %d\n",
    //    threadIdx.x, threadIdx.y, intoDevBlock, intoDevBlock, dev_dist[intoDevBlock], intoDevBlock, dev_prev[intoDevBlock]);

    dist[cell] = dev_dist[intoDevBlock];
    prev[cell] = dev_prev[intoDevBlock];


    __syncthreads(); // make sure shared memory is fully initialised

    /*if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("Before loop:\n");
        printArr(dist, prev, maxIndex * maxIndex);
        printf("\n");
    }*/


    block_loop(dist, prev, dist, dist, prev, rowIndex, maxIndex, cell);

    //printf("threadIdx.x = %d, threadIdx.y = %d, intoDevBlock = %d bl = %d\n", threadIdx.x, threadIdx.y
    //, intoDevBlock, bl);
    __syncthreads();

    /*if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("After loop:\n");
        printArr(dist, prev, maxIndex * maxIndex);
        printf("\n");
    }*/
    // write-back
    dev_dist[intoDevBlock] = dist[cell];
    dev_prev[intoDevBlock] = prev[cell];
}



__global__ void pdep_blocks (int b, int num_blocks, int bl, int graphLength, int rem, int* dev_dist, int* dev_prev) {
    // B[b, i], B[b, b], B[b, i]
    // B[i, b], B[i, b], B[b, b]

    //if (threadIdx.x == 0 && threadIdx.y == 0) printf("blockIdx.x = %d, blockIdx.y = %d\n", blockIdx.x, blockIdx.y);
    if ( threadIdx.x >= bl || threadIdx.y >= bl ) return;

    int blockSize = bl * bl;

    // find out which block we are
    int blockX; int blockY;
    if ( blockIdx.y == b ) {
        blockX = blockIdx.x < b ? blockIdx.x : blockIdx.x + 1;
        blockY = b;
    }
    else if ( blockIdx.x == b ) {
        blockY = blockIdx.y < b ? blockIdx.y : blockIdx.y + 1;
        blockX = b;
    }
    else {
        return;
    }

    // need to fetch our block, and block B[b, b]
    extern __shared__ int dist[];
    int* dist_i = dist + blockSize;
    int* prev = dist_i + blockSize;
    int* prev_i = prev + blockSize;

    int blocksDown = blockSize * num_blocks;
    int rowsDown = bl * num_blocks;

    int rowIndex = threadIdx.x * bl;

    int devIndexIn = graphLength * bl * b + graphLength * threadIdx.x + bl * b + threadIdx.y;
    //printf("b = %d, blockX = %d, blockY = %d, devIndexIn = %d\n", b, blockX, blockY, devIndexIn);
    int cell = rowIndex + threadIdx.y;

    dist[cell] = dev_dist[devIndexIn];
    prev[cell] = dev_prev[devIndexIn];


    int* dist_1; int* dist_2; int* dist_3;
    int* prev_1;              int* prev_3;

    // maybe we should have seperate kernels for the two types of pdep blocks? Avoids branching

    if (blockX == b) { // B[b, i], B[b, b], B[b, i]
        devIndexIn = graphLength * bl * b + graphLength * threadIdx.x + bl * blockY + threadIdx.y;
        dist_i[cell] = dev_dist[devIndexIn];
        prev_i[cell] = dev_prev[devIndexIn];

        dist_1 = dist_i; dist_2 = dist; dist_3 = dist_i;
        prev_1 = prev_i;                prev_3 = prev_i;
    }
    else if ( blockY == b ) { // B[i, b], B[i, b], B[b, b]
        devIndexIn = graphLength * bl * blockX + graphLength * threadIdx.x + bl * b + threadIdx.y;
        dist_i[cell] = dev_dist[devIndexIn];
        prev_i[cell] = dev_prev[devIndexIn];

        dist_1 = dist_i; dist_2 = dist_i; dist_3 = dist;
        prev_1 = prev_i;                  prev_3 = prev;
    }

    // limit index and prune threads outside
    int maxIndex = bl;
    if (blockX == num_blocks - 1 || blockY == num_blocks - 1) {
        maxIndex = rem;
        if (threadIdx.x > rem || threadIdx.y > rem) return;
    }

    __syncthreads(); // ensure all initialised

    block_loop(dist_1, prev_1, dist_2, dist_3, prev_3, rowIndex, maxIndex, cell);

    dev_dist[devIndexIn] = dist_i[cell];
    dev_prev[devIndexIn] = prev_i[cell];
}

__global__ void indep_blocks (int b, int num_blocks, int bl, int graphLength, int rem, int* dev_dist, int* dev_prev) {
    // B[i, j], B[i, b], B[b, j]
    //if ( threadIdx.x == 0 && threadIdx.y == 0) printf("blockIdx.x = %d, blockIdx.y = %d\n", blockIdx.x, blockIdx.y);
    if ( threadIdx.x >= bl || threadIdx.y >= bl ) return;

    int blockSize = bl * bl;

    // find out what block we are
    int blockX = blockIdx.x + (blockIdx.x >= b);
    int blockY = blockIdx.y + (blockIdx.y >= b);

    int rowIndex = threadIdx.x * bl;
    int cell = rowIndex + threadIdx.y;

    // fetch our block as well as our corresponding partially dependent blocks
    extern __shared__ int shared[];
    int* dist_1 = shared; int* prev_1 = dist_1 + blockSize;
    int* dist_2 = prev_1 + blockSize;
    int* dist_3 = dist_2 + blockSize; int* prev_3 = dist_3 + blockSize;

    int maxIndex = bl;
    if (blockX == num_blocks - 1 || blockY == num_blocks - 1 || b == num_blocks - 1) {
        maxIndex = rem;
        if (threadIdx.x > rem || threadIdx.y > rem) return;
    }

    // populate shared memory
    int sharedIndexIn = rowIndex + threadIdx.y;
    int devIndexIn = graphLength * bl * blockX + graphLength * threadIdx.x + bl * blockY + threadIdx.y;
    //printf("blockX = %d, blockY = %d, devIndexIn = %d\n", blockX, blockY, devIndexIn);
    dist_1[sharedIndexIn] = dev_dist[devIndexIn]; prev_1[sharedIndexIn] = dev_prev[devIndexIn];
    dist_2[sharedIndexIn] = dev_dist[graphLength * bl * blockX + graphLength * threadIdx.x + bl * b + threadIdx.y];
    int devIndexIn_3 = graphLength * bl * b + graphLength * threadIdx.x + bl * blockY + threadIdx.y;
    dist_3[sharedIndexIn] = dev_dist[devIndexIn_3]; prev_3[sharedIndexIn] = dev_prev[devIndexIn_3];

    __syncthreads();

    block_loop(dist_1, prev_1, dist_2, dist_3, prev_3, rowIndex, maxIndex, cell);

    dev_dist[devIndexIn] = dist_1[sharedIndexIn];
    dev_prev[devIndexIn] = prev_1[sharedIndexIn];
}

Result** cuda_FWarsh(GraphMatrix& graph, int block_length) {
    int graphSize = graph.GetSize();
    int matSize = graphSize * graphSize;

    int* dev_dist; gpuErrchk(cudaMalloc(&dev_dist, sizeof(int) * matSize));
    gpuErrchk(cudaMemcpy(dev_dist, graph.GetMatrix(), sizeof(int) * matSize, cudaMemcpyHostToDevice));

    int* dev_prev; gpuErrchk(cudaMalloc(&dev_prev, sizeof(int) * matSize));
    GraphMatrix prev = GraphMatrix(graph, -1);
    for (int r = 0; r < graphSize; r++) {
        int rowIndex = r * graphSize;
        for (int c = 0; c < graphSize; c++) {
            if (graph[rowIndex + c] != INT_MAX) { prev[rowIndex + c] = r; } // set previous as in graph
        }
    }
    //printArr(graph.GetMatrix(), prev.GetMatrix(), matSize);
    gpuErrchk(cudaMemcpy(dev_prev, &prev[0], sizeof(int) * matSize, cudaMemcpyHostToDevice));

    int num_blocks = (graph.GetSize() + block_length - 1) / block_length; // ceiling the value (1 axis)
    int rem = graph.GetSize() % block_length;
    if (rem == 0) { rem = block_length; }

    dim3 block_threads(block_length, block_length);

    int pdep_count = num_blocks * 2 - 2;
    dim3 pdep_dim(num_blocks - 1, num_blocks - 1);

    int indep_count = (num_blocks * num_blocks) - pdep_count - 1;
    dim3 indep_dim(num_blocks - 1, num_blocks - 1);

    size_t memsize = sizeof(int) * block_length * block_length * 2;

    size_t free, totalmem;
    for (int block = 0; block < num_blocks; block++) {
        dep_block<<<1, block_threads, memsize>>>(block, num_blocks, block_length, graphSize, rem, dev_dist, dev_prev);
        pdep_blocks<<<pdep_dim, block_threads, memsize * 2>>>(block, num_blocks, block_length, graphSize, rem, dev_dist, dev_prev);
        indep_blocks<<<indep_dim, block_threads, memsize * 2 + memsize / 2>>>(block, num_blocks, block_length, graphSize, rem, dev_dist, dev_prev);
    }

    cudaDeviceSynchronize();
    cudaPeekAtLastError();

    GraphMatrix dist = GraphMatrix(graph, INT_MAX);

    cudaMemcpy(&dist[0], dev_dist, sizeof(int) * matSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(&prev[0], dev_prev, sizeof(int) * matSize, cudaMemcpyDeviceToHost);
    cudaFree(dev_dist); cudaFree(dev_prev);

    Result** results = new Result*[graphSize];

    for (int i = 0; i < graphSize; i++ ) {
        results[i] = new Result;
        results[i]->dist = &dist[graphSize * i];
        results[i]->prev = &prev[graphSize * i];
    }

    return results;
}
