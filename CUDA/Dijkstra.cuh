#pragma once
#include "GraphParse.h"
#include "GraphMatrix.h"
#include <cuda_runtime.h>


Result** cuda_DijkstraAPSP(GraphMatrix& graph);

void fastmin(volatile const int* arr, volatile const int* queues, volatile int* in_idxs, int size, volatile int* out_vals, volatile int* out_idxs,
    volatile int* block_id_masks, volatile int grid_size, volatile int* out_min, volatile int* out_minid, cudaStream_t* stream);
void placeOnDevice(int* ptr, int size, int* src);
