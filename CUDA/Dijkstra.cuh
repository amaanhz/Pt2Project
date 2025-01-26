#pragma once
#include "GraphParse.h"
#include "GraphMatrix.h"


Result** cuda_DijkstraAPSP(GraphMatrix& graph);

void fastmin(const int* arr, const int* queues, int* idxs, int size, int* out_vals, int* out_idxs, int* block_id_masks,
    int grid_size, int* out_min, int* out_minid);
void placeOnDevice(int* ptr, int size, int* src);
void process_node(int* graph, int* dist, int* prev, int* queues, int* node, int dim, int grid_size);
