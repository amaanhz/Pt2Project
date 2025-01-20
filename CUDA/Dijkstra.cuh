#pragma once
#include "GraphParse.h"
#include "GraphMatrix.h"


Result** cuda_DijkstraAPSP(GraphMatrix& graph);

int fastmin(int* arr, int* queues, int size);
void placeOnDevice(int* ptr, int size, int* src);
void process_node(GraphMatrix& graph, GraphMatrix& dist,
    GraphMatrix& prev, int node);