#pragma once
#include "GraphParse.h"

typedef struct BMFResult {
    int* dist;
    int* prev;
} BMFResult;

void relax(int u, Node* v, int* dist, int* prev);

BMFResult* BMFordSSSP(const Graph* graph, int src);

