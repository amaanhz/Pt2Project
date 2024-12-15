#pragma once
#include "GraphParse.h"



void relax(int u, const Node* v, int* dist, int* prev);

Result* BMFordSSSP(const Graph* graph, int src);
void BMFordSSSP_t(const void* args);

Result** BMFordAPSP(const Graph* graph);
Result** BMFordAPSP_mt_a(const Graph* graph, int numthreads);
