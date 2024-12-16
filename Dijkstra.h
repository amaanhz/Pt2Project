#pragma once
#include "GraphParse.h"

Result* DijkstraSSSP(const Graph* graph, int src);
void DijkstraSSSP_t(const void* args); // Wrapper for pthread
Result** DijkstraAPSP(const Graph* graph);
Result** DijkstraAPSP_mt(const Graph* graph, int numthreads);