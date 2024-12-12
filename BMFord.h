#pragma once
#include "GraphParse.h"


void relax(int u, const Node* v, int* dist, int* prev);

Result* BMFordSSSP(const Graph* graph, int src);

