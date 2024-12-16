#pragma once
#include "GraphParse.h"

typedef struct BMF_b_args
{
    const Graph* graph;
    int start;
    int end;


} BMF_b_args;

void relax(int u, const Node* v, int* dist, int* prev);

Result* BMFordSSSP(const Graph* graph, int src);
void BMFordSSSP_t(const void* args);

void Relax_t(const void* args);

Result** BMFordAPSP(const Graph* graph);
Result** BMFordAPSP_mt_a(const Graph* graph, int numthreads);
// Embarassingly parallel
Result** BMFordAPSP_mt_b(const Graph* graph, int numthreads);
// Less embarassingly parallel