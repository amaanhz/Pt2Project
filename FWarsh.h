#pragma once
#include "GraphParse.h"

typedef struct FWarsh_args
{
    int block_size;
    int** m_dist;
    int** m_prev;
    int B1x; int B1y;
    int B2x; int B2y;
    int B3x; int B3y;
} FWarsh_args;

int** m_dist_init(const Graph* graph, int** m_dist, int** m_prev);

void repath(int u, int v, Result** results, const int** m_dist, const int** m_prev);
Result** FWarsh(const Graph* graph);

void do_blocks(void* args);
Result** FWarsh_mt(const Graph* graph, int numthreads);