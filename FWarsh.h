#pragma once
#include "GraphParse.h"

typedef struct FWarsh_args
{
    int nblocks;
    int rem;
    int block_length;
    int** m_dist;
    int** m_prev;
    int B1x; int B1y; // from
    int B2x; int B2y; // to
    int B3x; int B3y; // through
} FWarsh_args;

void m_dist_init(const Graph* graph, int** m_dist, int** m_prev);

void repath(int u, int v, Result** results, const int** m_dist, const int** m_prev);
Result** FWarsh(const Graph* graph);

void do_blocks(void* args);
FWarsh_args* construct_args(int g, int r, int l, const int** d, const int** prev, int b1x, int b1y, int b2x, int b2y,
    int b3x, int b3y);
Result** FWarsh_mt(const Graph* graph, int block_length, int numthreads);