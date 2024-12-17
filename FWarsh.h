#pragma once
#include "GraphParse.h"

int** m_dist_init(const Graph* graph, int** m_dist, int** m_prev);


void repath(int u, int v, Result** results, const int** m_dist, const int** m_prev);
Result** FWarsh(const Graph* graph);
