#pragma once
#include "GraphParse.h"

typedef struct Queue {
	int max;
	int tail;
	int* items;
} Queue;


Queue* createQueue(int size);
void enq(Queue* q, int item);
int dqmin(Queue* q, const int* dist);

Result* DijkstraSSSP(const Graph* graph, int src);
void DijkstraSSSP_t(const void* args); // Wrapper for pthread
Result** DijkstraAPSP(const Graph* graph);
Result** DijkstraAPSP_mt(const Graph* graph, int numthreads);