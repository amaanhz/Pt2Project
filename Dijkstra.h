#pragma once
#include "GraphParse.h"

typedef struct Queue {
	int max;
	int tail;
	int* items;
} Queue;

typedef struct DijkstraResult {
	int* dist;
	int* prev;
} DijkstraResult;

typedef struct DijkstraArgs {
	int* next_node;
	pthread_mutex_t* q_lock;
	pthread_mutex_t* r_lock;
	const Graph* graph;
	DijkstraResult** results;
} DijkstraArgs;

Queue* createQueue(int size);
void enq(Queue* q, int item);
int dqmin(Queue* q, const int* dist);
void printResult(const DijkstraResult* result, int src, int size);

DijkstraResult* DijkstraSSSP(const Graph* graph, int src);
void DijkstraSSSP_t(const void* args); // Wrapper for pthread
DijkstraResult** DijkstraAPSP(const Graph* graph);
DijkstraResult** DijkstraAPSP_mt(const Graph* graph, int numthreads);