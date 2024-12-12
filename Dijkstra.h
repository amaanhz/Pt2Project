#pragma once
#include "GraphParse.h"

typedef struct Queue {
	int max;
	int tail;
	int* items;
} Queue;



typedef struct DijkstraArgs {
	int* next_node;
	pthread_mutex_t* q_lock;
	pthread_mutex_t* r_lock;
	const Graph* graph;
	Result** results;
} DijkstraArgs;

Queue* createQueue(int size);
void enq(Queue* q, int item);
int dqmin(Queue* q, const int* dist);
void printResult(const Result* result, int src, int size);

Result* DijkstraSSSP(const Graph* graph, int src);
void DijkstraSSSP_t(const void* args); // Wrapper for pthread
Result** DijkstraAPSP(const Graph* graph);
Result** DijkstraAPSP_mt(const Graph* graph, int numthreads);