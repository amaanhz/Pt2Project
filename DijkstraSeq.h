#pragma once
#include "graphparse.h"

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
	const Graph* graph;
	int src;
	//DijkstraResult** results;
} DijkstraArgs;

Queue* createQueue(int size);
void enq(Queue* q, int item);
int dqmin(Queue* q, int* dist);
void printResult(const DijkstraResult* result, int src, int size);

DijkstraResult* DijkstraSSSP(const Graph* graph, int src);
void* DijkstraSSSP_t(void* args); // Wrapper for pthread
DijkstraResult** DijkstraAPSP(const Graph* graph);
DijkstraResult** DijkstraAPSP_mt(const Graph* graph);