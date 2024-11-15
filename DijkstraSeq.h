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

Queue* createQueue(int size);
void enq(Queue* q, int item);
int dqmin(Queue* q, int* dist);
void printResult(DijkstraResult* result, int size);

DijkstraResult* DijkstraSSSP(const Graph* graph, int src);
DijkstraResult** DijkstraAPSP(const Graph* graph);