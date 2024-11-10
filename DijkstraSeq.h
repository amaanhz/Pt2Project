#pragma once
#include "graphparse.h"

typedef struct Queue {
	int max;
	int tail;
	int* items;
} Queue;

typedef struct DijkstraResult {
	int size;
	int* dist;
	int* prev;
} DijkstraResult;

Queue* createQueue(int size);
void enq(Queue* q, int item);
int dqmin(Queue* q, int* dist);

int* DijkstraSSSP(const Graph* graph, int src);