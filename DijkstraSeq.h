#pragma once
#include "graphparse.h"

typedef struct Queue {
	int tail;
	int* items;
} Queue;

Queue* createQueue(int size) {

}
int* DijkstraSSSP(const Graph* graph, int src);