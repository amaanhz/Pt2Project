#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "graphparse.h"

int* DijkstraSSSP(const Graph* graph, int src) {
	int* dist = malloc(sizeof(int) * graph->size);
	int* prev = calloc(sizeof(int) * graph->size); // calloc to enforce null initial val


	// set initial values
	memset(dist, INT_MAX, sizeof(int) * graph->size)

	dist[src] = 0;

	return NULL;
}

