#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include "DijkstraSeq.h"
#include "graphparse.h"

Queue* createQueue(int size) {
	Queue* queue = malloc(sizeof(Queue));
	queue->max = size;  queue->tail = 0;
	queue->items = malloc(sizeof(int) * size); // empty positions in queue are null, so calloc
	return queue;
}

void enq(Queue* q, int item) {
	if (q->tail < q->max) {
		q->items[q->tail] = item;
		q->tail++;
	}
	else {
		printf("Tried to enqueue past max size!\n");
	}
}

int dqmin(Queue* q, int* dist) {
	int min = 0; int mindist = INT_MAX; int d;
	int j = -1; // save the position in array we need to remove
	for (int i = 0; i < q->tail; i++) {
		d = dist[q->items[i]];
		if (d <= mindist) {
			mindist = d;
			min = q->items[i];
			j = i;
		}
	}
	// need to reoragnise second half of queue
	memmove(&(q->items[j]), &(q->items[j + 1]), sizeof(int) * (q->tail - j)); // move j+1.. to j
	q->tail--;


	return min;
}

void printResult(const DijkstraResult* result, int src, int size) {
	for (int i = 0; i < size; i++) {
		if (i != src) {
			if (result->dist[i] < INT_MAX) {
				printf("Distance to %d: %d, ", i, result->dist[i]);
				printf("Path: ");
				int v = result->prev[i];
				printf("%d <- ", i);
				while (v != src && v != i) {
					printf("%d <- ", v);
					v = result->prev[v];
				}
				printf("%d\n", v);
			}
			else {
				printf("%d is unreachable.\n", i);
			}
		}
		else {
			printf("%d is the source node.\n", i);
		}
	}
	printf("\n");
}


DijkstraResult* DijkstraSSSP(const Graph* graph, int src) {
	int* dist = malloc(sizeof(int) * graph->size);
	int* prev = calloc(graph->size, sizeof(int)); // calloc to enforce null initial val
	Queue* queue = createQueue(graph->size);

	for (int i = 0; i < graph->size; i++) {
		dist[i] = INT_MAX;
		enq(queue, i);
	}

	dist[src] = 0;

	while (queue->tail > 0) {
		int u = dqmin(queue, dist);
		if (dist[u] == INT_MAX) { continue; }

		for (int i = 0; i < queue->tail; i++) {
			int v = queue->items[i];
			int d = neighbour(graph, u, v);
			if (d) {
				int thruU = dist[u] + d;
				if (thruU < dist[v]) {
					dist[v] = thruU;
					prev[v] = u;
				}
			}
		}
	}

	DijkstraResult* result = malloc(sizeof(DijkstraResult));
	result->dist = dist; result->prev = prev; //result->src = src;

	free(queue);
	return result;
}

DijkstraResult** DijkstraAPSP(const Graph* graph)
{
	DijkstraResult** results = malloc(sizeof(DijkstraResult*) * graph->size);
	for (int i = 0; i < graph->size; i++) {
		results[i] = DijkstraSSSP(graph, i);
	}
	return results;
}

DijkstraResult** DijkstraAPSP_mt(const Graph* graph)
{
}



