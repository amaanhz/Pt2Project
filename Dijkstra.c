#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <pthread.h>
#include "Dijkstra.h"
#include "GraphParse.h"

void enq(Queue* q, int item) {
	if (q->tail < q->max) {
		q->items[q->tail] = item;
		q->tail++;
	}
	else {
		printf("Tried to enqueue past max size!\n");
	}
}

int dqmin(Queue* q, const int* dist) {
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
			if (result->dist[i] >= 0 && result->dist[i] < INT_MAX) {
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
	int items[graph->size];
	Queue queue = {
		.items = items, .max = graph->size, .tail=0
	};

	for (int i = 0; i < graph->size; i++) {
		dist[i] = INT_MAX;
		enq(&queue, i);
	}

	dist[src] = 0;

	while (queue.tail > 0) {
		int u = dqmin(&queue, dist);
		if (dist[u] == INT_MAX) { continue; }

		for (int i = 0; i < queue.tail; i++) {
			int v = queue.items[i];
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
	
	return result;
}

void DijkstraSSSP_t(const void* args) {	// Initialise SSSP threads
	DijkstraArgs* a = (DijkstraArgs*) args;

	// unpacking on init //
	const Graph* graph = a->graph; pthread_mutex_t* q_lock = a->q_lock;
	pthread_mutex_t* r_lock = a->r_lock; int* next_node = a->next_node;
	DijkstraResult** results = a->results;
	///////////////////////

	while (1)
	{
		pthread_mutex_lock(q_lock);
		int src = *next_node;
		if (src < graph->size)
		{
			(*next_node)++;
			pthread_mutex_unlock(q_lock);

			DijkstraResult* result = malloc(sizeof(DijkstraResult));
			result = DijkstraSSSP(graph, src);

			pthread_mutex_lock(r_lock);
			results[src] = result;
			pthread_mutex_unlock(r_lock);
		}
		else
		{
			pthread_mutex_unlock(q_lock);
			pthread_exit(NULL);
		}

		pthread_mutex_unlock(q_lock);
	}
}

DijkstraResult** DijkstraAPSP(const Graph* graph)
{
	DijkstraResult** results = malloc(sizeof(DijkstraResult*) * graph->size);
	for (int i = 0; i < graph->size; i++) {
		results[i] = DijkstraSSSP(graph, i);
	}
	return results;
}

DijkstraResult** DijkstraAPSP_mt(const Graph* graph, int numthreads)
{
	pthread_t* threads[numthreads];
	DijkstraResult** results = malloc(sizeof(DijkstraResult*) * graph->size);

	int next_node = 0;
	pthread_mutex_t q_lock; pthread_mutex_init(&q_lock, NULL); // queue lock
	pthread_mutex_t r_lock; pthread_mutex_init(&r_lock, NULL); // result lock

	DijkstraArgs* args = malloc(sizeof(DijkstraArgs));
	args->next_node = &next_node; args->results = results;
	args->graph = graph; args->q_lock = &q_lock; args->r_lock = &r_lock;

	for (int t = 0; t < numthreads; t++)
	{
		pthread_create((pthread_t*)threads + t, NULL, DijkstraSSSP_t, (void*) args);
	}

	for (int t = 0; t < numthreads; t++)
	{
		pthread_join(threads[t], NULL);
	}

	pthread_mutex_destroy(&q_lock); pthread_mutex_destroy(&r_lock);
	free(args);
	return results;
}



