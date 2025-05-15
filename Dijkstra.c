#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <pthread.h>
#include "Dijkstra.h"
#include "GraphParse.h"


Result* DijkstraSSSP(const Graph* graph, int src) {
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

    Result* result = malloc(sizeof(Result));
    result->dist = dist;
    result->prev = prev; //result->src = src;

    return result;
}

void DijkstraSSSP_t(const void* args) { // Initialise SSSP threads
    MultiSSSPArgs* a = (MultiSSSPArgs*)args;

    // unpacking on init //
    const Graph* graph = a->graph;
    pthread_mutex_t* q_lock = a->q_lock;
    //pthread_mutex_t* r_lock = a->r_lock;
    int* next_node = a->next_node;
    Result** results = a->results;
    ///////////////////////

    while (1) {
        int err = pthread_mutex_trylock(q_lock);
        //if (err < 0) { printf("ERROR ON TRYLOCK!"); }
        int src = *next_node;
        if (src < graph->size) {
            (*next_node)++;
            pthread_mutex_unlock(q_lock);

            Result* result = malloc(sizeof(Result));
            result = DijkstraSSSP(graph, src);

            //pthread_mutex_lock(r_lock);
            results[src] = result;
            //pthread_mutex_unlock(r_lock);
        }
        else {
            pthread_mutex_unlock(q_lock);
            pthread_exit(NULL);
        }

        pthread_mutex_unlock(q_lock);
    }
}

Result** DijkstraAPSP(const Graph* graph) {
    Result** results = malloc(sizeof(Result*) * graph->size);
    for (int i = 0; i < graph->size; i++) {
        results[i] = DijkstraSSSP(graph, i);
    }
    return results;
}

Result** DijkstraAPSP_mt(const Graph* graph, int numthreads) {
    pthread_t* threads[numthreads];
    Result** results = malloc(sizeof(Result*) * graph->size);

    int next_node = 0;
    pthread_mutex_t q_lock;
    pthread_mutex_init(&q_lock, NULL); // queue lock
    //pthread_mutex_t r_lock;
    //pthread_mutex_init(&r_lock, NULL); // result lock

    MultiSSSPArgs* args = malloc(sizeof(MultiSSSPArgs));
    args->next_node = &next_node;
    args->results = results;
    args->graph = graph;
    args->q_lock = &q_lock;
    //args->r_lock = &r_lock;

    for (int t = 0; t < numthreads; t++) {
        pthread_create((pthread_t*)threads + t, NULL, DijkstraSSSP_t, (void*)args);
    }

    for (int t = 0; t < numthreads; t++) {
        pthread_join(threads[t], NULL);
    }

    pthread_mutex_destroy(&q_lock);
    //pthread_mutex_destroy(&r_lock);
    free(args);
    return results;
}
