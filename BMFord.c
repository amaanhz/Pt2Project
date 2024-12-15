#include <stdio.h>
#include <limits.h>
#include "BMFord.h"
#include <stdlib.h>
#include <pthread.h>
#include "GraphParse.h"

void relax(int u, const Node* v, int* dist, int* prev)
{
    if (dist[u] != INT_MAX && dist[u] + v->weight < dist[v->vertex])
    {
        // if dist[u] + w < dist[v]
        dist[v->vertex] = dist[u] + v->weight;
        prev[v->vertex] = u;
    }
}

Result* BMFordSSSP(const Graph* graph, int src) {
    int* dist = malloc(sizeof(int) * graph->size);
    int* prev = calloc(graph->size, sizeof(int));

    for (int i = 0; i < graph->size; i++) {
        dist[i] = INT_MAX;
    }

    dist[src] = 0;

    // Relax here
    for (int i = 0; i < graph->size - 1; i++) { // Repeat |V| - 1 times
        for (int n = 0; n < graph->size; n++) { // Loop through the graph
            Node* adj = graph->verts[n];
            if (adj != NULL)
            {
                while (adj->next != NULL) {
                    relax(n, adj, dist, prev);
                    adj = adj->next;
                }
                if (adj != NULL) { relax(n, adj, dist, prev); }
            }
        }
    }
    Result* result = malloc(sizeof(Result));
    result->dist = dist;
    result->prev = prev;

    return result;
}

void BMFordSSSP_t(const void* args)
{
    MultiSSSPArgs* a = (MultiSSSPArgs*) args;

    const Graph* graph = a->graph; pthread_mutex_t* q_lock = a->q_lock;
    pthread_mutex_t* r_lock = a->r_lock; int* next_node = a->next_node;
    Result** results = a->results;

    while (1)
    {
        pthread_mutex_lock(q_lock);
        int src = *next_node;
        if (src < graph->size)
        {
            (*next_node)++;
            pthread_mutex_unlock(q_lock);

            Result* result = malloc(sizeof(Result));
            result = BMFordSSSP(graph, src);

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


Result** BMFordAPSP(const Graph* graph)
{
    Result** results = malloc(sizeof(Result*) * graph->size);
    for (int n = 0; n < graph->size; n++)
    {
        results[n] = BMFordSSSP(graph, n);
    }
    return results;
}

Result** BMFordAPSP_mt_a(const Graph* graph, int numthreads)
{
    pthread_t* threads[numthreads];
    Result** results = malloc(sizeof(Result*) * graph->size);

    int next_node = 0;
    pthread_mutex_t q_lock; pthread_mutex_init(&q_lock, NULL); // queue lock
    pthread_mutex_t r_lock; pthread_mutex_init(&r_lock, NULL); // result lock

    MultiSSSPArgs* args = malloc(sizeof(MultiSSSPArgs));
    args->next_node = &next_node; args->results = results;
    args->graph = graph; args->q_lock = &q_lock; args->r_lock = &r_lock;

    for (int t = 0; t < numthreads; t++)
    {
        pthread_create((pthread_t*)threads + t, NULL, BMFordSSSP_t, (void*) args);
    }

    for (int t = 0; t < numthreads; t++)
    {
        pthread_join(threads[t], NULL);
    }

    pthread_mutex_destroy(&q_lock); pthread_mutex_destroy(&r_lock);
    free(args);
    return results;
}
