#include <stdio.h>
#include <limits.h>
#include "BMFord.h"
#include <stdlib.h>
#include <pthread.h>
#include "GraphParse.h"
#include <errno.h>

void relax(int u, const Node* v, int* dist, int* prev) {
    if (dist[u] != INT_MAX && dist[u] + v->weight < dist[v->vertex]) {
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
            if (adj != NULL) {
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

void BMFordSSSP_t(const void* args) {
    MultiSSSPArgs* a = (MultiSSSPArgs*)args;

    const Graph* graph = a->graph;
    pthread_mutex_t* q_lock = a->q_lock;
    pthread_mutex_t* r_lock = a->r_lock;
    int* next_node = a->next_node;
    Result** results = a->results;

    while (1) {
        pthread_mutex_lock(q_lock);
        int src = *next_node;
        if (src < graph->size) {
            (*next_node)++;
            pthread_mutex_unlock(q_lock);

            Result* result = malloc(sizeof(Result));
            result = BMFordSSSP(graph, src);

            pthread_mutex_lock(r_lock);
            results[src] = result;
            pthread_mutex_unlock(r_lock);
        }
        else {
            pthread_mutex_unlock(q_lock);
            pthread_exit(NULL);
        }

        pthread_mutex_unlock(q_lock);
    }
}

void Relax_t(const void* args) {
    BMF_b_args* a = (BMF_b_args*)args;
    const Graph* graph = a->graph;
    int** m_dist = a->m_dist;
    int** m_prev = a->m_prev;
    Result** results = a->results;

    pthread_mutex_t* q_lock = a->q_lock;
    pthread_mutex_t* i_lock = a->i_lock;
    pthread_mutex_t* v_locks = a->v_locks;
    // Synchronise iterations. Each dist depends on the last iteration's.
    while (1) {
        // check if we've reached V iterations
        pthread_mutex_lock(i_lock);
        if (*a->iter == graph->size) {
            pthread_mutex_unlock(i_lock);
            break;
        }
        pthread_mutex_unlock(i_lock);
        ///////////////

        // We haven't, lets get the next node to work on
        pthread_mutex_lock(q_lock);
        int src = *a->next_node;
        if (src >= graph->size) {
            // End of iteration, need to increment iter number
            // Need to make sure every node synchronises at this point though, and only one node makes changes
            if (pthread_mutex_trylock(i_lock) == 0) {
                pthread_mutex_lock(q_lock);
                *a->iter++;
                *a->next_node = 0;
                pthread_mutex_unlock(i_lock);
                pthread_mutex_unlock(q_lock);
            }
            pthread_barrier_wait(a->barrier);
            continue;
        }
        *a->next_node++;
        pthread_mutex_unlock(q_lock);

        Node* adj = graph->verts[src];
        while (adj != NULL) {
            // Lock source and target
            pthread_mutex_lock(v_locks + src);
            pthread_mutex_lock(v_locks + adj->vertex);
            int v = adj->vertex;

            // Perform the relaxation
            if (m_dist[src][v] != INT_MAX && m_dist[src][v] + adj->weight < m_dist[src][v]) {
                m_dist[src][v] = m_dist[src][v] + adj->weight;
                m_prev[src][v] = adj->vertex;
            }

            // Release locks
            pthread_mutex_unlock(v_locks + src);
            pthread_mutex_unlock(v_locks + adj->vertex);

            adj = adj->next; // Continue for all adjacent nodes
        }
    }
}

Result** BMFordAPSP(const Graph* graph) {
    Result** results = malloc(sizeof(Result*) * graph->size);
    for (int n = 0; n < graph->size; n++) {
        results[n] = BMFordSSSP(graph, n);
    }
    return results;
}

Result** BMFordAPSP_mt_a(const Graph* graph, int numthreads) {
    pthread_t* threads[numthreads];
    Result** results = malloc(sizeof(Result*) * graph->size);

    int next_node = 0;
    pthread_mutex_t q_lock;
    pthread_mutex_init(&q_lock, NULL); // queue lock
    pthread_mutex_t r_lock;
    pthread_mutex_init(&r_lock, NULL); // result lock

    MultiSSSPArgs* args = malloc(sizeof(MultiSSSPArgs));
    args->next_node = &next_node;
    args->results = results;
    args->graph = graph;
    args->q_lock = &q_lock;
    args->r_lock = &r_lock;

    for (int t = 0; t < numthreads; t++) {
        pthread_create((pthread_t*)threads + t, NULL, (void*)BMFordSSSP_t, (void*)args);
    }

    for (int t = 0; t < numthreads; t++) {
        pthread_join(threads[t], NULL);
    }

    pthread_mutex_destroy(&q_lock);
    pthread_mutex_destroy(&r_lock);
    free(args);
    return results;
}

Result** BMFordAPSP_mt_b(const Graph* graph, int numthreads) {
    pthread_t* threads[numthreads];
    Result** results = malloc(sizeof(Result*) * graph->size);

    pthread_mutex_t r_lock;
    pthread_mutex_init(&r_lock, NULL); // result lock
    pthread_mutex_t* v_locks = malloc(sizeof(pthread_mutex_t) * graph->size);
    for (int l = 0; l < graph->size; l++) {
        pthread_mutex_init(v_locks + l, NULL);
    }
    pthread_mutex_t q_lock;
    pthread_mutex_init(&q_lock, NULL); // queue lock
    pthread_mutex_t i_lock;
    pthread_mutex_init(&i_lock, NULL); // iteration lock
    pthread_barrier_t i_barrier;
    pthread_barrier_init(&i_barrier, NULL, numthreads);

    // m_dist -> n*n*n matrix of distances, global
    // dim 1 -> iteration #
    // dim 2 -> vertex u
    // dim 3 -> vertex v
    // i.e, m_dist[1][1][2] == minimum distance from 1 to 2 at the 1st iteration
    int** m_dist = malloc(sizeof(int*) * graph->size);
    int** m_prev = malloc(sizeof(int*) * graph->size);
    for (int u = 0; u < graph->size; u++) {
        m_dist[u] = malloc(sizeof(int) * graph->size);
        m_prev[u] = malloc(sizeof(int) * graph->size);
        for (int v = 0; v < graph->size; v++) {
            if (u == v) {
                m_dist[u][v] = 0;
                m_prev[u][v] = u;
            }
            else {
                m_dist[u][v] = INT_MAX;
                m_prev[u][v] = -1;
            }
        }
        // while we're here, initialise the results lists for every node
        results[u] = malloc(sizeof(Result));
        results[u]->dist = malloc(sizeof(int) * graph->size);
        results[u]->prev = malloc(sizeof(int) * graph->size);
    }

    int next_node = 0;
    int iter = 0;

    BMF_b_args args = {
        .graph = graph,
        .m_dist = m_dist,
        .m_prev = m_prev,
        .next_node = &next_node,
        .iter = &iter,
        .results = results,
        .r_lock = &r_lock,
        .q_lock = &q_lock,
        .i_lock = &i_lock,
        .v_locks = v_locks,
        .barrier = &i_barrier,
    };

    for (int t = 0; t < numthreads; t++) {
        pthread_create((pthread_t*)threads + t, NULL, Relax_t, (void*)&args);
    }

    for (int t = 0; t < numthreads; t++) {
        pthread_join(threads[t], NULL);
    }

    pthread_mutex_destroy(&r_lock);
    pthread_mutex_destroy(&q_lock);
    pthread_mutex_destroy(&i_lock);
    pthread_barrier_destroy(&i_barrier);
    for (int i = 0; i < graph->size; i++) {
        pthread_mutex_destroy(&v_locks[i]);
    }

    free(m_dist);

    return results;
}
