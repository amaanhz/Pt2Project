#include <stdio.h>
#include <limits.h>
#include "BMFord.h"
#include <stdlib.h>
#include "GraphParse.h"

void relax(int u, Node* v, int* dist, int* prev)
{
    if (dist[u] + v->weight < dist[v->vertex])
    {
        // if dist[u] + w < dist[v]
        dist[v->vertex] = dist[u] + v->weight;
        prev[v->vertex] = u;
    }
}

BMFResult* BMFordSSSP(const Graph* graph, int src) {
    int* dist = malloc(sizeof(int) * graph->size);
    int* prev = calloc(graph->size, sizeof(int));

    for (int i = 0; i < graph->size; i++) {
        dist[i] = INT_MAX;
    }

    dist[src] = 0;

    // Relax here
    for (int i = 0; i < graph->size - 1; i++) { // Repeat |V| - 1 times
        for (int n = 0; n < graph->size; i++) { // Loop through the graph
            Node* adj = graph->verts[n];
            if (adj != NULL)
            {
                while (adj->next != NULL) {
                    relax(n, adj, dist, prev);
                    adj = adj->next;
                }
                relax(n, adj, dist, prev);
            }
        }
    }
    BMFResult* result = malloc(sizeof(BMFResult));
    result->dist = dist;
    result->prev = prev;

    return result;
}
