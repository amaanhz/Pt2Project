#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "GraphParse.h"


void m_dist_init(const Graph* graph, int** m_dist, int** m_prev)
{
    for (int u = 0; u < graph->size; u++)
    {
        m_dist[u] = malloc(sizeof(int) * graph->size);
        m_prev[u] = malloc(sizeof(int) * graph->size);
        for (int v = 0; v < graph->size; v++)
        {
            if (u == v)
            {
                m_dist[u][v] = 0;
                m_prev[u][v] = u;

            }
            else
            {
                m_dist[u][v] = INT_MAX;
                m_prev[u][v] = -1;
            }
        }
    }
    for (int n = 0; n < graph->size; n++)
    {
        Node* adj = graph->verts[n];
        while (adj != NULL)
        {
            m_dist[n][adj->vertex] = adj->weight;
            m_prev[n][adj->vertex] = n;
            adj = adj->next;
        }
    }
}

void repath(int u, int v, Result** results, const int** m_dist, const int** m_prev)
{
    Result* result = results[u];

    if (m_prev[u][v] == -1)
    {
        result->dist[v] = INT_MAX;
        result->prev[v] = -1;
    }
    else
    {
        result->dist[v] = m_dist[u][v];
        result->prev[v] = m_prev[u][v];
        while (u != v)
        {
            v = m_prev[u][v];
            result->prev[v] = m_prev[u][v];
        }
    }
}


Result** FWarsh(const Graph* graph)
{
    Result** results = malloc(sizeof(Result*) * graph->size);
    int** m_dist = malloc(sizeof(int*) * graph->size);
    int** m_prev = malloc(sizeof(int*) * graph->size);

    m_dist_init(graph, m_dist, m_prev);

    for (int k = 0; k < graph->size; k++)
    {
        for (int i = 0; i < graph->size; i++)
        {
            for (int j = 0; j < graph->size; j++)
            {
                if (m_dist[i][k] != INT_MAX && m_dist[k][j] != INT_MAX && m_dist[i][j] > m_dist[i][k] + m_dist[k][j])
                {
                    m_dist[i][j] = m_dist[i][k] + m_dist[k][j];
                    m_prev[i][j] = m_prev[k][j];
                }
            }
        }
    }

    // Reconstruct paths
    for (int u = 0; u < graph->size; u++)
    {
        results[u] = malloc(sizeof(Result));
        results[u]->dist = malloc(sizeof(int) * graph->size);
        results[u]->prev = malloc(sizeof(int) * graph->size);

        for (int v = 0; v < graph->size; v++)
        {
            repath(u, v, results, m_dist, m_prev);
        }
    }
    free(m_dist);
    free(m_prev);


    return results;
}




// I realised this is actually just Floyd-Warshall.....


/*
    BMF_b_args* a = (BMF_b_args*) args;
    const Graph* graph = a->graph; int** m_dist = a->m_dist;
    Result** results = a->results;

    pthread_mutex_t* q_lock = a->q_lock; pthread_mutex_t* i_lock = a->i_lock;
    pthread_mutex_t* v_locks = a->v_locks;
    // Synchronise iterations. Each dist depends on the last iteration's.
    while (1)
    {
        // check if we've reached V iterations
        pthread_mutex_lock(i_lock);
        if (*a->iter == graph->size)
        {
            pthread_mutex_unlock(i_lock);
            break;
        }
        pthread_mutex_unlock(i_lock);
        ///////////////

        // We haven't, lets get the next node to work on
        pthread_mutex_lock(q_lock);
        int src = *a->next_node;
        if (src >= graph->size)
        {
            // End of iteration, need to increment iter number
            // Need to make sure every node synchronises at this point though, and only one node makes changes
            if (pthread_mutex_trylock(i_lock) == 0)
            {
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
        while (adj != NULL)
        {
            // Lock source and target
            pthread_mutex_lock(v_locks + src);
            pthread_mutex_lock(v_locks + adj->vertex);
            int v = adj->vertex;

            // Perform the relaxation
            if (m_dist[src][v] != INT_MAX && m_dist[src][v] + adj->weight < m_dist[src][v])
            {
                m_dist[src][v] = m_dist[src][v] + adj->weight;
            }

            // Release locks
            pthread_mutex_unlock(v_locks + src);
            pthread_mutex_unlock(v_locks + adj->vertex);

            adj = adj->next; // Continue for all adjacent nodes
        }
    }
 */
