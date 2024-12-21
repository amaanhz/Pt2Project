#include "FWarsh.h"

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <pthread.h>
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

void do_blocks(void* args)
{
    FWarsh_args* a = (FWarsh_args*)args;
    int bl = a->block_length;
    int** dist = a->m_dist; int** prev = a->m_prev;

    int B1x = a->B1x * bl; int B1y = a->B1y * bl;
    int B2x = a->B2x * bl; int B2y = a->B2y * bl;
    int B3x = a->B3x * bl; int B3y = a->B3y * bl;

    for (int k = 0; k < a->kmax; k++)
    {
        for (int i = 0; i < a->imax; i++)
        {
            for (int j = 0; j < a->jmax; j++)
            {
                if (dist[B2x + i][B2y + k] != INT_MAX && dist[B3x + k][B3y + j] != INT_MAX)
                {
                    int t = dist[B2x + i][B2y + k] + dist[B3x + k][B3y + j];
                    if (t < dist[B1x + i][B1y + j])
                    {
                        dist[B1x + i][B1y + j] = t;
                        prev[B1x + i][B1y + j] = prev[B3x + k][B3y + j];
                    }
                }
            }
        }
    }
    free(a);
}

FWarsh_args* construct_args(int nb, int r, int l, const int** d, const int** prev, int b1x, int b1y, int b2x,
    int b2y, int b3x, int b3y)
{
    FWarsh_args* args = malloc(sizeof(FWarsh_args));
    *args = (const FWarsh_args){.block_length = l, .m_dist = d, .m_prev = prev, .B1x = b1x, .B1y = b1y,
        .B2x = b2x, .B2y = b2y, .B3x = b3x, .B3y = b3y,};
    nb--;
    args->kmax = b2y == nb || b3x == nb ? r : l; // limit at edges
    args->imax = b1x == nb || b2x == nb ? r : l;
    args->jmax = b1y == nb || b3y == nb ? r : l;
    return args;
}

FWarsh_args_mt* construct_args_mt(int l, const int** d, const int** prev, pthread_mutex_t* dist_lock,
    pthread_mutex_t* prev_lock, pthread_cond_t* dep_cond)
{
    FWarsh_args_mt* args = malloc(sizeof(FWarsh_args_mt));
    *args = (const FWarsh_args_mt){.block_length = l, .m_dist = d, .m_prev = prev, .prev_lock = prev_lock,
        .dist_lock = dist_lock, .dep_cond = dep_cond
    };
    return args;
}

Result** FWarsh_blocking(const Graph* graph, int block_length)
{
    Result** results = malloc(sizeof(Result*) * graph->size);
    int** m_dist = malloc(sizeof(int*) * graph->size);
    int** m_prev = malloc(sizeof(int*) * graph->size);


    m_dist_init(graph, m_dist, m_prev);
    int num_blocks = (graph->size + block_length - 1) / block_length; // ceiling the value
    int rem = graph->size % block_length;
    if (rem == 0) { rem = block_length; }


    for (int b = 0; b < num_blocks; b++)
    {
        void* args = (void*)construct_args(num_blocks, rem, block_length, m_dist, m_prev,
            b, b, b, b, b, b); // Diagonal blocks
        do_blocks(args);
        for (int i = 0; i < num_blocks; i++)
        {
            // Horizontal and vertical blocks
            args = (void*)construct_args(num_blocks, rem, block_length, m_dist, m_prev,
                b, i, b, b, b, i);
            do_blocks(args);

            args = (void*)construct_args(num_blocks, rem, block_length, m_dist, m_prev,
                i, b, i, b, b, b);
            do_blocks(args);
        }
        // Peripheral blocks
        for (int i = 0; i < num_blocks; i++)
        {
            for (int j = 0; j < num_blocks; j++)
            {
                if (i != b && j != b)
                {
                    args = (void*)construct_args(num_blocks, rem, block_length, m_dist, m_prev,
                        i, j, i, b, b, j);
                    do_blocks(args);
                }
            }
        }
    }

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

work_pool* init_work_pool(int nblocks)
{
    work_pool* wp = malloc(sizeof(work_pool));
    *wp = (const work_pool){.items = malloc(sizeof(block_triplet) * nblocks * nblocks),
        .tail = 0};
    return wp;
}

void mt_blocks(const void* args)
{

}

Result** FWarsh_mt(const Graph* graph, int block_length, int numthreads)
{
    pthread_t* threads[numthreads];
    pthread_mutex_t dist_lock; pthread_mutex_init(&dist_lock, NULL);
    pthread_mutex_t prev_lock; pthread_mutex_init(&prev_lock, NULL);

    Result** results = malloc(sizeof(Result*) * graph->size);
    int** m_dist = malloc(sizeof(int*) * graph->size);
    int** m_prev = malloc(sizeof(int*) * graph->size);

    m_dist_init(graph, m_dist, m_prev);
    int num_blocks = (graph->size + block_length - 1) / block_length;
    int rem = graph->size % block_length;
    if (rem == 0) { rem = block_length; }

    work_pool* wp = init_work_pool(num_blocks);
    pthread_cond_t** dep_conds = malloc(sizeof(pthread_cond_t*) * num_blocks);
    FWarsh_args_mt* args = malloc(sizeof(FWarsh_args_mt));
    *args = (FWarsh_args_mt){ .block_length = block_length, .m_dist = m_dist, .m_prev = m_prev,
        .dist_lock = &dist_lock, .prev_lock = &prev_lock, .wp = wp, .dep_conds = dep_conds
    };

    for (int t = 0; t < numthreads; t++)
    {
        pthread_create((pthread_t*)threads + t, NULL, mt_blocks, (void*) args);
    }



    for (int t = 0; t < numthreads; t++)
    {
        pthread_join(threads[t], NULL);
    }


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

    free(wp->items);
    free(wp);
    free(m_dist);
    free(m_prev);
    return results;
}
