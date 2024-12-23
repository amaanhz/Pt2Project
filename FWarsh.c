#include "FWarsh.h"

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <pthread.h>
#include <time.h>
#include "GraphParse.h"

struct timespec start, end;

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

void do_blocks(const void* args)
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

index* point(int x, int y)
{
    index* p = malloc(sizeof(index));
    *p = (const index){ .x = x, .y = y};
    return p;
}

work_pool* init_work_pool(int nblocks)
{
    work_pool* wp = malloc(sizeof(work_pool));
    *wp = (const work_pool){.items = malloc(sizeof(block_triplet) * nblocks * nblocks * nblocks),
        .head = 0, .tail = 0, .max = nblocks * nblocks * nblocks, .empty = 1};

    // We already know what will be in the queue
    for (int i = 0; i < nblocks; i++)
    {
        // Dependent
        index* diag = point(i, i);
        wp_insert(wp, diag, diag, diag);

        // Partially dependent
        for (int j = 0; j < nblocks; j++)
        {
            if (i != j)
            {
                index* pd1 = point(i, j);
                wp_insert(wp, pd1, diag, pd1);

                index* pd2 = point(j, i);
                wp_insert(wp, pd2, pd2, diag);
            }
        }

        // Independent (Peripheral)
        for (int x = 0; x < nblocks; x++)
        {
            for (int y = 0; y < nblocks; y++)
            {
                if (x != i && y != i)
                {
                    index* from = point(x, y);
                    index* to = point(x, i);
                    index* thru = point(i, y);

                    wp_insert(wp, from, to, thru);
                }
            }
        }
    }


    return wp;
}

void wp_insert(work_pool* wp, index* b1, index* b2, index* b3)
{
    block_triplet* triplet = malloc(sizeof(block_triplet));
    *triplet = (block_triplet){.b1 = b1, .b2 = b2, .b3 = b3};
    wp_insert_trip(wp, triplet);
}

void wp_insert_trip(work_pool* wp, block_triplet* triplet)
{
    if (wp->tail < wp->max)
    {
        wp->items[wp->tail] = triplet;
        wp->tail++;
    }
    else
    {
        printf("Queue is full!\n");
    }
    wp->empty = 0;
}

block_triplet* wp_pop(work_pool* wp)
{
    if (wp->tail != wp->head)
    {
        block_triplet* item = wp->items[wp->head];
        wp->head++;
        if (wp->head == wp->tail) { wp->empty = 1; }
        return item;
    }
    printf("Queue is empty!\n");
    return NULL;
}

void mt_blocks(block_triplet* triplet, int bl, int** dist, int** prev, int kmax, int imax, int jmax,
    pthread_mutex_t* dist_lock, pthread_mutex_t* prev_lock)
{
    index* b1 = triplet->b1; index* b2 = triplet->b2; index* b3 = triplet->b3;

    int B1x = b1->x * bl; int B1y = b1->y * bl;
    int B2x = b2->x * bl; int B2y = b2->y * bl;
    int B3x = b3->x * bl; int B3y = b3->y * bl;

    for (int k = 0; k < kmax; k++)
    {
        for (int i = 0; i < imax; i++)
        {
            for (int j = 0; j < jmax; j++)
            {
                //pthread_mutex_lock(dist_lock);
                if (dist[B2x + i][B2y + k] != INT_MAX && dist[B3x + k][B3y + j] != INT_MAX)
                {
                    int t = dist[B2x + i][B2y + k] + dist[B3x + k][B3y + j];
                    if (t < dist[B1x + i][B1y + j])
                    {
                        dist[B1x + i][B1y + j] = t;
                        //pthread_mutex_lock(prev_lock);
                        prev[B1x + i][B1y + j] = prev[B3x + k][B3y + j];
                        //pthread_mutex_unlock(prev_lock);
                    }
                }
                //pthread_mutex_unlock(dist_lock);
            }
        }
    }
}


void FWarsh_t(const void* args)
{
    FWarsh_args_mt* a = (FWarsh_args_mt*)args;
    int nb = a->num_blocks; int l = a->block_length; int r = a->rem; int* deps = a->deps;
    int** dist = a->m_dist; int** prev = a->m_prev; work_pool* wp = a->wp; pthread_mutex_t* wp_lock = a->wp_lock;
    pthread_mutex_t* dist_lock = a->dist_lock; pthread_mutex_t* prev_lock = a->prev_lock;
    pthread_mutex_t* dep_locks = a->dep_locks; pthread_cond_t* dep_conds = a->dep_conds;

    int total_blocks = nb * nb;

    block_triplet* blocks;
    while (1)
    {
        // Get next block to work
        pthread_mutex_lock(wp_lock);
        if (!wp->empty)
        {
            blocks = wp_pop(wp);
            pthread_mutex_unlock(wp_lock);
            index* b1 = blocks->b1; index* b2 = blocks->b2; index* b3 = blocks->b3;

            int kmax = b2->y == nb - 1 || b3->x == nb - 1 ? r : l; // limit at edges
            int imax = b1->x == nb - 1 || b2->x == nb - 1 ? r : l;
            int jmax = b1->y == nb - 1 || b3->y == nb - 1 ? r : l;


            // Dependent
            if (b1->x == b1->y && b1->x == b2->x && b1->x == b3->x)
            {
                clock_gettime(CLOCK_MONOTONIC, &start);
                if (b1->x > 0) // need to check that the last phase completed before doing this one
                {
                    pthread_mutex_lock(&dep_locks[b1->x - 1]);
                    if (deps[b1->x - 1] < total_blocks)
                    {
                        while (1)
                        {
                            pthread_cond_wait(&dep_conds[b1->x - 1], &dep_locks[b1->x - 1]);
                            if (deps[b1->x - 1] == total_blocks)
                            {
                                pthread_mutex_unlock(&dep_locks[b1->x - 1]);
                                break;
                            }
                        }
                    }
                }
                clock_gettime(CLOCK_MONOTONIC, &end);
                double time_spent = (end.tv_sec - start.tv_sec);
                time_spent += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
                printf("Took %f seconds waiting at a dependent block (%d, %d)\n", time_spent, b1->x, b1->y);

                mt_blocks(blocks, l, dist, prev, kmax, imax, jmax, dist_lock, prev_lock);

                pthread_mutex_lock(&dep_locks[b1->x]);
                deps[b1->x]++;
                pthread_mutex_unlock(&dep_locks[b1->x]);

                pthread_cond_broadcast(&dep_conds[b1->x]);
            }

            // Partially Dependent
            else if (b2->x == b2->y || b3->x == b3->y)
            {
                clock_gettime(CLOCK_MONOTONIC, &start);
                index* diag = b2->x == b2->y ? b2 : b3; // find out which dep block this relates to
                pthread_mutex_lock(&dep_locks[diag->x]);
                if (deps[diag->x] == 0) // Dependent block hasn't been calculated yet
                {
                    while (1)
                    {
                        pthread_cond_wait(&dep_conds[diag->x], &dep_locks[diag->x]);
                        if (deps[diag->x] > 0) { break; }
                    }
                }
                pthread_mutex_unlock(&dep_locks[diag->x]);
                clock_gettime(CLOCK_MONOTONIC, &end);
                double time_spent = (end.tv_sec - start.tv_sec);
                time_spent += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
                printf("Took %f seconds waiting at a partially dependent block: (%d, %d) for diag (%d, %d)\n", time_spent,
                    b1->x, b1->y, diag->x, diag->y);

                mt_blocks(blocks, l, dist, prev, kmax, imax, jmax, dist_lock, prev_lock);

                pthread_mutex_lock(&dep_locks[diag->x]);
                deps[diag->x]++;
                pthread_mutex_unlock(&dep_locks[diag->x]);

                pthread_cond_broadcast(&dep_conds[diag->x]); // Inform any independents the value has changed

            }

            else
            {

                index* diag = point(b2->y, b2->y);
                clock_gettime(CLOCK_MONOTONIC, &start);
                pthread_mutex_lock(&dep_locks[diag->x]);
                if (deps[diag->x] < nb * 2 - 1) // Partially dependent hasn't finished yet
                {
                    while (1)
                    {
                        pthread_cond_wait(&dep_conds[diag->x], &dep_locks[diag->x]);
                        if (deps[diag->x] >= nb * 2 - 1) { break; }
                    }
                }
                pthread_mutex_unlock(&dep_locks[diag->x]);
                clock_gettime(CLOCK_MONOTONIC, &end);
                double time_spent = (end.tv_sec - start.tv_sec);
                time_spent += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
                printf("Took %f seconds waiting at an independent block: (%d, %d) for diag (%d, %d)\n", time_spent, b1->x,
                    b1->y, diag->x, diag->y);

                mt_blocks(blocks, l, dist, prev, kmax, imax, jmax, dist_lock, prev_lock);

                pthread_mutex_lock(&dep_locks[diag->x]);
                deps[diag->x]++;
                int t = deps[diag->x] == total_blocks;
                pthread_mutex_unlock(&dep_locks[diag->x]);

                pthread_cond_broadcast(&dep_conds[diag->x]);
                free(diag);
            }
        }
        else
        {
            pthread_mutex_unlock(wp_lock);
            break;
        }
    }
    pthread_exit(NULL);
}

Result** FWarsh_mt(const Graph* graph, int block_length, int numthreads)
{
    pthread_t* threads[numthreads];
    pthread_mutex_t wp_lock; pthread_mutex_init(&wp_lock, NULL);
    pthread_mutex_t dist_lock; pthread_mutex_init(&dist_lock, NULL);
    pthread_mutex_t prev_lock; pthread_mutex_init(&prev_lock, NULL);



    Result** results = malloc(sizeof(Result*) * graph->size);
    int** m_dist = malloc(sizeof(int*) * graph->size);
    int** m_prev = malloc(sizeof(int*) * graph->size);

    m_dist_init(graph, m_dist, m_prev);
    int num_blocks = (graph->size + block_length - 1) / block_length;
    int rem = graph->size % block_length;
    if (rem == 0) { rem = block_length; }

    int* deps = malloc(sizeof(int) * num_blocks); // diagonal length = side length
    pthread_mutex_t* dep_locks = malloc(sizeof(pthread_mutex_t) * num_blocks);

    // for each dep block [i, i], dep[i] == 1 when dep is done, then dep[i] == nblocks * 2 - 1 when
    // partially dependent blocks done

    pthread_cond_t* conds = malloc(sizeof(pthread_cond_t) * num_blocks);
    for (int i = 0; i < num_blocks; i++)
    {
        deps[i] = 0;
        pthread_cond_init(conds + i, NULL);
        pthread_mutex_init(dep_locks + i, NULL);
    }

    work_pool* wp = init_work_pool(num_blocks);
    FWarsh_args_mt* args = malloc(sizeof(FWarsh_args_mt));
    *args = (const FWarsh_args_mt){ .block_length = block_length, .num_blocks = num_blocks, .rem = rem, .deps = deps,
        .m_dist = m_dist, .m_prev = m_prev, .wp_lock = &wp_lock, .dist_lock = &dist_lock, .prev_lock = &prev_lock,
        .wp = wp, .dep_locks = dep_locks, .dep_conds = conds
    };

    for (int t = 0; t < numthreads; t++)
    {
        pthread_create((pthread_t*)threads + t, NULL, FWarsh_t, (void*) args);
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
