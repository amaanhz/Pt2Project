#pragma once
#include "GraphParse.h"
#include <pthread.h>

typedef struct index
{
    int x;
    int y;
} index;

typedef struct block_triplet
{
    index* b1;
    index* b2;
    index* b3;
} block_triplet;

typedef struct work_pool // This doesn't need to be a circular queue...
{
    block_triplet** items;
    int head;
    int tail;
    int max;
    int empty;
} work_pool;

typedef struct FWarsh_args_mt
{
    int block_length;
    int num_blocks;
    int rem;
    int* deps;
    int** m_dist;
    int** m_prev;
    work_pool* wp;
    pthread_mutex_t* wp_lock;
    pthread_mutex_t* dist_lock;
    pthread_mutex_t* prev_lock;
    pthread_mutex_t** dep_locks;
    pthread_cond_t** dep_conds;
} FWarsh_args_mt;

typedef struct FWarsh_args
{
    int block_length;
    int** m_dist;
    int** m_prev;
    int B1x; int B1y; // from
    int B2x; int B2y; // to
    int B3x; int B3y; // through
    int kmax; int imax; int jmax;
} FWarsh_args;

void m_dist_init(const Graph* graph, int** m_dist, int** m_prev);

void repath(int u, int v, Result** results, const int** m_dist, const int** m_prev);
Result** FWarsh(const Graph* graph);

void do_blocks(const void* args);
FWarsh_args* construct_args(int nb, int r, int l, const int** d, const int** prev, int b1x, int b1y, int b2x, int b2y,
    int b3x, int b3y);
Result** FWarsh_blocking(const Graph* graph, int block_length);

index* point(int x, int y);
work_pool* init_work_pool(int nblocks);
void wp_insert(work_pool* wp, index* b1, index* b2, index* b3);
void wp_insert_trip(work_pool* wp, block_triplet* triplet);
block_triplet* wp_pop(work_pool* wp);

void mt_blocks(block_triplet* triplet, int bl, int** dist, int** prev, int kmax, int imax, int jmax,
    pthread_mutex_t* dist_lock, pthread_mutex_t* prev_lock);
void FWarsh_t(const void* args);
Result** FWarsh_mt(const Graph* graph, int block_length, int numthreads);