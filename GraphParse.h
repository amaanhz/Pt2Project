#pragma once
#include <pthread.h>

typedef struct Node {
	int vertex;
	int weight;
	struct Node* next;
} Node;

typedef struct Graph {
	int size;
	Node** verts;
} Graph;

typedef struct Result {
	int* dist;
	int* prev;
} Result;

typedef struct MultiSSSPArgs {
	int* next_node;
	pthread_mutex_t* q_lock;
	pthread_mutex_t* r_lock;
	const Graph* graph;
	Result** results;
} MultiSSSPArgs;

int neighbour(const Graph* graph, int u, int v);
void addEdge(const Graph* graph, int i, int j, int w);
Graph* fileparse(const char* file);


void printGraph(const Graph* graph);
void printResult(const Result* result, int src, int size);
void printResults(const Result** results, int size);

int resultEq(const Result* r1, const Result* r2, int size);
int resultsEq(const Result** r1, const Result** r2, int size);

void freeGraph(Graph* graph);
void freeResults(Result** result, int size);