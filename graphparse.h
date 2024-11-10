#pragma once

typedef struct Node {
	int vertex;
	int weight;
	struct Node* next;
} Node;

typedef struct Graph {
	int size;
	Node** verts;
} Graph;

int neighbour(const Graph* graph, int u, int v);
void addEdge(const Graph* graph, int i, int j, int w);
Graph* fileparse(const char* file);
void printGraph(Graph* graph);
