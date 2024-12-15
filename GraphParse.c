#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include "GraphParse.h"

int neighbour(const Graph* graph, int u, int v)
{
	Node* n = graph->verts[u];
	if (n) {
		while (n->next) {
			if (n->vertex == v) {
				break;
			}
			n = n->next;
		}
		return n->vertex == v ? n->weight : 0;
	}
	return 0;
	
}

void addEdge(const Graph* graph, int i, int j, int w) {
	Node* n = graph->verts[i];
	Node* nxt = malloc(sizeof(Node)); // initialise new node, it's gonna go somewhere
	nxt->vertex = j; nxt->weight = w; nxt->next = NULL;
	if (n) { // traverse to tail
		while (n->next) {
			n = n->next;
		}
		n->next = nxt;
	}
	else { // otherwise, n is nullptr
		graph->verts[i] = nxt;
	}
}

Graph* fileparse(const char* file) {
	Graph* graph = malloc(sizeof(Graph));
	graph->size = 0;
	graph->verts = calloc(1, sizeof(Node**));

	char path[strlen(file) + 3];
	strcpy(path, "../"); strcat(path, file);

	FILE* fp = fopen(path, "r");

	if (fp) {
		int line = 0;
		while (!feof(fp)) {
			++line;
			//printf("Line: %d\n", line);
			// parse each edge
			int edge[3]; int part = 0;
			while (part < 3) {
				int c = fgetc(fp);
				if (c == ' ' || c == '\n') {
					continue;
				}
				else {
					int v = c - '0';
					c = fgetc(fp);
					while (c != ' ' && c != '\n' && !feof(fp))
					{
						v = 10 * v + c - '0';
						c = fgetc(fp);
					}
					edge[part] = v;
					if (part != 2) {
						int oldsize = graph->size; // save old size
						if (v + 1 > graph->size) {
							graph->verts = realloc(graph->verts, sizeof(Node**) * (v + 1)); // new maximum node number found
							memset(&(graph->verts[oldsize]), 0, sizeof(Node*) * ((v + 1) - oldsize)); // set new pointers to null (no recalloc !)
							graph->size = v + 1;
						}
					}
					part++;
				}
			}
			//printf("Adding edge %d -> %d (%d) \n\n", edge[0], edge[1], edge[2]);
			addEdge(graph, edge[0], edge[1], edge[2]);
		}
	}
	fclose(fp);
	return graph;
}

void printGraph(const Graph* graph) {
	for (int i = 0; i < graph->size; i++) {
		printf("%d -> {", i);
		Node* n = graph->verts[i];
		while (n) {
			if (n->next) {
				printf("%d: %d, ", n->vertex, n->weight);
			}
			else {
				printf("%d: %d", n->vertex, n->weight);
			}
			n = n->next;
		}
		printf("}\n");
	}
	printf("\n");
}

void printResult(const Result* result, int src, int size) {
	for (int i = 0; i < size; i++) {
		if (i != src) {
			if (result->dist[i] >= 0 && result->dist[i] < INT_MAX) {
				printf("Distance to %d: %d, ", i, result->dist[i]);
				printf("Path: ");
				int v = result->prev[i];
				printf("%d <- ", i);
				while (v != src && v != i) {
					printf("%d <- ", v);
					v = result->prev[v];
				}
				printf("%d\n", v);
			}
			else {
				printf("%d is unreachable.\n", i);
			}
		}
		else {
			printf("%d is the source node.\n", i);
		}
	}
	printf("\n");
}

void printResults(const Result** results, int size)
{
	for (int i = 0; i < size; i++) {
		printf("Result for node %d:\n", i);
		printResult(results[i], i, size);
	}
}

int resultEq(const Result* r1, const Result* r2, int size)
{
	for (int i = 0; i < size; i++)
	{
		if (r1->dist[i] != r2->dist[i]) // Don't care about exact route, as long as distance is the same
		{
			printf("Inequality between distance/routes for node %d\n", i);
			return 0;
		}
	}
	return 1;
}

int resultsEq(const Result** r1, const Result** r2, int size)
{
	for (int i = 0; i < size; i++)
	{
		if (!resultEq(r1[i], r2[i], size))
		{
			printf("Unequal results for APSP at node %d\n\n", i);
			return 0;
		}
	}
	return 1;
}

void freeGraph(Graph* graph)
{
	for (int i = 0; i < graph->size; i++)
	{
		Node* n = graph->verts[i];
		if (n)
		{
			Node* list[graph->size]; int x = 0;
			while (n->next)
			{
				list[x] = n;
				n = n->next;
				x++;
			}
			list[x] = n; x++;
			for (int y = 0; y < x; y++) { free(list[y]); }
		}
	}
	free(graph->verts);
	free(graph);
}

void freeResults(Result** results, int size)
{
	for (int n = 0; n < size; n++)
	{
		Result* result = results[n];
		free(result->dist);
		free(result->prev);
		free(result);
	}
	free(results);
}
