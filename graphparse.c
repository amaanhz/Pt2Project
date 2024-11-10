#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "graphparse.h"

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

	FILE* fp;
	errno_t err = fopen_s(&fp, "testgraph", "r");


	if (fp) {
		while (!feof(fp)) {
			// parse each edge
			int edge[3]; int part = 0;
			while (part < 3) {
				char c = fgetc(fp);
				if (c == ' ') {
					continue;
				}
				else {
					int v = c - '0';
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
			addEdge(graph, edge[0], edge[1], edge[2]);
			fgetc(fp); // discard \n or force into EOF
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
}