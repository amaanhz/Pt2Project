#include <stdio.h>
#include <stdlib.h>
#include "graphparse.h"
#include "DijkstraSeq.h"

int main(int argc, char* argv[]) {
	Graph* graph = fileparse("testgraph");
	printGraph(graph);
	//printf("Running DijkstraSSSP:\n");

	// SSSP test //
	//DijkstraResult* result = malloc(sizeof(DijkstraResult));
	//result = DijkstraSSSP(graph, 0);
	//printResult(result, 0, graph->size);
	///////////////

	// APSP test //
	
	DijkstraResult** results = DijkstraAPSP_mt(graph);
	for (int i = 0; i < graph->size; i++) {
		printf("Result for node %d:\n", i);
		printResult(results[i], i, graph->size);
	}

	///////////////

	// Freeing Memory so compiler stops shouting //
	for (int i = 0; i < graph->size; i++)
	{
		free(results[i]->dist);
		free(results[i]->prev);
		free(results[i]);

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
	free(graph); free(results);

	///////////////////////////////////////////////

	return 0;
}