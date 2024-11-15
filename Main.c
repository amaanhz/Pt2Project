#include <stdio.h>
#include <stdlib.h>
#include "graphparse.h"
#include "DijkstraSeq.h"

int main(int argc, char* argv[]) {
	Graph* graph = fileparse("testgraph");
	printGraph(graph);
	printf("Running DijkstraSSSP:\n");

	// SSSP test //
	//DijkstraResult* result = malloc(sizeof(DijkstraResult));
	//result = DijkstraSSSP(graph, 0);
	//printResult(result, 0, graph->size);
	///////////////

	// APSP test //
	
	DijkstraResult** results = DijkstraAPSP(graph);
	for (int i = 0; i < graph->size; i++) {
		printf("Result for node %d:\n", i);
		printResult(results[i], i, graph->size);
	}

	///////////////

	free(graph); free(results);
	return 0;
}