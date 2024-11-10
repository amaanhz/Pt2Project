#include <stdio.h>
#include <stdlib.h>
#include "graphparse.h"
#include "DijkstraSeq.h"

int main(int argc, char* argv[]) {
	Graph* graph = fileparse("testgraph");
	printGraph(graph);
	printf("Running DijkstraSSSP:\n");

	DijkstraResult* result = malloc(sizeof(DijkstraResult));
	result = DijkstraSSSP(graph, 0);

	printResult(result);

	free(graph);
	return 0;
}