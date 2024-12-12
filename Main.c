#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "BMFord.h"
#include "GraphParse.h"
#include "Dijkstra.h"

int main(int argc, char* argv[]) {
	Graph* graph = fileparse("USairport500");
	//printGraph(graph);
	//printf("Running DijkstraSSSP:\n");

	// SSSP test //
	//DijkstraResult* result = malloc(sizeof(DijkstraResult));
	//result = DijkstraSSSP(graph, 0);
	//printResult(result, 0, graph->size);
	///////////////

	// APSP test //

	Result** results = NULL;

	struct timespec start, end;

	clock_gettime(CLOCK_MONOTONIC, &start);
	DijkstraAPSP(graph);
	clock_gettime(CLOCK_MONOTONIC, &end);

	double time_spent = (end.tv_sec - start.tv_sec);
	time_spent += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
	printf("Runtime for sequential: %f\n", time_spent);

	clock_gettime(CLOCK_MONOTONIC, &start);
	results = DijkstraAPSP_mt(graph, 16);
	clock_gettime(CLOCK_MONOTONIC, &end);

	//printResult(results[1], 1, graph->size);

	time_spent = (end.tv_sec - start.tv_sec);
	time_spent += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
	printf("Runtime for MT: %f\n", time_spent);

	Result* result = NULL;

	clock_gettime(CLOCK_MONOTONIC, &start);
	result = BMFordSSSP(graph, 1);
	clock_gettime(CLOCK_MONOTONIC, &end);

	time_spent = (end.tv_sec - start.tv_sec);
	time_spent += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
	printf("Runtime for BMF_SSSP: %f\n", time_spent);

	//printResult(result, 1, graph->size);
	printf("\n");

	printf("Results for Dijkstra and BMFord are %s", resultsEq(result, results[1], graph->size) ? "equal" : "non-equal");
	/* for (int i = 0; i < graph->size; i++) {
		printf("Result for node %d:\n", i);
		printResult(results[i], i, graph->size);
	} */

	///////////////

	// Freeing Memory so compiler stops shouting //
	for (int i = 0; i < graph->size; i++)
	{
		if (results)
		{
			free(results[i]->dist);
			free(results[i]->prev);
			free(results[i]);
		}
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