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

	Result** d_results = NULL;

	struct timespec start, end;

	clock_gettime(CLOCK_MONOTONIC, &start);
	DijkstraAPSP(graph);
	clock_gettime(CLOCK_MONOTONIC, &end);

	double time_spent = (end.tv_sec - start.tv_sec);
	time_spent += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
	printf("Runtime for Dijkstra_APSP (Seq): %f\n", time_spent);

	clock_gettime(CLOCK_MONOTONIC, &start);
	d_results = DijkstraAPSP_mt(graph, 16);
	clock_gettime(CLOCK_MONOTONIC, &end);

	//printResult(results[1], 1, graph->size);

	time_spent = (end.tv_sec - start.tv_sec);
	time_spent += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
	printf("Runtime for Dijkstra_APSP (MT): %f\n", time_spent);

	Result** b_results = NULL;

	clock_gettime(CLOCK_MONOTONIC, &start);
	b_results = BMFordAPSP(graph);
	clock_gettime(CLOCK_MONOTONIC, &end);

	time_spent = (end.tv_sec - start.tv_sec);
	time_spent += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
	printf("Runtime for Bellman-Ford_APSP (Seq): %f\n", time_spent);

	clock_gettime(CLOCK_MONOTONIC, &start);
	b_results = BMFordAPSP_mt_a(graph, 16);
	clock_gettime(CLOCK_MONOTONIC, &end);

	time_spent = (end.tv_sec - start.tv_sec);
	time_spent += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
	printf("Runtime for Bellman-Ford_APSP (MT): %f\n", time_spent);



	//printf("\n");
	//printResult(d_results[14], 14, graph->size);
	//printResult(b_results[14], 14, graph->size);


	printf("Results for Dijkstra and BMFord are %s", resultsEq(d_results, b_results, graph->size) ? "equal" : "non-equal");


	///////////////

	// Freeing Memory so compiler stops shouting //

	freeResults(b_results, graph->size);
	freeResults(d_results, graph->size);
	freeGraph(graph);


	///////////////////////////////////////////////

	return 0;
}