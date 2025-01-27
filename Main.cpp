#include <stdio.h>
#include <cstdlib>
#include <random>
#include <time.h>
#include "GraphMatrix.h"
#include "GraphSearch.h"
#include "CUDA/Dijkstra.cuh"


int main(int argc, char* argv[]) {
    struct timespec start, end;
    //GraphSearch("graphs/USairport500");
    auto graph = GraphMatrix("graphs/testgraph");
    //graph.printGraph();

    clock_gettime(CLOCK_MONOTONIC, &start);
    Result** results = cuda_DijkstraAPSP(graph);
    clock_gettime(CLOCK_MONOTONIC, &end);

    //int test[13] = {-2, 1, 3, 3, 3, -9, -3, -1, 10, 11, 12,  2, 0};
    //int mask[13] = {1, 0, 1, 1, 1, 1, 1,  1,  1,  0,  1,  1, 1};
    //fastmin(test, mask, 13);

    //printResults(results, graph.GetSize());
    double time_spent = (end.tv_sec - start.tv_sec);
    time_spent += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("\nRuntime for Dijkstra_APSP (GPU): %f\n", time_spent);
    //printResult(results[498], 498, graph.GetSize());
    printf("Done\n");
}
