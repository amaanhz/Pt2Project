#include <stdio.h>
#include <cstdlib>
#include <random>
#include <time.h>
#include "FWarsh.h"
#include "GraphParse.h"
#include "GraphMatrix.h"
#include "GraphSearch.h"
#include "CUDA/Dijkstra.cuh"
#include "CUDA/FWarsh.cuh"


int main(int argc, char* argv[]) {

    const char* graph_path = "graphs/USairport_2010";

    struct timespec start, end;
    GraphSearch("graphs/USairport500");
    auto graph = GraphMatrix(graph_path);
    //graph.printGraph();
    Result** ground_truth = FWarsh_mt(fileparse(graph_path), 10, 16);

    clock_gettime(CLOCK_MONOTONIC, &start);
    //Result** results = cuda_DijkstraAPSP(graph);
    Result** results = cuda_FWarsh(graph, 1024);
    clock_gettime(CLOCK_MONOTONIC, &end);

    //int test[13] = {-2, 1, 3, 3, 3, -9, -3, -1, 10, 11, 12,  2, 0};
    //int mask[13] = {1, 0, 1, 1, 1, 1, 1,  1,  1,  0,  1,  1, 1};
    //fastmin(test, mask, 13);

    printResults(results, graph.GetSize());
    double time_spent = (end.tv_sec - start.tv_sec);
    time_spent += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("\nRuntime for Dijkstra_APSP (GPU): %f\n", time_spent);
    //printResult(results[498], 498, graph.GetSize());

    //printResult(ground_truth[7], 7, graph.GetSize());
    //printResult(results[7], 7, graph.GetSize());

    printf("Results for GPU_Dijkstra and Floyd-Warshall are %s\n",
           resultsEq(ground_truth, results, graph.GetSize()) ? "equal" : "non-equal");

    printf("Done\n");
}
