#include <stdio.h>
#include <cstdlib>
#include <random>
#include "GraphMatrix.h"
#include "GraphSearch.h"
#include "CUDA/Dijkstra.cuh"


int main(int argc, char* argv[]) {
    //GraphSearch("graphs/USairport500");
    auto graph = GraphMatrix("graphs/testgraph");
    //graph.printGraph();
    Result** results = cuda_DijkstraAPSP(graph);

    //int test[13] = {4, 1, 3, 2, 3, 4, 7, 9, 10, 11, 12, 13, 0};
    //int mask[13] = {1, 0, 1, 1, 1, 0, 1, 0,  1,  0,  1,  1, 0};
    //fastmin(test, mask, 13);

    //printResults(results, graph.GetSize());
    printf("Done\n");
}
