#include <stdio.h>
#include <cstdlib>
#include <random>
#include "GraphMatrix.h"
#include "GraphSearch.h"
#include "CUDA/Dijkstra.cuh"


int main(int argc, char* argv[]) {
    //GraphSearch("graphs/USairport500");
    auto graph = GraphMatrix("graphs/USairport500");

    Result** results = cuda_DijkstraAPSP(graph);

    printf("Done\n");
}
