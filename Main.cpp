#include <stdio.h>
#include <cstdlib>
#include "GraphMatrix.h"
#include "GraphSearch.h"
#include "CUDA/Dijkstra.cuh"


int main(int argc, char* argv[]) {
    srand(time(0));
    //GraphSearch("graphs/USairport500");
    GraphMatrix graph = GraphMatrix("graphs/USairport500");
    printf("Done\n");
    int test[3500];
    for (int i = 0; i < 3500; i++) {
        test[i] = rand();
    }
    printf("Result: %d\n", fastmin(test, 3500));
}
