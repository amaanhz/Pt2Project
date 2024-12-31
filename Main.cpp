#include <stdio.h>

#include "Graphs.h"
#include "GraphSearch.h"
#include "CUDA/Dijkstra.cuh"


int main(int argc, char *argv[])
{
    //GraphSearch("graphs/USairport500");
    GraphMatrix graph = GraphMatrix("graphs/USairport500");
    printf("Done");
}