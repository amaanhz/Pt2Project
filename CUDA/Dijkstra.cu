#include <stdio.h>
#include <limits.h>
#include <queue>
#include <cuda_runtime.h>
#include "Dijkstra.cuh"
#include "GraphParse.h"
#include "GraphMatrix.h"

using namespace std;

Result** cuda_DijkstraAPSP(const GraphMatrix& graph) {
    Result** results = new Result*[graph.GetSize()];
    queue<int> q;

    GraphMatrix dist = GraphMatrix(graph, INT_MAX);
    GraphMatrix prev = GraphMatrix(graph, -1);



    return results;
}
