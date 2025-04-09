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
    const char* graph_path = "graphs/USairport500";

    struct timespec start, end;
    //GraphSearch("graphs/USairport500");
    auto graph = GraphMatrix(graph_path);
    //graph.printGraph();
    clock_gettime(CLOCK_MONOTONIC, &start);
    Result** ground_truth = FWarsh_mt(fileparse(graph_path), 10, 8);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_spent = (end.tv_sec - start.tv_sec);
    time_spent += (end.tv_nsec - start.tv_nsec) / 1000000000.0;


    printf("\nRuntime for FWarsh (CPU): %f\n", time_spent);
    //printResults(ground_truth, graph.GetSize());

    size_t free, totalmem;
    int bl = 4;
    for (int bl = 1; bl <= 32; bl++) {
        printf("Trying block length = %d", bl);

        struct timespec start_cuda, end_cuda;

        clock_gettime(CLOCK_MONOTONIC, &start_cuda);
        //Result** results = cuda_DijkstraAPSP(graph);
        Result** results = cuda_FWarsh(graph, bl);
        clock_gettime(CLOCK_MONOTONIC, &end_cuda);

        //int test[13] = {-2, 1, 3, 3, 3, -9, -3, -1, 10, 11, 12,  2, 0};
        //int mask[13] = {1, 0, 1, 1, 1, 1, 1,  1,  1,  0,  1,  1, 1};
        //fastmin(test, mask, 13);

        //printResults(results, graph.GetSize());
        double time_cuda = (end_cuda.tv_sec - start_cuda.tv_sec);
        time_cuda += (end_cuda.tv_nsec - start_cuda.tv_nsec) / 1000000000.0;
        printf("\nRuntime for FWarsh (GPU): %f\n", time_cuda);
        //printResult(ground_truth[1], 1, graph.GetSize());
        //printResult(results[1], 1, graph.GetSize());

        //printResult(ground_truth[7], 7, graph.GetSize());
        //printResult(results[7], 7, graph.GetSize());

        printf("Results for GPU_Fwarsh and CPU_FWarsh are %s\n",
               resultsEq(ground_truth, results, graph.GetSize()) ? "equal" : "non-equal");
        cudaMemGetInfo(&free, &totalmem);
        printf("Memory Available: %ld/%ld\n", free, totalmem);
    }

    printf("Done\n");
}
