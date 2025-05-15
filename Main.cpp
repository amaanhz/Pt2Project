#include <stdio.h>
#include <cstdlib>
#include <random>
#include <time.h>
#include <unistd.h>

#include "BMFord.h"
#include "Dijkstra.h"
#include "FWarsh.h"
#include "GraphParse.h"
#include "GraphMatrix.h"
#include "GraphSearch.h"
#include "CUDA/BMFord.cuh"
#include "CUDA/Dijkstra.cuh"
#include "CUDA/FWarsh.cuh"

int parse_number(char* threads) {
    int total = 0;
    char* c = threads;
    int mult = 1;
    while (*c != '\0') {
        int val = (int)(*c - '0');
        total *= mult;
        total += val;
        mult *= 10;
        c++;
    }
    return total;
}

void run_algo(unordered_map<string, int> algos, const string& algo, Graph* graphLL, GraphMatrix graphMatrix,
        Result** ground_truth, int numthreads=16) {
    int mapped_algo = algos[algo];
    string algo_name = "ERROR!";
    Result** result = NULL;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    switch (mapped_algo) {
        case 0:
            result = DijkstraAPSP(graphLL);
        algo_name = "Dijkstra (Seq)";
        break;
        case 1:
            result = DijkstraAPSP_mt(graphLL, numthreads);
        algo_name = "Dijkstra (MT)";
        break;
        case 2:
            result = BMFordAPSP(graphLL);
        algo_name = "Bellman-Ford (Seq)";
        break;
        case 3:
            result = BMFordAPSP_mt_a(graphLL, numthreads);
        algo_name = "Bellman-Ford (MT)";
        break;
        case 4:
            result = FWarsh(graphLL);
        algo_name = "FWarsh (Seq)";
        break;
        case 5:
            result = FWarsh_blocking(graphLL, 32);
        algo_name = "FWarsh (Blocking, Seq)";
        break;
        case 6:
            result = FWarsh_mt(graphLL, 32, numthreads);
        algo_name = "FWarsh (MT)";
        break;
        case 7:
            result = cuda_DijkstraAPSP(graphMatrix);
        algo_name = "Dijkstra (GPU)";
        break;
        case 8:
            result = cuda_BMFord(graphMatrix, 32);
        algo_name = "Bellman-Ford (GPU)";
        break;
        case 9:
            result = cuda_FWarsh(graphMatrix, 4);
        algo_name = "FWarsh (GPU)";
        break;
        default:
            printf("Error! Unrecognized algorithm.\n");
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_spent = (double)(end.tv_sec - start.tv_sec);
    time_spent += (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    //printResult(result[49], 49, graphLL->size);
    //printResult(ground_truth[49], 49, graphLL->size);

    printf("%s correct: %d\n", algo.c_str(), resultsEq(ground_truth, result, graphLL->size));
    printf("%f\n", time_spent);

    if (result) {
        //freeResults(result, graphLL->size);
    }
}

int main(int argc, char* argv[]) {
    unordered_map<string, int> algos = {{"djseq", 0}, {"djmt", 1}, {"bmfseq", 2}, {"bmfmt", 3},
    {"fwarshseq", 4}, {"fwarshblockseq", 5}, {"fwarshmt", 6}, {"cuda_dj", 7},
    {"cuda_bmf", 8}, {"cuda_fwarsh", 9}};

    const char* graph_path = argv[1];

    auto graphLL = fileparse(graph_path);
    auto graphMatrix = GraphMatrix(graph_path);

    Result** ground_truth = BMFordAPSP(graphLL);

    int numthreads = 16;
    int argn = 2;
    while (argn  < argc) {
        // algo name
        string algo = argv[argn];
        argn++;
        // # of runs
        int nruns = parse_number(argv[argn]);
        argn++;
        if (algo == "djmt" || algo == "bmfmt" || algo == "fwarshmt") {
            numthreads = parse_number(argv[argn]);
            argn++;
        }
        for (int n = 0; n < nruns; n++) {
            run_algo(algos, algo, graphLL, graphMatrix, ground_truth, numthreads);
        }
    }
}
