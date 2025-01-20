#pragma once
#include "GraphParse.h"
#include "GraphMatrix.h"


Result** cuda_DijkstraAPSP(GraphMatrix& graph);

int fastmin(int* arr, int* queues, int size);