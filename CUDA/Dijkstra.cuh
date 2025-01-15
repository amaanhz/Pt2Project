#pragma once
#include "GraphParse.h"
#include "GraphMatrix.h"


Result** cuda_DijkstraAPSP(const GraphMatrix& graph);

int fastmin(int* arr, int size);