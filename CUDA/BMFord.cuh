#pragma once
#include "GraphMatrix.h"
#include "GraphParse.h"

Result** cuda_BMFord(GraphMatrix& graph, int block_length=32);
