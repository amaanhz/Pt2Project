#pragma once
#include "GraphMatrix.h"
#include "GraphParse.h"

struct Vec2 {
    int x, y;
    Vec2(int x, int y);
};

struct Triple {
    Vec2 p1, p2, p3;
    Triple(Vec2 p1, Vec2 p2, Vec2 p3);
};

void invoke_blocks(Vec2 b1, Vec2 b2, Vec2 b3, int rem, int* dev_dist, int* dev_prev);

Result** cuda_FWarsh(GraphMatrix& graph, int block_length);