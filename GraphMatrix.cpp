#include <climits>
#include <algorithm>
#include "GraphMatrix.h"

#include <cstdio>

#include "GraphParse.h"

GraphMatrix::GraphMatrix(const char* filename) {
    // instantiate with original distances
    Graph* gtemp = fileparse(filename);
    size = gtemp->size;
    matrix = new int[size*size]; // flattened 2d array

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[size * i + j] = INT_MAX;
        }
        matrix[size * i + i] = 0;
        Node* n = gtemp->verts[i];
        while (n != nullptr) {
            matrix[size * i + n->vertex] = n->weight;
            n = n->next;
        }
    }
    freeGraph(gtemp);
}

GraphMatrix::GraphMatrix(const GraphMatrix& graph, int initial) {
    size = graph.GetSize();
    matrix = new int[size * size];
    for (int i = 0; i < size; i++) {
        fill_n(matrix + size * i, size, initial);
    }
}

int GraphMatrix::GetSize() const {
    return size;
}

int* GraphMatrix::GetMatrix() {
    return matrix;
}

int& GraphMatrix::operator[](int idx) {
    return matrix[idx];
}

void GraphMatrix::printGraph() const {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (matrix[size * i + j] != INT_MAX) {
                printf("Graph[%d][%d] = %d\n", i, j, matrix[size * i + j]);
            }
        }
    }
    printf("\n");
}