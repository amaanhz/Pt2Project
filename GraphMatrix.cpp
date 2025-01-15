#include <limits.h>
#include <algorithm>
#include "GraphMatrix.h"
#include "GraphParse.h"

GraphMatrix::GraphMatrix(const char* filename) {
    Graph* gtemp = fileparse(filename);
    size = gtemp->size;
    matrix = new int*[size];

    for (int i = 0; i < size; i++) {
        matrix[i] = new int[size];
        for (int j = 0; j < size; j++) {
            matrix[i][j] = INT_MAX;
        }
        matrix[i][i] = 0;
        Node* n = gtemp->verts[i];
        while (n != NULL) {
            matrix[i][n->vertex] = n->weight;
            n = n->next;
        }
    }
}

GraphMatrix::GraphMatrix(const GraphMatrix& graph, int initial) {
    size = graph.GetSize();
    matrix = new int*[size];
    for (int i = 0; i < size; i++) {
        matrix[i] = new int[size];
        fill_n(matrix[i], size, initial);
    }
}

int GraphMatrix::GetSize() const {
    return size;
}

int& GraphMatrix::operator[](int idx) {
    return *matrix[idx];
}
