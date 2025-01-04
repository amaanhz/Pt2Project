#pragma once
#include <unordered_map>

using namespace std;

class GraphMatrix {
    public:
    GraphMatrix(const char* filename);
    GraphMatrix(const GraphMatrix& graph, int initial=0);
    int GetSize() const;

    private:
    int** matrix;
    int size;
};
