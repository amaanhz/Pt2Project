#pragma once
#include <unordered_map>

using namespace std;

class GraphMatrix {
    // flattened 2d array
    public:
    GraphMatrix(const char* filename);
    GraphMatrix(const GraphMatrix& graph, int initial=0);
    [[nodiscard]] int GetSize() const; // axis length
    [[nodiscard]] int* GetMatrix();
    int& operator[](int idx);
    void printGraph() const;


    private:
    int* matrix;
    int size;
};
