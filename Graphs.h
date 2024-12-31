#pragma once
#include <unordered_map>

using namespace std;

typedef enum GraphType
{
    adjmatrix,
    adjlist,
    hash
} GraphType;

class GraphMatrix {
    public:
        GraphMatrix(const char* filename);
    private:
        int** matrix;
        int size;
};

class GraphMap {
    public:
        GraphMap(const char* filename);
    private:
        unordered_map<int, int> map;
        int size;
};