#pragma once

typedef enum GraphType
{
    adjmatrix,
    adjlist,
    hash
} GraphType;

class GraphExt {
    public:
        GraphExt(const char* filename, GraphType graphType);
        GraphType type;

    private:
    int** matrix;
    int size;
    void ParseFile(const char* filename, GraphType type);
};
