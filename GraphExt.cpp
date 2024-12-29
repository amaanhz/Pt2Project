#include <limits.h>

#include "GraphExt.h"
#include "GraphParse.h"

GraphExt::GraphExt(const char* filename, GraphType graphType)
{
    Graph* gtemp = fileparse(filename);

    switch (graphType)
    {
        case adjmatrix:
            {
                type = graphType;
                matrix = new int*[gtemp->size];

                for (int i = 0; i < gtemp->size; i++)
                {
                    for (int j = 0; j < gtemp->size; j++)
                    {
                        matrix[i][j] = INT_MAX;
                    }
                    Node* n = gtemp->verts[i];
                    while (n != NULL)
                    {
                        matrix[i][n->vertex] = n->weight;
                    }
                }
                break;
            }
        case adjlist:

        default:
        break;
    }

}

void GraphExt::ParseFile(const char* filename, GraphType type)
{

}
