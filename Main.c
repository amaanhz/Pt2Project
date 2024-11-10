#include "graphparse.h"
#include "DijkstraSeq.h"

int main(int argc, char* argv[]) {
	Graph* graph = fileparse("testgraph");

	printGraph(graph);

	free(graph);
	return 0;
}