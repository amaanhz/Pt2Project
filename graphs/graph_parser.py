import csv
from pyvis.network import Network
import networkx as nx
import gravis as gv

def reddit_tsv(name, numlines):
    with open("raw/soc-redditHyperlinks-body.tsv", "r") as file:
        reader = csv.reader(file, delimiter='\t')
        reader.__next__() # skip column titles
        for i, row in enumerate(reader):
            if i == numlines:
                break
            vector = row[-1].split(',')
            print(f"{row[0]} -> {row[1]}, Pos: {vector[18]}, Neg: {vector[19]}, Comp: {vector[20]}")


def visualise(name, references=False):
    nx_graph = nx.Graph()
    refs = {}
    if references:
        with open(f"reference/{name}_ref", "r") as file:
            reader = csv.reader(file, delimiter=' ')
            for row in reader:
                refs[int(row[0])] = row[1].strip('"')
    with open(f"{name}", "r") as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            row = [int(x) for x in row]
            nx_graph.add_node(row[0], size=10, label=refs[row[0]] if references else row[0])
            nx_graph.add_node(row[1], size=10, label=refs[row[1]] if references else row[1])
            nx_graph.add_edge(row[0], row[1], weight=row[2])
    fig = gv.vis(nx_graph, graph_height=1000, show_node_label=True, show_edge=True, spring_length=1000,
                 spring_constant=0.5, avoid_overlap=1, node_label_data_source='label', edge_size_factor=0.01)
    fig.display()

def menu():
    resp = 0
    while (resp == 0):
        resp = int(input("1. Parse a graph\n2. Visualise a graph\n3. Exit\n"))
        if 0 < resp < 4:
            return resp

if __name__ == "__main__":
    visualise("USairport_2010", True)
    # while True:
    #     resp = menu()
    #     if resp == 3:
    #         break