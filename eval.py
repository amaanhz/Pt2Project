import os
import sys
import networkx as nx
import random
import subprocess
import shutil


if __name__ == "__main__":
    result_collect = {
        "Dijkstra_APSP": {
            "Seq": 0,
            "MT" : 0,
            "GPU": 0
        },
        "Bellman-Ford_APSP": {
            "Seq": 0,
            "MT": 0,
            "GPU": 0
        },
        "FWarsh": {
            "Seq": 0,
            "Seq-Blocking": 0,
            "MT": 0,
            "GPU": 0
        }
    }

    shutil.rmtree("../graphs/eval_graphs")
    os.mkdir("../graphs/eval_graphs")

    for i in range(2, 100, 3):
        print(f"Testing |V| = {i}")
        edgenum = int(i * (i - 1) / 3)
        G = nx.gnm_random_graph(i, edgenum, directed=True)
        #print(f"G has {len(G.edges())} edges")
        path = f"../graphs/eval_graphs/testeval_{i}_{edgenum}"
        with open(path, "w") as f:
            f.seek(0)
            edges = G.edges()
            for n, (u, v) in enumerate(edges):
                f.write(f"{u} {v} {random.randint(0, 50)}")
                if n < len(edges) - 1:
                    f.write("\n")

        out = subprocess.run(["./Pt2Project", path[1:]],
                             check=True, stdout=subprocess.PIPE).stdout.decode("ascii")
        #filtered = [x.split() for x in out.splitlines() if x != ""]
        print(out)