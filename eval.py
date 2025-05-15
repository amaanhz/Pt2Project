import os
import sys
import networkx as nx
import random
import subprocess
import shutil
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from networkx.utils import flatten
from numpy import unique

#def test_dijkstra()

algos = {
    "djseq" : {
        "name" : "Dijkstra's (Sequential)",
        "color" : "sienna"
    },
    "djmt" : {
        "name" : "Dijkstra's (Multi-Threaded)",
        "color" : "orange"
    },
    "bmfseq" : {
        "name": "Bellman-Ford (Sequential)",
        "color": "blue"
    },
    "bmfmt" : {
        "name" : "Bellman-Ford (Multi-Threaded)",
        "color": "cyan"
    },
    "fwarshseq" : {
        "name": "Floyd-Warshall (Sequential)",
        "color": "salmon"
    },
    "fwarshblockseq" : {
        "name" : "Floyd-Warshall (Sequential, Blocking)",
        "color": "rosybrown"
    },
    "fwarshmt": {
        "name" : "Floyd-Warshall (Multi-threaded)",
        "color": "darkred"
    },
    "cuda_dj" : {
        "name" : "Dijkstra's (GPU)",
        "color" : "forestgreen"
    },
    "cuda_bmf" : {
       "name" : "Bellman-Ford (GPU)",
       "color" : "lime"
    },
    "cuda_fwarsh" : {
        "name" : "Floyd-Warshall (GPU)",
        "color" : "springgreen"
    }
}

def generate_graphs(graph_path, min_nodes, max_nodes, interval, edgeformula):
    for i in range(min_nodes, max_nodes, interval):
        print(f"Generating |V| = {i}")
        edgenum = edgeformula(i)
        G = nx.gnm_random_graph(i, edgenum, directed=True)
        #print(f"G has {len(G.edges())} edges")
        path = f"../{graph_path}/testeval_{i}_{edgenum}"
        with open(path, "w") as f:
            f.seek(0)
            edges = G.edges()
            for n, (u, v) in enumerate(edges):
                f.write(f"{u} {v} {random.randint(1, 50)}")
                if n < len(edges) - 1:
                    f.write("\n")

def bmfseqfunc(x, a, b, c):
    return a * x ** 4 + b * x ** 2 + c

if __name__ == "__main__":
    graph_path = "graphs/eval_sparse_graphs"

    #shutil.rmtree("../" + graph_path)
    #os.mkdir("../" + graph_path)

    #generate_graphs(graph_path, 1004, 1008, 4, lambda x : int(x * (x - 1) / 20))
    reps = 4

    agg_results = {}

    algo_groupings = [["djseq"],
                      ["djseq", "djmt"],
                      ["djseq", "djmt", "cuda_dj"],
                      ["bmfseq"],
                      ["bmfseq", "bmfmt"],
                      ["bmfseq", "bmfmt", "cuda_bmf"],
                      ["fwarshseq", "fwarshblockseq"],
                      ["fwarshseq", "fwarshblockseq", "fwarshmt"],
                      ["fwarshseq", "fwarshmt", "cuda_fwarsh"],
                      ["cuda_dj", "cuda_bmf", "cuda_fwarsh"],
                      ["djmt", "cuda_dj"],
                      ["bmfmt", "cuda_bmf"],
                      ["fwarshmt", "cuda_fwarsh"],
                      ["djmt", "fwarshmt"],
                      ["cuda_dj", "cuda_fwarsh"],
                      ["djseq", "fwarshseq"]]

    # algo_groupings = [["djmt", "bmfmt", "fwarshmt"],
    #                   ["djmt", "fwarshmt"],
    #                   ["djmt", "cuda_dj"],
    #                   ["bmfmt", "cuda_bmf"],
    #                   ["fwarshmt", "cuda_fwarsh"],
    #                   ["cuda_dj", "cuda_bmf", "cuda_fwarsh"]]



    run_algos = unique(flatten(algo_groupings))

    for algo in run_algos:
        agg_results[algo] = {
            "errors" : [[], []],
            "vcounts" : [],
            "results" : [],
        }

    for i, f in enumerate(sorted(os.listdir("../" + graph_path), key=lambda x : int(x[9:].replace("_", "")))[0:125:5]):
        pvals = {}

        for algo in run_algos:
            pvals[algo] = []

        vertices = f.split(sep=os.sep)[-1].split(sep="_")[1]
        print(f"{run_algos} Graph #{i}: Testing {vertices} vertices");


        args = ["./Pt2Project", graph_path + "/" + f]
        for algo in run_algos:
            args.append(algo)
            args.append(str(reps))
            if (algo == "djmt" or algo == "fwarshmt" or algo == "bmfmt"):
                args.append("16")
        print(" ".join(args))
        out = subprocess.run(args, check=True, stdout=subprocess.PIPE).stdout.decode("ascii")

        filtered = [x.split() for x in out.splitlines() if x != ""]
        it = iter(filtered)
        runs = list(zip(it, it))


        for n, (info, runtime_out) in enumerate(runs):
            if (n % reps != 0):
                algo = info[0]

                point_values = pvals[algo]

                correct = bool(int(info[-1]))
                runtime = float(runtime_out[0])
                point_values.append(runtime)

        for algo in run_algos:
            errs = agg_results[algo]["errors"]
            vertex_counts = agg_results[algo]["vcounts"]
            results = agg_results[algo]["results"]
            point_values = pvals[algo]

            vertex_counts.append(vertices)
            errs[0].append(min(point_values))
            errs[1].append(max(point_values))
            results.append(np.mean(point_values))


    for n, group in enumerate(algo_groupings):
        for algo in group:
            plt.figure(n)

            plotinfo = algos[algo]
            results = np.array(agg_results[algo]["results"])
            vertex_counts = np.array(agg_results[algo]["vcounts"], dtype=int)
            params, _ = opt.curve_fit(bmfseqfunc, vertex_counts, results)
            errs = agg_results[algo]["errors"]
            print(params)
            plt.errorbar(vertex_counts,
                         results,
                         yerr=errs,
                         fmt="x",
                         color=plotinfo["color"],
                         ecolor=plotinfo["color"],
                         capsize=4,
                         linestyle="none",
                         errorevery=1)
            plt.plot(vertex_counts,
                     bmfseqfunc(vertex_counts, *params),
                     label=plotinfo["name"],
                     color=plotinfo["color"])

        plt.xlabel("Vertex Count")
        plt.ylabel("Runtime (s)")
        plt.title(f"Runtime against # of vertices")
        plt.ylim(bottom = 0)
        plt.legend()

    plt.show()
