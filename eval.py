import os
import sys
from subprocess import CalledProcessError

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
        "color": "deepskyblue"
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

def func1(x, a, b, c):
    return a * x ** 4 + b * x ** 2 + c

def func2(x, a, b, c, d):
    return a * np.exp(- b * x + c) + d
    #return -a * np.log(b * x + c) + d
    #return a * (x - b) ** 4 + c * x ** 2 + d

if __name__ == "__main__":
    graph_path = "graphs/eval_graphs"

    #shutil.rmtree("../" + graph_path)
    #os.mkdir("../" + graph_path)

    #generate_graphs(graph_path, 1004, 1008, 4, lambda x : int(x * (x - 1) / 20))
    reps = 5

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
                      ["djseq", "bmfseq", "fwarshseq"],
                      ["cuda_dj", "cuda_bmf", "cuda_fwarsh"],
                      ["djmt", "cuda_dj"],
                      ["bmfmt", "cuda_bmf"],
                      ["fwarshmt", "cuda_fwarsh"],
                      ["djmt", "fwarshmt"],
                      ["cuda_dj", "cuda_fwarsh"],
                      ["djseq", "fwarshseq"],
                      ["djmt", "bmfmt", "fwarshmt"]]

    # algo_groupings = [["djmt"],
    #                   ["bmfmt"],
    #                   ["fwarshmt"],
    #                   ["djmt", "bmfmt"],
    #                   ["djmt", "fwarshmt"],
    #                   ["bmfmt", "fwarshmt"],
    #                   ["djmt", "bmfmt", "fwarshmt"]]

    # algo_groupings = [["cuda_fwarsh"]]



    run_algos = unique(flatten(algo_groupings))

    for algo in run_algos:
        agg_results[algo] = {
            "errors" : [[], []],
            "vcounts" : [],
            "results" : [],
            "stddev" : []
        }

    bmf_bs = 32
    fwarsh_bs = 10

    for i, f in enumerate(sorted(os.listdir("../" + graph_path), key=lambda x : int(x[9:].replace("_", "")))[0:126:5]):
        vertices = f.split(sep=os.sep)[-1].split(sep="_")[1]
        print(f"{run_algos} Graph #{i}: Testing {vertices} vertices")

        pvals = {}

        for algo in run_algos:
            pvals[algo] = []

        args = ["./Pt2Project", graph_path + "/" + f]
        for algo in run_algos:
            args.append(algo)
            args.append(str(reps))
            if (algo == "djmt" or algo == "fwarshmt" or algo == "bmfmt"):
                args.append("16")
            elif (algo == "cuda_fwarsh"):
                args.append(str(fwarsh_bs))
            elif (algo == "cuda_bmf"):
                args.append(str(bmf_bs))
        print(" ".join(args))
        try :
            out = subprocess.run(args, check=True, stdout=subprocess.PIPE).stdout.decode("ascii")
        except CalledProcessError:
            continue

        filtered = [x.split() for x in out.splitlines() if x != ""]
        it = iter(filtered)
        runs = list(zip(it, it))


        for n, (info, runtime_out) in enumerate(runs):
            if (n % reps != 0):
                algo = info[0]

                point_values = pvals[algo]

                correct = bool(int(info[-1]))
                #print(f"{algo}: {correct}")
                runtime = float(runtime_out[0])
                point_values.append(runtime)

        for algo in run_algos:
            errs = agg_results[algo]["errors"]
            stddev = agg_results[algo]["stddev"]
            vertex_counts = agg_results[algo]["vcounts"]
            results = agg_results[algo]["results"]
            point_values = pvals[algo]

            vertex_counts.append(vertices)
            errs[0].append(min(point_values))
            errs[1].append(max(point_values))
            stddev.append(np.std(point_values, dtype=float))
            results.append(np.mean(point_values))

    plt.style.use("ggplot")

    for n, group in enumerate(algo_groupings):
        for algo in group:
            plt.figure(n)

            plotinfo = algos[algo]
            results = np.array(agg_results[algo]["results"])
            vertex_counts = np.array(agg_results[algo]["vcounts"], dtype=int)

            quint_params, _ = opt.curve_fit(func1, vertex_counts, results)
            #exp_params, _ = opt.curve_fit(func2, vertex_counts, results)

            errs = agg_results[algo]["errors"]
            stddev = agg_results[algo]["stddev"]
            #print(quint_params)
            plt.errorbar(vertex_counts,
                         results,
                         yerr=stddev,
                         fmt="x",
                         color=plotinfo["color"],
                         ecolor=plotinfo["color"],
                         capsize=4,
                         linestyle="none",
                         markevery=1,
                         errorevery=1)
            plt.plot(vertex_counts,
                     func1(vertex_counts, *quint_params),
                     label=plotinfo["name"],
                     color=plotinfo["color"],
                     linewidth=1.0)
            # plt.plot(vertex_counts,
            #         func2(vertex_counts, *exp_params),
            #         label=plotinfo["name"],
            #         color=plotinfo["color"],
            #          linewidth=1.0)

        #plt.axvline(16, color="k", linestyle="--", label="Max Concurrent Threads")

        plt.xlabel("Vertex Count")
        plt.ylabel("Runtime (s)")
        #plt.title(f"Runtime against Thread Count")
        plt.ylim(bottom = 0)
        plt.tight_layout()
        plt.legend()

    plt.show()