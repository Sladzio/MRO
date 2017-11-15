from collections import defaultdict

import networkx as nx
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import euclidean, pdist, correlation
import numpy as np
from sklearn.cluster import SpectralClustering


def plot_linkage(graph, linkage, title):
    figure = plt.figure(figsize=(10, 10))
    figure.suptitle(title, fontsize=14, fontweight='bold')

    # Plotting dendogram on top of the image
    dendrogram_subplot = figure.add_subplot('211')
    den = dendrogram(linkage, ax=dendrogram_subplot)
    clusters = get_classes_by_color(den)
    vertex_colors = {int(member): color for color, members in clusters.items() for member in members}
    graph_subplot = figure.add_subplot('212')
    graph_subplot.axis('off')
    colors = [vertex_colors[node] for node in graph.nodes()]
    nx.draw_spring(graph, with_labels=True, ax=graph_subplot,
                   node_color=colors, cmap=plt.get_cmap('Set1'))
    figure.tight_layout()
    figure.subplots_adjust(top=0.9)
    plt.show()


def get_classes_by_color(den, label='ivl'):
    cluster_idxs = defaultdict(list)
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))
    cluster_classes = {}
    for c, l in cluster_idxs.items():
        i_l = [den[label][i] for i in l]
        cluster_classes[c] = i_l
    return cluster_classes


# We have to remap nodes so that it matches the rest of graphs
def karate():
    karate_graph = nx.read_gml('karate/karate.gml')
    nodes_mapping = {n: int(n) - 1 for n in karate_graph.nodes()}
    nx.relabel_nodes(karate_graph, nodes_mapping, copy=False)
    return karate_graph


def compute_linkage(graph, metric, method):
    adjacency = nx.adjacency_matrix(graph).todense()
    if metric is 'shortest_path':
        a = np.zeros((len(adjacency), len(adjacency)))
        np.fill_diagonal(a, 1)
        metric = lambda u, v: nx.shortest_path_length(graph, np.argmax(u), np.argmax(v))
    else:
        a = adjacency
    return linkage(a, method=method, metric=metric)


def main():
    methods = ['single', 'complete', 'average']
    metrics = ['euclidean', 'correlation', 'shortest_path']
    graphs = [karate(), nx.read_gml('dolphins/dolphins.gml'), nx.read_gml('football/football.gml')]
    graph_names = ["Zachary’s karate club", "Dolphin social network", "American College football"]
    id = 1
    for graph, graph_name in zip(graphs, graph_names):
        for metric in metrics:
            for method in methods:
                title = 'Id - {id}, Graph - {graph}, Method - {method}, Metric - {metric}' \
                    .format(id=id,
                            graph=graph_name.capitalize(),
                            method=method.capitalize(),
                            metric=metric.capitalize())
                res = compute_linkage(graph, metric, method)

                id += 1
                plot_linkage(graph, res, title)


if __name__ == '__main__':
    main()
