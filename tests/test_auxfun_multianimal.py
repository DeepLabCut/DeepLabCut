import networkx as nx
import numpy as np
import pytest
from deeplabcut.utils import auxfun_multianimal
from itertools import combinations


def test_prune_paf_graph():
    n_bpts = 10  # This corresponds to 45 edges
    edges = [list(edge) for edge in combinations(range(n_bpts), 2)]
    with pytest.raises(ValueError):
        pruned_edges = auxfun_multianimal.prune_paf_graph(edges, n_bpts - 2)
        pruned_edges = auxfun_multianimal.prune_paf_graph(edges, len(edges))

    for target in range(20, 45, 5):
        pruned_edges = auxfun_multianimal.prune_paf_graph(edges, target)
        assert len(pruned_edges) == target

    for degree in (4, 6, 8):
        pruned_edges = auxfun_multianimal.prune_paf_graph(
            edges, average_degree=degree,
        )
        G = nx.Graph(pruned_edges)
        assert np.mean(list(dict(G.degree).values())) == degree
