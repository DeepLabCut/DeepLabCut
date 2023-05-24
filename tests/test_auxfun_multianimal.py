#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import networkx as nx
import numpy as np
import pandas as pd
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
            edges,
            average_degree=degree,
        )
        G = nx.Graph(pruned_edges)
        assert np.mean(list(dict(G.degree).values())) == degree


def test_reorder_individuals_in_df():
    import random

    # Load sample multi animal data
    df = pd.read_hdf("tests/data/montblanc_tracks.h5")
    individuals = df.columns.get_level_values("individuals").unique().to_list()

    # Generate a random permutation and reorder data. Ignore the unique bodypart
    permutation_indices = random.sample(
        range(len(individuals[:-1])), k=len(individuals[:-1])
    )
    permutation = [individuals[i] for i in permutation_indices]
    permutation.append("single")
    df_reordered = auxfun_multianimal.reorder_individuals_in_df(df, permutation)

    # Get inverse permutation and reorder the modified data to get back
    # to the original
    inverse_permutation_indices = np.argsort(permutation_indices).tolist()
    inverse_permutation = [individuals[i] for i in inverse_permutation_indices]
    inverse_permutation.append("single")
    df_inverse_reordering = auxfun_multianimal.reorder_individuals_in_df(
        df_reordered, inverse_permutation
    )

    # Check
    pd.testing.assert_frame_equal(df, df_inverse_reordering)
