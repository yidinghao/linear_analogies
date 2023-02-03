import itertools
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from _utils import timer

k = 10


def index_data(vectors_dict: Dict[str, Any]) -> Dict[Tuple[str, str], int]:
    shapes = vectors_dict["shapes"]
    textures = vectors_dict["textures"]
    return {(s, t): i for i, (s, t) in enumerate(zip(shapes, textures))}


def get_analogy_tests(shapes: List[str], textures: List[str],
                      data_index: Dict[Tuple[str, str], int]) -> \
        Tuple[List[int], List[int], List[int], List[int]]:
    tests = [(s1, s2, t1, t2) for s1, s2, t1, t2
             in itertools.product(shapes, shapes, textures, textures)
             if s1 != s2 and t1 != t2 and (s1, t1) in data_index and
             (s1, t2) in data_index and (s2, t1) in data_index and
             (s2, t2) in data_index]

    w1s = [data_index[(s, t)] for s, _, t, _ in tests]
    w2s = [data_index[(s, t)] for _, s, t, _ in tests]
    w3s = [data_index[(s, t)] for _, s, _, t in tests]
    w4s = [data_index[(s, t)] for s, _, _, t in tests]

    return w1s, w2s, w3s, w4s


if __name__ == "__main__":
    with timer("Loading vectors from pickle..."):
        with open("vectors.p", "rb") as f:
            vector_data = pickle.load(f)

    with timer("Indexing data..."):
        index = index_data(vector_data)

    vectors = vector_data["vectors"]
    all_shapes = sorted(list(set(vector_data["shapes"])))
    all_textures = sorted(list(set(vector_data["textures"])))

    with timer("Getting analogy tests..."):
        w1s, w2s, w3s, w4s = get_analogy_tests(all_shapes, all_textures, index)

    with timer("Computing scene algebra..."):
        w4_hats = vectors[w1s] - vectors[w2s] + vectors[w3s]

    with timer("Computing cosine similarities..."):
        cos = cosine_distances(w4_hats, vectors)

    with timer("Sorting..."):
        sorted_ = np.argpartition(cos, k)[:, :k]

    n_correct = sum(w in sorted_[i] for i, w in enumerate(w4s))
    print(n_correct / len(w4s))
