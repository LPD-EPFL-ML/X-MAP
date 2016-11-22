# -*- coding: utf-8 -*-
"""private mapping item in target domain to source domain."""
import numpy as np


class Generator:
    def __init__(self, mapping_range, privacy_epsilon, sim_method, rpo):
        """Initialize the parameters
        Args:
            num_mapping_range: the size of neighborhood used for mapping.
            privacy_epsilon: the level of privacy.
            sim_method: the method used to calculate item2item similarity.
            rpo: used to control the accuracy of modification.
        """
        self.mapping_range = mapping_range
        self.privacy_epsilon = privacy_epsilon
        self.sim_method = sim_method
        self.rpo = rpo

    def global_sentivity(self):
        """Return the value of global sentivity based on its methd:
            The global sensitivity of cosine similarity is 1 - 0 = 1
            The global sensitivity of adjuested cosine is 1 - (-1) = 2
        """
        return 1 if self.sim_method == "cosine" else 2

    def cross_private_mapping(self, rdd):
        """get current items' mapping item based on its neighborhood."""
        def helper(iter_items):
            def decide_num_selection(lines):
                """choose k=`mapping_range` non-duplicate item from pairs,
                    and deal with the case that k < # of non-zero in pairs:
                        avoid error in `np.random.choice`
                Args:
                    k: number of neighbor that we want to select.
                    pairs: normalized probabilities pairs
                """
                nnz = np.count_nonzero(map(lambda line: line[1], lines))
                return self.mapping_range \
                    if nnz >= self.mapping_range else nnz

            def calculate_prob(lines, replacement_selection_list):
                """calculate probability (not normalized)."""
                probs = []
                for line in zip(lines, replacement_selection_list):
                    tmp = 1.0 * np.exp(
                        self.privacy_epsilon * line[0][1] / (
                            2 * self.mapping_range * line[1][1]))
                    probs.append((line[0][0], tmp))
                return probs

            def normalize_prob(prob_pairs):
                """normalize probability."""
                sum_prob = sum([pair[1] for pair in prob_pairs])
                normalized_prob_pairs = [
                    (pair[0], pair[1] / sum_prob) for pair in prob_pairs]
                return normalized_prob_pairs

            def weighted_pick(weights, n_picks):
                """weighted random selection
                returns:
                    n_picks random indexes.
                        the chance to pick the index i
                        is give by the weight weights[i].
                """
                if type(weights) is not list:
                    return 0
                t = np.cumsum(weights)
                s = np.sum(weights)
                st = list(set(np.searchsorted(t, np.random.rand(n_picks) * s)))
                left = n_picks - len(st)
                new = []
                if left != 0:
                    new += weighted_pick(weights, left)
                return st + new

            for iid, lines in iter_items:
                lines = sorted(lines, key=lambda x: - abs(x[1]))[: 10]
                replacement_selection_list = [
                    (line[0], self.global_sentivity()) for line in lines]
                prob_pairs = calculate_prob(lines, replacement_selection_list)
                normalized_prob_pairs = normalize_prob(prob_pairs)
                num_selection = decide_num_selection(normalized_prob_pairs)
                index_choice = weighted_pick(
                    map(lambda line: line[1], normalized_prob_pairs),
                    num_selection)
                choice = np.take(
                    list(map(lambda l: l[0], normalized_prob_pairs)),
                    index_choice)
                yield iid, choice
        return rdd.mapPartitions(helper)

    def cross_nonprivate_mapping(self, rdd, topn=4):
        """get current items' mapping item based on its neighborhood.
        arg:
            topn: random select one item among top `topn` item.
                In non-private version, we first sort similarity,
                and then select top `topn` items, then randomly select one item
        """
        def helper(iter_items):
            for iid, lines in iter_items:
                pairs = sorted(lines, key=lambda x: - abs(x[1]))[: topn]
                yield iid, pairs[np.random.randint(0, len(pairs) - 1)][0]
        return rdd.mapPartitions(helper)

    def mapping_item(self, line, mapping_dict):
        """Use traini.
        Args:
            line: in the form of (uid, iid, rating, rating time).
            mapping_dict: {source item: target item}.
            mapping_key: it contains all keys of mapping.
        """
        return (line[0], mapping_dict[line[1]], line[2], line[3]) \
            if line[1] in mapping_dict else None

    def build_alterEgo(self, trainRDD, mapping_dict):
        """use this function to build alterEgo profile.
        Args:
            rdd: training dataset, contains source/target domain item,
                in the form of `iid, rating, rating time.`
            mapping_dict: `{target item: source item}`.
        Returns:
            alterEgoProfile in target domain.
                It has structure as follows: (uid, iid, rating, time)*
        """
        dataRDD = trainRDD.flatMap(
            lambda line: [(line[0], l[0], l[1], l[2]) for l in line[1]])
        alterEgo_profile = dataRDD.map(
            lambda line: self.mapping_item(line, mapping_dict)).filter(
            lambda line: line is not None)
        return dataRDD.filter(
            lambda line: "T:" in line[1]).union(alterEgo_profile)