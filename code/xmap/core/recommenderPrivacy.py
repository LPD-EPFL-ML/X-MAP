# -*- coding: utf-8 -*-
"""use private policy on alterEgo profile."""

from math import log, e

import numpy as np


class RecommenderPrivacy:
    def __init__(self, mapping_range, privacy_epsilon, rpo):
        """Initialize the parameters
        Args:
            mapping_range: The size of private neighborhood.
            privacy_epsilon: The level of privacy.
            rpo: used to control the accuracy of modification.
        """
        self.mapping_range = mapping_range
        self.privacy_epsilon = privacy_epsilon / 2
        self.rpo = rpo

    def find_neighbor(self, dataRDD):
        """For the rating pair of sim_pairs, we switch the position of id,
            and do groupBy operation to obtain each iid's neighbor.
        Args:
            dataRDD: in the form of ((iid1, iid2), [sim, RS])*
        """
        def key_on_first_item(iters):
            for id_pair, info_pair in iters:
                id1, id2 = id_pair
                yield id1, [(id2, info_pair)]
                # yield id2, [(id1, info_pair)]
        return dataRDD.mapPartitions(
            key_on_first_item).reduceByKey(lambda a, b: a + b)

    def prepare_private_selection(self, lines, max_replacement_selection):
        """prepare the procedure of mapping.
        Args:
            lines: in the form of (iid, [sim, sensitivity])*
            max_replacement_selection: max replacement selection of all pairs.
        """
        def get_k_sim(lines):
            """get k'th similarity, where k=mapping_range."""
            return lines[self.mapping_range - 1][1][0] \
                if len(lines) >= self.mapping_range else lines[-1][1][0]

        def split_to_sets(lines, k_sim, w):
            """split original list to two sets based on its similarity."""
            C1 = [line for line in lines if line[1][0] >= k_sim - w]
            C0 = [line for line in lines if line not in C1]
            return [C1, C0]

        def get_w(lines, max_replacement_selection, k_sim):
            """calculate w.
                note that there exists a condition where len(lines) < k.
            """
            count = len(lines)
            return min(
                k_sim,
                (2 * self.mapping_range * max_replacement_selection /
                 self.privacy_epsilon) * log(
                    self.mapping_range * (
                        count - self.mapping_range) / self.rpo, e)) \
                if count > self.mapping_range else k_sim

        k_sim = get_k_sim(lines)
        w = get_w(lines, max_replacement_selection, k_sim)
        splits = split_to_sets(lines, k_sim, w)
        modified_lines = [
            (line[0], [max(line[1][0], line[1][0] - w), line[1][1]])
            for line in lines]
        return splits, modified_lines

    def get_private_neighbor(self, lines):
        """get current items' PRIVATE neighbor user."""
        def decide_num_selection(lines):
            """choose k=`mapping_range` non-duplicate item from pairs,
                and deal with the case that k < # of non-zero in pairs:
                    avoid error in `np.random.choice`
            Args:
                lines: normalized probabilities pairs
            """
            nnz = np.count_nonzero(map(lambda line: line[1], lines))
            return self.mapping_range \
                if nnz >= self.mapping_range else nnz

        def calculate_prob(lines, replacement_selection_list):
            """calculate probability (not normalized)."""
            probs = []
            for line in zip(lines, replacement_selection_list):
                tmp = 1.0 * np.exp(
                    self.privacy_epsilon * line[0][1][0] / (
                        2 * self.mapping_range * line[1][1])
                    )
                probs.append((line[0][0], tmp))
            return probs

        def normalize_prob(probs):
            """normalize probability."""
            sum_prob = sum([prob[1] for prob in probs])
            normalized_prob = [(prob[0], prob[1] / sum_prob) for prob in probs]
            return normalized_prob

        def get_max_replacement_selection(replacement_selection_list):
            """get max replacement_selection (local)."""
            return sorted(
                replacement_selection_list, key=lambda x: - abs(x[1]))[0][1]

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

        lines = sorted(lines, key=lambda x: - abs(x[1][0]))
        replacement_selection_list = [(line[0], line[1][1]) for line in lines]
        max_replacement_selection = get_max_replacement_selection(
            replacement_selection_list)
        splites, modified_lines = self.prepare_private_selection(
            lines, max_replacement_selection)
        probs = calculate_prob(modified_lines, replacement_selection_list)
        normalized_probs = normalize_prob(probs)
        num_selection = decide_num_selection(normalized_probs)
        index_choice = weighted_pick(
            list(map(lambda line: line[1], normalized_probs)),
            num_selection)
        return map(lambda ind: lines[ind], index_choice)

    def private_neighbor_selection(self, rdd):
        """a function to private select neighbor.
        Args:
            rdd:                   (((id1, id2), (sim, RS)), ...)
        """
        return self.find_neighbor(rdd).map(
            lambda line: (line[0], self.get_private_neighbor(line[1])))

    def get_nonprivate_neighbor(self, pairs):
        """Use this function to get current items' priavte neighbor user.
        """
        pairs = sorted(pairs, key=lambda x: - abs(x[1][0]))
        return pairs[: self.mapping_range]

    def nonprivate_neighbor_selection(self, rdd):
        """a function to no-private select neighbor."""
        return self.find_neighbor(rdd).map(
            lambda line: (line[0], self.get_nonprivate_neighbor(line[1])))

    def noise_perturbation(self, rdd):
        """In the perturbation step, we add independent Laplace noise
            to the neighbor selected in previous step.
        Args:
            rdd: (id, [(id1, [sim, local_sensitivity])*])
        """
        def add_laplace_noise(info):
            """Use this function to calculate independent Laplace noise.
            Args:
                info:    (sim, RS)
            """
            return np.random.laplace(0, abs(info[1]) / self.privacy_epsilon)

        def helper(line):
            id, pairs = line
            perturbated_paris = [
                (neigh_id, info[0] + add_laplace_noise(info))
                for neigh_id, info in pairs]
            return id, perturbated_paris
        return rdd.map(helper)

    def nonnoise_perturbation(self, rdd):
        """It is no-noise perturbation step, we simply remove local sentivity.
        Args:
            rdd: (id, [(id1, [sim, local_sensitivity])*])
        """
        def helper(line):
            id, pairs = line
            cleaned_pairs = [(neigh_id, info[0]) for neigh_id, info in pairs]
            return id, cleaned_pairs
        return rdd.map(helper)
